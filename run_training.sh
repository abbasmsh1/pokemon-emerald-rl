#!/bin/bash
# Launcher for the systemd user service. Picks up where the last run stopped.
#
# This exists because training has been lost three separate ways: OOM-killed
# twice, and killed by a machine reboot twice more with nothing to bring it
# back. systemd handles the restarting; this script handles choosing the right
# checkpoint and refusing to start into conditions that will just fail again.

set -u
cd "/home/abbas/Documents/Projects/Pokemon Agent" || exit 1

# Never run from cached bytecode. A one-digit edit to env.py restored within
# the same second leaves a .pyc whose mtime and size still match, and a run
# started seconds later trained with LOITER_DISTINCT = 0 while the file said 6.
export PYTHONDONTWRITEBYTECODE=1
rm -rf __pycache__

# NatureCNN, not resnet18. The resnet lineage ran 82 million steps with random
# conv weights under ImageNet BatchNorm parameters: all 20 conv tensors differed
# from torchvision's by up to 1.03 while every BatchNorm tensor matched exactly,
# which drove the pooled features to magnitude 72 and a pairwise cosine
# similarity of 0.9991 across completely different screens. The policy was
# blind. A feature extractor cannot be swapped by resuming, so this run starts
# fresh in its own directory. The old zips stay exactly where they are, in
# checkpoints/ itself, which this glob no longer looks at.
# nature2, not nature: the state vector grew from 17 values to 29 so the agent
# can see its own last action and how far along each penalty counter it is. That
# changes the observation space, which a resume cannot cross, so the run starts
# fresh in a new directory. checkpoints/nature keeps the 17-value lineage.
# nature3: the observation gained a "text" key (16 hashed word ids of the
# on-screen text) and the extractor gained a learned text encoder and an
# inverse-dynamics head (extractor.py). New observation space, new policy.
# checkpoints/nature2 keeps the 54M-step lineage that learned the door pump.
# nature4: the observation gained "tiles" (an 11x15 window of tile classes
# read from the map grid) and "battle" (HP, PP, level ratio, move nature from
# the battle Pokemon array). New observation space, new policy; nature3 keeps
# the previous lineage.
CKPT_DIR=checkpoints/nature4
ARGS="--steps 100000000 --backbone agent --batch-size 128 --max-steps 0 --coverage-every 250000 --checkpoint-dir $CKPT_DIR"
MIN_FREE_MB=3000
LOG=service.log

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Orphaned workers outlive a killed parent and hold several GB. systemd's
# KillMode should prevent this, but a SIGKILLed parent can still leave them.
pkill -9 -f "[t]rain.py --steps" 2>/dev/null && sleep 5

# Wait for memory rather than starting into an OOM. systemd would restart us in
# a loop otherwise, and each attempt would make the pressure worse.
for _ in $(seq 1 30); do
    AVAIL=$(free -m | awk '/^Mem:/ {print $7}')
    [ "$AVAIL" -ge "$MIN_FREE_MB" ] && break
    log "only ${AVAIL}MB free, waiting for ${MIN_FREE_MB}MB"
    sleep 60
done

# Newest VALID checkpoint. A crash mid-write leaves a truncated or 0-byte zip,
# and plain `ls -t | head -1` hands that to --resume, PPO.load raises
# BadZipFile, and Restart=always loops on the same broken file forever. Seen for
# real: emerald_56700000_steps.zip was 0 bytes after a run died mid-checkpoint.
LATEST=""
for f in $(ls -t "$CKPT_DIR"/*_steps.zip 2>/dev/null); do
    if [ -s "$f" ] && .venv/bin/python -c "import zipfile; zipfile.ZipFile('$f')" 2>/dev/null; then
        LATEST="$f"
        break
    fi
    log "skipping unreadable checkpoint $f"
done

if [ -n "$LATEST" ]; then
    log "starting, resuming from $LATEST"
    exec .venv/bin/python train.py $ARGS --resume "$LATEST"
else
    log "starting fresh, no checkpoint found"
    exec .venv/bin/python train.py $ARGS
fi
