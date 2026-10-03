#!/bin/bash
# Launcher for the systemd user service. Picks up where the last run stopped.
#
# This exists because training has been lost three separate ways: OOM-killed
# twice, and killed by a machine reboot twice more with nothing to bring it
# back. systemd handles the restarting; this script handles choosing the right
# checkpoint and refusing to start into conditions that will just fail again.

set -u
cd "/home/abbas/Documents/Projects/Pokemon Agent" || exit 1

ARGS="--steps 100000000 --backbone resnet18 --batch-size 128 --max-steps 65536 --coverage-every 250000"
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

LATEST=$(ls -t checkpoints/*_steps.zip 2>/dev/null | head -1)

if [ -n "$LATEST" ]; then
    log "starting, resuming from $LATEST"
    exec .venv/bin/python train.py $ARGS --resume "$LATEST"
else
    log "starting fresh, no checkpoint found"
    exec .venv/bin/python train.py $ARGS
fi
