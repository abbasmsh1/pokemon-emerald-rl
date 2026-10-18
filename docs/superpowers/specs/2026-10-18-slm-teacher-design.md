# SLM teacher: a language model plays, the vision model and the policy learn from it

Date: 2026-10-18. Status: approved by Abbas 2026-10-18 ("approved, build it"); builder-check answers recorded in the plan's Global Constraints. Supersedes the 2026-10-04 "SLM
text advisor" design, whose latency premise (0.3 s per answer) was measured
today to be twenty times too pessimistic.

## Intent

Hand the main controls to a small language model that can read the game:
the on-screen text, the tile classes around the player, the battle state and
the natures of its moves, all of which the environment now exposes exactly.
Its play is the training signal for two students: the vision model (the
screen CNN with its inverse-dynamics and forward heads) and the policy,
which Abbas calls the RLHF model: fitted to imitate the teacher's choices,
with PPO continuing on top so it can exceed the teacher. The teacher keeps
playing throughout training (Abbas, 2026-10-18: "keep playing throughout"),
so the demonstration record grows and the students track it continuously
rather than being bootstrapped once.

Why: 54 million steps of reward-only PPO never learned to read a dialogue
box, leave a town on purpose or fight a battle on purpose. A 0.5B instruct
model does all three on the first prompt, badly but legibly ("Wild ZIGZAGOON
appeared!" at low HP: it answered A). Imitation gives the students a
competent prior to improve from instead of a random one.

## Measured facts this design rests on

- Qwen2.5-0.5B-Instruct, fp16 on the 4 GB card, beside the running
  8-worker training: 37 ms per decision for a 115-token prompt and a 4-token
  answer, 1.01 GB VRAM peak. On the CPU: 599 ms.
- The environment exposes, per step: `screen_text` (exact, from the text
  printer), `tiles` (11x15 classes), `battle` (12 values including move
  nature), `state` (29 values), and the reward ledger per term.
- Training runs 8 workers at ~260 steps/s. One extra emulator for the
  teacher costs one core.

## Goals

1. A teacher process that runs one environment with the SLM choosing every
   action from a prompt built from the exposed state, at 15+ decisions per
   second on the GPU, and appends every (observation, action, reward) to a
   demonstration store on disk.
2. The teacher's decision is always one of the 8 actions: the model is
   scored, not free-generated. One forward pass gives the logits of each
   action's first token; the argmax (or a temperature sample) is the action.
   No parsing, no invalid answers, and faster than generating.
3. The students train on the store continuously. `train.py` gains a callback
   that, after each PPO update, samples a batch of demonstrations and
   (a) minimises cross-entropy between the policy's action distribution and
   the teacher's action on those observations, and (b) runs the existing
   inverse-dynamics and forward losses on the demonstration screen pairs, so
   the vision model learns from competent play too. Gradients flow into the
   shared extractor exactly as the existing callback's do.
4. The imitation weight is small and constant, so PPO's own reward can
   overrule the teacher where the teacher is wrong. The store is a ring of
   the most recent N transitions, so the students track the teacher's recent
   play.
5. Resumable: no observation change, no network-shape change. The nature4
   run continues; the callback simply starts contributing.
6. Stage 0 gate, before any student code: the teacher alone, measured. Its
   reward per 1,000 steps, encounters, wins, distinct tiles and maps over
   20,000 steps, against the same numbers for a random policy and for the
   current nature4 checkpoint. If the teacher does not beat random on reward
   per 1,000 steps and distinct tiles, its demonstrations are worthless and
   the prompt is reworked before anything is built on it.
7. The viewer can show the teacher playing, with its prompt and the scores
   it gave each action on the HUD.

## Non-goals

- Running the SLM inside the 8 training workers' loop. It plays its own
  environment; the workers consume its record.
- Fine-tuning the SLM. It is frozen; only the prompt changes.
- A learned reward model or SLM-as-judge (RLAIF). If the imitation stage
  works, that is a separate design.
- OCR, or any image input to the SLM. It reads the exact state.
- Changing PPO's hyperparameters, the observation space or the extractor's
  shape.

## Anti-goals

- A teacher that is worse than random (Stage 0 exists to catch it).
- An imitation weight large enough that the student cannot leave the
  teacher's mistakes behind. Measured: PPO's own reward per 1,000 steps on
  nature4 must not fall after the callback starts.
- The teacher starving training: GPU memory above the ~1.9 GB currently
  free, or more than one CPU core.
- A store that grows without bound on disk, or that the workers block on.
  Append-only shards, ring-pruned, read by the trainer between rollouts.

## Design

### 1. Prompt (`teacher.py`, `build_prompt(s, obs, recent) -> str`)

Deterministic, short, under ~200 tokens. From the exposed state:

- The screen text, verbatim, or "no text".
- The tile window summarised rather than dumped: the class of the four
  adjacent tiles ("north: grass, south: tree, east: path, west: door"), the
  direction and distance of the nearest tall grass, door, NPC and water
  within the window, and which edges of the window are off the map.
- Battle, when on: enemy HP and own HP as percentages, each move with its
  name resolved from the move id (the ROM table holds no names; a small
  dict of the first ~50 move ids is enough for Routes 101-104 and the first
  gym, with "move N" as the fallback), PP, and D/S for damaging/status.
- Own situation: party level, HP fraction, money, the last three actions,
  and whether the last press moved the player.
- Navigation the student never sees (builder-check answer): the map's name
  and the player's position, the compass direction of the nearest section
  never visited on this map (the frontier computation the env already runs),
  and a one-line sequence of towns and routes to Rustboro in the goal. The
  student observes only the game state, so no route is scripted into the
  policy; the teacher is a demonstration source and may know the way.
- The instruction: the goal (reach Rustboro and win its gym badge), the 8
  buttons with one-line meanings, and "answer with the button only".

The chat template is applied once; only the user turn changes per step.

### 2. Decision (`teacher.py`, `choose_action(model, tok, prompt) -> (action, scores)`)

One forward pass on the prompt. The next-token logits are read at the
first token of each action word ("up", "down", "left", "right", "A", "B",
"start", "none"), giving 8 scores. Action = argmax, or a sample at
temperature 0.7 (the builder picks the default and the viewer shows both).
Ties and an all-low distribution are still valid actions. Target: under
25 ms per decision on the GPU.

### 3. Teacher loop (`teacher.py`, script)

One `EmeraldEnv` (same anchor as the workers), one model on the GPU in fp16.
Per step: build the prompt from the previous step's state and observation,
decide, `env.step`, record. Every K steps, flush a shard
`demos/<timestamp>_<n>.npz` holding `screen`, `state`, `text`, `tiles`,
`battle`, `action`, `reward`, `episode_start` arrays. A `demos/` ring keeps
the newest N shards (default: 50 shards of 2,048 steps, ~100k transitions)
and deletes older ones. The loop logs its own reward ledger per 1,000 steps
to stdout in the same shape as training's, so the two can be compared.

Runs as its own systemd user service, `pokemon-teacher.service`, with the
same restart rules as training.

### 4. Student callback (`train.py`, `ImitationCallback`)

After each PPO update (`_on_rollout_end`, like the inverse-dynamics
callback): load the newest shards' index lazily, sample B transitions, move
them to the device, and take a few optimiser steps on
`IMITATION_WEIGHT * cross_entropy(policy_logits(obs), teacher_action)` plus
the existing `idm` and `forward` losses on consecutive demonstration screens
(valid pairs only, as the existing callback does across episode starts).
The policy's logits come from SB3's `policy.get_distribution(obs)`. The
optimiser covers the extractor and the policy head; `IMITATION_WEIGHT = 0.1`
to start. Logs `aux/imitation_loss` and `aux/imitation_acc` (the share of
demonstration actions the policy would have chosen).

If `demos/` is empty or the teacher is down, the callback does nothing and
training proceeds exactly as today.

### 5. Viewer

`watch.py --teacher`: drives the environment with the SLM instead of a
checkpoint, shows the prompt's summary lines and the 8 scores on the HUD,
and highlights the chosen button. This is how a bad prompt is seen.

### 6. Stage 0 measurement (`tools/measure_teacher.py`)

Runs the teacher loop, a random policy and the newest nature4 checkpoint for
20,000 steps each from the same anchor, and prints reward per 1,000 steps by
term, encounters, wins (exp increases), whiteouts, distinct tiles and maps.
The go/no-go for building the student callback is: teacher beats random on
reward per 1,000 steps and on distinct tiles. Reported to Abbas with the
numbers before Stage 1.

### 7. Tests

- `test_teacher.py`: `build_prompt` is deterministic and under 220 tokens on
  the three fixtures (anchor, dialogue, battle); the battle fixture's prompt
  names SCRATCH as damaging and GROWL as status; `choose_action` returns an
  index in 0..7 for every fixture and the scores sum to a finite vector; a
  shard round-trips through write and read with the same arrays; the ring
  prunes to N shards.
- `test_train.py`: `ImitationCallback` on a synthetic shard where the
  teacher always presses "right" raises `imitation_acc` from chance to above
  0.9 within 50 steps on a fresh policy, with gradients reaching the
  extractor; with no shards present it changes no parameter.
- A smoke run of `train.py` with a shard present logs `aux/imitation_*`.

## Measurement after launch

On `reward/*` and `aux/*` from the day the callback starts, against the
nature4 numbers before it: `aux/imitation_acc` rising; `reward/bump` per 1k
falling faster than before; `reward/encounter`, `attack` and `exp` rising;
`reward/total` not falling; `progress/maps_visited` and the first arrival at
Petalburg (map 0,0) and Rustboro (0,3). The teacher's own ledger alongside,
so a student that overtakes the teacher is visible.

## Left to the builder

- The exact prompt wording, within the content list above.
- Argmax versus temperature sampling as the teacher's default.
- Shard size, ring size and the imitation batch size, within the stated
  defaults.
- Whether the move-name dict lives in `battle.py` or `teacher.py`.
- The systemd unit's exact settings, copied from the training unit.
