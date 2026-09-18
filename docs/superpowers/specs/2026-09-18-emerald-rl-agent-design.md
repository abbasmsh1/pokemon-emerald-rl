# Pokemon Emerald RL Agent - Design

Date: 2026-09-18

## Goal

Train a reinforcement learning agent that starts from the beginning of
gameplay in Pokemon Emerald and earns the first gym badge (Roxanne, Rustboro
City Gym).

Success is binary and verifiable: `game_state()["num_badges"] >= 1`.

## Why this substrate

The agent runs on mGBA through the `pygba` Python bindings. Measured on the
target machine (8 physical cores with hyperthreading, RTX 3050 Ti 4GB, 38GB
RAM of which roughly 12GB is free):

| Configuration | Throughput | Relative to realtime |
| --- | --- | --- |
| Raw emulation, single core | 3,198 frames/sec | 53x |
| Full env step, 1 process | 84.5 steps/sec | 34x |
| Full env step, 4 processes | 242 steps/sec | 97x |
| Full env step, 8 processes | 262 steps/sec | 105x |
| Full env step, 16 processes | 360 steps/sec | 144x |

Throughput scales poorly past 4 processes. Combined with limited free memory,
the design uses 8 workers rather than 16.

`pygba` ships a `PokemonEmerald` game wrapper that already decodes the game's
encrypted save blocks and exposes badges, gym and Elite Four progress, visited
cities, the full Pokedex, party, boxes, money, player position, map location,
and the raw script, trainer, and system event flag arrays. This removes the
usual multi-week effort of reverse engineering RAM offsets.

Two environment defects are known and handled:

1. The `mgba` wheel does not bundle `libmgba.so.0.10`. The system package
   `libmgba0.10t64` (version 0.10.2) supplies it and is installed.
2. `pygba` calls `Image.to_pil()`, which the `mgba` bindings define only when
   Pillow is importable. Pillow is therefore a hard dependency despite not
   being declared as one.

## Episode boundaries

Every episode resets from a savestate captured at the moment the player gains
control in the bedroom, after the intro cutscenes, truck sequence, and name
entry.

This is deliberate. The intro is a fixed scripted sequence containing a
character-grid name entry screen. It is not learnable by reinforcement
learning in any useful sense, and including it would spend a large fraction of
every episode replaying identical frames.

A related constraint: game state reads return an empty dictionary before the
save blocks are initialised. Roughly 3,000 frames of boot are required before
`game_state()` returns usable values. Resetting from a savestate avoids this
entirely, but any code path that cold-boots the ROM must account for it.

An episode ends when the agent earns the badge, whites out, or reaches 16,384
steps (approximately 109 minutes of in-game time).

## Observation space

A dictionary with two entries.

**Screen.** The 240x160 RGB framebuffer converted to grayscale, downscaled to
120x80, and stacked three deep. Shape `(3, 80, 120)`, dtype uint8. The stack
lets the policy distinguish a static screen from an animating one, which
matters for detecting menus, battle transitions, and dialogue.

**State vector.** Shape `(17,)`, dtype float32: eight badge bits, mean party
HP as a fraction, mean party level, normalised map ID, player x, player y, an
in-battle flag, log-scaled money, Pokedex seen count, Pokedex caught count.

Two of these need work beyond reading the wrapper's output:

- The wrapper does not expose an in-battle flag. It must be derived
  separately, by reading the battle-state address directly. Locating and
  verifying that address is an implementation task, not a solved one.
- The wrapper omits the `money` key entirely when money is zero, rather than
  reporting zero. Reading it must default to 0.0 on a missing key.

The state vector is a deliberate compromise. A pixels-only agent would be more
principled, but values like badge count and party HP are nearly impossible to
read reliably from a downscaled grayscale frame, and withholding them makes
credit assignment substantially harder without teaching the agent anything
interesting.

## Action space

`Discrete(7)`: up, down, left, right, A, B, start.

`pygba` defaults to `Discrete(35)`, which enumerates button combinations. Most
are meaningless. Select, L, and R have no useful function before Rustboro.

Frameskip is 24 frames, or 0.4 seconds per action. This corresponds to roughly
one tile of movement per action and advances dialogue at a reasonable rate
without wasting steps on held-button frames.

## Reward

The reward combines the `PokemonEmerald` wrapper's built-in signals with a
coordinate-novelty exploration term.

| Event | Reward |
| --- | --- |
| Roxanne's badge earned | +100 |
| New map entered | +2 |
| New `(map_id, x, y)` tile visited | +0.05 |
| Script event flag newly set | +1 |
| Trainer defeated | +2 |
| Party level gained | +0.2 |
| Pokemon newly seen | +0.1 |
| Pokemon newly caught | +0.5 |
| Whiteout | -5, episode ends |

### Why coordinate novelty

The badge is roughly 50,000 steps from the bedroom. No sparse reward bridges
that gap. A dense exploration signal is mandatory.

Tile novelty was chosen over frame-embedding novelty, the approach used by
PokemonRedExperiments. The game exposes exact coordinates and map ID in RAM,
so novelty reduces to a set insertion costing microseconds, with no false
positives. Frame novelty requires a nearest-neighbour search against a growing
buffer on every step, needs threshold and buffer-size tuning, and rewards
graphical noise such as weather effects and NPC animation cycles. The
signals frame novelty uniquely provides, chiefly progress inside battles and
menus, are available more cheaply from the script event flags.

The per-map cap of 400 novel tiles prevents the agent from farming reward by
pacing across a large open route indefinitely.

## Training

Proximal Policy Optimization from stable-baselines3, with `SubprocVecEnv`
across 8 workers and `n_steps=1024`, producing a rollout buffer of roughly
236MB.

The policy is a NatureCNN over the screen and a two-layer MLP over the state
vector, concatenated before the policy and value heads.

A callback logs per rollout: badges earned, furthest map ID reached, unique
tiles visited, and episode reward. Model checkpoints are written every 100,000
steps. Separately, whenever a run reaches a new furthest map, the emulator
savestate at that moment is written to disk. These savestates are the raw
material for the curriculum fallback described below.

## Components

| File | Responsibility |
| --- | --- |
| `env.py` | Gymnasium environment: observation construction, action mapping, reward, savestate reset |
| `train.py` | PPO setup, vectorised envs, callbacks, TensorBoard |
| `make_savestate.py` | One-off pygame frontend for playing the intro manually and dumping `boot.state` |
| `watch.py` | Load a checkpoint and render it playing |
| `test_env.py` | Assertion-based checks |
| `boot.state` | Committed artifact, the post-intro savestate |

Reward computation lives inside `env.py`. It is the component most subject to
iteration, but separating it into its own module at this stage adds an
interface without adding clarity.

`make_savestate.py` uses a pygame frontend and dumps state through the same
Python API the environment loads it with, rather than relying on savestate
files produced by the `mgba-qt` GUI. This guarantees format compatibility.

## Testing

`test_env.py` contains assertions only, no test framework:

1. `reset()` returns observations matching the declared space in shape and
   dtype.
2. Resetting twice from the savestate produces byte-identical observations.
3. The new-tile reward fires exactly once for a given tile and does not fire
   again on revisit.
4. Reward remains finite across 1,000 random-agent steps.

## Risks

**The agent may never leave the bedroom.** This is the first thing to measure
once the environment exists, before any training run. Procedure: run 1,000
random-agent episodes and record the furthest map ID reached. If Route 101 is
never reached, tile novelty alone cannot bridge the gap and the curriculum
fallback becomes mandatory rather than optional.

**mGBA writes its emulator log to stdout, not stderr.** A single 3,000-frame
boot produced 835KB of output. Across 8 workers this will bury training logs.
Suppression must happen at the file-descriptor level; Python-level stdout
redirection does not capture output from the C library.

**Sustained thermal throttling.** The target machine is a laptop. Multi-day
runs pinning 8 workers will not sustain the benchmarked throughput.

## Fallback: checkpoint curriculum

Held in reserve, to be adopted only if the exploration reward fails to produce
progress toward Rustboro.

Maintain savestates at several points along the route: bedroom, Route 101,
Oldale Town, Petalburg Woods, Rustboro City. Sample the episode start point
from this set, weighting later checkpoints more heavily as competence
increases.

This is a rescue rather than a foundation. It adds configuration surface and
makes it difficult to answer whether the agent genuinely learned the route or
merely learned each segment in isolation. The savestates written by the
furthest-map callback supply these checkpoints without extra manual work.
