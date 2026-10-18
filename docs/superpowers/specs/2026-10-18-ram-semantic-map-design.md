# RAM semantic map, battle state and frontier shaping

Date: 2026-10-18. Status: approved by Abbas 2026-10-18 ("approved, build it"); move-nature indices added the same day.

## Intent

Make the agent do three things it currently cannot do reliably: stop walking
into things, leave a town by the right exit and keep going, and win fights on
purpose. Today each of those is rewarded or charged as an outcome, but the
agent has to infer walls, exits and enemy health from 120x80 grayscale pixels,
and that inference is what keeps failing (a 400k policy spent 25% of its
actions pressing up into a blocked tile; fights are entered and abandoned at
random).

The chosen means is to read the game's own tile grid and battle state from RAM
and put them in the observation, in place of a learned segmentation model.
Emerald holds a behaviour byte per tile (walkable, tall grass, water, ledge
and its direction, door, warp) and the position of every NPC, which is a
perfect segmentation at zero cost and with no training data. Abbas agreed to
this on 2026-10-18 in place of a learned segmenter.

## Goals

1. A local tile-class map around the player in the observation, exact, read
   from RAM each step.
2. Battle state in the observation: whether a battle is on, enemy HP fraction,
   our active Pokemon's HP fraction and the PP fraction of each of its moves.
3. A reward for dealing damage (enemy HP falling), potential-based so a
   battle nets its damage dealt and nothing can be pumped.
4. A generic push toward unexplored ground: potential-based shaping on the
   distance to the nearest never-visited section. Not a scripted route.
5. Throughput: the added per-step read cost stays under 0.5 ms (one worker's
   step budget is ~7 ms at 150 fps across 8 workers).
6. Every new RAM address is found and verified empirically against the
   running game, with a committed fixture savestate and a test per address.
   No address is taken from a symbol table on trust; the ResNet weights and
   the text buffers both failed exactly that way.

## Non-goals

- A learned segmentation model. The RAM grid is the ground truth it would
  approximate.
- Scripted routes or waypoints toward Rustboro. The frontier term is distance
  to unvisited ground, wherever it is.
- Changing PPO hyperparameters, the CNN over the screen, the text encoder, the
  inverse-dynamics head or the curiosity bonus. They stay as they are.
- Resuming the current `nature3` policy. The observation space changes, so
  this is a fresh policy in `checkpoints/nature4`. `nature3` stays on disk.

## Anti-goals (what would make this a failure even if it works)

- Training throughput falling below ~100 fps across 8 workers.
- A reward term that can be farmed. Every new term must be either edge-
  triggered on a monotonic quantity or potential-based.
- A map read that produces garbage during save-block relocation and feeds it
  to the policy or the reward. Every read gates on the same map-stability
  check the rest of the environment uses.
- Silently wrong tiles. If the behaviour table cannot be resolved for the
  current map, the map channel is all "unknown", never a guess.

## Constraints

- Reads go through pygba's per-frame RAM copy for EWRAM/IWRAM and through
  mgba's `core.memory.u8` view for ROM slices. `PyGBA.read_memory` on the ROM
  region copies 16 MB and costs 16 ms; it must never be called per step.
- `import torch` before the first mGBA core hangs libmgba; every new entry
  point and test keeps the warmup core first.
- The observation stays a `gym.spaces.Dict`; new keys are added, existing
  keys keep their shapes, so `watch.py` and the callbacks keep working.
- Tests are assertion-based scripts with no framework, each guard is
  revert-checked (the test must fail with the fix removed) and `__pycache__`
  is cleared after any sed-and-restore.

## Design

### 1. Tile map observation (`tiles`)

A window of `TILE_H x TILE_W` = 11 x 15 tiles centred on the player, each tile
one of these classes, one-hot over a `C`-channel array shaped `(C, 11, 15)`,
`uint8`:

| class | source |
| --- | --- |
| walkable | metatile behaviour: normal |
| blocked | collision bit set, or any behaviour not listed below |
| tall grass | behaviour tall grass (encounters) |
| water | surfable water |
| ledge | jump-down ledge, any direction (the direction is left to the builder; four channels if they want it) |
| door or warp | doors, warps, stairs, arrows |
| npc | an object event (NPC) stands on the tile |
| out of map | beyond the map bounds, or a map connection tile |
| unknown | the behaviour could not be resolved (see anti-goals) |

Sources, each to be located by scanning RAM while the game is in a known
state and confirmed by tracking through movement, exactly as the text
printer was:

- The current map's grid: a `(width, height, pointer)` layout struct in IWRAM
  (pokeemerald names it `gBackupMapLayout`) pointing at a `u16` per tile in
  EWRAM; the low 10 bits are the metatile id, bits 10-11 collision, bits 12-15
  elevation.
- Behaviour per metatile: the current map header's layout points at a primary
  and a secondary tileset, each with an attribute table in ROM; the behaviour
  is the low byte of the attribute for that metatile id (primary tileset for
  ids below 512, secondary above). Read through the memory view, cached per
  tileset pointer so the ROM is touched once per map, not per step.
- NPCs: the object-event array in EWRAM (pokeemerald `gObjectEvents`), each
  with an active flag and a current tile position.

Candidate addresses from the pokeemerald US symbol table are a starting point
for the scan, not an answer: `gBackupMapLayout` 0x03005DC0, `gMapHeader`
0x02037318, `gObjectEvents` 0x02037350. Each is confirmed by (a) decoding a
window at a known position and comparing it to the frame, and (b) walking and
watching the window shift by one tile per move.

Gating: the window is computed only when the map id is stable (same test as
`_compute_reward`), otherwise the previous window is reused. If the tileset
pointers do not resolve to ROM, every tile is "unknown".

Cost budget: 165 tiles, one dict lookup each against a cached behaviour array,
plus a scan of up to 16 object events. Target under 0.3 ms.

### 2. Battle state observation (`battle`)

A float vector:

| index | value |
| --- | --- |
| 0 | in battle (1/0) |
| 1 | enemy active Pokemon HP fraction |
| 2 | our active Pokemon HP fraction |
| 3-6 | PP fraction of our four moves (0 for empty slots) |
| 7 | enemy level / our level, clipped to [0, 2] then halved |
| 8-11 | for each of our four move slots: 1 if the move is damaging (base power above 0), 0 if it is a status move or the slot is empty |

Indices 8-11 were added at Abbas's request on 2026-10-18 ("the nature of
moves, status changing or damaging"). Move nature comes from the ROM move
table (pokeemerald `gBattleMoves`, candidate 0x0831C898, 12-byte entries with
base power at +1), indexed by the four move ids stored in our battle Pokemon
entry; read through the memory view once per distinct move id and cached.

Source: the battle Pokemon array (pokeemerald `gBattleMons`, candidate
0x02024084, 0x58 bytes per entry, hp at +0x28, maxHP +0x2C, pp at +0x24, level
+0x2A); index 0 is ours, 1 the enemy in a single battle. "In battle" is taken
as: the array parses with plausible values (maxHP > 0, level in 1..100) AND
the screen text or the frame shows a battle; the builder picks the cheapest
reliable combination and proves it on a fixture where the agent is mid-battle.
Outside a battle the vector is all zeros.

The battle fixture is produced by driving the environment into a wild battle
(grass on Route 101/103, direction presses, a battle start detected as in
`scratchpad/run_probe.py`) and saving the raw state as `battle.state` next to
`dialogue.state`.

### 3. Damage reward

`DAMAGE_REWARD` per unit of enemy HP fraction lost, potential-based within a
battle: paid on decreases, charged back on increases (enemy healing), zero
across the battle's start and end so a new enemy's full bar is not a loss.
Weight such that taking a full enemy bar pays about one wild win's experience
(0.4-0.8), i.e. `DAMAGE_REWARD = 0.5`. Logged as `reward/damage`.

### 4. Frontier shaping

Potential-based: `FRONTIER_REWARD * (d_prev - d_now)` where `d` is the
Manhattan distance in sections from the player's current section to the
nearest section never visited in this process (the cumulative set that backs
`FIRST_SECTION_REWARD`). Evaluated only on stable map ids and only within the
current map (distance is undefined across maps; on a map change the term pays
nothing). Weight small, 0.1 per section of progress, so it tilts ties rather
than dominating; the goal is "when nothing else is paying, walk toward the
edge of what you know". Logged as `reward/frontier`. Potential-based means a
loop nets zero, so it cannot be farmed.

Section lookup is over the cumulative set of sections visited on the current
map; the nearest unvisited section is found by scanning section coordinates
in increasing ring distance from the player's section until one is absent
from the set, capped at a radius of 8 sections (64 tiles); if nothing is
unvisited within that radius, `d` is the cap and the term is zero.

### 5. Extractor

`AgentExtractor` gains a small CNN over `tiles` (two 3x3 conv layers, 32 then
64 channels, flattened to 128 features through a linear layer) and passes the
`battle` vector straight through. Features dimension becomes
512 + 29 + 32 + 128 + 8. Inverse dynamics and curiosity continue to run on the
screen features only. The observation space change means a fresh policy;
`run_training.sh` moves `CKPT_DIR` to `checkpoints/nature4`.

### 6. Viewer

`watch.py` renders the tile window as a small coloured grid beside the frame
and adds enemy HP and our PP to the HUD, so the semantic map can be checked by
eye against the screen, which is how the text reader's failure was caught.

### 7. Tests

Per address: a fixture-based test that the decoded value matches a known
truth (the window around a known position on `dialogue.state` contains the
wall the player is facing; `battle.state` reports in-battle with enemy HP in
(0, 1]). A movement test: step right, the window shifts by one column. A
timing test: the full state read with tiles and battle stays under the
budget. Reward tests for damage (pays on decrease, charges on increase, zero
across battle boundaries) and frontier (walking toward an unvisited section
pays, walking away charges, a loop nets zero), each revert-checked.

## Measurement

Watched in `reward/*` and `aux/*` over the first few million steps of
`nature4`, against `nature3` at the same step count:

- `reward/bump` per 1k: should fall faster than nature3's (it was -4.4 at
  the first rollout, -2.0 at 3M).
- Encounters started vs flees vs wins from the viewer counters: wins should
  rise as a share.
- `progress/maps_visited` and the first time a worker reaches Petalburg
  (map group 0, number 0) and Rustboro (0, 3).
- `aux/idm_acc` should not fall (the screen CNN is unchanged).
- `time/fps` must stay above 100.

## Left to the builder

- The exact tile class list beyond the table above (ledge direction as one
  class or four; whether "arrow" tiles join "door or warp").
- CNN layer sizes for the tile map within the stated shape.
- How "in battle" is determined, provided it is proven on `battle.state` and
  on `dialogue.state` (which must read as not in battle).
- The order in which addresses are located; the scan method is the one that
  found the text printer.
- Whether the frontier search is a ring scan or a BFS over the section grid.
