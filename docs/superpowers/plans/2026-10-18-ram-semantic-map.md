# RAM Semantic Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put an exact tile-class window and the battle state into the agent's observation, add a potential-based damage reward and a potential-based frontier push, and start a fresh policy on them.

**Architecture:** Two new RAM readers (`tilemap.py`, `battle.py`) owned by `GameState` like `TextReader` is, exposing one new observation key each; one pure function module (`frontier.py`) for the frontier distance; the reward terms and keys wired into `env.py`; a tile CNN and battle passthrough added to `AgentExtractor`. Every RAM address is located by a probe script and pinned by a fixture-based test.

**Tech Stack:** Python 3.11, pygba 0.2.9 / mgba 0.10.2 bindings, gymnasium 1.3, stable-baselines3 2.9, torch, numpy. Assertion-based test scripts, no framework.

**Spec:** `docs/superpowers/specs/2026-10-18-ram-semantic-map-design.md`

## Global Constraints

- Observation keys `screen (3,80,120) uint8`, `state (29,) float32`, `text (16,) int64` keep their shapes; new keys are `tiles` and `battle`.
- Added per-step read cost under 0.5 ms (tiles + battle together), measured idle.
- EWRAM/IWRAM through `gba.read_u8/u16/u32/read_memory`; ROM only through `gba.core.memory.u8[a:b]` slices, cached per tileset pointer. `PyGBA.read_memory` is never called on a ROM address per step.
- Every new entry point and test file creates a throwaway `PyGBA` core before any torch/SB3 import (see the top of `test_backbone.py`).
- Every reward term is booked through `self._pay(term, amount)`; every new reward is edge-triggered on a monotonic quantity or potential-based.
- Tile and battle reads are computed only when the map id is stable (`s["map"] == self._prev["map"]`); otherwise the previous value is reused.
- Each guard gets a revert-check (the test must fail with the fix removed), run with `PYTHONDONTWRITEBYTECODE=1` and `rm -rf __pycache__` after any sed-and-restore.
- Commits authored as the repo's configured user; message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Fresh policy: `run_training.sh` sets `CKPT_DIR=checkpoints/nature4`; `checkpoints/nature3` is untouched.
- The live tree is the working tree (this is how every change on this project has been made). The `nature3` service keeps running through Tasks 1-4, which do not change the observation space. Task 5 changes it, so Task 5 begins with `systemctl --user stop pokemon-rl.service` and lands the `CKPT_DIR` switch in the same commit as the observation change, so that an auto-resume can never load a nature3 checkpoint into the new space. `watchdog.sh` is not running; systemd is the only restarter.
- Timing tests (`under 0.3 ms`, `under 0.1 ms`, `fps above 100`) are measured with the service stopped. If a test must run while training is up, it measures the baseline `run_frame()` cost in the same process and asserts on the overhead, as `test_state.py` does.
- The deliverable ends at Task 7: smoke run, service switched to nature4, suites green, pushed. The multi-million-step comparison in the spec's Measurement section is the owner's to read over the following days. Go/no-go for the switch: the smoke run's `time/fps` must be above 100 and no traceback; otherwise the service stays on nature3 and the shortfall is reported.

## Review Focus

1. A map whose tileset pointers do not resolve to ROM (0x08000000..0x09FFFFFF): the whole window must read as the `unknown` class, never garbage. Test in Task 2.
2. The player standing within 5 tiles of the map edge: window positions outside `[0,width) x [0,height)` must read `outside`, not wrap or index past the grid. Test in Task 2.
3. A battle that ends on the same step as damage (enemy faints): the damage term must pay the final decrease once and nothing on the following out-of-battle step where the enemy vector is zeros. Test in Task 4.
4. The map changing between two steps while the frontier distance is evaluated: the term pays nothing on that step and the stored distance resets. Test in Task 4.
5. A worker resumed from a checkpoint whose extractor lacks the tile CNN: `PPO.load` must fail loudly rather than train on an extractor with random tile weights. Test in Task 5.

---

### Task 1: Battle fixture

**Files:**
- Create: `battle.state` (binary, committed like `dialogue.state`)
- Create: `tools/make_battle_state.py`

**Interfaces:**
- Produces: `battle.state`, a raw mGBA state with a wild battle in progress on Route 101 or 103, past the intro animation (the "What will X do?" menu or later).

- [ ] **Step 1: Write `tools/make_battle_state.py`**

Signature: script, no arguments. It builds `EmeraldEnv(state_path="checkpoints/furthest_5maps_0_16.state")` (fall back to the default anchor if that file is absent), walks with direction presses biased north, detects a battle start as `scratchpad/run_probe.py` did (position frozen for 3 steps while the mean absolute frame difference exceeds 25), then steps no-op until `env.state_reader.read()["screen_text"]` contains `"do?"` or 60 no-op steps have passed, and writes `bytes(ffi.buffer(env.gba.core.save_raw_state()))` to `battle.state`. Prints the step count, map, position and final screen text.

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=. .venv/bin/python tools/make_battle_state.py`
Expected: `battle.state` exists, under 1 MB, and the printed text contains `do?` or a move name.

- [ ] **Step 3: Verify by eye**

Run: a three-line script that loads `battle.state` into `EmeraldEnv(state_path="battle.state")`, calls `reset()`, saves `env._raw_frame()` as `/tmp/battle_fixture.png`. Open the image: it must show the battle screen with the enemy's HP bar.

- [ ] **Step 4: Commit**

```bash
git add battle.state tools/make_battle_state.py
git commit -m "test: battle savestate fixture"
```

---

### Task 2: Tile map reader

**Files:**
- Create: `tilemap.py`
- Create: `tools/probe_tilemap.py`
- Create: `test_tilemap.py`
- Modify: `state.py` (own a `TileMapReader`, expose nothing new in `read()` yet)

**Interfaces:**
- Produces in `tilemap.py`:
  - `TILE_H = 11`, `TILE_W = 15`
  - `CLASSES = ("walkable", "blocked", "grass", "water", "ledge", "warp", "npc", "outside", "unknown")`; channel index is position in this tuple.
  - `class TileMapReader:` `__init__(self, gba)`; `window(self) -> np.ndarray` returning `(len(CLASSES), TILE_H, TILE_W)` `uint8` one-hot centred on the player; `tile_class(self, x: int, y: int) -> str` for one map coordinate.
  - Module constants, filled in by the probe: `BACKUP_MAP_LAYOUT` (IWRAM address of `{s32 width; s32 height; u16 *map}`), `MAP_HEADER` (EWRAM address of the current map header, whose `mapLayout` pointer at +0 leads to `primaryTileset` at +0x10 and `secondaryTileset` at +0x14, each a ROM struct whose `metatileAttributes` pointer is at +0x10), `OBJECT_EVENTS` (EWRAM array), `OBJECT_EVENT_SIZE = 0x24`, `OBJECT_EVENT_COUNT = 16`, with the active flag in bit 0 of byte +0 and current x,y as `s16` at +0x10 and +0x12 (these offsets are the pokeemerald layout; the probe confirms each or corrects it).
- `state.py`: `GameState.__init__` gains `self.tiles = TileMapReader(gba)`.

- [ ] **Step 1: Write `tools/probe_tilemap.py` to locate the three addresses**

Script, no arguments. Loads `dialogue.state` into an env, then:
1. Scans IWRAM (0x03000000, 0x8000) for a 12-byte `{s32 w; s32 h; u32 ptr}` with `1 <= w, h <= 200` and `0x02000000 <= ptr < 0x02040000`; prints every candidate as `BACKUP_MAP_LAYOUT`.
2. For each candidate, reads the `u16` at `ptr + 2*(x + w*y)` for the player's `(x, y)` from `GameState.read()["pos"]`, steps the env right once, re-reads at the new position, and prints both metatile ids; the real layout gives ids that change when the position changes and a map width consistent with the player's x staying in range.
3. Scans EWRAM for a `u32` in ROM range whose target, read as a map layout struct, has `primaryTileset` and `secondaryTileset` pointers also in ROM and whose `width`/`height` at +0 and +4 equal the layout's `w`/`h`; prints candidates as `MAP_HEADER` (the address holding the layout pointer is `MAP_HEADER + 0`).
4. Scans EWRAM for 16 consecutive 0x24-byte records where at least one has bit 0 of byte 0 set and `s16` fields at +0x10, +0x12 within `[0, w) x [0, h)` (object events store tile coordinates offset by 7; the probe tries both raw and minus-7 and reports which lands NPCs on the map); prints candidates as `OBJECT_EVENTS`.

Prints a final block `BACKUP_MAP_LAYOUT = 0x...`, `MAP_HEADER = 0x...`, `OBJECT_EVENTS = 0x...` when exactly one candidate survives each check.

- [ ] **Step 2: Run the probe and pin the constants**

Run: `PYTHONPATH=. .venv/bin/python tools/probe_tilemap.py`
Expected: one candidate per constant. Paste the three values into `tilemap.py`. If a scan returns more than one survivor, add the discriminating check to the probe rather than choosing by hand.

- [ ] **Step 3: Write the failing tests in `test_tilemap.py`**

```python
def test_window_shape_and_one_hot():
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    w = env.state_reader.tiles.window()
    assert w.shape == (len(CLASSES), TILE_H, TILE_W) and w.dtype == np.uint8
    assert (w.sum(axis=0) == 1).all(), "every tile has exactly one class"

def test_player_tile_is_walkable_or_grass():
    # the player stands on it, so it cannot be blocked, water or outside
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    centre = env.state_reader.tiles.window()[:, TILE_H // 2, TILE_W // 2]
    assert CLASSES[int(centre.argmax())] in ("walkable", "grass", "warp")

def test_window_shifts_with_movement():
    # Walk until a step actually moves; the previous window's centre column
    # must then appear one column over in the new window.
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    ...  # press B to close the dialogue, then 'right' until pos changes
    assert np.array_equal(before[:, :, 1:], after[:, :, :-1])

def test_blocked_presses_face_non_walkable_tiles():
    # 300 random direction presses from the default anchor. Every press that
    # did not move the player, with no text on screen, must have faced a tile
    # whose class is not walkable/grass/warp; every press that moved must
    # have landed on walkable/grass/warp. This is the self-consistency check
    # that needs no hand labels.
    ...
    assert blocked_mismatches == 0 and moved_mismatches == 0

def test_route_103_window_contains_grass():
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    w = env.state_reader.tiles.window()
    assert w[CLASSES.index("grass")].sum() > 0

def test_edge_of_map_reads_outside():
    # Force the player's position in the reader to (0, 0) via tile_class
    # calls at negative coordinates.
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    assert env.state_reader.tiles.tile_class(-1, 0) == "outside"
    assert env.state_reader.tiles.tile_class(0, 10_000) == "outside"

def test_unresolvable_tileset_reads_unknown():
    # Monkeypatch the reader's tileset-pointer resolution to return 0 and
    # assert the whole window is the unknown class.
    ...
    assert (w[CLASSES.index("unknown")] == 1).all()

def test_npc_tiles_are_marked():
    # dialogue.state has Birch standing next to the player.
    env = EmeraldEnv(state_path="dialogue.state"); env.reset()
    assert env.state_reader.tiles.window()[CLASSES.index("npc")].sum() >= 1

def test_window_read_is_cheap():
    # 200 reads after a frame each; mean under 0.3 ms.
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python test_tilemap.py`
Expected: FAIL with `ImportError` / `AttributeError` on `TileMapReader`.

- [ ] **Step 5: Implement `tilemap.py`**

`TileMapReader.window()`:
1. Read `w, h, grid_ptr` from `BACKUP_MAP_LAYOUT`; read the player's `(x, y)` from `read_save_block_1(gba)["pos"]`.
2. Resolve the behaviour table once per `(primary, secondary)` tileset pointer pair: read `MAP_HEADER + 0` → layout; `layout + 0x10`, `+ 0x14` → tileset structs; each `+ 0x10` → `metatileAttributes` ROM pointer; slice `core.memory.u8[attr : attr + 4 * 512]` for the primary and `[attr : attr + 4 * 512]` for the secondary (attributes are `u32` per metatile in Emerald; behaviour is the low byte). Cache `{(primary, secondary): (np.ndarray(512), np.ndarray(512))}`. If any pointer is outside ROM range, return a window that is all `unknown`.
3. For each window cell `(dy, dx)`, map coordinate `(x - 7 + dx, y - 5 + dy)`; outside the grid → `outside`; else read the `u16`, `metatile = v & 0x3FF`, `collision = (v >> 10) & 3`; behaviour from the primary table if `metatile < 512` else secondary at `metatile - 512`; class by this order: `collision != 0` → `blocked`; behaviour in `GRASS` → `grass`; in `WATER` → `water`; in `LEDGE` → `ledge`; in `WARP` → `warp`; `0x00` → `walkable`; anything else → `blocked`.
4. Overlay `npc`: for each active object event whose `(x, y)` (minus the 7-tile offset if the probe found it applies) falls inside the window and is not the player's own tile.

The behaviour sets come from pokeemerald `include/constants/metatile_behaviors.h` (MB_TALL_GRASS, MB_LONG_GRASS and the other grass variants; MB_POND_WATER, MB_DEEP_WATER, MB_OCEAN_WATER and the other surfable waters; MB_JUMP_EAST..MB_JUMP_SOUTH and the diagonal jumps; the door, warp, stairs and arrow behaviours). Fetch the header and copy the values rather than recalling them; `test_blocked_presses_face_non_walkable_tiles` and `test_route_103_window_contains_grass` catch a wrong set.

Read the 165 grid `u16`s with one `gba.read_memory` of the covered rows rather than 165 calls.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python test_tilemap.py`
Expected: all PASS, timing test prints under 0.3 ms.

- [ ] **Step 7: Revert-check**

Swap the `walkable` and `blocked` branches; `test_blocked_presses_face_non_walkable_tiles` must fail. Restore, `rm -rf __pycache__`.

- [ ] **Step 8: Commit**

```bash
git add tilemap.py tools/probe_tilemap.py test_tilemap.py state.py
git commit -m "feat: tile-class window read from the game's map grid"
```

---

### Task 3: Battle reader

**Files:**
- Create: `battle.py`
- Create: `tools/probe_battle.py`
- Create: `test_battle.py`
- Modify: `state.py` (own a `BattleReader`; `read()` gains key `"battle": np.ndarray (8,) float32` and `"in_battle": bool`)
- Modify: `test_state.py` (expected key set gains `battle`, `in_battle`)

**Interfaces:**
- Produces in `battle.py`: `BATTLE_SIZE = 12`; `class BattleReader:` `__init__(self, gba)`; `in_battle(self) -> bool`; `read(self) -> np.ndarray` float32 `(12,)` laid out as the spec's table (0 in-battle, 1 enemy HP fraction, 2 our HP fraction, 3-6 our PP fractions, 7 `min(enemy_level / our_level, 2) / 2`) plus 8-11: for each of our four move slots, `1.0` if the move is damaging (base power above 0) and `0.0` if it is a status move or the slot is empty. Added on 2026-10-18 at Abbas's request ("the nature of moves, status changing or damaging"). All zeros when not in battle.
- Move nature comes from the ROM move table (pokeemerald `gBattleMoves`, candidate 0x0831C898, 12-byte entries: effect u8 +0, power u8 +1, type u8 +2, accuracy u8 +3, pp u8 +4), indexed by our mon's four move ids, which are `u16` at entry offset +0x0C in the battle Pokemon array. Read through `core.memory.u8` once per distinct move id and cached in a dict; never per step. Module constants from the probe: `BATTLE_MONS` (candidate 0x02024084, entry size 0x58, our mon index 0, enemy index 1; `pp` four `u8` at +0x24, `hp` u16 +0x28, `level` u8 +0x2A, `maxHP` u16 +0x2C; max PP is read from the same entry only if the probe finds it, otherwise PP fraction is `pp / 35`) and `IN_BATTLE_FLAG = (address, bitmask)`.
- `GameState.read()` returns `"battle"` and `"in_battle"` alongside the existing keys.

- [ ] **Step 1: Write `tools/probe_battle.py`**

Loads `battle.state` and `dialogue.state` into two envs. For `BATTLE_MONS`: reads the candidate entries and prints species, hp, maxHP, level, pp for indices 0 and 1 in both states; the battle state must show `0 < hp <= maxHP`, `1 <= level <= 100` for both, the dialogue state need not. For `IN_BATTLE_FLAG`: scans EWRAM and IWRAM for bytes that are non-zero in `battle.state` and zero in `dialogue.state` AND zero in the default anchor state, and additionally zero in `battle.state` after 200 no-op steps if the battle has ended by then (print whether it has); prints the surviving `(address, value)` pairs. Prefer a byte whose set bit is a single bit.

- [ ] **Step 2: Run the probe and pin the constants**

Run: `PYTHONPATH=. .venv/bin/python tools/probe_battle.py`
Expected: battle mons parse plausibly at the candidate address (else adjust by scanning for the pair `(hp, maxHP)` shown on screen); one in-battle byte survives. Paste into `battle.py`.

The probe also verifies `BATTLE_MOVES_TABLE`: for our mon's four move ids it prints the table's `pp` and `power`; the table is confirmed when its `pp` for each known move equals that move's maximum PP (the fixture's Torchic knows Scratch, pp 35, power 40, and Growl, pp 40, power 0). If the candidate address does not satisfy that, scan ROM for a 12-byte stride where entry 10 (Scratch) has power 40 / pp 35 and entry 45 (Growl) has power 0 / pp 40.

- [ ] **Step 3: Write the failing tests in `test_battle.py`**

```python
def test_in_battle_on_the_fixture_and_not_in_dialogue():
    assert BattleReader(battle_env.gba).in_battle()
    assert not BattleReader(dialogue_env.gba).in_battle()

def test_vector_layout_in_battle():
    v = reader.read()
    assert v.shape == (12,) and v.dtype == np.float32 and v[0] == 1.0
    assert 0.0 < v[1] <= 1.0 and 0.0 < v[2] <= 1.0
    assert (0.0 <= v[3:7]).all() and (v[3:7] <= 1.0).all()
    assert 0.0 <= v[7] <= 1.0
    assert set(v[8:12].tolist()) <= {0.0, 1.0}

def test_move_nature_marks_scratch_damaging_and_growl_status():
    # the fixture's Torchic: Scratch (power 40) -> 1.0, Growl (power 0) -> 0.0,
    # found by move id in the mon's four slots
    ...

def test_vector_is_zero_outside_battle():
    assert not BattleReader(dialogue_env.gba).read().any()

def test_state_read_exposes_battle():
    s = dialogue_env.state_reader.read()
    assert s["battle"].shape == (12,) and s["in_battle"] is False

def test_battle_read_is_cheap():
    # mean of 200 reads under 0.1 ms
```

- [ ] **Step 4: Run to verify they fail** — `ImportError` on `BattleReader`.

- [ ] **Step 5: Implement `battle.py` and wire `state.py`**

`in_battle()` reads the flag byte and tests the mask. `read()` returns zeros unless in battle; otherwise reads both entries with one `gba.read_memory(BATTLE_MONS, 2 * 0x58)` and unpacks with `struct`. Clip every fraction to `[0, 1]`; if `maxHP == 0` treat as not in battle.

- [ ] **Step 6: Run the tests to verify they pass**, then run `test_state.py` and `test_text.py`.

- [ ] **Step 7: Revert-check** — make `in_battle()` return `True` unconditionally; `test_vector_is_zero_outside_battle` must fail. Restore.

- [ ] **Step 8: Commit** — `git commit -m "feat: battle state read from the battle Pokemon array"`

---

### Task 4: Frontier distance, damage reward, frontier reward

**Files:**
- Create: `frontier.py`
- Create: `test_frontier.py`
- Modify: `env.py` (constants, two reward terms, two counters, reset)
- Modify: `test_env.py` (four new tests)
- Modify: `README.md` (reward table rows and two short paragraphs)

**Interfaces:**
- `frontier.py`: `def nearest_unvisited_distance(section: tuple[int, int], visited: set[tuple[int, int]], cap: int = 8) -> int` returning the Manhattan distance in sections to the nearest `(sx, sy)` not in `visited`, searching rings of increasing distance from `section` and stopping at `cap`; returns `cap` if none found.
- `env.py` constants: `DAMAGE_REWARD = 0.5`, `FRONTIER_REWARD = 0.1`, `FRONTIER_CAP = 8`. Ledger terms `"damage"` and `"frontier"`. State: `self._enemy_hp_prev: float | None`, `self._frontier_prev: int | None`, both reset in `reset()` and cleared on a map change.

- [ ] **Step 1: Write the failing tests in `test_frontier.py`**

```python
def test_unvisited_neighbour_is_distance_one():
    assert nearest_unvisited_distance((5, 5), {(5, 5)}) == 1
def test_distance_counts_rings():
    visited = {(x, y) for x in range(3, 8) for y in range(3, 8)}   # 5x5 block
    assert nearest_unvisited_distance((5, 5), visited) == 3
def test_cap_when_everything_nearby_is_visited():
    visited = {(x, y) for x in range(-10, 21) for y in range(-10, 21)}
    assert nearest_unvisited_distance((5, 5), visited, cap=8) == 8
def test_current_section_unvisited_is_zero():
    assert nearest_unvisited_distance((0, 0), set()) == 0
```

- [ ] **Step 2: Run to verify they fail**, implement `frontier.py` with the ring scan (for `d` in `0..cap`, iterate the `4d` cells at Manhattan distance exactly `d`), **run to verify they pass.**

- [ ] **Step 3: Write the failing env tests in `test_env.py`**

```python
def test_damage_pays_on_enemy_hp_falling_and_charges_on_healing():
    # two in-battle states with battle vectors; enemy hp 1.0 -> 0.6 pays
    # 0.4 * DAMAGE_REWARD; 0.6 -> 0.8 charges 0.2 * DAMAGE_REWARD
def test_damage_pays_nothing_across_battle_start_and_end():
    # out-of-battle zeros -> in-battle 1.0: nothing; in-battle 0.3 -> zeros
    # (enemy fainted, battle over): nothing; ledger has no "damage"
def test_frontier_pays_toward_unvisited_and_charges_away_and_loops_net_zero():
    # visited sections fixed; move one section toward the frontier: pays
    # FRONTIER_REWARD; move back: charges it; sum of a loop == 0
def test_frontier_pays_nothing_on_a_map_change():
```

Each uses `_compute_reward` on dicts with explicit `"battle"`, `"in_battle"`, `"pos"`, `"map"` keys, after a priming call and `drain_reward_terms()`, and asserts via the ledger.

- [ ] **Step 4: Run to verify they fail**, then implement in `env.py`:

Damage, inside `if prev is not None:`: if `s["in_battle"] and prev["in_battle"]`: `delta = prev["battle"][1] - s["battle"][1]`; `reward += self._pay("damage", self.DAMAGE_REWARD * delta)` (negative when the enemy heals). Nothing otherwise.

Frontier, after the section bookkeeping: `section = (s["pos"][0] // SECTION_SIZE, s["pos"][1] // SECTION_SIZE)`; `visited = {(sx, sy) for (g, n, sx, sy) in self._sections_ever if (g, n) == map_key}`; `d = nearest_unvisited_distance(section, visited, self.FRONTIER_CAP)`; if `self._frontier_prev is not None and map_key == prev["map"]`: `reward += self._pay("frontier", self.FRONTIER_REWARD * (self._frontier_prev - d))`; then `self._frontier_prev = d`. In the map-change early return, set `self._frontier_prev = None`.

Note `_sections_ever` is updated earlier in the same call, so the current section is always visited when `d` is computed; `d == 0` therefore never occurs after the first step, which is intended.

- [ ] **Step 5: Run `test_env.py` to verify everything passes.**

- [ ] **Step 6: Revert-checks** — `DAMAGE_REWARD = 0.0` fails the damage test; drop the `prev["in_battle"]` condition fails the start/end test; `FRONTIER_REWARD = 0.0` fails the frontier test. Restore, clear `__pycache__`.

- [ ] **Step 7: README** — add rows `| Damage dealt | +0.5 per enemy HP bar, charged back on healing |` and `| Walking toward the nearest never-visited section | +0.1 per section, potential-based |`, and one paragraph each under the reward section in the existing style (what, why, how it cannot be farmed).

- [ ] **Step 8: Commit** — `git commit -m "feat: damage reward and frontier shaping"`

---

### Task 5: Observation keys and extractor

**Files:**
- Modify: `env.py` (observation space, `_observation`, stability gating of the two reads)
- Modify: `extractor.py`
- Modify: `test_env.py`, `test_extractor.py`
- Modify: `run_training.sh` (`CKPT_DIR=checkpoints/nature4`)
- Modify: `README.md` (observation paragraph, layout table)

**Interfaces:**
- `EmeraldEnv.observation_space` gains `"tiles": Box(0, 1, (9, 11, 15), uint8)` and `"battle": Box(0.0, 1.0, (12,), float32)`.
- `AgentExtractor`: `TILE_FEATURES = 128`; `features_dim = 512 + state_dim + 32 + 128 + 12`; `forward` concatenates `[visual, state, words, tile_feats, battle]` in that order. `idm_logits`, `forward_error`, `encode_screen` unchanged.

- [ ] **Step 1: Write the failing tests**

In `test_env.py`:
```python
def test_observation_has_tiles_and_battle():
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["tiles"].shape == (9, 11, 15) and obs["battle"].shape == (12,)
def test_tiles_reuse_previous_window_during_map_transition():
    # feed _observation a state whose map differs from _prev; the tiles
    # array must equal the one from the previous stable step
```
In `test_extractor.py`:
```python
def test_features_dim_includes_tiles_and_battle():
    assert ext.features_dim == 512 + 29 + 32 + 128 + 12
def test_tile_cnn_reacts_to_tiles():
    # same screen/state/text, two different tiles arrays -> different outputs
def test_loading_an_extractor_without_tile_weights_fails_loudly():
    # build an AgentExtractor, delete its tile keys from the state_dict,
    # load_state_dict(strict=True) must raise
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

`env.py`: in `step()`, after the stability check, `self._tiles = self.state_reader.tiles.window() if stable or self._tiles is None else self._tiles`; `_observation` returns `"tiles": self._tiles` and `"battle": s["battle"]`; `reset()` computes the first window. `extractor.py`: `self.tiles = nn.Sequential(Conv2d(9, 32, 3, padding=1), ReLU, Conv2d(32, 64, 3, padding=1), ReLU, Flatten, Linear(64 * 11 * 15, 128), ReLU)`; input `observations["tiles"].float()` (SB3 passes it unnormalised since the space is not an image space by its channel count; divide by nothing, values are already 0/1).

- [ ] **Step 4: Run `test_env.py`, `test_extractor.py`, `test_train.py` to verify they pass.**

- [ ] **Step 5: `run_training.sh`** — `CKPT_DIR=checkpoints/nature4` with a comment naming the observation change; `mkdir -p checkpoints/nature4`.

- [ ] **Step 6: README** — observation paragraph gains the two keys and the layout table gains `tilemap.py`, `battle.py`, `frontier.py`.

- [ ] **Step 7: Commit** — `git commit -m "feat: tile window and battle state in the observation; nature4 run"`

---

### Task 6: Viewer HUD

**Files:**
- Modify: `watch.py`

**Interfaces:**
- Consumes `obs["tiles"]`, `obs["battle"]`, `tilemap.CLASSES`.

- [ ] **Step 1: Implement**

Render the 11x15 window as a grid of 12 px squares to the right of the frame (window width grows by `15 * 12 + 8`), one colour per class (walkable grey, blocked black, grass green, water blue, ledge brown, warp yellow, npc red, outside dark, unknown magenta), the player's cell outlined white. Add to the HUD line 2: `enemy HP 0.63  our HP 0.80  PP 1.0/0.9/0.4/0.0  moves D/S/S/-` when `obs["battle"][0] == 1`, where D is a damaging move, S a status move and - an empty slot.

- [ ] **Step 2: Verify by eye**

Run: `DISPLAY=:0 .venv/bin/python watch.py <any nature3 checkpoint> --steps 300` with `EmeraldEnv` unchanged (a nature3 checkpoint cannot load into the new extractor; for this step load no model and drive with random actions via a `--random` flag added to `watch.py`). The grid must show blocked tiles where the frame shows trees and fences, and grass where it shows grass.

- [ ] **Step 3: Commit** — `git commit -m "feat: viewer shows the tile window and battle state"`

---

### Task 7: Smoke run and switch

**Files:**
- None new.

- [ ] **Step 1: Smoke run**

Run: `PYTHONDONTWRITEBYTECODE=1 nohup .venv/bin/python train.py --steps 16384 --backbone agent --batch-size 128 --max-steps 0 --coverage-every 10000000 --checkpoint-dir /tmp/claude-1000/-home-abbas/abb9f38b-72d2-4401-b1e5-38b0b397b66e/scratchpad/smoke4 > .../smoke4.log 2>&1 &`
Expected after two rollouts: `reward/damage` or `reward/frontier` present, `aux/idm_acc` logged, `time/fps` above 100, no traceback.

- [ ] **Step 2: Switch the service**

Run: `systemctl --user restart pokemon-rl.service`; `service.log` shows `starting fresh, no checkpoint found` and `/proc/<MainPID>/cmdline` contains `--checkpoint-dir checkpoints/nature4`.

- [ ] **Step 3: Full suites**

Run every `test_*.py`; all pass.

- [ ] **Step 4: Push** — `git push origin coverage-overlay`.
