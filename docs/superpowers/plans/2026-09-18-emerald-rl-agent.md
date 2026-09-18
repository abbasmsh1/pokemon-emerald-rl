# Pokemon Emerald RL Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a PPO agent that starts from a post-intro savestate in Pokemon Emerald and earns Roxanne's gym badge.

**Architecture:** A Gymnasium environment wraps mGBA through `pygba`. Observations are a stack of three downscaled grayscale frames plus a 17-value decoded state vector. Reward combines game-progress events with a coordinate-novelty exploration bonus. Training uses stable-baselines3 PPO across 8 subprocess workers.

**Tech Stack:** Python 3.11, pygba 0.2.9, mgba 0.10.2, gymnasium 1.3.0, stable-baselines3 2.9.0, torch 2.14.0+cu130, numpy, pillow, pygame

**Spec:** `docs/superpowers/specs/2026-09-18-emerald-rl-agent-design.md`

## Global Constraints

- Python 3.11 exactly. The `mgba` 0.10.2 Linux x86_64 wheels exist only for cp310 and cp311. The venv at `.venv/` is already Python 3.11.15.
- System package `libmgba0.10t64` version 0.10.2 must be installed. It supplies `libmgba.so.0.10`, which the wheel does not bundle.
- Pillow is a hard dependency. `pygba` calls `Image.to_pil()`, which `mgba` defines only when Pillow is importable, even though nothing declares the dependency.
- **Never read ROM memory (addresses `0x08000000` and above) during a step.** `PyGBA.read_memory` copies the entire memory region into Python on first touch per frame. For ROM that is a 16MB copy costing 1.28ms per step, against a 7.36ms total step budget. ROM is immutable: read it once at construction and cache the result on the instance.
- Do not call `PokemonEmerald.reward()`. It reads ROM every step and loops over ~400 species. This project computes its own reward.
- The ROM file is `Pokemon - Emerald Version (USA, Europe).gba` in the repo root and is gitignored. Never commit it.
- mGBA writes its emulator log to **stdout**, not stderr. Suppression must be at the file-descriptor level.
- All measured performance figures: raw emulation 3,198 frames/sec/core; 24-frame step 7.36ms; savestate 397,312 bytes; 8 workers ~262 steps/sec aggregate.

---

### Task 1: Fast game state reader

Decodes everything the environment needs from RAM, with ROM tables cached at construction.

**Files:**
- Create: `state.py`
- Test: `test_state.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class GameState` with `__init__(self, gba: PyGBA)` and `read(self) -> dict`.
  - `read()` returns a dict with keys: `badges` (int, 0-8), `map` (tuple `(mapGroup, mapNum)`), `pos` (tuple `(x, y)`), `party_levels` (list[int]), `party_hp_frac` (float, 0.0-1.0), `money` (int), `seen` (int), `caught` (int), `script_flag_count` (int), `trainer_flag_count` (int), `whiteout` (bool).
  - Module constant `POKEDEX_CAPACITY = 386`.

- [ ] **Step 1: Write the failing test**

Create `test_state.py`:

```python
import numpy as np
from pygba import PyGBA
from state import GameState, POKEDEX_CAPACITY

ROM = "Pokemon - Emerald Version (USA, Europe).gba"


def test_read_returns_expected_keys_and_types():
    gba = PyGBA.load(ROM)
    gba.wait(3000)
    gs = GameState(gba)
    s = gs.read()

    expected = {
        "badges", "map", "pos", "party_levels", "party_hp_frac",
        "money", "seen", "caught", "script_flag_count",
        "trainer_flag_count", "whiteout",
    }
    assert set(s) == expected, f"key mismatch: {set(s) ^ expected}"

    assert isinstance(s["badges"], int) and 0 <= s["badges"] <= 8
    assert isinstance(s["map"], tuple) and len(s["map"]) == 2
    assert isinstance(s["pos"], tuple) and len(s["pos"]) == 2
    assert isinstance(s["party_levels"], list)
    assert isinstance(s["party_hp_frac"], float) and 0.0 <= s["party_hp_frac"] <= 1.0
    assert isinstance(s["money"], int) and s["money"] >= 0
    assert 0 <= s["seen"] <= POKEDEX_CAPACITY
    assert 0 <= s["caught"] <= POKEDEX_CAPACITY
    assert isinstance(s["whiteout"], bool)
    print("test_read_returns_expected_keys_and_types PASSED")


def test_read_before_boot_returns_safe_defaults():
    """Save blocks do not exist at frame 0. read() must not raise."""
    gba = PyGBA.load(ROM)
    gs = GameState(gba)
    s = gs.read()
    assert s["badges"] == 0
    assert s["party_levels"] == []
    assert s["party_hp_frac"] == 0.0
    assert s["whiteout"] is False
    print("test_read_before_boot_returns_safe_defaults PASSED")


def test_read_does_not_touch_rom():
    """A ROM read costs 1.28ms. read() must stay far under that."""
    import time
    gba = PyGBA.load(ROM)
    gba.wait(3000)
    gs = GameState(gba)
    gs.read()

    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
        gs.read()
    per_call = (time.perf_counter() - t0) / 200

    baseline_frame = 0.307e-3
    overhead = per_call - baseline_frame
    assert overhead < 0.5e-3, f"read() overhead {overhead*1e3:.3f}ms suggests a ROM read"
    print(f"test_read_does_not_touch_rom PASSED (overhead {overhead*1e3:.3f}ms)")


if __name__ == "__main__":
    test_read_before_boot_returns_safe_defaults()
    test_read_returns_expected_keys_and_types()
    test_read_does_not_touch_rom()
    print("\nall state tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_state.py 2>/dev/null`
Expected: FAIL with `ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3: Write the implementation**

Create `state.py`:

```python
"""Fast RAM-only game state reader for Pokemon Emerald.

Deliberately avoids pygba's PokemonEmerald.game_state(), which reads ROM
tables on every call. PyGBA.read_memory copies an entire memory region into
Python on first touch per frame; for the 16MB ROM region that costs 1.28ms
per step against a 7.36ms step budget.
"""

from pygba import PyGBA
from pygba.game_wrappers.utils.emerald_utils import (
    FLAG_BADGE01_GET,
    SCRIPT_FLAGS_START,
    TRAINER_FLAGS_START,
    SYSTEM_FLAGS_START,
    read_save_block_1,
    read_save_block_2,
)

POKEDEX_CAPACITY = 386


def _get_flag(flags, flag_id: int) -> bool:
    if flag_id < 0 or flag_id // 8 >= len(flags):
        return False
    return bool((flags[flag_id // 8] >> (flag_id % 8)) & 1)


def _popcount(buf) -> int:
    return sum(b.bit_count() for b in buf)


class GameState:
    def __init__(self, gba: PyGBA):
        self.gba = gba

    def read(self) -> dict:
        sb1 = read_save_block_1(self.gba)
        sb2 = read_save_block_2(self.gba)

        if sb1 is None:
            return {
                "badges": 0,
                "map": (0, 0),
                "pos": (0, 0),
                "party_levels": [],
                "party_hp_frac": 0.0,
                "money": 0,
                "seen": 0,
                "caught": 0,
                "script_flag_count": 0,
                "trainer_flag_count": 0,
                "whiteout": False,
            }

        flags = sb1["flags"]
        badges = sum(_get_flag(flags, FLAG_BADGE01_GET + i) for i in range(8))

        loc = sb1["location"]
        pos = sb1["pos"]

        party = sb1["playerParty"]
        levels = [m["level"] for m in party]
        total_hp = sum(m["hp"] for m in party)
        total_max = sum(m["maxHp"] for m in party)
        hp_frac = (total_hp / total_max) if total_max > 0 else 0.0

        money = 0
        if sb2 is not None:
            money = sb1["money"] ^ sb2["encryptionKey"]
            # A corrupt XOR mid-write can produce absurd values; clamp to the
            # game's own maximum rather than feeding garbage to the network.
            money = max(0, min(money, 999_999))

        seen = caught = 0
        if sb2 is not None:
            seen = _popcount(sb2["pokedex"]["seen"])
            caught = _popcount(sb2["pokedex"]["owned"])

        script = flags[SCRIPT_FLAGS_START // 8:TRAINER_FLAGS_START // 8]
        trainer = flags[TRAINER_FLAGS_START // 8:SYSTEM_FLAGS_START // 8]

        return {
            "badges": badges,
            "map": (loc["mapGroup"], loc["mapNum"]),
            "pos": (pos["x"], pos["y"]),
            "party_levels": levels,
            "party_hp_frac": hp_frac,
            "money": money,
            "seen": seen,
            "caught": caught,
            "script_flag_count": _popcount(script),
            "trainer_flag_count": _popcount(trainer),
            "whiteout": len(party) > 0 and total_hp == 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_state.py 2>/dev/null`
Expected: PASS, all three tests, with the overhead figure printed.

If `test_read_does_not_touch_rom` fails, something in the import chain is reading ROM. Check that `read_save_block_1` is called with default `parse_items=False`.

- [ ] **Step 5: Commit**

```bash
git add state.py test_state.py
git commit -m "feat: RAM-only game state reader"
```

---

### Task 2: Boot savestate

Produces the post-intro savestate every episode resets from. Requires a human to play the intro once.

**Files:**
- Create: `make_savestate.py`
- Create: `boot.state` (committed artifact, 397,312 bytes)

**Interfaces:**
- Consumes: nothing.
- Produces: `boot.state` on disk, loadable via `gba.core.load_raw_state(open("boot.state","rb").read())`.

- [ ] **Step 1: Write the savestate tool**

Create `make_savestate.py`:

```python
"""Play the Emerald intro manually, then dump a savestate.

The intro is cutscenes, a truck sequence, and a character-grid name entry.
None of it is learnable by RL, so every training episode starts after it.

Controls: arrow keys, Z=A, X=B, Enter=Start, Backspace=Select.
Press S to write boot.state and exit.
"""

import os
import sys

import numpy as np
import pygame
from mgba._pylib import ffi
import mgba.image
from pygba import PyGBA

ROM = "Pokemon - Emerald Version (USA, Europe).gba"
OUT = "boot.state"
SCALE = 3

KEYS = {
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_z: "A",
    pygame.K_x: "B",
    pygame.K_RETURN: "start",
    pygame.K_BACKSPACE: "select",
}


def main():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    os.dup2(devnull, 1)  # mgba logs to stdout at the C level

    gba = PyGBA.load(ROM)
    width, height = gba.core.desired_video_dimensions()
    framebuffer = mgba.image.Image(width, height)
    gba.core.set_video_buffer(framebuffer)
    gba.core.reset()

    os.dup2(saved_stdout, 1)

    pygame.init()
    screen = pygame.display.set_mode((width * SCALE, height * SCALE))
    pygame.display.set_caption("Play the intro, then press S to save")
    clock = pygame.time.Clock()

    from pygba.utils import KEY_MAP

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                blob = bytes(ffi.buffer(gba.core.save_raw_state()))
                with open(OUT, "wb") as f:
                    f.write(blob)
                print(f"wrote {OUT} ({len(blob)} bytes)")
                running = False

        pressed = pygame.key.get_pressed()
        held = [KEY_MAP[name] for key, name in KEYS.items() if pressed[key]]
        gba.core.set_keys(*held)

        os.dup2(devnull, 1)
        gba.core.run_frame()
        os.dup2(saved_stdout, 1)

        arr = np.frombuffer(ffi.buffer(framebuffer.buffer), dtype=np.uint8)
        arr = arr.reshape(height, framebuffer.stride, 4)[:, :width, :3]

        surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (width * SCALE, height * SCALE))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Play the intro and save**

Run: `.venv/bin/python make_savestate.py`

Play through: title screen, the truck, stepping out, the clock-setting, going downstairs and back up, until you have free control standing in the bedroom. Press **S**.

This is a manual step. There is no way around it: name entry is a character grid that cannot be mashed through reliably.

- [ ] **Step 3: Verify the savestate loads and is at the right point**

Run:

```bash
.venv/bin/python -c "
from pygba import PyGBA
from state import GameState
gba = PyGBA.load('Pokemon - Emerald Version (USA, Europe).gba')
gba.core.load_raw_state(open('boot.state','rb').read())
gba.core.run_frame()
s = GameState(gba).read()
print('map', s['map'], 'pos', s['pos'], 'party', len(s['party_levels']))
" 2>/dev/null
```

Expected: a non-zero `map` tuple and a plausible `pos`. Party will be empty — you do not have a Pokemon yet at this point, which is correct.

- [ ] **Step 4: Commit**

```bash
git add make_savestate.py boot.state
git commit -m "feat: intro savestate tool and boot.state artifact"
```

---

### Task 3: Environment observations and actions

The Gymnasium environment, without reward. Reward arrives in Task 4.

**Files:**
- Create: `env.py`
- Test: `test_env.py`

**Interfaces:**
- Consumes: `GameState` and `POKEDEX_CAPACITY` from `state.py`.
- Produces:
  - `class EmeraldEnv(gym.Env)` with `__init__(self, rom_path="Pokemon - Emerald Version (USA, Europe).gba", state_path="boot.state", frameskip=24, max_steps=16384, render_mode=None)`.
  - `EmeraldEnv.ACTIONS`: list of 7 key-name strings or `None`.
  - `observation_space`: `gym.spaces.Dict` with `"screen"` as `Box(0, 255, (3, 80, 120), uint8)` and `"state"` as `Box(-1.0, 1.0, (17,), float32)`.
  - `action_space`: `Discrete(7)`.
  - `EmeraldEnv.save_state() -> bytes`, dumping the 397,312-byte emulator state.
  - Module function `suppress_stdout()`, a context manager doing fd-level suppression.

- [ ] **Step 1: Write the failing test**

Create `test_env.py`:

```python
import numpy as np

from env import EmeraldEnv


def test_observation_matches_declared_space():
    env = EmeraldEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "obs outside declared space"
    assert obs["screen"].shape == (3, 80, 120)
    assert obs["screen"].dtype == np.uint8
    assert obs["state"].shape == (17,)
    assert obs["state"].dtype == np.float32
    env.close()
    print("test_observation_matches_declared_space PASSED")


def test_reset_is_deterministic():
    """Two resets from the same savestate must give identical observations."""
    env = EmeraldEnv()
    a, _ = env.reset()
    b, _ = env.reset()
    assert np.array_equal(a["screen"], b["screen"]), "screen differs across resets"
    assert np.array_equal(a["state"], b["state"]), "state differs across resets"
    env.close()
    print("test_reset_is_deterministic PASSED")


def test_action_space_is_seven_discrete():
    env = EmeraldEnv()
    assert env.action_space.n == 7
    assert len(EmeraldEnv.ACTIONS) == 7
    env.close()
    print("test_action_space_is_seven_discrete PASSED")


def test_step_returns_valid_transition():
    env = EmeraldEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()
    print("test_step_returns_valid_transition PASSED")


def test_frame_stack_advances():
    """The three stacked frames must not all be the same frame forever."""
    env = EmeraldEnv()
    obs, _ = env.reset()
    for _ in range(10):
        obs, *_ = env.step(4)  # press A, advances dialogue
    newest, oldest = obs["screen"][2], obs["screen"][0]
    assert not np.array_equal(newest, oldest), "frame stack is not advancing"
    env.close()
    print("test_frame_stack_advances PASSED")


if __name__ == "__main__":
    test_action_space_is_seven_discrete()
    test_observation_matches_declared_space()
    test_reset_is_deterministic()
    test_step_returns_valid_transition()
    test_frame_stack_advances()
    print("\nall env tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_env.py 2>/dev/null`
Expected: FAIL with `ModuleNotFoundError: No module named 'env'`

- [ ] **Step 3: Write the implementation**

Create `env.py`:

```python
"""Gymnasium environment for Pokemon Emerald on mGBA."""

import contextlib
import os
from collections import deque

import gymnasium as gym
import mgba.image
import numpy as np
from mgba._pylib import ffi
from pygba import PyGBA
from pygba.utils import KEY_MAP

from state import POKEDEX_CAPACITY, GameState

SCREEN_SHAPE = (3, 80, 120)
STATE_SIZE = 17


@contextlib.contextmanager
def suppress_stdout():
    """mGBA logs from C at fd 1. Python-level redirection does not catch it."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class EmeraldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    # No in-battle flag: pygba exposes no battle address, and a full-screen
    # battle is trivially visible to the CNN given the 3-frame stack.
    ACTIONS = [None, "up", "down", "left", "right", "A", "B"]

    def __init__(
        self,
        rom_path: str = "Pokemon - Emerald Version (USA, Europe).gba",
        state_path: str = "boot.state",
        frameskip: int = 24,
        max_steps: int = 16384,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.frameskip = frameskip
        self.max_steps = max_steps
        self.render_mode = render_mode

        with open(state_path, "rb") as f:
            self._boot_state = f.read()

        with suppress_stdout():
            self.gba = PyGBA.load(rom_path)
            self.width, self.height = self.gba.core.desired_video_dimensions()
            self._framebuffer = mgba.image.Image(self.width, self.height)
            self.gba.core.set_video_buffer(self._framebuffer)
            self.gba.core.reset()

        self.state_reader = GameState(self.gba)
        self._frames = deque(maxlen=3)
        self._step_count = 0

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, SCREEN_SHAPE, dtype=np.uint8),
            "state": gym.spaces.Box(-1.0, 1.0, (STATE_SIZE,), dtype=np.float32),
        })

    def _raw_frame(self) -> np.ndarray:
        """Read the framebuffer directly. Byte-identical to the PIL path and
        76x faster (0.001ms vs 0.076ms)."""
        arr = np.frombuffer(ffi.buffer(self._framebuffer.buffer), dtype=np.uint8)
        return arr.reshape(self.height, self._framebuffer.stride, 4)[:, :self.width, :3]

    def _gray_downscaled(self) -> np.ndarray:
        """240x160 RGB -> 120x80 grayscale. Green channel with 2x stride: one
        slice, no copy, no extra dependency. Green is a fine luminance proxy."""
        return np.ascontiguousarray(self._raw_frame()[::2, ::2, 1])

    def _encode_state(self, s: dict) -> np.ndarray:
        levels = s["party_levels"]
        vec = np.zeros(STATE_SIZE, dtype=np.float32)
        for i in range(8):
            vec[i] = 1.0 if i < s["badges"] else 0.0
        vec[8] = s["party_hp_frac"]
        vec[9] = (sum(levels) / len(levels) / 100.0) if levels else 0.0
        vec[10] = s["map"][0] / 50.0
        vec[11] = s["map"][1] / 100.0
        vec[12] = s["pos"][0] / 100.0
        vec[13] = s["pos"][1] / 100.0
        vec[14] = np.log1p(s["money"]) / np.log1p(999_999)
        vec[15] = s["seen"] / POKEDEX_CAPACITY
        vec[16] = s["caught"] / POKEDEX_CAPACITY
        return np.clip(vec, -1.0, 1.0)

    def _observation(self, s: dict) -> dict:
        return {
            "screen": np.stack(self._frames, axis=0),
            "state": self._encode_state(s),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        with suppress_stdout():
            self.gba.core.reset()
            self.gba.core.load_raw_state(self._boot_state)
            self.gba.core.run_frame()

        frame = self._gray_downscaled()
        self._frames.clear()
        for _ in range(3):
            self._frames.append(frame)

        s = self.state_reader.read()
        return self._observation(s), {}

    def step(self, action: int):
        key = self.ACTIONS[action]
        with suppress_stdout():
            if key is None:
                self.gba.core.set_keys()
            else:
                self.gba.core.set_keys(KEY_MAP[key])
            for _ in range(self.frameskip):
                self.gba.core.run_frame()

        self._frames.append(self._gray_downscaled())
        self._step_count += 1

        s = self.state_reader.read()
        obs = self._observation(s)

        reward = 0.0  # Task 4
        terminated = False
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._raw_frame().copy()
        return None

    def save_state(self) -> bytes:
        """Dump the emulator state. Used across process boundaries by the
        training callback via SubprocVecEnv.env_method."""
        with suppress_stdout():
            return bytes(ffi.buffer(self.gba.core.save_raw_state()))

    def close(self):
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_env.py 2>/dev/null`
Expected: PASS, all five tests.

If `test_frame_stack_advances` fails, the game is on a static screen where pressing A changes nothing. Confirm `boot.state` was saved with the player in free control, not mid-cutscene.

- [ ] **Step 5: Commit**

```bash
git add env.py test_env.py
git commit -m "feat: Emerald gym env with observations and actions"
```

---

### Task 4: Reward

Adds the reward function and episode termination to the existing environment.

**Files:**
- Modify: `env.py`
- Modify: `test_env.py`

**Interfaces:**
- Consumes: `EmeraldEnv` from Task 3, `GameState.read()` from Task 1.
- Produces:
  - `EmeraldEnv.TILE_CAP_PER_MAP = 400`
  - `EmeraldEnv._compute_reward(self, s: dict) -> float`
  - `info` dict from `step()` gains keys: `badges` (int), `maps_visited` (int), `tiles_visited` (int), `furthest_map` (tuple).

- [ ] **Step 1: Write the failing test**

Append to `test_env.py`, and add the new calls to the `__main__` block:

```python
def test_new_tile_rewards_once():
    """Revisiting a tile must not pay again."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    key = (s["map"][0], s["map"][1], s["pos"][0], s["pos"][1])

    env._visited_tiles.clear()
    env._tiles_per_map.clear()
    first = env._compute_reward(s)
    second = env._compute_reward(s)

    assert first > second, f"first visit {first} should beat revisit {second}"
    assert key in env._visited_tiles
    env.close()
    print("test_new_tile_rewards_once PASSED")


def test_tile_reward_capped_per_map():
    env = EmeraldEnv()
    env.reset()
    base = env.state_reader.read()
    m = base["map"]

    total = 0.0
    for i in range(EmeraldEnv.TILE_CAP_PER_MAP + 50):
        s = dict(base, pos=(i % 256, i // 256))
        total += env._compute_reward(s)

    assert env._tiles_per_map[m] == EmeraldEnv.TILE_CAP_PER_MAP, (
        f"cap not enforced: {env._tiles_per_map[m]}"
    )
    env.close()
    print("test_tile_reward_capped_per_map PASSED")


def test_reward_finite_over_random_rollout():
    env = EmeraldEnv()
    env.reset()
    total = 0.0
    for _ in range(1000):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r), "non-finite reward"
        total += r
        if term or trunc:
            env.reset()
    print(f"test_reward_finite_over_random_rollout PASSED (total {total:.2f})")
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_env.py 2>/dev/null`
Expected: FAIL with `AttributeError: 'EmeraldEnv' object has no attribute '_visited_tiles'`

- [ ] **Step 3: Write the implementation**

In `env.py`, add the constants to the class body next to `ACTIONS`:

```python
    TILE_CAP_PER_MAP = 400

    BADGE_REWARD = 100.0
    NEW_MAP_REWARD = 2.0
    NEW_TILE_REWARD = 0.05
    SCRIPT_FLAG_REWARD = 1.0
    TRAINER_REWARD = 2.0
    LEVEL_REWARD = 0.2
    SEEN_REWARD = 0.1
    CAUGHT_REWARD = 0.5
    WHITEOUT_PENALTY = -5.0
```

In `__init__`, after `self._step_count = 0`, add:

```python
        self._visited_tiles: set[tuple[int, int, int, int]] = set()
        self._visited_maps: set[tuple[int, int]] = set()
        self._tiles_per_map: dict[tuple[int, int], int] = {}
        self._prev: dict | None = None
        self._furthest_map = (0, 0)
```

Add the reward method:

```python
    def _compute_reward(self, s: dict) -> float:
        reward = 0.0
        prev = self._prev

        map_key = s["map"]
        if map_key not in self._visited_maps:
            self._visited_maps.add(map_key)
            reward += self.NEW_MAP_REWARD

        tile_key = (map_key[0], map_key[1], s["pos"][0], s["pos"][1])
        if tile_key not in self._visited_tiles:
            count = self._tiles_per_map.get(map_key, 0)
            # Cap stops the agent farming reward by pacing a large open route.
            if count < self.TILE_CAP_PER_MAP:
                self._visited_tiles.add(tile_key)
                self._tiles_per_map[map_key] = count + 1
                reward += self.NEW_TILE_REWARD

        if prev is not None:
            reward += self.BADGE_REWARD * max(0, s["badges"] - prev["badges"])
            reward += self.SCRIPT_FLAG_REWARD * max(
                0, s["script_flag_count"] - prev["script_flag_count"]
            )
            reward += self.TRAINER_REWARD * max(
                0, s["trainer_flag_count"] - prev["trainer_flag_count"]
            )
            reward += self.LEVEL_REWARD * max(
                0, sum(s["party_levels"]) - sum(prev["party_levels"])
            )
            reward += self.SEEN_REWARD * max(0, s["seen"] - prev["seen"])
            reward += self.CAUGHT_REWARD * max(0, s["caught"] - prev["caught"])
            if s["whiteout"] and not prev["whiteout"]:
                reward += self.WHITEOUT_PENALTY

        if map_key > self._furthest_map:
            self._furthest_map = map_key

        self._prev = s
        return reward
```

Replace the body of `reset()` after the frame-stack fill with:

```python
        self._visited_tiles.clear()
        self._visited_maps.clear()
        self._tiles_per_map.clear()
        self._furthest_map = (0, 0)
        self._prev = None

        s = self.state_reader.read()
        self._prev = s
        return self._observation(s), self._info(s)
```

Replace the tail of `step()` (from `reward = 0.0`) with:

```python
        reward = self._compute_reward(s)
        terminated = bool(s["badges"] >= 1 or s["whiteout"])
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, self._info(s)
```

Add the info helper:

```python
    def _info(self, s: dict) -> dict:
        return {
            "badges": s["badges"],
            "maps_visited": len(self._visited_maps),
            "tiles_visited": len(self._visited_tiles),
            "furthest_map": self._furthest_map,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_env.py 2>/dev/null`
Expected: PASS, all eight tests.

- [ ] **Step 5: Commit**

```bash
git add env.py test_env.py
git commit -m "feat: reward with coordinate-novelty exploration bonus"
```

---

### Task 5: Random-agent exploration baseline

The spec's primary risk gate. Run this before any training. If a random agent never leaves the starting area, tile novelty alone cannot bridge the gap to Rustboro and the curriculum fallback becomes mandatory.

**Files:**
- Create: `baseline.py`

**Interfaces:**
- Consumes: `EmeraldEnv` from Task 4.
- Produces: `baseline.json` with keys `episodes`, `steps_per_episode`, `maps_reached` (list of `[group, num]`), `mean_tiles`, `max_tiles`, `left_start_map_fraction`.

- [ ] **Step 1: Write the baseline script**

Create `baseline.py`:

```python
"""Measure how far a random agent gets. This is the go/no-go gate on the
exploration reward, per the spec's primary risk."""

import json
from collections import Counter

from env import EmeraldEnv

EPISODES = 20
STEPS = 4096


def main():
    env = EmeraldEnv(max_steps=STEPS)
    maps_seen = Counter()
    tiles = []
    start_map = None
    left_start = 0

    for ep in range(EPISODES):
        obs, info = env.reset()
        if start_map is None:
            start_map = env.state_reader.read()["map"]

        episode_maps = set()
        for _ in range(STEPS):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            episode_maps.add(info["furthest_map"])
            if term or trunc:
                break

        maps_seen.update(episode_maps)
        tiles.append(info["tiles_visited"])
        if episode_maps - {start_map}:
            left_start += 1
        print(f"episode {ep+1}/{EPISODES}: {info['tiles_visited']} tiles, "
              f"{len(episode_maps)} maps, furthest {info['furthest_map']}")

    result = {
        "episodes": EPISODES,
        "steps_per_episode": STEPS,
        "maps_reached": sorted([list(m) for m in maps_seen]),
        "mean_tiles": sum(tiles) / len(tiles),
        "max_tiles": max(tiles),
        "left_start_map_fraction": left_start / EPISODES,
    }
    with open("baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nleft the starting map in {left_start}/{EPISODES} episodes")
    print(f"mean tiles {result['mean_tiles']:.1f}, max {result['max_tiles']}")
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python baseline.py 2>/dev/null`

Takes roughly 20 minutes (20 episodes x 4096 steps at ~85 steps/sec).

- [ ] **Step 3: Interpret the result and decide**

Read `left_start_map_fraction` in `baseline.json`:

- **Above 0.5**: random exploration escapes the starting map readily. Proceed to Task 6 unchanged.
- **Between 0.1 and 0.5**: workable but slow. Proceed to Task 6, and raise `NEW_TILE_REWARD` to `0.1`.
- **Below 0.1**: tile novelty cannot bridge the gap. **Stop and report to the user.** The curriculum fallback in the spec becomes mandatory, which is a design change requiring approval, not a silent implementation decision.

- [ ] **Step 4: Commit**

```bash
git add baseline.py baseline.json
git commit -m "test: random-agent exploration baseline"
```

---

### Task 6: Training

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `EmeraldEnv` from Task 4.
- Produces: `checkpoints/emerald_<steps>_steps.zip`, `runs/` TensorBoard logs, `checkpoints/furthest_<group>_<num>.state` savestates.

- [ ] **Step 1: Write the training script**

Create `train.py`:

```python
"""PPO training for the Emerald agent.

8 workers, not 16: measured throughput barely improves past 8 (262 vs 360
steps/sec) and the machine has ~12GB free RAM, so 16 risks swapping.
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import EmeraldEnv

N_ENVS = 8
N_STEPS = 1024  # 8 x 1024 x (3x80x120 + 17) ~ 236MB rollout buffer


def make_env(rank: int):
    def _init():
        return EmeraldEnv()
    return _init


class ProgressCallback(BaseCallback):
    """Logs progress metrics and dumps a savestate at each new furthest map.

    The savestates are the raw material for the spec's curriculum fallback.
    Workers live in separate processes, so the state is pulled back with
    env_method rather than read directly.
    """

    def __init__(self, save_dir: str = "checkpoints"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_map = (0, 0)

    def _on_step(self) -> bool:
        best_badges = 0
        best_tiles = 0
        for i, info in enumerate(self.locals.get("infos", [])):
            if "furthest_map" not in info:
                continue
            best_badges = max(best_badges, info["badges"])
            best_tiles = max(best_tiles, info["tiles_visited"])
            if info["furthest_map"] > self.best_map:
                self.best_map = info["furthest_map"]
                group, num = self.best_map
                blob = self.training_env.env_method("save_state", indices=[i])[0]
                path = self.save_dir / f"furthest_{group}_{num}.state"
                path.write_bytes(blob)
                self.logger.record("progress/new_furthest_map", float(num))
                print(f"new furthest map: group {group} num {num} "
                      f"at {self.num_timesteps} steps -> {path}")

        self.logger.record("progress/badges", best_badges)
        self.logger.record("progress/tiles_visited", best_tiles)
        self.logger.record("progress/furthest_map_num", float(self.best_map[1]))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log="runs")
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy",  # handles Dict obs: CNN for screen, MLP for state
            env,
            n_steps=N_STEPS,
            batch_size=512,
            n_epochs=3,
            gamma=0.999,       # long horizon: the badge is ~50k steps away
            ent_coef=0.01,     # keep exploring
            learning_rate=2.5e-4,
            tensorboard_log="runs",
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path="checkpoints",
            name_prefix="emerald",
        ),
        ProgressCallback(),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks,
                reset_num_timesteps=args.resume is None)
    model.save("checkpoints/emerald_final")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a tiny run**

Run: `.venv/bin/python train.py --steps 20000 2>/dev/null`

Expected: PPO prints its rollout table, no crash, `checkpoints/` gains a zip. Confirm memory stays under control with `free -g` in another shell while it runs.

- [ ] **Step 3: Verify TensorBoard logging works**

Run: `.venv/bin/python -c "
from pathlib import Path
runs = sorted(Path('runs').glob('*'))
assert runs, 'no tensorboard run directory created'
events = list(runs[-1].glob('events.*'))
assert events, 'no event file written'
print('tensorboard logging OK:', events[0])
"`

Expected: prints the event file path.

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: PPO training script"
```

---

### Task 7: Watch a trained agent

**Files:**
- Create: `watch.py`

**Interfaces:**
- Consumes: `EmeraldEnv` from Task 4, checkpoints from Task 6.
- Produces: nothing on disk; renders to a pygame window.

- [ ] **Step 1: Write the viewer**

Create `watch.py`:

```python
"""Render a trained checkpoint playing. Usage:
    python watch.py checkpoints/emerald_final.zip
"""

import argparse

import numpy as np
import pygame
from stable_baselines3 import PPO

from env import EmeraldEnv

SCALE = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=16384)
    args = parser.parse_args()

    env = EmeraldEnv(render_mode="rgb_array")
    model = PPO.load(args.checkpoint)

    pygame.init()
    screen = pygame.display.set_mode((env.width * SCALE, env.height * SCALE))
    pygame.display.set_caption("Emerald agent")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    for _ in range(args.steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))

        frame = env.render()
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (env.width * SCALE, env.height * SCALE))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        if terminated or truncated:
            print(f"episode end: badges {info['badges']}, "
                  f"tiles {info['tiles_visited']}, map {info['furthest_map']}")
            obs, info = env.reset()

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the smoke-test checkpoint**

Run: `.venv/bin/python watch.py checkpoints/emerald_final.zip 2>/dev/null`

Expected: a window showing the game being played. The agent will be terrible, having trained for 20k steps. You are verifying the plumbing, not the policy.

- [ ] **Step 3: Commit**

```bash
git add watch.py
git commit -m "feat: viewer for trained checkpoints"
```

---

## Deviations from the spec

Recorded here because they were decided during plan-writing, after measurement:

1. **The in-battle flag is dropped from the observation.** The spec flagged locating its address as an open implementation task. Rather than reverse-engineer it, the flag is omitted: a full-screen battle is visually unmistakable and the 3-frame stack gives the CNN the motion cues it needs. The state vector stays at 17 values because map ID splits into `mapGroup` and `mapNum`.

2. **`PokemonEmerald.reward()` is not used.** Measurement showed it reads ROM every step at 1.28ms per touch. The reward is reimplemented from RAM-only reads in `state.py`.

3. **A separate `state.py` exists**, where the spec folded everything into `env.py`. The ROM-caching constraint makes memory reading a distinct concern worth isolating, and it keeps both files small enough to hold in context.

4. **The policy uses SB3's `MultiInputPolicy`** rather than a hand-written
   extractor. Its `CombinedExtractor` runs NatureCNN over the screen and
   flattens the 17-value vector, concatenating both before the default
   two-layer policy MLP. The spec described an MLP applied to the state vector
   before concatenation. The difference is negligible for 17 values and avoids
   a custom extractor class.

5. **Pokedex counts come from popcounting the save block bitfields** rather than the species-to-dex-number mapping, which lives in ROM.
