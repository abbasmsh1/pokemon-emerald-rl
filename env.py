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

    TILE_CAP_PER_MAP = 400

    # ponytail: real play sets at most a handful of event flags per 24-frame step.
    # A jump far above that is a mid-relocation save-block parse, not progress.
    # Ceiling: a legitimate burst above this cap is silently dropped.
    MAX_FLAG_DELTA = 16

    BADGE_REWARD = 100.0
    NEW_MAP_REWARD = 2.0
    NEW_TILE_REWARD = 0.05
    SCRIPT_FLAG_REWARD = 1.0
    TRAINER_REWARD = 2.0
    LEVEL_REWARD = 0.2
    SEEN_REWARD = 0.1
    CAUGHT_REWARD = 0.5
    WHITEOUT_PENALTY = -5.0

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
        self._visited_tiles: set[tuple[int, int, int, int]] = set()
        self._visited_maps: set[tuple[int, int]] = set()
        self._tiles_per_map: dict[tuple[int, int], int] = {}
        self._prev: dict | None = None
        self._furthest_map = (0, 0)
        self._flag_baseline: dict[str, int] | None = None

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
        # 127.0 is exact for signed-byte map identifiers (mapGroup/mapNum are
        # "b", -128..127) and cannot saturate; 255.0 gives roughly 2x headroom
        # over the largest Hoenn maps for tile coordinates (x/y are "H").
        vec[10] = s["map"][0] / 127.0
        vec[11] = s["map"][1] / 127.0
        vec[12] = s["pos"][0] / 255.0
        vec[13] = s["pos"][1] / 255.0
        vec[14] = np.log1p(s["money"]) / np.log1p(999_999)
        vec[15] = s["seen"] / POKEDEX_CAPACITY
        vec[16] = s["caught"] / POKEDEX_CAPACITY
        return np.clip(vec, -1.0, 1.0)

    def _observation(self, s: dict) -> dict:
        return {
            "screen": np.stack(self._frames, axis=0),
            "state": self._encode_state(s),
        }

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

            # ponytail: a save-block relocation parses as garbage for one step, swinging
            # the flag popcount wildly in one direction and back the next. Measuring
            # against the last plausible reading means the garbage step pays nothing AND
            # does not poison the baseline, so the recovery step reads as a zero delta.
            # Ceiling: a legitimate burst above MAX_FLAG_DELTA is dropped, not deferred.
            if self._flag_baseline is not None:
                script_delta = s["script_flag_count"] - self._flag_baseline["script"]
                if abs(script_delta) <= self.MAX_FLAG_DELTA:
                    reward += self.SCRIPT_FLAG_REWARD * max(0, script_delta)
                    self._flag_baseline["script"] = s["script_flag_count"]

                trainer_delta = s["trainer_flag_count"] - self._flag_baseline["trainer"]
                if abs(trainer_delta) <= self.MAX_FLAG_DELTA:
                    reward += self.TRAINER_REWARD * max(0, trainer_delta)
                    self._flag_baseline["trainer"] = s["trainer_flag_count"]

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

    def _info(self, s: dict) -> dict:
        return {
            "badges": s["badges"],
            "maps_visited": len(self._visited_maps),
            "tiles_visited": len(self._visited_tiles),
            "furthest_map": self._furthest_map,
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

        self._visited_tiles.clear()
        self._visited_maps.clear()
        self._tiles_per_map.clear()
        self._furthest_map = (0, 0)
        self._prev = None

        s = self.state_reader.read()
        self._prev = s
        self._flag_baseline = {
            "script": s["script_flag_count"],
            "trainer": s["trainer_flag_count"],
        }
        return self._observation(s), self._info(s)

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

        reward = self._compute_reward(s)
        terminated = bool(s["badges"] >= 1 or s["whiteout"])
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, self._info(s)

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
