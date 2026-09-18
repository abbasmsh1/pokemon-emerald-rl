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
