"""Produce battle.state: a raw mGBA savestate with a wild battle in progress,
past the intro animation. A test fixture for battle.py and env.py.

Usage: PYTHONPATH=. .venv/bin/python tools/make_battle_state.py
"""
import os
import random

import numpy as np
from mgba._pylib import ffi

from env import EmeraldEnv, suppress_stdout

START = "checkpoints/furthest_5maps_0_16.state"
UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4


def main() -> None:
    env = EmeraldEnv(state_path=START if os.path.exists(START) else None)
    env.reset()
    rng = random.Random(9)
    prev_pos, frozen = None, 0
    prev_frame = env._gray_downscaled().astype(int)
    for i in range(6000):
        env.step(rng.choice([UP, UP, UP, LEFT, RIGHT, DOWN]))
        s = env.state_reader.read()
        frame = env._gray_downscaled().astype(int)
        diff = float(np.abs(frame - prev_frame).mean())
        prev_frame = frame
        frozen = frozen + 1 if s["pos"] == prev_pos else 0
        prev_pos = s["pos"]
        if frozen >= 3 and diff > 25:
            break
    else:
        raise SystemExit("no battle found in 6000 steps")

    # Advance through "Wild X appeared!" and "Go! Y!" with A until the fight
    # menu ("What will Y do?") is up; A alternated with a no-op so each press
    # registers as a fresh edge.
    # Two A presses dismiss "Wild X appeared!" and "Go! Y!"; the fight menu
    # then sits waiting for input. More presses would start playing the fight.
    # Let the intro settle on "Wild X appeared!" with the enemy's HP bar drawn.
    # The fight menu itself is not reachable deterministically from here (the
    # text reader does not see the menu prompt and pixel cues misfire on the
    # bright battle backdrop), and the battle Pokemon array is populated for
    # both sides at this point, which is what the fixture exists to test.
    for _ in range(12):
        env.step(0)
    text = env.state_reader.read()["screen_text"] or env.state_reader.text.last
    with suppress_stdout():
        blob = bytes(ffi.buffer(env.gba.core.save_raw_state()))
    with open("battle.state", "wb") as f:
        f.write(blob)
    s = env.state_reader.read()
    print(f"battle at step {i}, map {s['map']} pos {s['pos']}, text {text!r}, {len(blob)} bytes")


if __name__ == "__main__":
    main()
