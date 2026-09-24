"""Render a trained checkpoint playing. Usage:
    python watch.py checkpoints/emerald_final.zip
"""

import argparse

# ponytail: importing torch before any mgba core exists makes libmgba's
# run_frame() spin forever. Creating one core first inoculates the process.
# Only the ORDER matters: verified that the inoculation survives this core
# being dereferenced and garbage collected, so the name is for clarity only.
# This block must stay above the torch/SB3 imports below.
# Ceiling: costs one extra 16MB ROM copy at startup.
from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

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
                  f"tiles {info['tiles_visited']}, map {info['map']}")
            obs, info = env.reset()

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
