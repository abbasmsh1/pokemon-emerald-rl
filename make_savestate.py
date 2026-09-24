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
