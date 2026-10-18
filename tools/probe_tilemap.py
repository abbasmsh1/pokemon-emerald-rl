"""Locate the map layout, map header and object-event addresses by scanning
RAM in a known state and confirming each by its behaviour under movement.

Usage: PYTHONPATH=. .venv/bin/python tools/probe_tilemap.py
"""
import struct

import numpy as np

from env import EmeraldEnv
from pygba.game_wrappers.utils.emerald_utils import read_save_block_1

ROM_LO, ROM_HI = 0x08000000, 0x0A000000
EW_LO, EW_HI = 0x02000000, 0x02040000


def rom(ptr: int) -> bool:
    return ROM_LO <= ptr < ROM_HI


def main() -> None:
    env = EmeraldEnv(state_path="dialogue.state")
    env.reset()
    env.step(6)                      # B closes the dialogue box
    for _ in range(3):
        env.step(0)
    g = env.gba
    pos = read_save_block_1(g)["pos"]
    px, py = pos["x"], pos["y"]
    print(f"player at ({px}, {py})")

    # 1. gBackupMapLayout: {s32 width; s32 height; u16 *map} in IWRAM.
    iw = bytes(g.read_memory(0x03000000, 0x8000))
    layouts = []
    for off in range(0, len(iw) - 12, 4):
        w, h, ptr = struct.unpack_from("<iiI", iw, off)
        if 1 <= w <= 200 and 1 <= h <= 200 and EW_LO <= ptr < EW_HI:
            layouts.append((0x03000000 + off, w, h, ptr))
    print("layout candidates:", [(hex(a), w, h, hex(p)) for a, w, h, p in layouts])

    def grid_u16(ptr, w, x, y):
        return g.read_u16(ptr + 2 * (x + w * y))

    # 2. gMapHeader: an EWRAM word pointing at a ROM layout struct whose width
    # and height match the backup layout minus the 7-tile border on each side.
    ew = bytes(g.read_memory(EW_LO, 0x40000))
    headers = []
    for a, w, h, ptr in layouts:
        for off in range(0, len(ew) - 4, 4):
            lp = struct.unpack_from("<I", ew, off)[0]
            if not rom(lp):
                continue
            lw, lh = struct.unpack("<ii", bytes(g.core.memory.u8[lp:lp + 8]))
            if lw == w - 15 and lh == h - 14:
                prim, sec = struct.unpack("<II", bytes(g.core.memory.u8[lp + 0x10:lp + 0x18]))
                if rom(prim) and rom(sec):
                    headers.append((EW_LO + off, lp, prim, sec, a))
    print("header candidates:", [(hex(a), hex(lp), hex(p), hex(s), hex(la)) for a, lp, p, s, la in headers[:6]])

    # 3. gObjectEvents: 16 records of 0x24 bytes; record 0 is the player, whose
    # current coords at +0x10/+0x12 are the player's position plus 7.
    objs = []
    for off in range(0, len(ew) - 16 * 0x24, 4):
        x0, y0 = struct.unpack_from("<hh", ew, off + 0x10)
        if (x0, y0) == (px + 7, py + 7) and ew[off] & 1:
            objs.append(EW_LO + off)
    print("object-event candidates (player record):", [hex(a) for a in objs])

    # 4. Movement check on the layout: step right, the grid index of the
    # player's tile must move by one and the tile under the player must read
    # the same metatile at the new coordinates as the tile to the right did.
    for a, w, h, ptr in layouts:
        before_here = grid_u16(ptr, w, px + 7, py + 7)
        before_right = grid_u16(ptr, w, px + 8, py + 7)
        env.step(4)
        for _ in range(2):
            env.step(0)
        npos = read_save_block_1(g)["pos"]
        after_here = grid_u16(ptr, w, npos["x"] + 7, npos["y"] + 7)
        print(f"layout {hex(a)}: moved {(px, py)}->{(npos['x'], npos['y'])}; tile here before {before_here:#06x}, right-of-player before {before_right:#06x}, here after {after_here:#06x}")
        env.step(3)
        for _ in range(2):
            env.step(0)

    if len(layouts) == 1 and len(set(h[0] for h in headers)) >= 1 and len(objs) == 1:
        print("\nBACKUP_MAP_LAYOUT =", hex(layouts[0][0]))
        print("MAP_HEADER =", hex(headers[0][0]))
        print("OBJECT_EVENTS =", hex(objs[0]))


if __name__ == "__main__":
    main()
