"""A tile-class window around the player, read from the game's own map grid.

Emerald keeps the current map's grid in RAM (one u16 per tile: metatile id in
the low 10 bits, collision in bits 10-11) and each tileset's attribute table
in ROM (one u16 per metatile; the behaviour is the low byte). Resolving the
tile under any coordinate is two lookups, and the behaviour byte says what
the tile is: tall grass, surfable water, a ledge, a door. This is the exact
segmentation a learned model would only approximate, and it is free.

All three addresses were found by tools/probe_tilemap.py, not taken on
trust: the layout struct by scanning IWRAM for {s32 w; s32 h; u16 *map}, the
map header by scanning EWRAM for a ROM layout whose size matches the backup
layout minus its 7-tile border, and the object-event array by finding the
record whose coordinates equal the player's position plus 7.
"""

import struct

import numpy as np

from pygba.game_wrappers.utils.emerald_utils import read_save_block_1

BACKUP_MAP_LAYOUT = 0x03005DC0   # {s32 width; s32 height; u16 *map}; width = map width + 15
MAP_HEADER = 0x02037318          # +0: const struct MapLayout *, in ROM
OBJECT_EVENTS = 0x02037350       # 16 records of OBJECT_EVENT_SIZE
OBJECT_EVENT_SIZE = 0x24
OBJECT_EVENT_COUNT = 16
BORDER = 7                       # the backup grid pads the map by 7 tiles each side...
MAP_OFFSET_W = 15                # ...plus one extra column: pokeemerald MAP_OFFSET_W = 2*7+1
MAP_OFFSET_H = 14                # and MAP_OFFSET_H = 2*7. Caught by the consistency test:
                                 # with 14 for both, x = width was read as a real tile.
ROM_LO, ROM_HI = 0x08000000, 0x0A000000

TILE_H = 11
TILE_W = 15
CLASSES = ("walkable", "blocked", "grass", "water", "ledge", "warp", "npc", "outside", "unknown")

# Behaviour bytes from pokeemerald include/constants/metatile_behaviors.h
# (enum order; MB_NORMAL is 0).
GRASS = {0x02, 0x03, 0x07, 0x09, 0x24}                       # tall, long, short, long-south-edge, ash
WATER = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x17, 0x18, 0x1A}
LEDGE = set(range(0x38, 0x40))                               # MB_JUMP_EAST .. MB_JUMP_SOUTHWEST
WARP = {0x0B, 0x0D, 0x0E, 0x1B, 0x29, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x67, 0x68,
        0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x8B, 0x8C, 0x8D, 0xBE, 0xEA}
NORMAL = 0x00


class TileMapReader:
    """One per emulator core, like GameState and TextReader."""

    def __init__(self, gba):
        self.gba = gba
        self._attr_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray] | None] = {}

    def _layout(self) -> tuple[int, int, int]:
        w, h, ptr = struct.unpack("<iiI", bytes(self.gba.read_memory(BACKUP_MAP_LAYOUT, 12)))
        return w, h, ptr

    def _tileset_pointers(self) -> tuple[int, int]:
        layout = self.gba.read_u32(MAP_HEADER)
        if not (ROM_LO <= layout < ROM_HI):
            return 0, 0
        return struct.unpack("<II", bytes(self.gba.core.memory.u8[layout + 0x10:layout + 0x18]))

    def _attributes(self):
        key = self._tileset_pointers()
        if key not in self._attr_cache:
            tables = []
            for ts in key:
                if not (ROM_LO <= ts < ROM_HI):
                    tables = None
                    break
                attr = struct.unpack("<I", bytes(self.gba.core.memory.u8[ts + 0x10:ts + 0x14]))[0]
                if not (ROM_LO <= attr < ROM_HI):
                    tables = None
                    break
                raw = bytes(self.gba.core.memory.u8[attr:attr + 2 * 512])
                tables.append(np.frombuffer(raw, dtype="<u2") & 0xFF)
            self._attr_cache[key] = tuple(tables) if tables else None
        return self._attr_cache[key]

    @staticmethod
    def _classify(value: int, tables) -> str:
        metatile = value & 0x3FF
        collision = (value >> 10) & 3
        if metatile < 512:
            behaviour = int(tables[0][metatile])
        else:
            behaviour = int(tables[1][metatile - 512])
        # Behaviour before collision for doors and ledges: a door tile carries
        # the collision bit yet the player walks into it (seen live: standing
        # in an Oldale doorway on a tile the bit called blocked), and a ledge
        # is passable in its jump direction.
        if behaviour in WARP:
            return "warp"
        if behaviour in LEDGE:
            return "ledge"
        if collision:
            return "blocked"
        if behaviour in GRASS:
            return "grass"
        if behaviour in WATER:
            return "water"
        if behaviour == NORMAL:
            return "walkable"
        return "blocked"

    def tile_class(self, x: int, y: int) -> str:
        """Class of one map coordinate (map coordinates, not grid-offset)."""
        w, h, ptr = self._layout()
        if not (0 <= x < w - MAP_OFFSET_W and 0 <= y < h - MAP_OFFSET_H):
            return "outside"
        tables = self._attributes()
        if tables is None:
            return "unknown"
        value = self.gba.read_u16(ptr + 2 * ((x + BORDER) + w * (y + BORDER)))
        return self._classify(value, tables)

    def _npc_tiles(self, player: tuple[int, int]) -> set[tuple[int, int]]:
        raw = bytes(self.gba.read_memory(OBJECT_EVENTS, OBJECT_EVENT_SIZE * OBJECT_EVENT_COUNT))
        tiles = set()
        for i in range(OBJECT_EVENT_COUNT):
            rec = raw[i * OBJECT_EVENT_SIZE:(i + 1) * OBJECT_EVENT_SIZE]
            if not rec[0] & 1:
                continue
            x, y = struct.unpack_from("<hh", rec, 0x10)
            tile = (x - BORDER, y - BORDER)
            if tile != player:
                tiles.add(tile)
        return tiles

    def window(self) -> np.ndarray:
        """(len(CLASSES), TILE_H, TILE_W) uint8 one-hot, centred on the player."""
        out = np.zeros((len(CLASSES), TILE_H, TILE_W), dtype=np.uint8)
        sb1 = read_save_block_1(self.gba)
        w, h, ptr = self._layout()
        tables = self._attributes() if sb1 is not None else None
        # No save block yet (the game has not started) or no resolvable
        # tileset: the whole window is "unknown", never a guess.
        if tables is None or not (0x02000000 <= ptr < 0x02040000):
            out[CLASSES.index("unknown")] = 1
            return out
        px, py = sb1["pos"]["x"], sb1["pos"]["y"]
        map_w, map_h = w - MAP_OFFSET_W, h - MAP_OFFSET_H
        x0, y0 = px - TILE_W // 2, py - TILE_H // 2
        # One read covering the window's rows of the padded grid.
        row0 = max(y0 + BORDER, 0)
        row1 = min(y0 + TILE_H + BORDER, h)
        grid = np.frombuffer(bytes(self.gba.read_memory(ptr + 2 * w * row0, 2 * w * (row1 - row0))), dtype="<u2")
        npcs = self._npc_tiles((px, py))
        for dy in range(TILE_H):
            y = y0 + dy
            for dx in range(TILE_W):
                x = x0 + dx
                if not (0 <= x < map_w and 0 <= y < map_h):
                    cls = "outside"
                elif (x, y) in npcs:
                    cls = "npc"
                else:
                    value = int(grid[(y + BORDER - row0) * w + (x + BORDER)])
                    cls = self._classify(value, tables)
                out[CLASSES.index(cls), dy, dx] = 1
        return out
