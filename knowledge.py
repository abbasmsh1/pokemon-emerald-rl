"""Game knowledge the agent cannot learn cheaply, read from the ROM: the
"wiki" and the walkthrough.

Everything here comes from the game's own tables, so nothing is downloaded and
nothing can disagree with the game. Each address was verified against facts
that are not in doubt (tools probe, 2026-10-19):

    species table  0x083203CC  28 bytes/species, type1 +6, type2 +7
                               Torchic 280 Fire/Fire, Wingull 309 Water/Flying
    move table     0x0831C898  12 bytes/move, power +1, type +2
                               Ember 52 Fire 40, Water Gun 55 Water 40
    type chart     0x0831ACE8  (attacker, defender, x10) triples, 0xFE
                               separator, 0xFF end; Fire->Grass 20, Normal->Ghost 0
    map header     0x02037318  +0x0C -> {s32 count; MapConnection *list},
                               12 bytes each: dir +0, mapGroup +8, mapNum +9
                               Route 103 south -> Oldale (dir 1)
"""

import struct

import numpy as np

SPECIES_TABLE = 0x083203CC
SPECIES_ENTRY = 28
MOVES_TABLE = 0x0831C898
MOVE_ENTRY = 12
TYPE_CHART = 0x0831ACE8
MAP_HEADER = 0x02037318
CONNECTION_DIRS = {1: "south", 2: "north", 3: "west", 4: "east"}   # dive/emerge (5, 6) ignored
DIRECTIONS = ("north", "south", "west", "east")
MAX_POWER = 150.0

# The story's route to the first badge, as map ids. Repeats are real: the
# rival battle on Route 103 sends the player back through Oldale before the
# footprints man stops blocking Route 102. Petalburg Woods sits inside Route
# 104 and the gym inside Rustboro; both are found from there by exploring.
WAYPOINTS = (
    (0, 9),    # Littleroot Town
    (0, 16),   # Route 101
    (0, 10),   # Oldale Town
    (0, 18),   # Route 103 (rival)
    (0, 10),   # Oldale Town again
    (0, 17),   # Route 102
    (0, 0),    # Petalburg City
    (0, 19),   # Route 104
    (0, 3),    # Rustboro City
)


class Knowledge:
    """ROM lookups, cached: the tables never change during a run."""

    def __init__(self, gba):
        self.gba = gba
        self._species: dict[int, tuple[int, int]] = {}
        self._moves: dict[int, tuple[int, int]] = {}
        self._chart: dict[tuple[int, int], float] | None = None

    def _rom(self, addr: int, n: int) -> bytes:
        return bytes(self.gba.core.memory.u8[addr:addr + n])

    def species_types(self, species: int) -> tuple[int, int]:
        if species not in self._species:
            e = self._rom(SPECIES_TABLE + SPECIES_ENTRY * species, SPECIES_ENTRY)
            self._species[species] = (e[6], e[7])
        return self._species[species]

    def move(self, move_id: int) -> tuple[int, int]:
        """(type, base power); (0, 0) for an empty slot."""
        if move_id == 0:
            return 0, 0
        if move_id not in self._moves:
            e = self._rom(MOVES_TABLE + MOVE_ENTRY * move_id, MOVE_ENTRY)
            self._moves[move_id] = (e[2], e[1])
        return self._moves[move_id]

    def _type_chart(self) -> dict[tuple[int, int], float]:
        if self._chart is None:
            raw, chart = self._rom(TYPE_CHART, 600), {}
            for i in range(0, len(raw) - 2, 3):
                if raw[i] == 0xFF:
                    break
                if raw[i] == 0xFE:                 # Foresight separator, not an entry
                    continue
                chart[(raw[i], raw[i + 1])] = raw[i + 2] / 10.0
            self._chart = chart
        return self._chart

    def effectiveness(self, move_type: int, defender_types: tuple[int, int]) -> float:
        chart = self._type_chart()
        m = chart.get((move_type, defender_types[0]), 1.0)
        if defender_types[1] != defender_types[0]:
            m *= chart.get((move_type, defender_types[1]), 1.0)
        return m

    def move_features(self, move_ids, enemy_species: int) -> np.ndarray:
        """(8,) float32: effectiveness of each of our four moves against the
        enemy divided by 4 (so x4 is 1.0, x1 is 0.25, immune 0), then base
        power / 150. All zeros outside a battle or for empty slots, and a
        status move (power 0) reads effectiveness 0 since it deals no damage."""
        out = np.zeros(8, dtype=np.float32)
        if not enemy_species:
            return out
        enemy = self.species_types(enemy_species)
        for k, mid in enumerate(move_ids):
            mtype, power = self.move(mid)
            if mid and power:
                out[k] = self.effectiveness(mtype, enemy) / 4.0
                out[4 + k] = min(power / MAX_POWER, 1.0)
        return out

    def exit_direction(self, target_map: tuple[int, int]) -> str | None:
        """Which edge of the current map leads straight to `target_map`, or
        None when it is not a direct connection (reached through a door, or
        not adjacent)."""
        ptr = self.gba.read_u32(MAP_HEADER + 0x0C)
        if not 0x08000000 <= ptr < 0x0A000000:
            return None
        count, lst = struct.unpack("<iI", self._rom(ptr, 8))
        if not 0 <= count <= 8 or not 0x08000000 <= lst < 0x0A000000:
            return None
        for k in range(count):
            e = self._rom(lst + 12 * k, 12)
            if (e[8], e[9]) == tuple(target_map) and e[0] in CONNECTION_DIRS:
                return CONNECTION_DIRS[e[0]]
        return None
