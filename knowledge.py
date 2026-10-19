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
                               +0x04 -> MapEvents: warpCount +1, warps* +8,
                               8 bytes each {s16 x, s16 y, u8 elev, u8 id,
                               u8 mapNum, u8 mapGroup}; Route 103 (45,6)->(24,106)
    wild headers   0x08552D48  20 bytes/map {u8 group, u8 num, pad, land*, ...}
                               0xFF-terminated; land* -> {u8 rate; mons*},
                               12 x {u8 min, u8 max, u16 species};
                               Route 101 levels 2-3, Petalburg Woods 5-6
    evolutions     0x0832531C  40 bytes/species = 5 x {u16 method, u16 param,
                               u16 target, pad}; method 4 = by level;
                               Torchic/Treecko -> 16
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
WILD_HEADERS = 0x08552D48
EVOLUTION_TABLE = 0x0832531C
EVO_LEVEL = 4
BACKUP_MAP_LAYOUT = 0x03005DC0     # {s32 w; s32 h; ...}; map size = w - 15, h - 14

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

    # --- doors, encounters, evolutions (2026-10-19) -----------------------

    def warps(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """((x, y), destination map) for every warp on the current map."""
        ev = self.gba.read_u32(MAP_HEADER + 0x04)
        if not 0x08000000 <= ev < 0x0A000000:
            return []
        head = self._rom(ev, 12)
        count, ptr = head[1], struct.unpack_from("<I", head, 8)[0]
        if not 0x08000000 <= ptr < 0x0A000000 or count > 64:
            return []
        raw = self._rom(ptr, 8 * count)
        out = []
        for k in range(count):
            x, y, _el, _id, num, grp = struct.unpack_from("<hhBBBB", raw, 8 * k)
            out.append(((x, y), (grp, num)))
        return out

    def _map_size(self) -> tuple[int, int]:
        w, h = struct.unpack("<ii", bytes(self.gba.read_memory(BACKUP_MAP_LAYOUT, 8)))
        return w - 15, h - 14

    def route_to(self, target_map, pos) -> tuple[str | None, int, bool]:
        """(compass direction, distance in tiles, via a door) from `pos` on the
        current map toward `target_map`: off a connected edge, else through a
        warp that leads there, else (None, 0, False)."""
        x, y = pos
        edge = self.exit_direction(target_map)
        if edge is not None:
            w, h = self._map_size()
            dist = {"north": y, "south": h - 1 - y, "west": x, "east": w - 1 - x}[edge]
            return edge, max(dist, 0), False
        doors = [(abs(wx - x) + abs(wy - y), wx, wy) for (wx, wy), dest in self.warps() if dest == tuple(target_map)]
        if doors:
            dist, wx, wy = min(doors)
            return compass(wx - x, wy - y), dist, True
        return None, 0, False

    def point_to(self, target_xy, pos) -> tuple[str | None, int]:
        dx, dy = target_xy[0] - pos[0], target_xy[1] - pos[1]
        return compass(dx, dy), abs(dx) + abs(dy)

    def _wild_table(self) -> dict:
        if not hasattr(self, "_wild"):
            self._wild = {}
            a = WILD_HEADERS
            for _ in range(512):
                e = self._rom(a, 20)
                if e[0] == 0xFF:
                    break
                land = struct.unpack_from("<I", e, 4)[0]
                if land:
                    mons = struct.unpack_from("<I", self._rom(land, 8), 4)[0]
                    lv = self._rom(mons, 48)
                    levels = [lv[4 * q] for q in range(12)] + [lv[4 * q + 1] for q in range(12)]
                    self._wild[(e[0], e[1])] = (min(levels), max(levels))
                a += 20
        return self._wild

    def wild_levels(self, map_key) -> tuple[int, int] | None:
        """(min, max) wild level in this map's grass, or None without grass."""
        return self._wild_table().get(tuple(map_key))

    def levels_to_evolve(self, species: int, level: int) -> int | None:
        """Levels until `species` evolves by level-up; None if it does not."""
        if not species:
            return None
        raw = self._rom(EVOLUTION_TABLE + 40 * species, 40)
        need = [struct.unpack_from("<HH", raw, 8 * k)[1] for k in range(5)
                if struct.unpack_from("<H", raw, 8 * k)[0] == EVO_LEVEL]
        return max(min(need) - level, 0) if need else None


def compass(dx: int, dy: int) -> str | None:
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"
