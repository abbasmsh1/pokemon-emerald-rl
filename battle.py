"""Battle state from RAM: whether a battle is on, both sides' HP, our PP, the
level ratio, and whether each of our moves does damage.

Addresses verified by tools/probe_battle.py against battle.state, not taken
on trust. The battle Pokemon array is STALE outside a battle (it still holds
the last fight), so an in-battle flag gates everything: a byte in gMain that
is non-zero on the battle fixture, zero on the dialogue fixture and the
anchor, and zero again once the fight is played out.
"""

import struct

import numpy as np

BATTLE_MONS = 0x02024084        # gBattleMons; index 0 ours, 1 the enemy
ENTRY = 0x58
IN_BATTLE_FLAG = (0x030026F9, 0x02)   # gMain + 0x439, the bit set during a battle
MOVES_TABLE = 0x0831C898        # gBattleMoves in ROM, 12 bytes per move
MOVE_ENTRY = 12
MOVE_POWER_OFFSET = 1
MOVE_PP_OFFSET = 4
BATTLE_SIZE = 12
LEVEL_RATIO_CAP = 2.0


class BattleReader:
    """One per emulator core, like GameState's other readers."""

    def __init__(self, gba):
        self.gba = gba
        self._move_cache: dict[int, tuple[int, int]] = {}   # id -> (power, max pp)

    def in_battle(self) -> bool:
        addr, mask = IN_BATTLE_FLAG
        return bool(self.gba.read_u8(addr) & mask)

    def _entries(self) -> tuple[dict, dict]:
        raw = bytes(self.gba.read_memory(BATTLE_MONS, 2 * ENTRY))
        out = []
        for i in range(2):
            e = raw[i * ENTRY:(i + 1) * ENTRY]
            moves = list(struct.unpack_from("<4H", e, 0x0C))
            pp = list(e[0x24:0x28])
            hp, maxhp = struct.unpack_from("<H", e, 0x28)[0], struct.unpack_from("<H", e, 0x2C)[0]
            species = struct.unpack_from("<H", e, 0x00)[0]
            personality = struct.unpack_from("<I", e, 0x48)[0]
            out.append(dict(moves=moves, pp=pp, hp=hp, maxhp=maxhp, level=e[0x2A],
                            species=species, personality=personality))
        return out[0], out[1]

    def _move(self, move_id: int) -> tuple[int, int]:
        """(base power, max pp) from the ROM table; one memory-view read per
        distinct id, then cached. Id 0 is an empty slot."""
        if move_id == 0:
            return 0, 0
        if move_id not in self._move_cache:
            a = MOVES_TABLE + MOVE_ENTRY * move_id
            rec = bytes(self.gba.core.memory.u8[a:a + MOVE_ENTRY])
            self._move_cache[move_id] = (rec[MOVE_POWER_OFFSET], rec[MOVE_PP_OFFSET])
        return self._move_cache[move_id]

    def enemy_identity(self) -> tuple[int, int]:
        """(species, personality) of the current enemy, (0, 0) outside a battle.

        The array is stale between battles and a battle's first in-battle step
        still shows the previous foe, so the damage term needs to know when
        the foe changed: a new wild Pokemon or a trainer's switch-in always
        has a different personality."""
        if not self.in_battle():
            return 0, 0
        enemy = self._entries()[1]
        return enemy["species"], enemy["personality"]

    def our_move_ids(self) -> list[int]:
        return self._entries()[0]["moves"]

    def read(self) -> np.ndarray:
        v = np.zeros(BATTLE_SIZE, dtype=np.float32)
        if not self.in_battle():
            return v
        ours, enemy = self._entries()
        if ours["maxhp"] == 0 or enemy["maxhp"] == 0 or not (1 <= ours["level"] <= 100):
            return v
        v[0] = 1.0
        v[1] = min(enemy["hp"] / enemy["maxhp"], 1.0)
        v[2] = min(ours["hp"] / ours["maxhp"], 1.0)
        for k, (mid, pp) in enumerate(zip(ours["moves"], ours["pp"])):
            power, max_pp = self._move(mid)
            v[3 + k] = min(pp / max_pp, 1.0) if max_pp else 0.0
            v[8 + k] = 1.0 if power > 0 else 0.0
        v[7] = min(enemy["level"] / ours["level"], LEVEL_RATIO_CAP) / LEVEL_RATIO_CAP
        return v
