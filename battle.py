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

# Move names for the teacher's prompt: the ROM table holds no names. Ids
# 1-150 in the Gen 3 numbering (the Gen 1 list, unchanged); anything else
# falls back to "move <id>". Checked against the ROM's power/PP table on the
# fixture: 10 SCRATCH (40/35), 45 GROWL (0/40).
MOVE_NAMES: dict[int, str] = {
    1: "POUND",
    2: "KARATE CHOP",
    3: "DOUBLE SLAP",
    4: "COMET PUNCH",
    5: "MEGA PUNCH",
    6: "PAY DAY",
    7: "FIRE PUNCH",
    8: "ICE PUNCH",
    9: "THUNDER PUNCH",
    10: "SCRATCH",
    11: "VICE GRIP",
    12: "GUILLOTINE",
    13: "RAZOR WIND",
    14: "SWORDS DANCE",
    15: "CUT",
    16: "GUST",
    17: "WING ATTACK",
    18: "WHIRLWIND",
    19: "FLY",
    20: "BIND",
    21: "SLAM",
    22: "VINE WHIP",
    23: "STOMP",
    24: "DOUBLE KICK",
    25: "MEGA KICK",
    26: "JUMP KICK",
    27: "ROLLING KICK",
    28: "SAND-ATTACK",
    29: "HEADBUTT",
    30: "HORN ATTACK",
    31: "FURY ATTACK",
    32: "HORN DRILL",
    33: "TACKLE",
    34: "BODY SLAM",
    35: "WRAP",
    36: "TAKE DOWN",
    37: "THRASH",
    38: "DOUBLE-EDGE",
    39: "TAIL WHIP",
    40: "POISON STING",
    41: "TWINEEDLE",
    42: "PIN MISSILE",
    43: "LEER",
    44: "BITE",
    45: "GROWL",
    46: "ROAR",
    47: "SING",
    48: "SUPERSONIC",
    49: "SONIC BOOM",
    50: "DISABLE",
    51: "ACID",
    52: "EMBER",
    53: "FLAMETHROWER",
    54: "MIST",
    55: "WATER GUN",
    56: "HYDRO PUMP",
    57: "SURF",
    58: "ICE BEAM",
    59: "BLIZZARD",
    60: "PSYBEAM",
    61: "BUBBLE BEAM",
    62: "AURORA BEAM",
    63: "HYPER BEAM",
    64: "PECK",
    65: "DRILL PECK",
    66: "SUBMISSION",
    67: "LOW KICK",
    68: "COUNTER",
    69: "SEISMIC TOSS",
    70: "STRENGTH",
    71: "ABSORB",
    72: "MEGA DRAIN",
    73: "LEECH SEED",
    74: "GROWTH",
    75: "RAZOR LEAF",
    76: "SOLAR BEAM",
    77: "POISON POWDER",
    78: "STUN SPORE",
    79: "SLEEP POWDER",
    80: "PETAL DANCE",
    81: "STRING SHOT",
    82: "DRAGON RAGE",
    83: "FIRE SPIN",
    84: "THUNDER SHOCK",
    85: "THUNDERBOLT",
    86: "THUNDER WAVE",
    87: "THUNDER",
    88: "ROCK THROW",
    89: "EARTHQUAKE",
    90: "FISSURE",
    91: "DIG",
    92: "TOXIC",
    93: "CONFUSION",
    94: "PSYCHIC",
    95: "HYPNOSIS",
    96: "MEDITATE",
    97: "AGILITY",
    98: "QUICK ATTACK",
    99: "RAGE",
    100: "TELEPORT",
    101: "NIGHT SHADE",
    102: "MIMIC",
    103: "SCREECH",
    104: "DOUBLE TEAM",
    105: "RECOVER",
    106: "HARDEN",
    107: "MINIMIZE",
    108: "SMOKESCREEN",
    109: "CONFUSE RAY",
    110: "WITHDRAW",
    111: "DEFENSE CURL",
    112: "BARRIER",
    113: "LIGHT SCREEN",
    114: "HAZE",
    115: "REFLECT",
    116: "FOCUS ENERGY",
    117: "BIDE",
    118: "METRONOME",
    119: "MIRROR MOVE",
    120: "SELF-DESTRUCT",
    121: "EGG BOMB",
    122: "LICK",
    123: "SMOG",
    124: "SLUDGE",
    125: "BONE CLUB",
    126: "FIRE BLAST",
    127: "WATERFALL",
    128: "CLAMP",
    129: "SWIFT",
    130: "SKULL BASH",
    131: "SPIKE CANNON",
    132: "CONSTRICT",
    133: "AMNESIA",
    134: "KINESIS",
    135: "SOFT-BOILED",
    136: "HI JUMP KICK",
    137: "GLARE",
    138: "DREAM EATER",
    139: "POISON GAS",
    140: "BARRAGE",
    141: "LEECH LIFE",
    142: "LOVELY KISS",
    143: "SKY ATTACK",
    144: "TRANSFORM",
    145: "BUBBLE",
    146: "DIZZY PUNCH",
    147: "SPORE",
    148: "FLASH",
    149: "PSYWAVE",
    150: "SPLASH",
}


def move_name(move_id: int) -> str:
    return MOVE_NAMES.get(int(move_id), f"move {move_id}")


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
