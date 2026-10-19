"""Fast RAM-only game state reader for Pokemon Emerald.

Deliberately avoids pygba's PokemonEmerald.game_state(), which reads ROM
tables on every call. PyGBA.read_memory copies an entire memory region into
Python on first touch per frame; for the 16MB ROM region that costs 1.28ms
per step against a 7.36ms step budget.
"""

import struct

import numpy as np

from pygba import PyGBA

from text import TextReader
from tilemap import TileMapReader
from battle import BATTLE_SIZE, BattleReader
from knowledge import STORY
from pygba.game_wrappers.utils.emerald_utils import (
    ADRESSES,
    SaveBlock2_spec,
    FLAG_BADGE01_GET,
    SCRIPT_FLAGS_START,
    TRAINER_FLAGS_START,
    SYSTEM_FLAGS_START,
    read_save_block_1,
    read_save_block_2,
)

POKEDEX_CAPACITY = 386

# Offset of the packed options half-word inside SaveBlock2, derived from
# pygba's own struct spec rather than hardcoded so it cannot drift if the
# spec changes. Resolves to 0x14, which matches pokeemerald.
_OPTIONS_OFFSET = struct.calcsize(
    "<" + "".join(
        fmt for name, fmt in SaveBlock2_spec[
            :[n for n, _ in SaveBlock2_spec].index("options")
        ]
    )
)
# Bits 0-2 of that half-word. 2 is OPTIONS_TEXT_SPEED_FAST.
TEXT_SPEED_MASK = 0b111
TEXT_SPEED_FAST = 2


def force_fast_text(gba: PyGBA) -> bool:
    """Pin TEXT SPEED to FAST, returning True if it had to write.

    The agent reached the OPTION screen and set text speed to SLOW, which
    multiplies the number of steps every mandatory dialogue costs, and most of
    what stands between it and the first badge is dialogue. Charging for the
    change is not enough on its own: the cost is paid once and the slow text
    then persists for millions of steps, so the setting is held down instead.

    Writes go through mgba's live memory view; PyGBA.read_* reads a per-frame
    cached copy, so a read back in the same frame still shows the old value.
    """
    ptr = gba.read_u32(ADRESSES["gSaveBlock2Ptr"])
    if ptr == 0:
        return False
    addr = ptr + _OPTIONS_OFFSET
    cur = gba.core.memory.u16[addr]
    if cur & TEXT_SPEED_MASK == TEXT_SPEED_FAST:
        return False
    gba.core.memory.u16[addr] = (cur & ~TEXT_SPEED_MASK) | TEXT_SPEED_FAST
    return True


def _get_flag(flags, flag_id: int) -> bool:
    if flag_id < 0 or flag_id // 8 >= len(flags):
        return False
    return bool((flags[flag_id // 8] >> (flag_id % 8)) & 1)


def _popcount(buf) -> int:
    return sum(b.bit_count() for b in buf)


class GameState:
    def __init__(self, gba: PyGBA):
        self.gba = gba
        self.text = TextReader(gba)
        self.tiles = TileMapReader(gba)
        self.battle = BattleReader(gba)
        self._last_good = None

    def reset(self) -> None:
        """Call at the start of each episode. Without this, a negative-map
        read on the new episode's first step would return the PREVIOUS
        episode's cached state (including its badges), which env.py's
        termination check would act on."""
        self._last_good = None

    def read(self) -> dict:
        sb1 = read_save_block_1(self.gba)
        sb2 = read_save_block_2(self.gba)

        if sb1 is None:
            return {
                "badges": 0,
                "map": (0, 0),
                "pos": (0, 0),
                "party_levels": [],
                "party_exp": 0,
                "lead": (0, 0),
                "heal_location": None,
                "story": (False, False, False),
                "screen_text": "",
                "party_hp_frac": 0.0,
                "money": 0,
                "seen": 0,
                "caught": 0,
                "script_flag_count": 0,
                "trainer_flag_count": 0,
                "whiteout": False,
                "options": (0, 0),
                "battle": np.zeros(BATTLE_SIZE, dtype=np.float32),
                "in_battle": False,
                "enemy_id": (0, 0),
                "our_moves": (0, 0, 0, 0),
            }

        flags = sb1["flags"]
        badges = sum(_get_flag(flags, FLAG_BADGE01_GET + i) for i in range(8))

        loc = sb1["location"]
        pos = sb1["pos"]

        party = sb1["playerParty"]
        # A slot can parse with box or substructs None while the game is
        # writing it (seen live: a worker died on it after 3.2M steps, and with
        # the fork start method the trainer then hung rather than crashed).
        # Treat it as the same kind of transient as a negative map id.
        if any(m.get("box") is None or m["box"].get("substructs") is None for m in party):
            if self._last_good is not None:
                return self._last_good
            party = []
        levels = [m["level"] for m in party]
        # Experience is the fine-grained battle signal. A level-up needs many
        # wins, so level alone is far too sparse to teach battling, and fleeing
        # earns exactly zero experience.
        party_exp = sum(m["box"]["substructs"][0]["experience"] for m in party)
        total_hp = sum(m["hp"] for m in party)
        total_max = sum(m["maxHp"] for m in party)
        hp_frac = (total_hp / total_max) if total_max > 0 else 0.0

        money = 0
        if sb2 is not None:
            money = sb1["money"] ^ sb2["encryptionKey"]
            # A corrupt XOR mid-write can produce absurd values; clamp to the
            # game's own maximum rather than feeding garbage to the network.
            money = max(0, min(money, 999_999))

        # The settings from the OPTION screen: optionsButtonMode plus the
        # packed half-word holding text speed, battle scene, battle style,
        # sound and frame. (0, 0) is the "no save block" sentinel, never a real
        # reading, so env.py can tell a failed read from a genuine change.
        options = (0, 0)
        if sb2 is not None:
            options = (sb2["optionsButtonMode"], sb2["options"])

        seen = caught = 0
        if sb2 is not None:
            seen = _popcount(sb2["pokedex"]["seen"])
            caught = _popcount(sb2["pokedex"]["owned"])

        script = flags[SCRIPT_FLAGS_START // 8:TRAINER_FLAGS_START // 8]
        trainer = flags[TRAINER_FLAGS_START // 8:SYSTEM_FLAGS_START // 8]

        result = {
            "badges": badges,
            "map": (loc["mapGroup"], loc["mapNum"]),
            "pos": (pos["x"], pos["y"]),
            "party_levels": levels,
            "party_exp": party_exp,
            # (species, level) of the lead Pokemon, for evolution knowledge.
            "lead": (party[0]["box"]["substructs"][0]["species"], party[0]["level"]) if party else (0, 0),
            # Where a whiteout returns the player: the last Pokemon Center used.
            "heal_location": ((sb1["lastHealLocation"]["mapGroup"], sb1["lastHealLocation"]["mapNum"]),
                              (sb1["lastHealLocation"]["x"], sb1["lastHealLocation"]["y"])),
            # Story flags, one per knowledge.STORY stage, in order.
            "story": tuple(_get_flag(flags, st["flag"]) for st in STORY),
            "screen_text": self.text.read(),
            "party_hp_frac": hp_frac,
            "options": options,
            "battle": self.battle.read(),
            "in_battle": self.battle.in_battle(),
            "enemy_id": self.battle.enemy_identity(),
            "our_moves": tuple(self.battle.our_move_ids()) if self.battle.in_battle() else (0, 0, 0, 0),
            "money": money,
            "seen": seen,
            "caught": caught,
            "script_flag_count": _popcount(script),
            "trainer_flag_count": _popcount(trainer),
            # Heuristic, not an engine flag: true whenever the party is
            # non-empty and total HP is zero. Good enough to gate episode
            # termination in env.py, but do not assume it mirrors any
            # in-game "whiteout" state/flag.
            "whiteout": len(party) > 0 and total_hp == 0,
        }

        # ponytail: save blocks relocate during map transitions and parse as garbage
        # for one step. mapGroup/mapNum are signed bytes, so a negative value is
        # impossible in a real map. Hold the last good read through the transient.
        # Ceiling: only catches transients that corrupt the map id; the reward's
        # delta clamp in env.py is the backstop for the rest.
        if result["map"][0] < 0 or result["map"][1] < 0:
            if self._last_good is not None:
                return self._last_good
        else:
            self._last_good = result
        return result
