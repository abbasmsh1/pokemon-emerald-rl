"""Fast RAM-only game state reader for Pokemon Emerald.

Deliberately avoids pygba's PokemonEmerald.game_state(), which reads ROM
tables on every call. PyGBA.read_memory copies an entire memory region into
Python on first touch per frame; for the 16MB ROM region that costs 1.28ms
per step against a 7.36ms step budget.
"""

from pygba import PyGBA

from text import read_screen_text
from pygba.game_wrappers.utils.emerald_utils import (
    FLAG_BADGE01_GET,
    SCRIPT_FLAGS_START,
    TRAINER_FLAGS_START,
    SYSTEM_FLAGS_START,
    read_save_block_1,
    read_save_block_2,
)

POKEDEX_CAPACITY = 386


def _get_flag(flags, flag_id: int) -> bool:
    if flag_id < 0 or flag_id // 8 >= len(flags):
        return False
    return bool((flags[flag_id // 8] >> (flag_id % 8)) & 1)


def _popcount(buf) -> int:
    return sum(b.bit_count() for b in buf)


class GameState:
    def __init__(self, gba: PyGBA):
        self.gba = gba
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
                "screen_text": "",
                "party_hp_frac": 0.0,
                "money": 0,
                "seen": 0,
                "caught": 0,
                "script_flag_count": 0,
                "trainer_flag_count": 0,
                "whiteout": False,
            }

        flags = sb1["flags"]
        badges = sum(_get_flag(flags, FLAG_BADGE01_GET + i) for i in range(8))

        loc = sb1["location"]
        pos = sb1["pos"]

        party = sb1["playerParty"]
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
            "screen_text": read_screen_text(self.gba),
            "party_hp_frac": hp_frac,
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
