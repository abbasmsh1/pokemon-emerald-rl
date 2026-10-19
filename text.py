"""Reads the text the game is printing on screen, straight from RAM. No OCR.

Emerald composes every dialogue line, sign, battle message and menu label
into a string and hands it to a text printer, and the printer walks a
pointer through that string one character per frame. gTextPrinters[0]'s
first field (0x020201B0 in the US ROM) is that pointer. Follow it to the
string it is inside and you have exactly what is on screen, with every
substitution already applied, whether the string lives in gStringVar4
(dialogue), gDisplayedStringBattle (battles) or straight in the ROM (menus,
signs).

An earlier version dumped two fixed RAM windows instead. That read the FIRST
60 bytes of gStringVar4 and every stale fragment around it, so the agent was
fed 'TACKLE! RGY! ENERGY EXIT' for days while the screen said something else.
Found by looking at the viewer's text line against the screen.

Both addresses were found empirically, not assumed: a scan of RAM for
pointers whose target decodes as text, taken while a dialogue box was open,
returned 0x020201B0 -> "You throw a BALL now, right?", and tracking it through
a conversation showed it advancing through gStringVar4 a line at a time.
"""

import re

from pygba.game_wrappers.utils.emerald_utils import EmeraldCharmap

PRINTER_CURRENT_CHAR = 0x020201B0   # gTextPrinters[0].printerTemplate.currentChar
TERMINATOR = 0xFF
MAX_SCAN = 400                      # longest string we will follow, either way
MAX_TEXT = 160
# A string stays "on screen" this many steps after it was last written, unless
# the caller can see a text box and keeps it alive longer. Battle messages
# auto-advance within a few steps; dialogue waits for A and is kept by the box.
FRESH_STEPS = 12

_PRINTABLE = re.compile(r"[A-Za-z0-9 .,!?'\"\-:;/]+")
_CHARMAP = EmeraldCharmap()


def _clean(decoded: str) -> str:
    # Control codes decode to odd glyphs; keep the readable runs, joined by spaces.
    return " ".join(_PRINTABLE.findall(decoded)).strip()


class TextReader:
    """Follows the text printer. One per emulator core, like GameState."""

    def __init__(self, gba):
        self.gba = gba
        self._start = None          # address of the string being printed
        self._end = None
        self.last = ""              # most recent decoded string, stale or not
        self._fresh = 0

    def _string_bounds(self, ptr: int) -> tuple[int, int]:
        """Walk back to the previous terminator and forward to the next."""
        mem = self.gba.core.memory.u8
        back = bytes(mem[max(ptr - MAX_SCAN, 0):ptr])
        start = ptr - (len(back) - back.rfind(bytes([TERMINATOR])) - 1) if TERMINATOR in back else ptr - len(back)
        fwd = bytes(mem[ptr:ptr + MAX_SCAN])
        end = ptr + (fwd.find(bytes([TERMINATOR])) if TERMINATOR in fwd else len(fwd))
        return start, end

    def read(self) -> str:
        """Text on screen right now, or '' once it has gone stale."""
        ptr = self.gba.read_u32(PRINTER_CURRENT_CHAR)
        in_ram = 0x02000000 <= ptr < 0x02040000
        in_rom = 0x08000000 <= ptr < 0x0A000000
        if not (in_ram or in_rom):
            self._fresh = max(self._fresh - 1, 0)
            return self.last if self._fresh else ""

        if self._start is None or not (self._start <= ptr <= self._end):
            self._start, self._end = self._string_bounds(ptr)
            raw = bytes(self.gba.core.memory.u8[self._start:self._end])
            try:
                text = _clean(_CHARMAP.decode(raw + bytes([TERMINATOR])))[:MAX_TEXT]
            except Exception:
                text = ""
            if text != self.last:
                self.last = text
                self._fresh = FRESH_STEPS
            return self.last
        self._fresh = max(self._fresh - 1, 0)
        return self.last if self._fresh else ""

    def keep_alive(self) -> str:
        """For the caller that can see a text box: the last string, regardless
        of freshness."""
        self._fresh = FRESH_STEPS
        return self.last


# The player choosing to run. Distinct from a wild Pokemon fleeing on its own,
# which is not the agent's doing and must not be penalised.
_PLAYER_FLED = re.compile(r"got away safely", re.IGNORECASE)
# Our Pokemon attacking: "TORCHIC used SCRATCH!" but not "Wild ZIGZAGOON used
# TACKLE!" or "Foe MAY's TORCHIC used ...".
_OUR_ATTACK = re.compile(r"^(?!wild\b)(?!foe\b)\S+ used \S", re.IGNORECASE)


# A wild battle starting. Encounters happen only in tall grass, water and
# caves, so this is "the agent is roaming where wild Pokemon are" without
# resolving the metatile under the player.
# Not anchored: the printer hands the line over with a stale fragment in front
# (": 0     Wild WINGULL appeared!" on the battle fixture), and the anchored
# version never matched a real battle start.
_WILD_APPEARED = re.compile(r"\bwild \S+ appeared", re.IGNORECASE)


_EVOLVING = re.compile(r"\bis evolving", re.IGNORECASE)
_EVOLVED = re.compile(r"\bevolved into\b", re.IGNORECASE)
_SUPER = re.compile(r"super effective", re.IGNORECASE)
_NOT_VERY = re.compile(r"not very effective", re.IGNORECASE)
_ANY_MOVE = re.compile(r"\S+ used \S", re.IGNORECASE)


def evolving(text: str) -> bool:
    return bool(_EVOLVING.search(text))


def evolved(text: str) -> bool:
    return bool(_EVOLVED.search(text))


def super_effective(text: str) -> bool:
    return bool(_SUPER.search(text))


def not_very_effective(text: str) -> bool:
    return bool(_NOT_VERY.search(text))


def someone_attacked(text: str) -> bool:
    return bool(_ANY_MOVE.search(text))


def wild_encounter(text: str) -> bool:
    return bool(_WILD_APPEARED.search(text))


def player_fled(text: str) -> bool:
    return bool(_PLAYER_FLED.search(text))


_MOVE_NAME = re.compile(r"\bused ([A-Z0-9 .'-]+?)!", re.IGNORECASE)


def move_used(text: str) -> str | None:
    """The move name in "X used NAME!", upper-cased, or None."""
    m = _MOVE_NAME.search(text)
    return m.group(1).strip().upper() if m else None


def our_attack(text: str) -> bool:
    return bool(_OUR_ATTACK.search(text))
