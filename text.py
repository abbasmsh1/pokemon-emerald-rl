"""Read the game's on-screen text straight out of RAM.

No OCR. Pokemon Emerald stores displayed text as charmap-encoded bytes, and
pygba already ships the decoder, so the text can be read exactly and for free
rather than inferred from a 240x160 framebuffer.

The two windows below were found empirically, by scanning EWRAM while a battle
was on screen and decoding every run of charmap bytes. That scan turned up
'ZIGZAGOON fled!' at 0x2022e34 and 'MUDKIP' at 0x2021e98. They are deliberately
windows rather than exact symbol addresses: this project has already lost hours
to guessed emulator addresses, and a window that brackets the observed hits
survives small layout differences that a single guessed pointer would not.

test_text.py pins them by asserting a known string decodes from a committed
savestate, so if they ever stop containing the text the tests fail loudly.
"""

import re

from pygba.game_wrappers.utils.emerald_utils import EmeraldCharmap

# String-variable region: species names and other substituted words.
STRINGVAR_START = 0x02021C00
STRINGVAR_LEN = 0x400

# Battle and dialogue message buffer.
MESSAGE_START = 0x02022C00
MESSAGE_LEN = 0x400

TERMINATOR = 0xFF
MIN_RUN = 4
MAX_TEXT = 64

_PRINTABLE = re.compile(r"[A-Za-z0-9 .,!?'\"\-:;/]+")
_CHARMAP = EmeraldCharmap()


def _decode_window(raw: bytes) -> list[str]:
    """Every plausible string in a window, in address order."""
    found = []
    i = 0
    while i < len(raw):
        if raw[i] == TERMINATOR:
            i += 1
            continue
        j = i
        while j < len(raw) and raw[j] != TERMINATOR and j - i < 120:
            j += 1
        if j - i >= MIN_RUN:
            try:
                decoded = _CHARMAP.decode(raw[i:j + 1])
            except Exception:
                decoded = ""
            text = decoded.strip()
            if text and _PRINTABLE.fullmatch(text) and sum(c.isalpha() for c in text) >= 3:
                found.append(text)
        i = j + 1
    return found


# ponytail: decoding both windows every step costs ~1ms of a ~3.5ms budget,
# because the scan is a Python loop over 2KB. Dialogue changes rarely, so the
# raw bytes are compared against the previous step and the decode is skipped on
# a match. Comparing 2KB is a memcmp; decoding it is not.
# Ceiling: one cache slot, so two alternating texts would miss every step.
_CACHE_RAW: bytes | None = None
_CACHE_TEXT: str = ""


def read_screen_text(gba) -> str:
    """Concatenated on-screen text, truncated to MAX_TEXT characters.

    Both windows sit in EWRAM, which the state reader already touches every
    step, so pygba's per-frame region cache is warm and the reads are slices.
    """
    global _CACHE_RAW, _CACHE_TEXT

    raw = (gba.read_memory(MESSAGE_START, MESSAGE_LEN)
           + gba.read_memory(STRINGVAR_START, STRINGVAR_LEN))
    if raw == _CACHE_RAW:
        return _CACHE_TEXT

    parts = _decode_window(raw[:MESSAGE_LEN]) + _decode_window(raw[MESSAGE_LEN:])
    _CACHE_RAW = raw
    _CACHE_TEXT = " ".join(parts)[:MAX_TEXT]
    return _CACHE_TEXT


# The player choosing to run. Distinct from a wild Pokemon fleeing on its own,
# which is not the agent's doing and must not be penalised.
_PLAYER_FLED = re.compile(r"got away safely", re.IGNORECASE)


def player_fled(text: str) -> bool:
    return bool(_PLAYER_FLED.search(text))
