"""Read the game's on-screen message out of RAM and decode it.

The address was found by measurement, not guessed: EWRAM was scanned with
EmeraldCharmap while known dialogue was displayed, and 0x02021FC4 was the
address whose contents tracked the dialogue as it advanced.

EWRAM only. A ROM read would cost 1.28ms against a 7.36ms step budget.
"""

import re
import unicodedata

from pygba.game_wrappers.utils.emerald_utils import EmeraldCharmap

MESSAGE_ADDR = 0x02021FC4
MAX_LEN = 256
TERMINATOR = 0xFF

_CHARMAP = EmeraldCharmap()

# Emerald's control bytes (line break, paragraph, scroll, and the
# parameterised 0xFC/0xFD forms) all decode to U+FFFD via the charmap.
CONTROL_BYTES = frozenset({0xFA, 0xFB, 0xFC, 0xFD, 0xFE})

_MIN_ALPHA = 3
_MIN_ALPHA_RATIO = 0.5


def _looks_like_text(text: str) -> bool:
    """Reject graphics data and kana that would pollute cache keys."""
    if "�" in text:
        return False
    # Kana in this buffer means we decoded graphics data, not English
    # dialogue. The ROM is the English release.
    if any("぀" <= c <= "ヿ" for c in text):
        return False
    ascii_alpha = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)
    return ascii_alpha >= _MIN_ALPHA and ascii_alpha >= len(text) * _MIN_ALPHA_RATIO


def decode_at(gba, addr: int, max_len: int = MAX_LEN) -> str | None:
    """Decode an Emerald string at `addr`, or None if it does not look like text."""
    raw = gba.read_memory(addr, max_len)

    chars = []
    saw_terminator = False
    for byte in raw:
        if byte == TERMINATOR:
            saw_terminator = True
            break
        if byte in CONTROL_BYTES:
            chars.append(" ")
            continue
        mapped = _CHARMAP.charmap[byte] if byte < len(_CHARMAP.charmap) else None
        chars.append(mapped if mapped and len(mapped) == 1 else " ")

    if not saw_terminator:
        # No 0xFF within MAX_LEN: either a longer page than we read, or not a
        # string at all. A truncated key would collide with nothing and
        # pollute the cache, so decline rather than guess.
        return None

    text = "".join(chars).strip()
    if not text:
        return None

    if not _looks_like_text(text):
        return None

    return text


def normalise(text: str) -> str:
    """Cache-key form: one line, single spaces, accents folded.

    Folding means "POKeMON" and "POKéMON" share an entry, and a line-break
    difference does not create a second entry for the same sentence.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("’", "'")
    return re.sub(r"\s+", " ", folded).strip()


def read_message(gba) -> str | None:
    """The current on-screen message, normalised, or None if there is none."""
    text = decode_at(gba, MESSAGE_ADDR)
    return normalise(text) if text else None
