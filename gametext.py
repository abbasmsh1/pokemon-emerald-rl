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

# Enough real letters to be text rather than a coincidental decode of graphics.
_MIN_ALPHA = 6
_MIN_ALPHA_RATIO = 0.5


def decode_at(gba, addr: int, max_len: int = MAX_LEN) -> str | None:
    """Decode an Emerald string at `addr`, or None if it does not look like text."""
    raw = gba.read_memory(addr, max_len)

    chars = []
    for byte in raw:
        if byte == TERMINATOR:
            break
        mapped = _CHARMAP.charmap[byte] if byte < len(_CHARMAP.charmap) else None
        # Multi-character entries are glyph names like "Pk"; control bytes map to
        # None. Both become a space so words do not run together.
        chars.append(mapped if mapped and len(mapped) == 1 else " ")

    text = "".join(chars).strip()
    if not text:
        return None

    alpha = sum(c.isalpha() for c in text)
    if alpha < _MIN_ALPHA or alpha < len(text) * _MIN_ALPHA_RATIO:
        # Graphics and save-block garbage decode to sparse letter soup. Rejecting
        # it here keeps nonsense out of the cache keys.
        return None

    return text


def normalise(text: str) -> str:
    """Cache-key form: one line, single spaces, accents folded.

    Folding means "POKeMON" and "POKéMON" share an entry, and a line-break
    difference does not create a second entry for the same sentence.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("'", "'")
    return re.sub(r"\s+", " ", folded).strip()


def read_message(gba) -> str | None:
    """The current on-screen message, normalised, or None if there is none."""
    text = decode_at(gba, MESSAGE_ADDR)
    return normalise(text) if text else None
