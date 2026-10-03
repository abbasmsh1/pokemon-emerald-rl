"""Assertion-based checks for game text extraction. No framework."""

from gametext import MESSAGE_ADDR, normalise, decode_at, MAX_LEN


class _StubGBA:
    """Feeds decode_at exact bytes, so these tests pin behaviour rather
    than whatever the emulator happens to hold."""
    def __init__(self, data: bytes):
        self._data = data
    def read_memory(self, addr, size):
        return (self._data + b"\xff" * size)[:size]


def test_address_is_the_measured_one():
    """Pinned by measurement; a silent change would break every cache key."""
    assert MESSAGE_ADDR == 0x02021FC4
    print("test_address_is_the_measured_one PASSED")


def test_normalise_collapses_whitespace_and_breaks():
    raw = "The mover's POKeMON do all the work\n\nof moving us in.   Really!"
    out = normalise(raw)
    assert "\n" not in out, "line breaks survived normalisation"
    assert "  " not in out, "runs of spaces survived normalisation"
    assert out == normalise(raw + "   "), "trailing space changed the key"
    print("test_normalise_collapses_whitespace_and_breaks PASSED")


def test_control_byte_becomes_a_separator():
    # 0xD5='a', 0xD6='b', 0xFB=U+FFFD (control), 0xD7='c', 0xD8='d'
    # "ab" 0xFB "cd" -> the line break must not fuse the words
    from gametext import CONTROL_BYTES
    data = bytes([0xD5, 0xD6, 0xFB, 0xD7, 0xD8, 0xFF])
    out = decode_at(_StubGBA(data), 0)
    assert out is not None, "valid text was rejected"
    assert "�" not in out, f"replacement char reached the key: {out!r}"
    assert " " in out, f"control byte did not separate words: {out!r}"
    print("test_control_byte_becomes_a_separator PASSED")


def test_unterminated_read_returns_none():
    from gametext import MAX_LEN, decode_at
    data = bytes([0xD5]) * (MAX_LEN + 10)   # no 0xFF anywhere
    assert decode_at(_StubGBA(data), 0) is None, "truncated text became a key"
    print("test_unterminated_read_returns_none PASSED")


def test_kana_is_rejected_as_graphics():
    from gametext import decode_at
    # 0x2F is 'あ' in the charmap: kana here means we decoded graphics
    data = bytes([0x2F] * 12 + [0xFF])
    assert decode_at(_StubGBA(data), 0) is None, "kana soup became a key"
    print("test_kana_is_rejected_as_graphics PASSED")


def test_short_real_dialogue_is_kept():
    from gametext import decode_at
    # "Okay!" (0xC9='O', 0xDF='k', 0xD5='a', 0xED='y', 0xAB='!') must survive
    data = bytes([0xC9, 0xDF, 0xD5, 0xED, 0xAB, 0xFF])
    out = decode_at(_StubGBA(data), 0)
    assert out is not None, "short real dialogue was rejected"
    print("test_short_real_dialogue_is_kept PASSED")


def test_typographic_apostrophe_is_folded():
    from gametext import normalise
    out = normalise("It’s a POKéMON")
    assert "’" not in out, f"typographic apostrophe survived: {out!r}"
    assert "'" in out, f"apostrophe lost entirely: {out!r}"
    print("test_typographic_apostrophe_is_folded PASSED")


if __name__ == "__main__":
    test_address_is_the_measured_one()
    test_normalise_collapses_whitespace_and_breaks()
    test_control_byte_becomes_a_separator()
    test_unterminated_read_returns_none()
    test_kana_is_rejected_as_graphics()
    test_short_real_dialogue_is_kept()
    test_typographic_apostrophe_is_folded()
    print("\nall gametext tests passed")
