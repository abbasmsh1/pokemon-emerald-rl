"""Assertion-based checks for game text extraction. No framework."""

from pygba import PyGBA

from env import suppress_stdout
from gametext import MESSAGE_ADDR, normalise, read_message

ROM = "Pokemon - Emerald Version (USA, Europe).gba"


def _booted():
    """A savestate with free player control and no dialogue showing."""
    with suppress_stdout():
        gba = PyGBA.load(ROM)
        gba.core.reset()
        gba.core.load_raw_state(open("boot.state", "rb").read())
        gba.core.run_frame()
    return gba


def test_address_is_the_measured_one():
    """Pinned by measurement; a silent change would break every cache key."""
    assert MESSAGE_ADDR == 0x02021FC4
    print("test_address_is_the_measured_one PASSED")


def test_read_message_returns_str_or_none():
    gba = _booted()
    msg = read_message(gba)
    assert msg is None or isinstance(msg, str)
    if msg is not None:
        assert "\x00" not in msg, "raw terminator leaked into the text"
    print(f"test_read_message_returns_str_or_none PASSED ({msg!r})")


def test_normalise_collapses_whitespace_and_breaks():
    raw = "The mover's POKeMON do all the work\n\nof moving us in.   Really!"
    out = normalise(raw)
    assert "\n" not in out, "line breaks survived normalisation"
    assert "  " not in out, "runs of spaces survived normalisation"
    assert out == normalise(raw + "   "), "trailing space changed the key"
    print("test_normalise_collapses_whitespace_and_breaks PASSED")


def test_normalise_handles_non_ascii_without_raising():
    """Review Focus 1. The real buffer contains POKeMON with an accent and a
    typographic apostrophe; neither may raise or produce replacement junk."""
    raw = "The mover's POKéMON do all the work"
    out = normalise(raw)
    assert out, "non-ascii text normalised to nothing"
    assert "�" not in out, "replacement character leaked into the key"
    print("test_normalise_handles_non_ascii_without_raising PASSED")


def test_garbage_buffer_reads_as_none():
    """Review Focus 2. During a map transition the buffer parses as garbage.

    Simulated by reading an address that holds non-text data; the function must
    return None rather than a nonsense string that would pollute the cache.
    """
    gba = _booted()
    from gametext import decode_at

    junk = decode_at(gba, 0x03000000)
    assert junk is None or len(junk) == 0 or not junk.strip(), (
        f"non-text region decoded to {junk!r}, which would become a cache key"
    )
    print("test_garbage_buffer_reads_as_none PASSED")


if __name__ == "__main__":
    test_address_is_the_measured_one()
    test_normalise_collapses_whitespace_and_breaks()
    test_normalise_handles_non_ascii_without_raising()
    test_read_message_returns_str_or_none()
    test_garbage_buffer_reads_as_none()
    print("\nall gametext tests passed")
