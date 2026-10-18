"""Checks for the on-screen text reader. No framework."""

from pathlib import Path

from pygba import PyGBA

from text import TextReader, FRESH_STEPS, our_attack, player_fled, wild_encounter
from env import EmeraldEnv, suppress_stdout

ROM = "Pokemon - Emerald Version (USA, Europe).gba"
DIALOGUE = Path("dialogue.state")   # Birch on the route, text box open, saved by a probe


def _env_at(state: Path) -> EmeraldEnv:
    env = EmeraldEnv(state_path=str(state))
    env.reset()
    return env


def test_reader_returns_the_printed_dialogue():
    """The string the printer is walking through, whole, substitutions applied.
    The old two-window dump returned 'TACKLE! RGY! ENERGY EXIT' here."""
    env = _env_at(DIALOGUE)
    text = env.state_reader.read()["screen_text"]
    low = text.lower()
    assert "footprints" in low and "rare pok" in low, f"dialogue not read: {text!r}"
    assert "tackle" not in low and "energy" not in low, f"stale fragments leaked in: {text!r}"
    env.close()
    print(f"test_reader_returns_the_printed_dialogue PASSED ({text[:50]!r})")


def test_text_goes_stale_but_a_visible_box_keeps_it():
    env = _env_at(DIALOGUE)
    reader = env.state_reader.text
    first = reader.read()
    assert first
    for _ in range(FRESH_STEPS + 2):
        reader.read()                      # nothing new is written
    assert reader.read() == "", "text never went stale"
    assert reader.keep_alive() == first, "keep_alive lost the last string"
    env.close()
    print("test_text_goes_stale_but_a_visible_box_keeps_it PASSED")


def test_env_keeps_dialogue_while_the_box_is_on_screen():
    """Through the env: pressing nothing for 20 steps with the box open must
    still report the dialogue, because the pixel gate keeps it alive."""
    env = _env_at(DIALOGUE)
    assert env._text_box_visible(), "fixture does not show a text box"
    text = ""
    for _ in range(20):
        env.step(0)
    text = env._prev["screen_text"]
    assert "footprints" in text.lower(), f"dialogue dropped while the box was open: {text!r}"
    env.close()
    print("test_env_keeps_dialogue_while_the_box_is_on_screen PASSED")


def test_flee_and_attack_detectors():
    assert player_fled("Got away safely!")
    assert not player_fled("Wild ZIGZAGOON fled!"), "a wild Pokemon fleeing is not our doing"
    assert our_attack("TORCHIC used SCRATCH!")
    assert our_attack("MUDKIP used WATER GUN!")
    assert not our_attack("Wild ZIGZAGOON used TACKLE!"), "the enemy's move is not our attack"
    assert not our_attack("Foe POOCHYENA used TACKLE!")
    assert not our_attack("What will TORCHIC do?")
    assert wild_encounter("Wild ZIGZAGOON appeared!")
    # As the printer actually delivers it: a stale fragment precedes the line.
    # An anchored pattern missed this on every battle, so the encounter reward
    # never paid and Stage 0 counted 0 encounters for all three agents.
    assert wild_encounter(": 0     Wild WINGULL appeared!")
    assert not wild_encounter("ZIGZAGOON appeared!") and not wild_encounter("Wild ZIGZAGOON fled!")
    print("test_flee_and_attack_detectors PASSED")


def test_state_read_stays_within_step_budget():
    """A step has ~3.5ms at 8 workers; the state read must not eat it."""
    import time
    env = _env_at(DIALOGUE)
    gba, gs = env.gba, env.state_reader
    gs.read()
    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
    base = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
        gs.read()
    per = (time.perf_counter() - t0) / 200 * 1e3
    overhead = per - base
    assert overhead < 1.5, f"state read overhead {overhead:.3f}ms is too costly"
    env.close()
    print(f"test_state_read_stays_within_step_budget PASSED (overhead {overhead:.3f}ms)")


if __name__ == "__main__":
    test_reader_returns_the_printed_dialogue()
    test_text_goes_stale_but_a_visible_box_keeps_it()
    test_env_keeps_dialogue_while_the_box_is_on_screen()
    test_flee_and_attack_detectors()
    test_state_read_stays_within_step_budget()
    print("\nall text tests passed")
