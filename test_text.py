"""Assertion-based checks for reading on-screen text from RAM. No framework."""

import time
from pathlib import Path

import text as textmod
from env import suppress_stdout
from pygba import PyGBA
from state import GameState
from text import MAX_TEXT, player_fled, read_screen_text

ROM = "Pokemon - Emerald Version (USA, Europe).gba"
# Committed savestate from a run that was mid-battle. The window addresses in
# text.py were found by scanning EWRAM here, so this pins them.
BATTLE_STATE = Path("checkpoints/furthest_7maps_1_4.state")


def _gba(state_path: Path):
    with suppress_stdout():
        gba = PyGBA.load(ROM)
        gba.core.reset()
        gba.core.load_raw_state(state_path.read_bytes())
        for _ in range(4):
            gba.core.run_frame()
    return gba


def test_known_battle_text_decodes():
    """The addresses in text.py must still contain readable game text.

    If this fails, the windows no longer bracket the message buffers and
    read_screen_text is returning nothing useful.
    """
    if not BATTLE_STATE.exists():
        print("test_known_battle_text_decodes SKIPPED (no battle savestate)")
        return
    textmod._CACHE_RAW = None
    out = read_screen_text(_gba(BATTLE_STATE))
    assert out, "no text decoded from the battle savestate"
    assert any(c.isalpha() for c in out), f"decoded garbage: {out!r}"
    assert len(out) <= MAX_TEXT, f"text not truncated: {len(out)}"
    print(f"test_known_battle_text_decodes PASSED ({out!r})")


def test_player_fled_matches_only_the_player():
    """A wild Pokemon fleeing is not the agent running away."""
    assert player_fled("Got away safely!")
    assert player_fled("BRETT got away safely!")
    assert not player_fled("ZIGZAGOON fled!"), "opponent fleeing must not count"
    assert not player_fled("TACKLE!")
    assert not player_fled("")
    print("test_player_fled_matches_only_the_player PASSED")


def test_cache_returns_same_text_and_is_faster():
    if not BATTLE_STATE.exists():
        print("test_cache_returns_same_text_and_is_faster SKIPPED")
        return
    gba = _gba(BATTLE_STATE)
    textmod._CACHE_RAW = None

    t0 = time.perf_counter()
    cold = read_screen_text(gba)
    cold_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for _ in range(50):
        warm = read_screen_text(gba)
    warm_ms = (time.perf_counter() - t0) * 1e3 / 50

    assert warm == cold, "cache returned different text"
    assert warm_ms < cold_ms, f"cache not faster: {warm_ms:.3f} vs {cold_ms:.3f}"
    print(f"test_cache_returns_same_text_and_is_faster PASSED "
          f"(cold {cold_ms:.3f}ms, warm {warm_ms:.3f}ms)")


def test_state_read_stays_within_step_budget():
    """A step has ~3.5ms at 8 workers; the state read must not eat it."""
    if not BATTLE_STATE.exists():
        print("test_state_read_stays_within_step_budget SKIPPED")
        return
    gba = _gba(BATTLE_STATE)
    gs = GameState(gba)
    gs.read()

    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
        gs.read()
    per_ms = (time.perf_counter() - t0) / 200 * 1e3

    baseline_frame_ms = 0.307
    overhead = per_ms - baseline_frame_ms
    assert overhead < 1.5, f"state read overhead {overhead:.3f}ms is too costly"
    print(f"test_state_read_stays_within_step_budget PASSED "
          f"(overhead {overhead:.3f}ms)")


def test_state_exposes_exp_and_text():
    if not BATTLE_STATE.exists():
        print("test_state_exposes_exp_and_text SKIPPED")
        return
    s = GameState(_gba(BATTLE_STATE)).read()
    assert "party_exp" in s and isinstance(s["party_exp"], int)
    assert s["party_exp"] >= 0
    assert "screen_text" in s and isinstance(s["screen_text"], str)
    print(f"test_state_exposes_exp_and_text PASSED (exp={s['party_exp']})")


def test_safe_defaults_before_boot():
    """Pre-boot reads must not raise or omit the new keys."""
    with suppress_stdout():
        gba = PyGBA.load(ROM)
    s = GameState(gba).read()
    assert s["party_exp"] == 0
    assert s["screen_text"] == ""
    print("test_safe_defaults_before_boot PASSED")


if __name__ == "__main__":
    test_player_fled_matches_only_the_player()
    test_safe_defaults_before_boot()
    test_known_battle_text_decodes()
    test_cache_returns_same_text_and_is_faster()
    test_state_exposes_exp_and_text()
    test_state_read_stays_within_step_budget()
    print("\nall text tests passed")
