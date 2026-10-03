import numpy as np
from pygba import PyGBA
from state import GameState, POKEDEX_CAPACITY

ROM = "Pokemon - Emerald Version (USA, Europe).gba"


def test_read_returns_expected_keys_and_types():
    gba = PyGBA.load(ROM)
    gba.wait(3000)
    gs = GameState(gba)
    s = gs.read()

    expected = {
        "badges", "map", "pos", "party_levels", "party_hp_frac",
        "money", "seen", "caught", "script_flag_count",
        "trainer_flag_count", "whiteout", "party_exp", "screen_text",
    }
    assert set(s) == expected, f"key mismatch: {set(s) ^ expected}"

    assert isinstance(s["badges"], int) and 0 <= s["badges"] <= 8
    assert isinstance(s["map"], tuple) and len(s["map"]) == 2
    assert isinstance(s["pos"], tuple) and len(s["pos"]) == 2
    assert isinstance(s["party_levels"], list)
    assert isinstance(s["party_hp_frac"], float) and 0.0 <= s["party_hp_frac"] <= 1.0
    assert isinstance(s["money"], int) and s["money"] >= 0
    assert isinstance(s["party_exp"], int) and s["party_exp"] >= 0
    assert isinstance(s["screen_text"], str)
    assert 0 <= s["seen"] <= POKEDEX_CAPACITY
    assert 0 <= s["caught"] <= POKEDEX_CAPACITY
    assert isinstance(s["whiteout"], bool)
    print("test_read_returns_expected_keys_and_types PASSED")


def test_read_before_boot_returns_safe_defaults():
    """Save blocks do not exist at frame 0. read() must not raise."""
    gba = PyGBA.load(ROM)
    gs = GameState(gba)
    s = gs.read()
    assert s["badges"] == 0
    assert s["party_levels"] == []
    assert s["party_hp_frac"] == 0.0
    assert s["whiteout"] is False
    print("test_read_before_boot_returns_safe_defaults PASSED")


def test_read_does_not_touch_rom():
    """A ROM read costs 1.28ms. read() must stay far under that."""
    import time
    gba = PyGBA.load(ROM)
    gba.wait(3000)
    gs = GameState(gba)
    gs.read()

    # Measure baseline for bare run_frame() calls
    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
    baseline_frame = (time.perf_counter() - t0) / 200

    # Measure with read() calls
    t0 = time.perf_counter()
    for _ in range(200):
        gba.core.run_frame()
        gs.read()
    per_call = (time.perf_counter() - t0) / 200

    overhead = per_call - baseline_frame
    assert overhead < 0.5e-3, f"read() overhead {overhead*1e3:.3f}ms suggests a ROM read"
    print(f"test_read_does_not_touch_rom PASSED (overhead {overhead*1e3:.3f}ms)")


if __name__ == "__main__":
    test_read_before_boot_returns_safe_defaults()
    test_read_returns_expected_keys_and_types()
    test_read_does_not_touch_rom()
    print("\nall state tests passed")
