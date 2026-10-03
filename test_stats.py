"""Assertion-based checks for the stats reporter. No framework.

Tests run against boot.state, which is committed and never changes, rather than
the live run's checkpoints, which move under you while training.
"""

from pathlib import Path

from stats import KNOWN_MAPS, map_name, milestone_of, read_stats

BOOT = Path("boot.state")


def test_milestone_parses_train_naming():
    assert milestone_of(Path("furthest_7maps_0_16.state")) == 7
    assert milestone_of(Path("furthest_12maps_1_4.state")) == 12
    assert milestone_of(Path("unrelated.state")) == 0, "unnamed files must sort first"
    print("test_milestone_parses_train_naming PASSED")


def test_map_name_falls_back_for_unknown_ids():
    assert map_name(0, 9) == "Littleroot Town"
    assert map_name(99, 99) == "map 99-99", "unknown maps must degrade, not raise"
    assert (0, 9) in KNOWN_MAPS
    print("test_map_name_falls_back_for_unknown_ids PASSED")


def test_read_stats_on_boot_state():
    """boot.state is Littleroot, no badges, no party, nothing caught."""
    r = read_stats(BOOT)

    expected = {
        "file", "map", "map_name", "pos", "badges", "money",
        "seen", "caught", "party", "cities", "gyms", "champion",
    }
    assert set(r) == expected, f"key mismatch: {set(r) ^ expected}"

    assert r["map"] == (0, 9), f"boot.state should be Littleroot, got {r['map']}"
    assert r["map_name"] == "Littleroot Town"
    assert r["badges"] == 0
    assert r["party"] == [], "no starter is obtained at boot.state"
    assert r["caught"] == 0
    assert r["champion"] is False
    assert isinstance(r["money"], int) and r["money"] >= 0
    print("test_read_stats_on_boot_state PASSED")


def test_party_entries_have_species_and_levels():
    """Shape check that survives boot.state having an empty party.

    Guards the decode path: if a party entry ever appears it must carry a named
    species and a plausible level, not a raw id or a garbage number.
    """
    r = read_stats(BOOT)
    for mon in r["party"]:
        assert isinstance(mon["species"], str) and mon["species"]
        assert 1 <= mon["level"] <= 100, f"implausible level {mon['level']}"
        assert 0 <= mon["hp"] <= mon["max_hp"]
    print(f"test_party_entries_have_species_and_levels PASSED ({len(r['party'])} mons)")


if __name__ == "__main__":
    test_milestone_parses_train_naming()
    test_map_name_falls_back_for_unknown_ids()
    test_read_stats_on_boot_state()
    test_party_entries_have_species_and_levels()
    print("\nall stats tests passed")
