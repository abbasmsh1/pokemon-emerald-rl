"""Checks for the ROM-backed game knowledge ("wiki"). No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import os as _os
_os.environ["EMERALD_START"] = "route103.state"   # tests start where they were written for: Route 103

from env import EmeraldEnv  # noqa: E402
from knowledge import STORY, Knowledge  # noqa: E402

NORMAL, FLYING, GHOST, FIRE, WATER, GRASS, DARK = 0, 2, 7, 10, 11, 12, 17


def _k(state=None):
    env = EmeraldEnv(state_path=state)
    env.reset()
    return env, Knowledge(env.gba)


def test_species_and_move_tables():
    env, k = _k()
    assert k.species_types(280) == (FIRE, FIRE)          # Torchic
    assert k.species_types(309) == (WATER, FLYING)       # Wingull
    assert k.species_types(286) == (DARK, DARK)          # Poochyena
    assert k.move(52) == (FIRE, 40) and k.move(55) == (WATER, 40) and k.move(45) == (NORMAL, 0)
    env.close()
    print("test_species_and_move_tables PASSED")


def test_type_chart():
    env, k = _k()
    assert k.effectiveness(FIRE, (GRASS, GRASS)) == 2.0
    assert k.effectiveness(WATER, (FIRE, FIRE)) == 2.0
    assert k.effectiveness(FIRE, (WATER, WATER)) == 0.5
    assert k.effectiveness(NORMAL, (GHOST, GHOST)) == 0.0
    assert k.effectiveness(NORMAL, (FIRE, FIRE)) == 1.0
    assert k.effectiveness(GRASS, (WATER, FLYING)) == 1.0       # x2 water, x0.5 flying
    env.close()
    print("test_type_chart PASSED")


def test_move_features_on_the_battle_fixture():
    """Torchic (Scratch, Growl, Focus Energy) against Wingull (Water/Flying):
    Scratch is neutral and damaging, Growl has no power."""
    env, k = _k("battle.state")
    s = env.state_reader.read()
    f = k.move_features(s["our_moves"], s["enemy_id"][0])
    assert f.shape == (8,)
    eff, power = f[:4], f[4:]
    i = list(s["our_moves"]).index(10)
    assert abs(eff[i] - 0.25) < 1e-6 and power[i] > 0, (eff, power)        # x1 -> 1/4
    assert power[list(s["our_moves"]).index(45)] == 0.0
    assert (f[:4] >= 0).all() and (f[:4] <= 1).all() and (f[4:] <= 1).all()
    assert not k.move_features((0, 0, 0, 0), 0).any(), "no battle: all zeros"
    env.close()
    print("test_move_features_on_the_battle_fixture PASSED")


def test_waypoints_and_exit_direction():
    env, k = _k()                                        # anchor: Route 103
    assert [st["name"] for st in STORY] == ["rival", "pokedex", "badge"]
    assert STORY[0]["flag"] == 0x82 and STORY[1]["flag"] == 0x861
    assert STORY[0]["goal"] == ((0, 18), (10, 3)) and STORY[2]["route"][-1] == (0, 3)
    assert k.exit_direction((0, 10)) == "south", k.exit_direction((0, 10))   # Route 103 -> Oldale
    assert k.exit_direction((0, 3)) is None              # Rustboro is not adjacent to Route 103
    env.close()
    print("test_waypoints_and_exit_direction PASSED")


def test_warps_wild_levels_and_evolutions():
    env, k = _k()                                        # Route 103
    assert ((45, 6), (24, 106)) in k.warps(), k.warps()
    assert k.wild_levels((0, 16)) == (2, 3)              # Route 101
    assert k.wild_levels((24, 11)) == (5, 6)             # Petalburg Woods
    assert k.wild_levels((0, 10)) is None                # Oldale: no grass
    assert k.levels_to_evolve(280, 7) == 9               # Torchic evolves at 16
    assert k.levels_to_evolve(277, 16) == 0              # Treecko at 16: due now
    env.close()
    print("test_warps_wild_levels_and_evolutions PASSED")


def test_route_to_a_map_by_edge_and_by_door():
    env, k = _k()
    pos = env.state_reader.read()["pos"]
    d, dist, door = k.route_to((0, 10), pos)             # Oldale is off the south edge
    assert d == "south" and not door and dist >= 0, (d, dist, door)
    d, dist, door = k.route_to((24, 106), pos)           # only reachable through the warp
    assert door and d in ("north", "south", "west", "east") and dist == abs(45 - pos[0]) + abs(6 - pos[1]), (d, dist, door)
    assert k.route_to((7, 7), pos) == (None, 0, False)   # unreachable from here
    env.close()
    print("test_route_to_a_map_by_edge_and_by_door PASSED")


if __name__ == "__main__":
    test_species_and_move_tables()
    test_type_chart()
    test_move_features_on_the_battle_fixture()
    test_waypoints_and_exit_direction()
    test_warps_wild_levels_and_evolutions()
    test_route_to_a_map_by_edge_and_by_door()
    print("\nall knowledge tests passed")
