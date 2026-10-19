"""Checks for the ROM-backed game knowledge ("wiki"). No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

from env import EmeraldEnv  # noqa: E402
from knowledge import WAYPOINTS, Knowledge  # noqa: E402

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
    assert WAYPOINTS[0] == (0, 9) and WAYPOINTS[-1] == (0, 3)
    assert k.exit_direction((0, 10)) == "south", k.exit_direction((0, 10))   # Route 103 -> Oldale
    assert k.exit_direction((0, 3)) is None              # Rustboro is not adjacent to Route 103
    env.close()
    print("test_waypoints_and_exit_direction PASSED")


if __name__ == "__main__":
    test_species_and_move_tables()
    test_type_chart()
    test_move_features_on_the_battle_fixture()
    test_waypoints_and_exit_direction()
    print("\nall knowledge tests passed")
