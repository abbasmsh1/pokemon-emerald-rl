"""Assertion-based checks for the battle state reader. No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import os as _os
_os.environ["EMERALD_START"] = "route103.state"   # tests start where they were written for: Route 103

import time  # noqa: E402

import numpy as np  # noqa: E402

from battle import BATTLE_SIZE, BattleReader  # noqa: E402
from env import EmeraldEnv  # noqa: E402


def _env(state: str) -> EmeraldEnv:
    env = EmeraldEnv(state_path=state)
    env.reset()
    return env


def test_in_battle_on_the_fixture_and_not_in_dialogue():
    b, d = _env("battle.state"), _env("dialogue.state")
    assert BattleReader(b.gba).in_battle(), "battle.state not read as in battle"
    assert not BattleReader(d.gba).in_battle(), "dialogue.state read as in battle"
    b.close(); d.close()
    print("test_in_battle_on_the_fixture_and_not_in_dialogue PASSED")


def test_vector_layout_in_battle():
    env = _env("battle.state")
    v = BattleReader(env.gba).read()
    assert v.shape == (BATTLE_SIZE,) and BATTLE_SIZE == 12 and v.dtype == np.float32
    assert v[0] == 1.0
    assert 0.0 < v[1] <= 1.0, f"enemy hp fraction {v[1]}"
    assert 0.0 < v[2] <= 1.0, f"our hp fraction {v[2]}"
    assert (0.0 <= v[3:7]).all() and (v[3:7] <= 1.0).all(), f"pp fractions {v[3:7]}"
    assert 0.0 <= v[7] <= 1.0, f"level ratio {v[7]}"
    assert set(v[8:12].tolist()) <= {0.0, 1.0}, f"move nature flags {v[8:12]}"
    env.close()
    print(f"test_vector_layout_in_battle PASSED ({np.round(v, 2).tolist()})")


def test_move_nature_marks_scratch_damaging_and_growl_status():
    """The fixture's Torchic knows Scratch (power 40) and Growl (power 0)."""
    env = _env("battle.state")
    r = BattleReader(env.gba)
    v = r.read()
    ids = r.our_move_ids()
    assert 10 in ids and 45 in ids, f"fixture moves changed: {ids}"
    assert v[8 + ids.index(10)] == 1.0, "Scratch not marked damaging"
    assert v[8 + ids.index(45)] == 0.0, "Growl marked damaging"
    assert v[8 + ids.index(0)] == 0.0 if 0 in ids else True, "empty slot marked damaging"
    env.close()
    print("test_move_nature_marks_scratch_damaging_and_growl_status PASSED")


def test_enemy_identity_changes_with_the_foe_and_is_zero_outside_battle():
    """Species and personality of the current enemy, so the damage term can
    tell a new foe's full bar from the last foe healing. Zero outside a
    battle, even though the array still holds the last foe."""
    b, d = _env("battle.state"), _env("dialogue.state")
    ident = BattleReader(b.gba).enemy_identity()
    assert ident[0] == 309 and ident[1] != 0, f"fixture enemy identity {ident}"
    assert BattleReader(d.gba).enemy_identity() == (0, 0), "identity reported outside a battle"
    assert b.state_reader.read()["enemy_id"] == ident
    b.close(); d.close()
    print("test_enemy_identity_changes_with_the_foe_and_is_zero_outside_battle PASSED")


def test_vector_is_zero_outside_battle():
    env = _env("dialogue.state")
    assert not BattleReader(env.gba).read().any(), "battle vector non-zero outside a battle"
    env.close()
    print("test_vector_is_zero_outside_battle PASSED")


def test_state_read_exposes_battle():
    env = _env("dialogue.state")
    s = env.state_reader.read()
    assert s["battle"].shape == (BATTLE_SIZE,) and s["in_battle"] is False
    b = _env("battle.state")
    assert b.state_reader.read()["in_battle"] is True
    env.close(); b.close()
    print("test_state_read_exposes_battle PASSED")


def test_battle_read_is_cheap():
    env = _env("battle.state")
    r = BattleReader(env.gba)
    r.read()
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame(); r.read()
    per = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame()
    base = (time.perf_counter() - t0) / 200 * 1e3
    assert per - base < 0.2, f"battle read costs {per - base:.3f} ms"
    env.close()
    print(f"test_battle_read_is_cheap PASSED ({per - base:.3f} ms)")


def test_tiles_and_battle_together_fit_the_budget():
    """The spec's bound: both new reads together under 0.5 ms per step."""
    env = _env("battle.state")
    gs = env.state_reader
    gs.tiles.window(); gs.battle.read()
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame(); gs.tiles.window(); gs.battle.read()
    per = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame()
    base = (time.perf_counter() - t0) / 200 * 1e3
    assert per - base < 0.5, f"tiles + battle cost {per - base:.3f} ms"
    env.close()
    print(f"test_tiles_and_battle_together_fit_the_budget PASSED ({per - base:.3f} ms)")


if __name__ == "__main__":
    test_in_battle_on_the_fixture_and_not_in_dialogue()
    test_vector_layout_in_battle()
    test_move_nature_marks_scratch_damaging_and_growl_status()
    test_enemy_identity_changes_with_the_foe_and_is_zero_outside_battle()
    test_vector_is_zero_outside_battle()
    test_state_read_exposes_battle()
    test_battle_read_is_cheap()
    test_tiles_and_battle_together_fit_the_budget()
    print("\nall battle tests passed")
