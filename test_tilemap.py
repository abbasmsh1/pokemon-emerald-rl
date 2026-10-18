"""Assertion-based checks for the tile-class window. No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import time  # noqa: E402

import numpy as np  # noqa: E402

from env import EmeraldEnv  # noqa: E402
from tilemap import CLASSES, TILE_H, TILE_W, TileMapReader  # noqa: E402

PASSABLE = ("walkable", "grass", "warp")


def _dialogue_env() -> EmeraldEnv:
    env = EmeraldEnv(state_path="dialogue.state")
    env.reset()
    return env


def _cls(onehot_column: np.ndarray) -> str:
    return CLASSES[int(onehot_column.argmax())]


def test_window_shape_and_one_hot():
    env = _dialogue_env()
    w = env.state_reader.tiles.window()
    assert w.shape == (len(CLASSES), TILE_H, TILE_W) and w.dtype == np.uint8, w.shape
    assert (w.sum(axis=0) == 1).all(), "every tile has exactly one class"
    env.close()
    print("test_window_shape_and_one_hot PASSED")


def test_player_tile_is_passable():
    """The player stands on it, so it cannot be blocked, water or outside."""
    env = _dialogue_env()
    centre = env.state_reader.tiles.window()[:, TILE_H // 2, TILE_W // 2]
    assert _cls(centre) in PASSABLE, f"player stands on {_cls(centre)}"
    env.close()
    print("test_player_tile_is_passable PASSED")


def test_window_shifts_with_movement():
    """Walk until a step moves the player one tile; the old window's columns
    must reappear shifted by one in the new one."""
    # From the default anchor, not dialogue.state: that fixture is inside a
    # scripted scene where the player is held still until the text ends.
    env = EmeraldEnv()
    env.reset()
    for _ in range(3):
        env.step(0)
    before = env.state_reader.tiles.window()
    pos0 = env.state_reader.read()["pos"]
    moved = None
    for action in (4, 4, 3, 3, 2, 2, 1, 1, 4, 3, 2, 1):   # right, left, down, up
        env.step(action)
        pos = env.state_reader.read()["pos"]
        if pos != pos0:
            moved = (pos[0] - pos0[0], pos[1] - pos0[1]); break
        before = env.state_reader.tiles.window()
    # One env step holds the key for 23 frames, a tile takes 16, and the game
    # finishes the tile it started, so a press moves one or two tiles.
    assert moved is not None and 1 <= abs(moved[0]) + abs(moved[1]) <= 2, f"no move happened: {moved}"
    after = env.state_reader.tiles.window()
    dx, dy = moved
    if dx > 0:   same = np.array_equal(before[:, :, dx:], after[:, :, :-dx])
    elif dx < 0: same = np.array_equal(before[:, :, :dx], after[:, :, -dx:])
    elif dy > 0: same = np.array_equal(before[:, dy:, :], after[:, :-dy, :])
    else:        same = np.array_equal(before[:, :dy, :], after[:, -dy:, :])
    assert same, f"window did not shift by {moved} with the move"
    env.close()
    print("test_window_shifts_with_movement PASSED")


def test_blocked_presses_face_non_passable_tiles():
    """Self-consistency against the game itself, no hand labels: a direction
    press that did not move the player (no text on screen) must have faced a
    non-passable tile, and a press that moved must have landed on a passable
    one."""
    import random
    env = EmeraldEnv()
    env.reset()
    rng = random.Random(0)
    step_to = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    blocked_bad = moved_bad = checked = 0
    for _ in range(300):
        s = env.state_reader.read()
        if s["screen_text"] or env._text_box_visible():
            env.step(6); continue
        a = rng.choice([1, 2, 3, 4])
        dx, dy = step_to[a]
        facing = env.state_reader.tiles.tile_class(s["pos"][0] + dx, s["pos"][1] + dy)
        env.step(a)
        s2 = env.state_reader.read()
        if s2["map"] != s["map"] or s2["screen_text"]:
            continue
        checked += 1
        if s2["pos"] == s["pos"]:
            if facing in PASSABLE and facing != "warp":
                blocked_bad += 1
        else:
            if env.state_reader.tiles.tile_class(*s2["pos"]) not in PASSABLE:
                moved_bad += 1
    assert checked > 100, f"only {checked} presses checked"
    assert blocked_bad == 0, f"{blocked_bad} blocked presses faced a tile the map calls walkable"
    assert moved_bad == 0, f"{moved_bad} moves landed on a tile the map calls impassable"
    env.close()
    print(f"test_blocked_presses_face_non_passable_tiles PASSED ({checked} presses)")


def test_encounter_tile_is_grass():
    """battle.state was captured where a wild encounter started, so the tile
    under the player must be tall grass. (dialogue.state sits at x=1 on the
    map's west edge with no grass within seven tiles, so it cannot serve.)"""
    env = EmeraldEnv(state_path="battle.state")
    env.reset()
    w = env.state_reader.tiles.window()
    centre = _cls(w[:, TILE_H // 2, TILE_W // 2])
    assert centre == "grass", f"the encounter tile reads {centre}"
    assert w[CLASSES.index("grass")].sum() >= 4, "a grass patch should surround an encounter"
    env.close()
    print("test_encounter_tile_is_grass PASSED")


def test_edge_of_map_reads_outside():
    env = _dialogue_env()
    r = env.state_reader.tiles
    assert r.tile_class(-1, 0) == "outside"
    assert r.tile_class(0, 10_000) == "outside"
    env.close()
    print("test_edge_of_map_reads_outside PASSED")


def test_unresolvable_tileset_reads_unknown():
    env = _dialogue_env()
    r = env.state_reader.tiles
    r._tileset_pointers = lambda: (0, 0)
    w = r.window()
    assert (w[CLASSES.index("unknown")] == 1).all(), "unresolvable tileset did not read as unknown"
    env.close()
    print("test_unresolvable_tileset_reads_unknown PASSED")


def test_window_before_the_game_starts_is_unknown():
    """boot.state's first frames have no save block; a worker steps there
    before the game is up and must get an all-unknown window, not a crash."""
    import tilemap
    env = _dialogue_env()
    real = tilemap.read_save_block_1
    tilemap.read_save_block_1 = lambda gba: None
    try:
        w = env.state_reader.tiles.window()
    finally:
        tilemap.read_save_block_1 = real
    assert (w[CLASSES.index("unknown")] == 1).all(), "no save block did not read as unknown"
    env.close()
    print("test_window_before_the_game_starts_is_unknown PASSED")


def test_behaviour_sets_match_the_header():
    """Values from pokeemerald include/constants/metatile_behaviors.h. The
    first version matched names by substring and swept MB_INDOOR_ENCOUNTER
    (a floor) and three closed doors into the passable warp set."""
    from tilemap import GRASS, LEDGE, WARP, WATER
    for closed in (0x0B, 0x8B, 0xBE, 0xEA):          # indoor-encounter floor, closed doors
        assert closed not in WARP, f"{closed:#x} is impassable but classed warp"
    assert 0x1C in WARP, "MB_SHOAL_CAVE_ENTRANCE missing from warp"
    assert 0x17 not in WATER, "MB_SHALLOW_WATER is walked on foot"
    assert 0x19 in WATER, "MB_NO_SURFACING is surfable deep water"
    assert 0x07 not in GRASS, "MB_SHORT_GRASS has no encounters"
    assert 0x08 in GRASS, "MB_CAVE floors have encounters"
    assert LEDGE == set(range(0x38, 0x40))
    print("test_behaviour_sets_match_the_header PASSED")


def test_npc_tiles_are_marked():
    """dialogue.state has Prof. Birch standing next to the player."""
    env = _dialogue_env()
    w = env.state_reader.tiles.window()
    assert w[CLASSES.index("npc")].sum() >= 1, "no NPC in the window"
    assert w[CLASSES.index("npc"), TILE_H // 2, TILE_W // 2] == 0, "the player's own tile was marked npc"
    env.close()
    print("test_npc_tiles_are_marked PASSED")


def test_window_read_is_cheap():
    env = _dialogue_env()
    r = env.state_reader.tiles
    r.window()
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame()
        r.window()
    per = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        env.gba.core.run_frame()
    base = (time.perf_counter() - t0) / 200 * 1e3
    assert per - base < 0.3, f"window read costs {per - base:.3f} ms"
    env.close()
    print(f"test_window_read_is_cheap PASSED ({per - base:.3f} ms)")


if __name__ == "__main__":
    test_window_shape_and_one_hot()
    test_player_tile_is_passable()
    test_window_shifts_with_movement()
    test_blocked_presses_face_non_passable_tiles()
    test_encounter_tile_is_grass()
    test_edge_of_map_reads_outside()
    test_unresolvable_tileset_reads_unknown()
    test_window_before_the_game_starts_is_unknown()
    test_behaviour_sets_match_the_header()
    test_npc_tiles_are_marked()
    test_window_read_is_cheap()
    print("\nall tilemap tests passed")
