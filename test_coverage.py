"""Assertion-based checks for the coverage overlay. No framework."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from coverage import CELL, group_by_map, panel_bounds, render_coverage, render_to_array


def test_group_by_map_splits_by_map_id():
    tiles = [(0, 9, 1, 1), (0, 9, 2, 1), (1, 4, 5, 5)]
    by_map = group_by_map(tiles)
    assert set(by_map) == {(0, 9), (1, 4)}
    assert sorted(by_map[(0, 9)]) == [(1, 1), (2, 1)]
    assert by_map[(1, 4)] == [(5, 5)]
    print("test_group_by_map_splits_by_map_id PASSED")


def test_panel_bounds_is_inclusive():
    """A single tile is a 1x1 box, not 0x0."""
    assert panel_bounds([(3, 7)]) == (3, 7, 1, 1)
    assert panel_bounds([(2, 2), (4, 5)]) == (2, 2, 3, 4)
    print("test_panel_bounds_is_inclusive PASSED")


def test_bounding_box_expands_for_outlying_tile():
    base = [(10, 10), (11, 11)]
    assert panel_bounds(base) == (10, 10, 2, 2)
    assert panel_bounds(base + [(40, 10)]) == (10, 10, 31, 2)
    assert panel_bounds(base + [(10, 40)]) == (10, 10, 2, 31)
    print("test_bounding_box_expands_for_outlying_tile PASSED")


def test_render_writes_nonempty_png_with_one_panel_per_map():
    tiles = [(0, 9, x, 0) for x in range(5)] + [(1, 4, 0, y) for y in range(3)]
    with tempfile.TemporaryDirectory() as d:
        out = render_coverage(tiles, Path(d) / "cov.png")
        assert out.exists() and out.stat().st_size > 0, "no PNG written"
        with Image.open(out) as im:
            w, h = im.size
        # two panels, so the sheet must be at least as wide as the wider one
        assert w >= 5 * CELL, f"sheet too narrow for its panels: {w}"
        assert h >= 3 * CELL, f"sheet too short for its panels: {h}"
    print("test_render_writes_nonempty_png_with_one_panel_per_map PASSED")


def test_empty_input_still_writes_a_file():
    with tempfile.TemporaryDirectory() as d:
        out = render_coverage([], Path(d) / "empty.png")
        assert out.exists() and out.stat().st_size > 0
    print("test_empty_input_still_writes_a_file PASSED")


def test_current_position_is_marked_distinctly():
    tiles = [(0, 9, x, 0) for x in range(4)]
    a = render_to_array(tiles)
    b = render_to_array(tiles, current=(0, 9, 2, 0))
    assert a.shape == b.shape, "marking the agent changed the sheet size"
    assert not np.array_equal(a, b), "current position produced no visible change"
    print("test_current_position_is_marked_distinctly PASSED")


def test_env_coverage_is_uncapped_and_survives_reset():
    """The regression this file exists for.

    _visited_tiles is capped at TILE_CAP_PER_MAP and cleared every reset, so
    reusing it would draw a truncated, single-episode picture. _coverage must do
    neither.
    """
    from env import EmeraldEnv

    env = EmeraldEnv()
    env.reset()

    fake = {(0, 9, x, y) for x in range(40) for y in range(15)}  # 600 > cap of 400
    env._coverage |= fake
    assert len(env.coverage()) > EmeraldEnv.TILE_CAP_PER_MAP, (
        f"coverage capped at {len(env.coverage())}"
    )

    before = len(env.coverage())
    env.reset()
    assert len(env.coverage()) >= before, "reset cleared the coverage set"
    env.close()
    print("test_env_coverage_is_uncapped_and_survives_reset PASSED")


if __name__ == "__main__":
    test_group_by_map_splits_by_map_id()
    test_panel_bounds_is_inclusive()
    test_bounding_box_expands_for_outlying_tile()
    test_render_writes_nonempty_png_with_one_panel_per_map()
    test_empty_input_still_writes_a_file()
    test_current_position_is_marked_distinctly()
    test_env_coverage_is_uncapped_and_survives_reset()
    print("\nall coverage tests passed")
