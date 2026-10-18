"""Checks for the SLM teacher: prompt, decision, loop. No framework."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import numpy as np  # noqa: E402

from battle import move_name  # noqa: E402
from env import EmeraldEnv  # noqa: E402
from teacher import MAX_PROMPT_TOKENS, build_prompt, format_chat, map_name, summarise_tiles  # noqa: E402
from tilemap import CLASSES  # noqa: E402


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)


def _fixture(state):
    env = EmeraldEnv(state_path=state)
    obs, _ = env.reset()
    s = env.state_reader.read()
    return env, s, obs


def test_move_names():
    assert move_name(10) == "SCRATCH" and move_name(45) == "GROWL" and move_name(33) == "TACKLE"
    assert move_name(55) == "WATER GUN" and move_name(116) == "FOCUS ENERGY" and move_name(999) == "move 999"
    assert map_name((0, 18)) == "Route 103" and map_name((0, 3)) == "Rustboro City" and map_name((2, 4)) == "map 2-4"
    print("test_move_names PASSED")


def test_tile_summary_names_adjacent_and_nearest():
    tiles = np.zeros((9, 11, 15), np.uint8)
    tiles[CLASSES.index("walkable")] = 1
    tiles[:, 5, 8] = 0; tiles[CLASSES.index("blocked"), 5, 8] = 1      # east of the centre (5,7)
    tiles[:, 2, 7] = 0; tiles[CLASSES.index("grass"), 2, 7] = 1        # 3 north
    s = summarise_tiles(tiles)
    assert "east: blocked" in s and "north: walkable" in s, s
    assert "nearest grass: 3 tiles north" in s, s
    assert "door" not in s and "map edge" not in s, s
    print("test_tile_summary_names_adjacent_and_nearest PASSED")


def test_prompt_is_deterministic_and_bounded_on_fixtures():
    tok = _tok()
    for state in (None, "dialogue.state", "battle.state"):
        env, s, obs = _fixture(state)
        p1 = build_prompt(s, obs, ["up", "A", "none"], True, "north")
        p2 = build_prompt(s, obs, ["up", "A", "none"], True, "north")
        assert p1 == p2, "prompt is not deterministic"
        n = len(tok(format_chat(tok, p1))["input_ids"])
        assert n <= MAX_PROMPT_TOKENS, f"{state}: {n} tokens"
        assert "Rustboro" in p1 and "location:" in p1 and "unexplored: north" in p1
        env.close()
    print("test_prompt_is_deterministic_and_bounded_on_fixtures PASSED")


def test_battle_prompt_marks_move_natures():
    env, s, obs = _fixture("battle.state")
    p = build_prompt(s, obs, [], False)
    assert "SCRATCH (D)" in p and "GROWL (S)" in p and "enemy HP" in p, p
    env.close()
    env, s, obs = _fixture("dialogue.state")
    assert "enemy HP" not in build_prompt(s, obs, [], False), "battle block shown outside a battle"
    env.close()
    print("test_battle_prompt_marks_move_natures PASSED")


def test_prompt_survives_hostile_text():
    tok = _tok()
    env, s, obs = _fixture(None)
    s2 = dict(s, screen_text='{"}' + "A" * 160 + "\n\n")
    p = build_prompt(s2, obs, [], False)
    n = len(tok(format_chat(tok, p))["input_ids"])
    assert n <= MAX_PROMPT_TOKENS, f"{n} tokens"
    assert "\n\n\n" not in p
    env.close()
    print("test_prompt_survives_hostile_text PASSED")


if __name__ == "__main__":
    test_move_names()
    test_tile_summary_names_adjacent_and_nearest()
    test_prompt_is_deterministic_and_bounded_on_fixtures()
    test_battle_prompt_marks_move_natures()
    test_prompt_survives_hostile_text()
    print("\nall teacher tests passed")
