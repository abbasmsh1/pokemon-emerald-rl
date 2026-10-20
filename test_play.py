"""Checks for the human-play recorder. No framework, no window."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import os as _os
_os.environ["EMERALD_START"] = "route103.state"   # tests start where they were written for: Route 103

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from demos import load_recent  # noqa: E402
from env import EmeraldEnv  # noqa: E402
from play import REPEAT_DELAY, InputQueue, Recorder, keys_to_action  # noqa: E402


def test_keys_map_to_the_agents_actions():
    A = EmeraldEnv.ACTIONS
    assert keys_to_action(set()) == A.index(None)
    assert keys_to_action({"up"}) == A.index("up")
    assert keys_to_action({"z"}) == A.index("A") and keys_to_action({"x"}) == A.index("B")
    assert keys_to_action({"return"}) == A.index("start")
    # Two keys at once: a button beats a direction, so mashing A while
    # walking still talks; directions resolve in a fixed order.
    assert keys_to_action({"up", "z"}) == A.index("A")
    assert keys_to_action({"left", "up"}) == A.index("up")
    print("test_keys_map_to_the_agents_actions PASSED")


def test_recorder_keeps_every_step_including_the_last_partial_shard():
    env = EmeraldEnv()
    obs, _ = env.reset()
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(Path(d), shard_steps=50)
        start = True
        for i in range(120):
            a = [1, 2, 3, 4][i % 4]
            nxt, r, *_ = env.step(a)
            rec.add(obs, a, r, start)
            obs, start = nxt, False
        rec.close()                                  # the 20 left over must not be lost
        data = load_recent(Path(d))
        assert len(data["action"]) == 120, len(data["action"])
        assert set(obs) <= set(data), f"observation keys missing: {set(obs) - set(data)}"
        assert bool(data["episode_start"][0]) and int(data["episode_start"].sum()) == 1
        assert rec.steps == 120
    env.close()
    print("test_recorder_keeps_every_step_including_the_last_partial_shard PASSED")


def test_a_tapped_arrow_moves_exactly_once():
    """In the battle menu one tap of down must move the cursor one place; a
    held arrow repeating every step skipped POKEMON (switch) every time."""
    q = InputQueue()
    q.keydown("down", t=0.0)
    acts = [q.next_action(now=0.05 * i, held={"down"} if i < 3 else set()) for i in range(10)]
    assert acts.count(EmeraldEnv.ACTIONS.index("down")) == 1, acts
    print("test_a_tapped_arrow_moves_exactly_once PASSED")


def test_a_held_arrow_repeats_after_the_delay_so_walking_works():
    q = InputQueue()
    q.keydown("up", t=0.0)
    down = EmeraldEnv.ACTIONS.index("up")
    early = [q.next_action(now=t, held={"up"}) for t in (0.0, 0.1, 0.2)]
    late = [q.next_action(now=REPEAT_DELAY + 0.05 + 0.25 * i, held={"up"}) for i in range(4)]
    assert early.count(down) == 1, early
    assert late.count(down) == 4, late
    print("test_a_held_arrow_repeats_after_the_delay_so_walking_works PASSED")


def test_buttons_are_one_per_press():
    q = InputQueue()
    q.keydown("z", t=0.0); q.keydown("z", t=0.1)
    acts = [q.next_action(now=0.2, held={"z"}) for _ in range(4)]
    assert acts.count(EmeraldEnv.ACTIONS.index("A")) == 2, acts
    print("test_buttons_are_one_per_press PASSED")


def test_arrows_never_repeat_in_a_battle():
    """A slow, deliberate press in the battle menu lasted past the repeat delay
    and moved the cursor twice, skipping POKEMON. In battle a press is one move."""
    q = InputQueue()
    q.keydown("down", t=0.0)
    down = EmeraldEnv.ACTIONS.index("down")
    acts = [q.next_action(now=0.25 * i, held={"down"}, repeat=False) for i in range(6)]
    assert acts.count(down) == 1, acts
    print("test_arrows_never_repeat_in_a_battle PASSED")


if __name__ == "__main__":
    test_keys_map_to_the_agents_actions()
    test_recorder_keeps_every_step_including_the_last_partial_shard()
    test_a_tapped_arrow_moves_exactly_once()
    test_a_held_arrow_repeats_after_the_delay_so_walking_works()
    test_buttons_are_one_per_press()
    test_arrows_never_repeat_in_a_battle()
    print("\nall play tests passed")
