"""Checks for the human-play recorder. No framework, no window."""

from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from demos import load_recent  # noqa: E402
from env import EmeraldEnv  # noqa: E402
from play import Recorder, keys_to_action  # noqa: E402


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


if __name__ == "__main__":
    test_keys_map_to_the_agents_actions()
    test_recorder_keeps_every_step_including_the_last_partial_shard()
    print("\nall play tests passed")
