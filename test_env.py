import numpy as np

from env import EmeraldEnv


def test_observation_matches_declared_space():
    env = EmeraldEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "obs outside declared space"
    assert obs["screen"].shape == (3, 80, 120)
    assert obs["screen"].dtype == np.uint8
    assert obs["state"].shape == (17,)
    assert obs["state"].dtype == np.float32
    env.close()
    print("test_observation_matches_declared_space PASSED")


def test_reset_is_deterministic():
    """Two resets from the same savestate must give identical observations."""
    env = EmeraldEnv()
    a, _ = env.reset()
    b, _ = env.reset()
    assert np.array_equal(a["screen"], b["screen"]), "screen differs across resets"
    assert np.array_equal(a["state"], b["state"]), "state differs across resets"
    env.close()
    print("test_reset_is_deterministic PASSED")


def test_action_space_is_seven_discrete():
    env = EmeraldEnv()
    assert env.action_space.n == 7
    assert len(EmeraldEnv.ACTIONS) == 7
    env.close()
    print("test_action_space_is_seven_discrete PASSED")


def test_step_returns_valid_transition():
    env = EmeraldEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()
    print("test_step_returns_valid_transition PASSED")


def test_frame_stack_advances():
    """The three stacked frames must not all be the same frame forever."""
    env = EmeraldEnv()
    obs, _ = env.reset()
    for _ in range(10):
        obs, *_ = env.step(5)  # press A, advances dialogue
    newest, oldest = obs["screen"][2], obs["screen"][0]
    assert not np.array_equal(newest, oldest), "frame stack is not advancing"
    env.close()
    print("test_frame_stack_advances PASSED")


if __name__ == "__main__":
    test_action_space_is_seven_discrete()
    test_observation_matches_declared_space()
    test_reset_is_deterministic()
    test_step_returns_valid_transition()
    test_frame_stack_advances()
    print("\nall env tests passed")
