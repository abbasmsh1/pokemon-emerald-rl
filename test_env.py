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


def test_state_vector_does_not_saturate():
    """Location fields must not clip to 1.0 for plausible in-game values."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    probe = dict(s, map=(3, 120), pos=(200, 240))
    vec = env._encode_state(probe)
    assert vec[10] < 1.0, f"mapGroup saturated: {vec[10]}"
    assert vec[11] < 1.0, f"mapNum saturated: {vec[11]}"
    assert vec[12] < 1.0, f"pos x saturated: {vec[12]}"
    assert vec[13] < 1.0, f"pos y saturated: {vec[13]}"
    assert env.observation_space["state"].contains(vec), "vec outside declared bounds"
    env.close()
    print("test_state_vector_does_not_saturate PASSED")


if __name__ == "__main__":
    test_action_space_is_seven_discrete()
    test_observation_matches_declared_space()
    test_reset_is_deterministic()
    test_step_returns_valid_transition()
    test_frame_stack_advances()
    test_state_vector_does_not_saturate()
    print("\nall env tests passed")
