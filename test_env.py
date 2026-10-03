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


def test_action_space_is_eight_discrete():
    """start is index 7; indices 0-6 must keep their original meaning."""
    env = EmeraldEnv()
    assert env.action_space.n == 8
    assert len(EmeraldEnv.ACTIONS) == 8
    assert EmeraldEnv.ACTIONS == [None, "up", "down", "left", "right", "A", "B", "start"]
    assert EmeraldEnv.ACTIONS[5] == "A", "index 5 must stay A; tests rely on it"
    env.close()
    print("test_action_space_is_eight_discrete PASSED")


def test_new_section_pays_once():
    """A section pays on first entry and never again that episode."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._visited_sections.clear()

    first = env._compute_reward(s)
    second = env._compute_reward(s)
    assert first - second >= EmeraldEnv.NEW_SECTION_REWARD - 1e-9, (
        f"section bonus not paid on first entry: {first} then {second}"
    )
    assert len(env._visited_sections) == 1
    env.close()
    print("test_new_section_pays_once PASSED")


def test_section_boundary_is_section_size_tiles():
    """Moving one tile stays in the section; moving SECTION_SIZE crosses it."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._visited_sections.clear()
    env._visited_tiles.clear()
    env._tiles_per_map.clear()
    env._visited_maps.clear()

    base = dict(s, pos=(0, 0))
    env._compute_reward(base)
    assert len(env._visited_sections) == 1

    env._compute_reward(dict(s, pos=(1, 0)))
    assert len(env._visited_sections) == 1, "one tile should not cross a section"

    env._compute_reward(dict(s, pos=(EmeraldEnv.SECTION_SIZE, 0)))
    assert len(env._visited_sections) == 2, "SECTION_SIZE tiles should cross one"
    env.close()
    print("test_section_boundary_is_section_size_tiles PASSED")


def test_sections_clear_on_reset():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    for x in range(0, 40, EmeraldEnv.SECTION_SIZE):
        env._compute_reward(dict(s, pos=(x, 0)))
    assert len(env._visited_sections) > 1
    env.reset()
    assert env._visited_sections == set(), "sections survived a reset"
    env.close()
    print("test_sections_clear_on_reset PASSED")


def test_section_reward_is_gated_on_stability():
    """An in-flight read must not register a section, same as every other term."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._visited_sections.clear()
    env._prev = dict(s, map=(0, 9))

    garbage = dict(s, map=(0, 0), pos=(65535, 65535))
    r = env._compute_reward(garbage)
    assert r == 0.0, f"in-flight step paid {r}"
    assert env._visited_sections == set(), "garbage read recorded a section"
    env.close()
    print("test_section_reward_is_gated_on_stability PASSED")


def test_whiteout_does_not_end_the_episode():
    """A whiteout warps the player to a Pokemon Center healed; it is not death.

    Ending the episode on it discarded all exploration and restarted from
    Littleroot, which is why ep_len_mean sat at 18,300 against a 65,536 cap.
    """
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    whited = dict(s, badges=0, whiteout=True)
    assert env._should_terminate(whited, True) is False, (
        "whiteout still ends the episode"
    )
    env.close()
    print("test_whiteout_does_not_end_the_episode PASSED")


def test_whiteout_still_costs_the_penalty():
    """Not fatal must not mean free."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = dict(s, whiteout=False)
    env._visited_tiles.clear()
    env._visited_maps.clear()
    env._visited_sections.clear()
    env._tiles_per_map.clear()

    before = env._compute_reward(dict(s, whiteout=False))
    env._prev = dict(s, whiteout=False)
    after = env._compute_reward(dict(s, whiteout=True))

    assert after < before, f"whiteout was free: {after} vs {before}"
    assert after - before <= EmeraldEnv.WHITEOUT_PENALTY + 1e-6, (
        f"whiteout cost {after - before}, expected {EmeraldEnv.WHITEOUT_PENALTY}"
    )
    env.close()
    print("test_whiteout_still_costs_the_penalty PASSED")


def test_badge_still_ends_the_episode():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    assert env._should_terminate(dict(s, badges=1), True) is True
    assert env._should_terminate(dict(s, badges=0), True) is False
    env.close()
    print("test_badge_still_ends_the_episode PASSED")


def test_stall_penalty_charges_after_limit():
    """Going STALL_LIMIT steps without earning anything must cost something."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    # a state that earns nothing: same map, already-visited tile, no deltas
    env._compute_reward(s)
    env._steps_since_reward = 0

    charged = None
    for i in range(EmeraldEnv.STALL_LIMIT + 2):
        r = env._compute_reward(s)
        if r < 0:
            charged = i + 1
            break

    assert charged == EmeraldEnv.STALL_LIMIT, (
        f"stall charged at step {charged}, expected {EmeraldEnv.STALL_LIMIT}"
    )
    assert env._steps_since_reward == 0, "counter must reset after charging"
    env.close()
    print("test_stall_penalty_charges_after_limit PASSED")


def test_progress_resets_the_stall_counter():
    """Any positive reward must clear the stall counter."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._steps_since_reward = EmeraldEnv.STALL_LIMIT - 1

    env._visited_tiles.clear()
    env._visited_maps.clear()
    env._tiles_per_map.clear()
    r = env._compute_reward(s)

    assert r > 0, f"expected new-map/new-tile reward, got {r}"
    assert env._steps_since_reward == 0, "progress did not reset the counter"
    env.close()
    print("test_progress_resets_the_stall_counter PASSED")


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


def test_new_tile_rewards_once():
    """Revisiting a tile must not pay again."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    key = (s["map"][0], s["map"][1], s["pos"][0], s["pos"][1])

    env._visited_tiles.clear()
    env._tiles_per_map.clear()
    first = env._compute_reward(s)
    second = env._compute_reward(s)

    assert first > second, f"first visit {first} should beat revisit {second}"
    assert key in env._visited_tiles
    env.close()
    print("test_new_tile_rewards_once PASSED")


def test_tile_reward_capped_per_map():
    env = EmeraldEnv()
    env.reset()
    base = env.state_reader.read()
    m = base["map"]

    total = 0.0
    for i in range(EmeraldEnv.TILE_CAP_PER_MAP + 50):
        s = dict(base, pos=(i % 256, i // 256))
        total += env._compute_reward(s)

    assert env._tiles_per_map[m] == EmeraldEnv.TILE_CAP_PER_MAP, (
        f"cap not enforced: {env._tiles_per_map[m]}"
    )
    env.close()
    print("test_tile_reward_capped_per_map PASSED")


def test_reward_finite_over_random_rollout():
    env = EmeraldEnv()
    env.reset()
    total = 0.0
    for _ in range(1000):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r), "non-finite reward"
        total += r
        if term or trunc:
            env.reset()
    print(f"test_reward_finite_over_random_rollout PASSED (total {total:.2f})")
    env.close()


def test_implausible_flag_delta_scores_zero():
    """A mid-relocation save-block parse must not pay out."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._flag_baseline = {"script": 4, "trainer": 0}
    env._prev = dict(s)
    spike = dict(s, script_flag_count=181, trainer_flag_count=64)
    env._visited_tiles.clear()
    env._tiles_per_map.clear()
    env._visited_maps.clear()
    env._visited_sections.clear()
    r = env._compute_reward(spike)
    # The exploration terms may legitimately fire here; the flag spike must not.
    exploration_ceiling = (
        EmeraldEnv.NEW_MAP_REWARD
        + EmeraldEnv.NEW_TILE_REWARD
        + EmeraldEnv.NEW_SECTION_REWARD
    )
    assert r < EmeraldEnv.BADGE_REWARD, f"transient paid {r}, more than a badge"
    assert r <= exploration_ceiling + 1e-6, (
        f"transient paid {r}, above the map+tile+section ceiling "
        f"{exploration_ceiling}; the flag spike was paid"
    )
    env.close()
    print("test_implausible_flag_delta_scores_zero PASSED")


def test_plausible_flag_delta_still_pays():
    """The clamp must not suppress real progress."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._flag_baseline = {"script": 10, "trainer": 0}
    env._prev = dict(s)
    real = dict(s, script_flag_count=13, trainer_flag_count=0)
    env._visited_tiles.clear()
    env._tiles_per_map.clear()
    env._visited_maps.clear()
    r = env._compute_reward(real)
    assert r >= 3 * EmeraldEnv.SCRIPT_FLAG_REWARD, f"real progress underpaid: {r}"
    env.close()
    print("test_plausible_flag_delta_still_pays PASSED")


def test_garbage_read_does_not_poison_baseline():
    """A rejected implausible read must not corrupt the reference used for
    future credit, and a clean recovery to the pre-garbage value must not
    pay out."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._flag_baseline = {"script": 181, "trainer": 64}
    env._prev = dict(s, script_flag_count=181, trainer_flag_count=64)
    env._visited_tiles.clear(); env._tiles_per_map.clear(); env._visited_maps.clear()

    # implausible dip (delta -177, well past MAX_FLAG_DELTA): must be
    # rejected and must not become the new baseline.
    garbage = dict(s, script_flag_count=4, trainer_flag_count=64)
    env._compute_reward(garbage)
    assert env._flag_baseline["script"] == 181, "garbage poisoned the baseline"

    # a clean recovery back to the exact pre-garbage value must pay nothing
    recovery = dict(s, script_flag_count=181, trainer_flag_count=64)
    r_recovery = env._compute_reward(recovery)
    assert r_recovery <= 1e-6, f"recovery paid {r_recovery}, expected 0"

    # genuine subsequent progress must still be measured against the
    # unpoisoned baseline (181), not against the garbage value (4): this is
    # what actually distinguishes a preserved baseline from one that gets
    # overwritten unconditionally on every read.
    real_progress = dict(s, script_flag_count=183, trainer_flag_count=64)
    r_progress = env._compute_reward(real_progress)
    assert r_progress >= 2 * EmeraldEnv.SCRIPT_FLAG_REWARD - 1e-6, (
        f"genuine progress underpaid after a rejected garbage read: {r_progress}"
    )
    env.close()
    print("test_garbage_read_does_not_poison_baseline PASSED")


def test_map_transition_step_scores_nothing():
    """A step where the map id is changing is mid-relocation: score nothing."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._visited_tiles.clear(); env._tiles_per_map.clear(); env._visited_maps.clear()
    env._flag_baseline = {"script": 181, "trainer": 64}
    env._prev = dict(s, map=(0, 9), script_flag_count=181, trainer_flag_count=64)

    # the garbage read: bogus map, flags shifted between slices
    garbage = dict(s, map=(0, 0), script_flag_count=167, trainer_flag_count=78)
    assert env._compute_reward(garbage) == 0.0, "in-flight step paid out"
    assert (0, 0) not in env._visited_maps, "bogus map recorded as visited"
    assert env._flag_baseline["script"] == 181, "baseline poisoned in flight"

    # arrival, still changing
    arriving = dict(s, map=(1, 4), script_flag_count=181, trainer_flag_count=64)
    assert env._compute_reward(arriving) == 0.0, "arrival step paid before settling"

    # first stable step credits the real new map exactly once
    settled = dict(s, map=(1, 4), script_flag_count=181, trainer_flag_count=64)
    r = env._compute_reward(settled)
    assert r >= EmeraldEnv.NEW_MAP_REWARD, f"real new map not credited: {r}"
    assert (1, 4) in env._visited_maps
    env.close()
    print("test_map_transition_step_scores_nothing PASSED")


def test_transition_step_cannot_terminate():
    """A fabricated badge from an in-flight read must not end the episode."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    prev = dict(s, map=(0, 9), badges=0)

    # in-flight read: bogus map, garbage badge bit set
    garbage = dict(s, map=(0, 0), badges=3, whiteout=False)
    stable = garbage["map"] == prev["map"]
    assert not stable
    assert env._should_terminate(garbage, stable) is False, (
        "in-flight badge/whiteout terminated the episode"
    )

    # same badge reading, but on a stable (non-transitioning) step
    settled = dict(s, map=prev["map"], badges=3, whiteout=False)
    stable_settled = settled["map"] == prev["map"]
    assert stable_settled
    assert env._should_terminate(settled, stable_settled) is True, (
        "stable badge reading failed to terminate"
    )
    env.close()
    print("test_transition_step_cannot_terminate PASSED")


def test_repeated_press_registers_multiple_times():
    """Two consecutive identical actions must register as two fresh presses,
    not one continuous hold that Gen-3 reads as a single press. Exercised
    from a genuine cold boot (the constructor already leaves the emulator
    there; env.reset() would load boot.state instead, which is a
    post-intro, free-movement state with no dialogue open to advance
    through, so it can't distinguish these two scenarios)."""
    N = 300

    repeated = EmeraldEnv()
    for _ in range(N):
        repeated.step(5)  # A, N times
    s_repeated = repeated.state_reader.read()
    repeated.close()

    single = EmeraldEnv()
    single.step(5)  # a single A press
    for _ in range(N - 1):
        single.step(0)  # no-op
    s_single = single.state_reader.read()
    single.close()

    assert s_repeated["script_flag_count"] > s_single["script_flag_count"], (
        f"repeated A ({s_repeated['script_flag_count']} flags) did not "
        f"out-progress a single A + no-ops ({s_single['script_flag_count']} flags)"
    )
    print("test_repeated_press_registers_multiple_times PASSED")


if __name__ == "__main__":
    test_action_space_is_eight_discrete()
    test_observation_matches_declared_space()
    test_reset_is_deterministic()
    test_step_returns_valid_transition()
    test_frame_stack_advances()
    test_new_section_pays_once()
    test_section_boundary_is_section_size_tiles()
    test_sections_clear_on_reset()
    test_section_reward_is_gated_on_stability()
    test_whiteout_does_not_end_the_episode()
    test_whiteout_still_costs_the_penalty()
    test_badge_still_ends_the_episode()
    test_stall_penalty_charges_after_limit()
    test_progress_resets_the_stall_counter()
    test_state_vector_does_not_saturate()
    test_new_tile_rewards_once()
    test_tile_reward_capped_per_map()
    test_reward_finite_over_random_rollout()
    test_implausible_flag_delta_scores_zero()
    test_plausible_flag_delta_still_pays()
    test_garbage_read_does_not_poison_baseline()
    test_map_transition_step_scores_nothing()
    test_transition_step_cannot_terminate()
    test_repeated_press_registers_multiple_times()
    print("\nall env tests passed")
