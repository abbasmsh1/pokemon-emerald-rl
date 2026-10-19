import numpy as np

from env import EmeraldEnv


def test_observation_matches_declared_space():
    env = EmeraldEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "obs outside declared space"
    assert obs["screen"].shape == (3, 80, 120)
    assert obs["screen"].dtype == np.uint8
    assert obs["state"].shape == (29,)
    assert obs["text"].shape == (16,) and obs["text"].dtype == np.int64
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


def _clear_exploration(env):
    """Exploration terms pay once, so a second _compute_reward call sees a
    different baseline than the first. Measured calls must each start from the
    same cleared state or the comparison is meaningless, which silently made an
    earlier version of these tests pass for the wrong reason.
    """
    env._visited_tiles.clear()
    env._visited_maps.clear()
    env._visited_sections.clear()
    env._tiles_per_map.clear()
    # The first-ever section bonus is cumulative for the process and must be
    # cleared too, or the first of two "identical" measured calls pays +1 more.
    env._sections_ever.clear()


def test_experience_gain_is_rewarded():
    """Winning a battle must pay. This is the reason the agent used to flee."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._max_party_exp = 100
    env._prev = dict(s, party_exp=100)
    before = env._compute_reward(dict(s, party_exp=100))

    _clear_exploration(env)
    env._max_party_exp = 100
    env._prev = dict(s, party_exp=100)
    after = env._compute_reward(dict(s, party_exp=130))

    gained = after - before
    # Absolute floor first. An earlier version asserted only against
    # EmeraldEnv.EXP_REWARD on both sides, so zeroing the constant made the
    # whole test vacuous and it passed with the feature disabled.
    assert gained > 0.1, f"30 exp paid {gained}; experience is not rewarded"
    assert abs(gained - 30 * EmeraldEnv.EXP_REWARD) < 1e-6, (
        f"30 exp paid {gained}, expected {30 * EmeraldEnv.EXP_REWARD}"
    )
    env.close()
    print("test_experience_gain_is_rewarded PASSED")


def test_experience_uses_a_high_water_mark():
    """A PC deposit then withdraw must not re-pay banked experience."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._max_party_exp = 0
    env._prev = dict(s, party_exp=0)

    env._compute_reward(dict(s, party_exp=200))
    assert env._max_party_exp == 200

    env._prev = dict(s, party_exp=200)
    dipped = env._compute_reward(dict(s, party_exp=0))
    env._prev = dict(s, party_exp=0)
    restored = env._compute_reward(dict(s, party_exp=200))

    assert env._max_party_exp == 200, "high-water mark regressed"
    assert restored <= dipped + 1e-6, "withdrawing re-paid the same experience"
    env.close()
    print("test_experience_uses_a_high_water_mark PASSED")


def test_running_away_is_penalised_once():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._was_fleeing = False
    env._prev = dict(s)
    quiet = env._compute_reward(dict(s, screen_text=""))

    _clear_exploration(env)
    env._was_fleeing = False
    env._prev = dict(s)
    fled = env._compute_reward(dict(s, screen_text="Got away safely!"))

    _clear_exploration(env)
    env._prev = dict(s)
    still = env._compute_reward(dict(s, screen_text="Got away safely!"))

    assert fled < quiet, f"fleeing was free: {fled} vs {quiet}"
    assert fled - quiet <= EmeraldEnv.FLEE_PENALTY + 1e-6, (
        f"flee cost {fled - quiet}, expected {EmeraldEnv.FLEE_PENALTY}"
    )
    assert still > fled, "flee penalty charged again while the text lingered"
    env.close()
    print("test_running_away_is_penalised_once PASSED")


def test_fleeing_costs_more_than_a_battle_wins():
    """The whole point of the flee penalty.

    A wild win is worth roughly 20-40 experience. If running costs less than a
    win pays, running stays near break-even and the agent keeps doing it, which
    is exactly the behaviour this penalty exists to stop.
    """
    best_case_win = 40 * EmeraldEnv.EXP_REWARD
    assert abs(EmeraldEnv.FLEE_PENALTY) > best_case_win, (
        f"fleeing costs {abs(EmeraldEnv.FLEE_PENALTY)} but a good win pays "
        f"{best_case_win}; running is still near break-even"
    )
    print("test_fleeing_costs_more_than_a_battle_wins PASSED")


def test_opponent_fleeing_is_not_penalised():
    """A wild Pokemon running is not the agent's doing."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._was_fleeing = False
    env._prev = dict(s)
    quiet = env._compute_reward(dict(s, screen_text=""))

    _clear_exploration(env)
    env._was_fleeing = False
    env._prev = dict(s)
    opponent = env._compute_reward(dict(s, screen_text="ZIGZAGOON fled!"))

    assert abs(opponent - quiet) < 1e-6, (
        f"opponent fleeing changed reward by {opponent - quiet}"
    )
    env.close()
    print("test_opponent_fleeing_is_not_penalised PASSED")


def test_zero_max_steps_removes_the_limit():
    """max_steps=0 must never truncate, however many steps elapse."""
    env = EmeraldEnv(max_steps=0)
    env.reset()
    for _ in range(40):
        _, _, terminated, truncated, _ = env.step(0)
        assert not truncated, "truncated despite max_steps=0"
        assert not terminated, "terminated without a badge"
    env._step_count = 10_000_000
    _, _, _, truncated, _ = env.step(0)
    assert not truncated, "truncated at a huge step count despite max_steps=0"
    env.close()
    print("test_zero_max_steps_removes_the_limit PASSED")


def test_positive_max_steps_still_truncates():
    env = EmeraldEnv(max_steps=5)
    env.reset()
    truncations = 0
    for _ in range(6):
        _, _, _, truncated, _ = env.step(0)
        truncations += int(truncated)
    assert truncations >= 1, "a positive max_steps no longer truncates"
    env.close()
    print("test_positive_max_steps_still_truncates PASSED")


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

    # Earning nothing for 2,500 steps is now only reachable one way. Standing
    # still trips the freeze penalty at 600 steps and pacing inside a block
    # trips the loiter penalty at 30, so this walks a step that changes BOTH
    # tile and block every time, over ground already visited. That is the
    # niche stall still covers: crossing an exhausted map earning nothing.
    # Alternating two tiles would trip loiter (2 distinct tiles over 30 moves),
    # so walk a long line and back: 40 distinct tiles, never fewer than 7 in
    # any 30-move window.
    path = [(x, 5) for x in range(40)] + [(x, 5) for x in range(38, 0, -1)]
    for pos in path:                      # spend the novelty on both tiles
        env._compute_reward(dict(s, pos=pos))
    env._steps_since_reward = 0

    charged = None
    for i in range(EmeraldEnv.STALL_LIMIT + 2):
        r = env._compute_reward(dict(s, pos=path[i % len(path)]))
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
    # Walk rather than press A. Episodes now start from the furthest progress
    # savestate, which is an ordinary overworld tile with nothing to talk to,
    # so A leaves the screen static and the old version of this test failed for
    # a reason that had nothing to do with frame stacking.
    for i in range(10):
        obs, *_ = env.step(1 if i % 2 == 0 else 2)  # up, down, up...
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
    env._max_flags = {"script": 4, "trainer": 0}
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
        + EmeraldEnv.FIRST_SECTION_REWARD
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
    env._max_flags = {"script": 10, "trainer": 0}
    env._prev = dict(s)
    real = dict(s, script_flag_count=13, trainer_flag_count=0)
    env.drain_reward_terms()
    _settle(env, real)        # a real increment persists, so it settles and pays
    paid = env.drain_reward_terms().get("script_flag", 0.0)
    assert abs(paid - 3 * EmeraldEnv.SCRIPT_FLAG_REWARD) < 1e-6, f"real progress underpaid: {paid}"
    env.close()
    print("test_plausible_flag_delta_still_pays PASSED")


def test_garbage_read_does_not_poison_baseline():
    """A rejected implausible read must not corrupt the reference used for
    future credit, and a clean recovery to the pre-garbage value must not
    pay out."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._max_flags = {"script": 181, "trainer": 64}
    env._prev = dict(s, script_flag_count=181, trainer_flag_count=64)
    env._visited_tiles.clear(); env._tiles_per_map.clear(); env._visited_maps.clear()

    # implausible dip (delta -177, well past MAX_FLAG_DELTA): must be
    # rejected and must not become the new high-water mark.
    garbage = dict(s, script_flag_count=4, trainer_flag_count=64)
    env._compute_reward(garbage)
    assert env._max_flags["script"] == 181, "garbage poisoned the high-water mark"

    # a clean recovery back to the exact pre-garbage value must pay nothing
    recovery = dict(s, script_flag_count=181, trainer_flag_count=64)
    r_recovery = env._compute_reward(recovery)
    assert r_recovery <= 1e-6, f"recovery paid {r_recovery}, expected 0"

    # genuine subsequent progress must still be measured against the
    # unpoisoned baseline (181), not against the garbage value (4): this is
    # what actually distinguishes a preserved baseline from one that gets
    # overwritten unconditionally on every read.
    real_progress = dict(s, script_flag_count=183, trainer_flag_count=64)
    env.drain_reward_terms()
    _settle(env, real_progress)
    r_progress = env.drain_reward_terms().get("script_flag", 0.0)
    assert abs(r_progress - 2 * EmeraldEnv.SCRIPT_FLAG_REWARD) < 1e-6, (
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
    env._max_flags = {"script": 181, "trainer": 64}
    env._prev = dict(s, map=(0, 9), script_flag_count=181, trainer_flag_count=64)

    # the garbage read: bogus map, flags shifted between slices
    garbage = dict(s, map=(0, 0), script_flag_count=167, trainer_flag_count=78)
    assert env._compute_reward(garbage) == 0.0, "in-flight step paid out"
    assert (0, 0) not in env._visited_maps, "bogus map recorded as visited"
    assert env._max_flags["script"] == 181, "high-water mark poisoned in flight"

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


def test_healing_is_rewarded():
    """Restoring HP must pay, or the Start menu has no reason to be pressed."""
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), party_levels=[5])

    _clear_exploration(env)
    env._prev = dict(s, party_hp_frac=0.2)
    before = env._compute_reward(dict(s, party_hp_frac=0.2))

    _clear_exploration(env)
    env._prev = dict(s, party_hp_frac=0.2)
    after = env._compute_reward(dict(s, party_hp_frac=1.0))

    gained = after - before
    # Absolute floor first: asserting only against HEAL_REWARD on both sides
    # would pass with the constant zeroed, which is how three earlier tests on
    # this project managed to be vacuous.
    assert gained > 0.5, f"healing 80% of HP paid {gained}; healing is not rewarded"
    assert abs(gained - 0.8 * EmeraldEnv.HEAL_REWARD) < 1e-6, (
        f"healing 0.8 paid {gained}, expected {0.8 * EmeraldEnv.HEAL_REWARD}"
    )
    env.close()
    print("test_healing_is_rewarded PASSED")


def test_damage_then_heal_nets_zero():
    """The farm loop: get hurt for free, heal at a Center for reward, repeat.

    Free Center healing is unlimited, so the HP term is symmetric and a full
    hurt/heal cycle must come out at zero.
    """
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), party_levels=[5])

    # One priming call to spend the new-map/tile/section bonuses, which would
    # otherwise land on the first measured step only and fake an imbalance.
    env._prev = dict(s, party_hp_frac=1.0)
    env._compute_reward(dict(s, party_hp_frac=1.0))

    hurt = env._compute_reward(dict(s, party_hp_frac=0.1))
    healed = env._compute_reward(dict(s, party_hp_frac=1.0))

    assert hurt < 0, f"losing 90% of HP paid {hurt}; the HP term is one-way"
    assert abs(hurt + healed) < 1e-6, (
        f"hurt {hurt} + healed {healed} = {hurt + healed}, must cancel"
    )
    env.close()
    print("test_damage_then_heal_nets_zero PASSED")


def test_empty_party_does_not_move_hp_reward():
    """A failed save-block read reports hp_frac 0.0 with no party. Billing that
    as damage would charge -HEAL_REWARD and refund it a step later."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    # Primed, so exploration income cannot mask a charge.
    env._prev = dict(s, party_hp_frac=1.0, party_levels=[5])
    env._compute_reward(dict(s, party_hp_frac=1.0, party_levels=[5]))
    r = env._compute_reward(dict(s, party_hp_frac=0.0, party_levels=[]))

    assert r > -1.0, f"a garbage empty-party read was billed as damage: {r}"
    env.close()
    print("test_empty_party_does_not_move_hp_reward PASSED")


def test_starts_from_furthest_progress_state(tmp_path=None):
    """Episodes must resume at the furthest map count reached, not the intro."""
    import os
    from pathlib import Path

    from env import latest_progress_state

    d = Path("/tmp/pokemon_progress_states")
    if d.exists():
        for f in d.iterdir():
            f.unlink()
    d.mkdir(exist_ok=True)
    for name in ("furthest_2maps_1_0.state", "furthest_14maps_2_4.state",
                 "furthest_9maps_0_18.state", "emerald_100_steps.zip"):
        (d / name).write_bytes(b"x")

    picked = latest_progress_state(str(d))
    assert picked is not None, "no progress savestate picked"
    # The 14-map state is map group 2, an Oldale interior. A measured rollout
    # from the real one never left the room it starts in, so the deepest INDOOR
    # state must lose to a shallower outdoor one.
    assert picked.endswith("furthest_9maps_0_18.state"), (
        f"picked {picked}, expected the 9-map outdoor state"
    )
    assert latest_progress_state(str(d / "empty")) is None, (
        "a directory with no savestates must fall back to boot.state"
    )

    # Indoor-only directory: nothing outdoors to anchor on, so fall back.
    indoor = Path("/tmp/pokemon_progress_indoor")
    if indoor.exists():
        for f in indoor.iterdir():
            f.unlink()
    indoor.mkdir(exist_ok=True)
    (indoor / "furthest_14maps_2_4.state").write_bytes(b"x")
    assert latest_progress_state(str(indoor)) is None, (
        "an indoor-only directory must fall back to boot.state"
    )

    # And the env must actually use it rather than hardcoding boot.state.
    env = EmeraldEnv()
    assert "furthest_" in env.state_path, (
        f"env started from {env.state_path}, not a progress savestate"
    )
    group = env.state_path.split("_")[-2]
    assert group == "0", (
        f"env anchored on map group {group}, an interior: {env.state_path}"
    )
    env.close()
    print("test_starts_from_furthest_progress_state PASSED")


def test_changing_options_is_penalised():
    """The agent had set TEXT SPEED to SLOW, which inflates every dialogue."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, options=(1, 1624))
    env._compute_reward(dict(s, options=(1, 1624)))   # prime exploration
    charged = env._compute_reward(dict(s, options=(1, 1625)))

    # Absolute floor first, so zeroing OPTIONS_PENALTY cannot make this vacuous.
    assert charged < -5.0, f"changing an option paid {charged}; it is not penalised"
    assert abs(charged - EmeraldEnv.OPTIONS_PENALTY) < 1e-6, (
        f"changing an option paid {charged}, expected {EmeraldEnv.OPTIONS_PENALTY}"
    )
    env.close()
    print("test_changing_options_is_penalised PASSED")


def test_unchanged_options_cost_nothing():
    """Walking around must not be billed as menu fiddling."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, options=(1, 1624))
    env._compute_reward(dict(s, options=(1, 1624)))
    r = env._compute_reward(dict(s, options=(1, 1624)))

    assert r > -1.0, f"unchanged options were billed {r}"
    env.close()
    print("test_unchanged_options_cost_nothing PASSED")


def test_failed_options_read_is_not_billed():
    """state.py reports (0, 0) when the save block cannot be read. Treating
    that as a change would charge on the way out and again on the way back."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, options=(1, 1624))
    env._compute_reward(dict(s, options=(1, 1624)))
    out = env._compute_reward(dict(s, options=(0, 0)))
    back = env._compute_reward(dict(s, options=(1, 1624)))

    assert out > -1.0, f"a failed options read was billed {out}"
    assert back > -1.0, f"recovering from a failed options read was billed {back}"
    env.close()
    print("test_failed_options_read_is_not_billed PASSED")


def test_text_speed_is_pinned_to_fast():
    """The progress savestate has TEXT SPEED on SLOW, set there by the agent."""
    from state import TEXT_SPEED_FAST, TEXT_SPEED_MASK
    from pygba.game_wrappers.utils.emerald_utils import read_save_block_2

    env = EmeraldEnv()
    env.reset()
    env.step(0)
    packed = read_save_block_2(env.gba)["options"]
    assert packed & TEXT_SPEED_MASK == TEXT_SPEED_FAST, (
        f"text speed is {packed & TEXT_SPEED_MASK}, expected FAST "
        f"({TEXT_SPEED_FAST})"
    )
    env.close()
    print("test_text_speed_is_pinned_to_fast PASSED")


def test_pinning_text_speed_leaves_other_options_alone():
    """Only bits 0-2 may move. Battle scene off and frame type are the agent's
    to keep, and clobbering them would be an unasked-for second change."""
    from state import TEXT_SPEED_MASK, force_fast_text
    from pygba.game_wrappers.utils.emerald_utils import read_save_block_2

    env = EmeraldEnv()
    env.reset()
    env.step(0)
    before = read_save_block_2(env.gba)["options"]
    force_fast_text(env.gba)
    env.step(0)
    after = read_save_block_2(env.gba)["options"]

    assert before & ~TEXT_SPEED_MASK == after & ~TEXT_SPEED_MASK, (
        f"options outside text speed changed: {before:#x} -> {after:#x}"
    )
    env.close()
    print("test_pinning_text_speed_leaves_other_options_alone PASSED")


def test_pinned_text_speed_is_not_billed_as_a_change():
    """The revert must not be charged. The agent sets SLOW, the env puts it
    back, and billing both edges would charge -2 x OPTIONS_PENALTY for one
    visit to the OPTION screen."""
    from pygba.game_wrappers.utils.emerald_utils import ADRESSES
    from state import _OPTIONS_OFFSET, TEXT_SPEED_MASK

    env = EmeraldEnv()
    env.reset()
    env.step(0)          # pin applied, reading settles on FAST
    _clear_exploration(env)
    env.step(0)          # spend the exploration bonuses
    before = env._prev["options"]

    # Put TEXT SPEED back to SLOW behind the env's back, exactly as a visit to
    # the OPTION screen would. The next step must notice, write FAST back, and
    # charge nothing for its own correction.
    addr = env.gba.read_u32(ADRESSES["gSaveBlock2Ptr"]) + _OPTIONS_OFFSET
    env.gba.core.memory.u16[addr] = env.gba.core.memory.u16[addr] & ~TEXT_SPEED_MASK
    r = env.step(0)[1]

    assert env._prev["options"] == before, (
        f"corrected reading disagrees with the previous step: "
        f"{before} -> {env._prev['options']}"
    )
    assert r > -1.0, f"the env was billed {r} for reverting its own pin"
    env.close()
    print("test_pinned_text_speed_is_not_billed_as_a_change PASSED")


def _walk_in_block(env, s, steps):
    """Step the reward function along a path that shuffles over two tiles,
    moving every step. Returns the per-step rewards."""
    return [env._compute_reward(dict(s, pos=(i % 2, 0))) for i in range(steps)]


def test_pacing_a_few_tiles_is_penalised():
    """30 moves over 6 or fewer tiles is shuffling. Measured on the 400k policy:
    73% of steps repeated the position two steps earlier, on a 3-tile stretch
    of Route 103."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._prev = dict(s, pos=(0, 0))
    env._recent_moves.clear()

    rewards = _walk_in_block(env, s, EmeraldEnv.LOITER_LIMIT * 2)
    charged = [i for i, r in enumerate(rewards) if r < 0]
    assert charged, "shuffling two tiles for twice the limit was never charged"
    first = charged[0]
    assert EmeraldEnv.LOITER_LIMIT - 1 <= first <= EmeraldEnv.LOITER_LIMIT + 1, (
        f"charged at step {first}, expected at the {EmeraldEnv.LOITER_LIMIT}-move limit"
    )
    assert rewards[first] <= EmeraldEnv.LOITER_PENALTY + 1e-6
    env.close()
    print("test_pacing_a_few_tiles_is_penalised PASSED")


def test_pacing_across_a_block_boundary_is_penalised():
    """The loophole the 400k policy found: shuffle across x=8, the boundary
    between two 8x8 blocks, and a per-block counter resets on every crossing.
    The window has no boundary to straddle."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._prev = dict(s, pos=(7, 13))
    env._recent_moves.clear()
    env.drain_reward_terms()
    for i in range(EmeraldEnv.LOITER_LIMIT * 2):
        env._compute_reward(dict(s, pos=((7, 13), (8, 13))[i % 2]))   # straddles x=8
    booked = env.drain_reward_terms().get("loiter", 0.0)
    assert booked <= EmeraldEnv.LOITER_PENALTY + 1e-6, (
        f"shuffling across a block boundary booked {booked}; the boundary is still a loophole"
    )
    env.close()
    print("test_pacing_across_a_block_boundary_is_penalised PASSED")


def test_walking_somewhere_is_not_loitering():
    """30 moves over 7 or more distinct tiles is going somewhere."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = dict(s, pos=(0, 0))
    env._compute_reward(dict(s, pos=(0, 0)))       # spend exploration
    env._recent_moves.clear()
    env.drain_reward_terms()
    for i in range(1, EmeraldEnv.LOITER_LIMIT * 3):
        env._compute_reward(dict(s, pos=(i % 12, 0)))   # a 12-tile line, back and forth
    assert "loiter" not in env.drain_reward_terms(), "walking a 12-tile line was billed as loitering"
    env.close()
    print("test_walking_somewhere_is_not_loitering PASSED")


def test_standing_still_is_never_billed_as_loitering():
    """A battle, a dialogue and an open menu all freeze the position. Charging
    them would undo the experience and flee rewards."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    _clear_exploration(env)
    env._prev = dict(s, pos=(3, 3))
    env._recent_moves.clear()
    env.drain_reward_terms()
    for _ in range(EmeraldEnv.LOITER_LIMIT * 3):
        env._compute_reward(dict(s, pos=(3, 3)))
    assert len(env._recent_moves) == 0, f"a frozen position accrued {len(env._recent_moves)} moves"
    assert "loiter" not in env.drain_reward_terms(), "standing still was billed as loitering"
    env.close()
    print("test_standing_still_is_never_billed_as_loitering PASSED")


def test_catching_outweighs_seeing():
    """Catching costs balls, money and a battle held at low HP. Seeing is free."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    _clear_exploration(env)
    env._prev = dict(s, seen=10, caught=5)
    seen_only = env._compute_reward(dict(s, seen=11, caught=5))

    _clear_exploration(env)
    env._prev = dict(s, seen=10, caught=5)
    caught_too = env._compute_reward(dict(s, seen=11, caught=6))

    assert caught_too - seen_only > 1.0, (
        f"catching added only {caught_too - seen_only} over seeing"
    )
    assert abs(caught_too - seen_only - EmeraldEnv.CAUGHT_REWARD) < 1e-6
    env.close()
    print("test_catching_outweighs_seeing PASSED")


def test_a_level_beats_a_handful_of_tiles():
    """The gym is unwinnable underlevelled, so a level must not be worth four
    new tiles the way it was at 0.2."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._max_level_sum = 20
    env._prev = dict(s, party_levels=[20])
    env._compute_reward(dict(s, party_levels=[20]))   # prime exploration
    gained = env._compute_reward(dict(s, party_levels=[21]))

    assert gained > 10 * EmeraldEnv.NEW_TILE_REWARD, (
        f"a level paid {gained}, less than ten tiles"
    )
    assert abs(gained - EmeraldEnv.LEVEL_REWARD) < 1e-6, (
        f"a level paid {gained}, expected {EmeraldEnv.LEVEL_REWARD}"
    )
    env.close()
    print("test_a_level_beats_a_handful_of_tiles PASSED")


def test_reward_ledger_matches_reward_paid():
    """The breakdown is the only live reward signal on this run, so it must
    equal the reward actually paid, not an approximation of it."""
    import random

    env = EmeraldEnv()
    obs, _ = env.reset()
    env.drain_reward_terms()

    rng = random.Random(0)
    paid = 0.0
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(rng.randrange(env.action_space.n))
        paid += r
        if term or trunc:
            obs, _ = env.reset()

    booked = sum(env.drain_reward_terms().values())
    assert abs(paid) > 0.0, "300 random steps earned nothing; test proves nothing"
    assert abs(booked - paid) < 1e-6, (
        f"ledger booked {booked} against {paid} paid"
    )
    env.close()
    print(f"test_reward_ledger_matches_reward_paid PASSED (paid {paid:.2f})")


def test_draining_the_ledger_clears_it():
    """The callback drains once per rollout and reports a rate. A drain that
    did not clear would report a running total climbing forever."""
    env = EmeraldEnv()
    env.reset()
    for _ in range(20):
        env.step(1)

    first = env.drain_reward_terms()
    second = env.drain_reward_terms()
    assert first, "nothing booked over 20 steps"
    assert second == {}, f"ledger survived a drain: {second}"
    env.close()
    print("test_draining_the_ledger_clears_it PASSED")


def test_novelty_refresh_restores_income():
    """With --max-steps 0 an episode only ends on a badge, so without a refresh
    the visited sets fill once and never clear. Measured for real: 9 million
    steps with tiles_visited frozen at 486 and every reward term at zero."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    first = env._compute_reward(s)
    assert first > 0, f"the first visit to a tile paid {first}"
    repeat = env._compute_reward(s)
    assert repeat <= 0, f"an already-visited tile still paid {repeat}"

    env._refresh_novelty()
    again = env._compute_reward(s)
    assert again > 0, f"after a refresh the tile paid {again}, expected income"
    # Everything re-pays except the first-ever section bonus, which is
    # cumulative for the process and is the one thing a refresh must not revive.
    assert abs(again - (first - EmeraldEnv.FIRST_SECTION_REWARD)) < 1e-6, (
        f"refresh paid {again} against {first} for the same tile"
    )
    env.close()
    print("test_novelty_refresh_restores_income PASSED")


def test_refresh_fires_on_schedule_during_an_endless_episode():
    """The refresh must be driven by the step counter, not by reset(), which
    never runs on this configuration."""
    env = EmeraldEnv(max_steps=0)
    env.reset()
    assert EmeraldEnv.NOVELTY_REFRESH_STEPS > 0

    for i in range(40):
        env.step([1, 3, 2, 4][i % 4])   # up, left, down, right
    before = len(env._visited_tiles)
    assert before > 1, f"only {before} tiles walked; the test would prove nothing"

    # Drive the counter to one step short of a refresh without the cost of
    # actually emulating that many frames.
    env._step_count = EmeraldEnv.NOVELTY_REFRESH_STEPS - 1
    env.step(0)
    assert env._step_count % EmeraldEnv.NOVELTY_REFRESH_STEPS == 0
    # The refresh runs before the step scores, so the tile standing on is
    # credited again straight away. Everything else must be gone.
    assert len(env._visited_tiles) < before, (
        f"refresh did not clear on schedule: {len(env._visited_tiles)} tiles "
        f"held against {before} before"
    )
    assert len(env._visited_tiles) <= 1, (
        f"refresh left {len(env._visited_tiles)} tiles behind"
    )
    env.close()
    print("test_refresh_fires_on_schedule_during_an_endless_episode PASSED")


def test_refresh_does_not_clear_the_coverage_record():
    """_coverage drives the exploration overlay and is cumulative. Clearing it
    with the reward bookkeeping would make the rendered map wrong."""
    env = EmeraldEnv()
    env.reset()
    for _ in range(5):
        env.step(1)
    before = set(env.coverage())
    assert before, "no coverage recorded over 5 steps"

    env._refresh_novelty()
    assert set(env.coverage()) >= before, (
        "refresh discarded coverage the overlay depends on"
    )
    env.close()
    print("test_refresh_does_not_clear_the_coverage_record PASSED")


def test_pressing_start_costs_more_than_other_buttons():
    """Sampling the 100k policy put 7 of 20 frames in a menu. Every one of those
    screens is reached through Start, so the press is where the tax goes."""
    start = EmeraldEnv.ACTIONS.index("start")
    b = EmeraldEnv.ACTIONS.index("B")

    env = EmeraldEnv()
    env.reset()
    env.step(0)                      # spend the first-step exploration income
    env.drain_reward_terms()
    plain = env.step(b)[1]

    env2 = EmeraldEnv()
    env2.reset()
    env2.step(0)
    env2.drain_reward_terms()
    pressed = env2.step(start)[1]

    assert pressed < plain, f"Start paid {pressed}, B paid {plain}; Start is free"
    assert pressed - plain <= EmeraldEnv.START_PENALTY + 1e-6, (
        f"Start cost {pressed - plain}, expected {EmeraldEnv.START_PENALTY}"
    )
    assert env2.drain_reward_terms().get("start_press") == EmeraldEnv.START_PENALTY, (
        "the charge is missing from the reward breakdown"
    )
    env.close(); env2.close()
    print("test_pressing_start_costs_more_than_other_buttons PASSED")


def test_only_start_is_charged():
    """Walking and talking must stay free, or this becomes a tax on playing."""
    env = EmeraldEnv()
    env.reset()
    env.step(0)
    for action, name in enumerate(EmeraldEnv.ACTIONS):
        if name == "start":
            continue
        env.drain_reward_terms()
        env.step(action)
        assert "start_press" not in env.drain_reward_terms(), (
            f"action {action} ({name}) was billed as a Start press"
        )
    env.close()
    print("test_only_start_is_charged PASSED")


def _hold_still(env, s, steps):
    """Score `steps` steps that never move the player. Returns per-step rewards."""
    return [env._compute_reward(dict(s, pos=(5, 5))) for _ in range(steps)]


def test_standing_against_a_wall_is_charged():
    """Measured on the 300k policy: position frozen on 94.7% of steps, longest
    freeze 1,170 steps against one wall, total cost about -1.4."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, pos=(5, 5))
    env._frozen_steps = 0
    rewards = _hold_still(env, s, EmeraldEnv.FREEZE_LIMIT * 2)
    charged = [i for i, r in enumerate(rewards) if r <= EmeraldEnv.FREEZE_PENALTY + 1e-6]
    assert charged, (
        f"{EmeraldEnv.FREEZE_LIMIT * 2} steps frozen in place were never charged"
    )
    first = charged[0]
    assert first >= EmeraldEnv.FREEZE_LIMIT - 1, (
        f"charged at step {first}, before the {EmeraldEnv.FREEZE_LIMIT}-step limit"
    )
    env.close()
    print("test_standing_against_a_wall_is_charged PASSED")


def test_moving_clears_the_freeze_counter():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, pos=(5, 5))
    env._frozen_steps = 0
    _hold_still(env, s, EmeraldEnv.FREEZE_LIMIT - 2)
    assert env._frozen_steps > 0, "freeze counter is not counting"

    env._compute_reward(dict(s, pos=(6, 5)))
    assert env._frozen_steps == 0, (
        f"counter survived a step that moved: {env._frozen_steps}"
    )
    env.close()
    print("test_moving_clears_the_freeze_counter PASSED")


def test_a_battle_is_not_billed_as_a_freeze():
    """A battle holds the overworld position frozen for its whole length. The
    experience reward exists to buy exactly that, so it must not be taxed."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()

    env._prev = dict(s, pos=(5, 5), party_exp=1000)
    env._max_party_exp = 1000
    env._frozen_steps = 0

    env.drain_reward_terms()
    total = 0.0
    for i in range(EmeraldEnv.FREEZE_LIMIT * 2):
        # Winning experience every so often, never moving: a long battle.
        exp = 1000 + (i // 50) * 20
        total += env._compute_reward(dict(s, pos=(5, 5), party_exp=exp))

    # The ledger, not the counter: the counter resets to 0 on charging, so
    # reading it after the fact cannot tell a charge from an exemption. An
    # earlier version of this test checked the counter and passed with the
    # exemption deleted.
    terms = env.drain_reward_terms()
    assert "freeze" not in terms, (
        f"a fight that paid experience was billed {terms['freeze']} as a freeze"
    )
    assert total > 0, f"a long winning battle was billed {total}"
    env.close()
    print("test_a_battle_is_not_billed_as_a_freeze PASSED")


def test_observation_reports_the_last_action():
    """Gen-3 animates walking on the spot when the way is blocked, so a step
    that moved and a step into a wall look alike for several frames. Without
    the action in the observation the agent cannot tell them apart."""
    from env import N_ACTION_SLOTS, STATE_SIZE

    env = EmeraldEnv()
    obs, _ = env.reset()
    slots = slice(STATE_SIZE - 12, STATE_SIZE - 12 + N_ACTION_SLOTS)
    assert obs["state"][slots].sum() == 0.0, "reset must report no last action"

    for action in range(env.action_space.n):
        obs, *_ = env.step(action)
        onehot = obs["state"][slots]
        assert onehot.sum() == 1.0, f"action {action}: {onehot.sum()} slots set"
        assert onehot[action] == 1.0, (
            f"action {action} reported as {int(onehot.argmax())}"
        )
    env.close()
    print("test_observation_reports_the_last_action PASSED")


def test_observation_reports_the_penalty_counters():
    """The penalties were being charged on state the agent could not see:
    _frozen_steps, _recent_moves and _steps_since_reward all lived only in the
    environment, so the tile it was about to be charged -5 on looked identical
    to the tile it had just arrived at."""
    env = EmeraldEnv()
    obs, _ = env.reset()
    assert obs["state"][26] == 0.0 and obs["state"][28] == 0.0

    env._frozen_steps = EmeraldEnv.FREEZE_LIMIT // 2
    for k in range(EmeraldEnv.LOITER_LIMIT // 2):
        env._recent_moves.append(((0, 0), (k, 0)))
    env._steps_since_reward = EmeraldEnv.STALL_LIMIT // 2
    s = env.state_reader.read()
    vec = env._encode_state(s)

    assert abs(vec[26] - 0.5) < 0.01, f"freeze counter reported as {vec[26]}"
    assert abs(vec[27] - 0.5) < 0.01, f"loiter counter reported as {vec[27]}"
    assert abs(vec[28] - 0.5) < 0.01, f"stall counter reported as {vec[28]}"

    # Past the limit the fraction must saturate, not run away past 1.0 and
    # clip the whole vector's meaning.
    env._frozen_steps = EmeraldEnv.FREEZE_LIMIT * 10
    assert env._encode_state(s)[26] == 1.0
    env.close()
    print("test_observation_reports_the_penalty_counters PASSED")


def test_observation_reports_whether_the_step_moved():
    env = EmeraldEnv()
    obs, _ = env.reset()
    assert obs["state"][25] == 0.0, "reset must report not having moved"

    moved_seen = False
    for i in range(40):
        obs, *_ = env.step([1, 2, 3, 4][i % 4])
        if obs["state"][25] == 1.0:
            moved_seen = True
            break
    assert moved_seen, "40 steps of walking never reported a move"
    env.close()
    print("test_observation_reports_whether_the_step_moved PASSED")


def _settle(env, s, steps=None):
    """Score enough identical stable steps for flag credit to be allowed."""
    for _ in range(steps or EmeraldEnv.FLAG_SETTLE_STEPS):
        env._compute_reward(s)


def test_warp_door_pump_pays_nothing():
    """The exploit that ate the 54M-step nature2 run. Standing on Birch's lab
    door and warping in and out produces, per warp, a two-step garbage read
    with trainer 0->7 on an identical bogus map id, which the stability gate
    accepted as settled and paid +14. Traced live: 393 such flips in 12,000
    steps, trainer +472, experience 0."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    real = dict(s, map=(1, 4), pos=(6, 12), script_flag_count=187, trainer_flag_count=0)
    garbage = dict(s, map=(0, 0), pos=(0, 512), script_flag_count=167, trainer_flag_count=7)

    env._max_flags = {"script": 187, "trainer": 0}
    env._prev = dict(real)
    _settle(env, real)
    env.drain_reward_terms()

    # Garbage arrives at many levels, each a step of 16 or less above the
    # last, which is how a plain high-water mark got ratcheted up through them
    # at 1.5-10 per 1,000 steps after the first fix. Windows of one and two
    # reads, like the live trace showed.
    for warp in range(40):
        level = 7 + 4 * (warp % 10)
        bogus = dict(garbage, trainer_flag_count=level, script_flag_count=167 + warp)
        env._compute_reward(bogus)             # transition: gated
        if warp % 2:
            env._compute_reward(bogus)         # same bogus map: "stable"
        env._compute_reward(real)              # back: transition, gated
        _settle(env, real)                     # settles on the real map

    terms = env.drain_reward_terms()
    assert terms.get("trainer", 0.0) == 0.0, (
        f"the warp-door pump still pays: trainer booked {terms['trainer']} over 40 warps"
    )
    assert terms.get("script_flag", 0.0) == 0.0, (
        f"script flags paid {terms['script_flag']} on garbage"
    )
    env.close()
    print("test_warp_door_pump_pays_nothing PASSED")


def test_flag_credit_waits_for_the_map_to_settle():
    """Even a plausible-looking rise is refused inside FLAG_SETTLE_STEPS of a
    map change; a real one is paid once the map has stayed put."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._visited_tiles.clear(); env._tiles_per_map.clear(); env._visited_maps.clear()
    env._max_flags = {"script": 187, "trainer": 0}
    env._prev = dict(s, map=(0, 9))

    arrived = dict(s, map=(1, 4), trainer_flag_count=1, script_flag_count=187)
    env._compute_reward(arrived)                   # transition
    env.drain_reward_terms()
    early = [env._compute_reward(arrived) for _ in range(EmeraldEnv.FLAG_SETTLE_STEPS - 1)]
    assert "trainer" not in env.drain_reward_terms(), (
        "a trainer flag was credited before the map settled"
    )
    env._compute_reward(arrived)                   # now settled
    paid = env.drain_reward_terms().get("trainer", 0.0)
    assert abs(paid - EmeraldEnv.TRAINER_REWARD) < 1e-6, (
        f"a real trainer defeat paid {paid} after settling, expected {EmeraldEnv.TRAINER_REWARD}"
    )
    env.close()
    print("test_flag_credit_waits_for_the_map_to_settle PASSED")


def test_flags_pay_only_for_a_new_high():
    """A count that drops and recovers must net exactly one payment: the drop
    is charged and the recovery refunded, so the excursion cancels."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    base = dict(s, trainer_flag_count=5, script_flag_count=187)
    env._max_flags = {"script": 187, "trainer": 5}
    env._prev = dict(base)
    _settle(env, base)
    env.drain_reward_terms()

    # Each value is read FLAG_SETTLE_STEPS times so it counts as settled; the
    # point under test is the mark, not the settle window.
    _settle(env, dict(base, trainer_flag_count=6))              # new high: pays once
    _settle(env, dict(base, trainer_flag_count=0))              # dip: nothing
    _settle(env, dict(base, trainer_flag_count=6))              # recovery: nothing
    paid = env.drain_reward_terms().get("trainer", 0.0)
    assert abs(paid - EmeraldEnv.TRAINER_REWARD) < 1e-6, (
        f"dip and recovery re-paid the same flag: {paid}"
    )
    env.close()
    print("test_flags_pay_only_for_a_new_high PASSED")


def test_counters_clear_on_income_not_net_reward():
    """A wild win that cost HP: +0.6 exp, -0.9 hp, net -0.3. The battle must
    still count as earning, or the freeze counter runs through every fight.

    Net NEGATIVE on purpose. A first version used an exactly-cancelling pair
    and came out +2e-16 from float rounding, so the old net-reward logic still
    cleared the counters and the test passed with the fix reverted."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    base = dict(s, pos=(5, 5), party_levels=[7], party_hp_frac=1.0, party_exp=1000)
    env._prev = dict(base)
    env._max_party_exp = 1000
    env._compute_reward(base)                      # spend exploration
    env._frozen_steps = EmeraldEnv.FREEZE_LIMIT - 5
    env._steps_since_reward = EmeraldEnv.STALL_LIMIT - 5

    # exp +30 (+0.6) against hp -0.3 of the bar (-0.9): net -0.3
    drop = 0.9 / EmeraldEnv.HEAL_REWARD
    r = env._compute_reward(dict(base, party_exp=1030, party_hp_frac=1.0 - drop))
    assert r < -0.1, f"test setup wrong: net reward {r}, wanted clearly negative"
    assert env._frozen_steps == 0, (
        f"a step that earned experience left the freeze counter at {env._frozen_steps}"
    )
    assert env._steps_since_reward == 0, (
        f"a step that earned experience left the stall counter at {env._steps_since_reward}"
    )
    env.close()
    print("test_counters_clear_on_income_not_net_reward PASSED")


def test_long_garbage_window_nets_zero():
    """The hole the settle windows alone leave open. A relocation that stays
    garbage for longer than FLAG_SETTLE_STEPS, on one bogus map id and one
    bogus value, passes both settles and is paid. It is the only shape that
    explains trainer still paying 1.5-10 per 1,000 steps after the first fix.
    Potential-based credit charges the fall back to the real value at the same
    rate, so the excursion nets zero however long it lasts."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    real = dict(s, map=(1, 4), pos=(6, 12), script_flag_count=187, trainer_flag_count=0)
    env._max_flags = {"script": 187, "trainer": 0}
    env._prev = dict(real)
    _settle(env, real)
    env.drain_reward_terms()

    for warp in range(20):
        bogus = dict(real, map=(0, 0), pos=(0, 0), trainer_flag_count=7 + warp)
        for _ in range(EmeraldEnv.FLAG_SETTLE_STEPS + 3):   # long, constant window
            env._compute_reward(bogus)
        for _ in range(EmeraldEnv.FLAG_SETTLE_STEPS + 3):   # back to the real value
            env._compute_reward(real)

    booked = env.drain_reward_terms().get("trainer", 0.0)
    assert abs(booked) < 1e-6, (
        f"a long garbage window still nets {booked} over 20 warps"
    )
    assert env._max_flags["trainer"] == 0, f"mark left at {env._max_flags['trainer']}"
    env.close()
    print("test_long_garbage_window_nets_zero PASSED")


def test_our_attack_pays_once_per_message():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    base = dict(s, screen_text="")
    env._prev = dict(base)
    env._compute_reward(base)                     # spend exploration
    env.drain_reward_terms()

    hit = dict(base, screen_text="TORCHIC used SCRATCH!")
    env._compute_reward(hit)                      # new message: pays
    env._compute_reward(hit)                      # same line still on screen: no
    env._compute_reward(dict(base, screen_text="Wild ZIGZAGOON used TACKLE!"))   # theirs: no
    paid = env.drain_reward_terms().get("attack", 0.0)
    assert abs(paid - EmeraldEnv.ATTACK_REWARD) < 1e-6, f"attack paid {paid}, expected one {EmeraldEnv.ATTACK_REWARD}"
    assert EmeraldEnv.ATTACK_REWARD > 0.1
    env.close()
    print("test_our_attack_pays_once_per_message PASSED")


def test_pressing_into_a_wall_is_charged():
    """The 400k policy spent 25% of its actions pressing up into a blocked
    tile with nothing billing it."""
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), screen_text="")
    env._moved = False
    env._text_box_visible = lambda: False
    env.drain_reward_terms()
    r = env._bump("up", s, stable=True, had_prev=True)
    assert r <= EmeraldEnv.BUMP_PENALTY + 1e-9 < 0, f"a blocked press paid {r}"
    assert env.drain_reward_terms().get("bump") == EmeraldEnv.BUMP_PENALTY
    env.close()
    print("test_pressing_into_a_wall_is_charged PASSED")


def test_bump_exempts_battles_menus_and_moves():
    """Direction presses move a cursor, not the player, whenever text is on
    screen; and a press that moved is not a bump."""
    env = EmeraldEnv()
    env.reset()
    base = dict(env.state_reader.read(), screen_text="")
    env._text_box_visible = lambda: False
    cases = {
        "text on screen":     (dict(base, screen_text="What will TORCHIC do?"), False, "up", True, True),
        "dialogue box":       (base, False, "up", True, True),
        "it moved":           (base, True, "up", True, True),
        "not a direction":    (base, False, "A", True, True),
        "map in transition":  (base, False, "up", False, True),
        "first step":         (base, False, "up", True, False),
    }
    for name, (s, moved, key, stable, had_prev) in cases.items():
        env._moved = moved
        env._text_box_visible = (lambda: True) if name == "dialogue box" else (lambda: False)
        env.drain_reward_terms()
        r = env._bump(key, s, stable, had_prev)
        assert r == 0.0 and "bump" not in env.drain_reward_terms(), f"{name}: billed {r} as a bump"
    env.close()
    print("test_bump_exempts_battles_menus_and_moves PASSED")


def test_wild_encounter_pays_once_per_message():
    """The grass reward: encounters only happen where wild Pokemon are."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    base = dict(s, screen_text="")
    env._prev = dict(base)
    env._compute_reward(base)
    env.drain_reward_terms()
    hit = dict(base, screen_text="Wild ZIGZAGOON appeared!")
    env._compute_reward(hit)
    env._compute_reward(hit)                                        # same line: no
    # A stale "appeared!" fragment carried ahead of the next message must not
    # re-pay: the edge is on the text changing, and the detector is unanchored.
    env._compute_reward(dict(base, screen_text="Wild ZIGZAGOON appeared! Go! TORCHIC!"))
    env._compute_reward(dict(base, screen_text="Go! TORCHIC!"))     # not an encounter
    paid = env.drain_reward_terms().get("encounter", 0.0)
    assert abs(paid - 2 * EmeraldEnv.ENCOUNTER_REWARD) < 1e-6 and paid > 0.5, f"encounter paid {paid}"
    assert EmeraldEnv.FLEE_PENALTY + EmeraldEnv.ENCOUNTER_REWARD < -1.0, "encounter-and-flee must stay clearly negative"
    env.close()
    print("test_wild_encounter_pays_once_per_message PASSED")


def test_first_ever_section_pays_extra_and_survives_refresh():
    """After a novelty refresh a known section re-pays NEW_SECTION_REWARD only;
    the first-ever bonus is paid once for the life of the process."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = dict(s)
    env._compute_reward(s)
    env._refresh_novelty()
    env._sections_ever.clear()
    env.drain_reward_terms()

    far = dict(s, pos=(s["pos"][0] + 5 * EmeraldEnv.SECTION_SIZE, s["pos"][1]))
    env._compute_reward(far)
    first = env.drain_reward_terms()
    assert abs(first.get("first_section", 0.0) - EmeraldEnv.FIRST_SECTION_REWARD) < 1e-6, f"first visit paid {first}"
    assert first.get("first_section", 0.0) > 0.5

    env._refresh_novelty()                     # the per-episode set forgets it
    env._compute_reward(far)
    again = env.drain_reward_terms()
    assert "first_section" not in again, f"the first-ever bonus was re-paid after a refresh: {again}"
    assert abs(again.get("new_section", 0.0) - EmeraldEnv.NEW_SECTION_REWARD) < 1e-6, "the per-episode section bonus should still re-pay"
    env.close()
    print("test_first_ever_section_pays_extra_and_survives_refresh PASSED")


def _battle_state(s, in_battle: bool, enemy_hp: float) -> dict:
    v = np.zeros(12, dtype=np.float32)
    if in_battle:
        v[0], v[1], v[2] = 1.0, enemy_hp, 0.8
    return dict(s, in_battle=in_battle, battle=v, enemy_id=(309, 777) if in_battle else (0, 0))


def test_damage_pays_on_enemy_hp_falling_and_charges_on_healing():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = _battle_state(s, True, 1.0)
    env._compute_reward(_battle_state(s, True, 1.0))        # prime exploration
    env.drain_reward_terms()

    env._compute_reward(_battle_state(s, True, 0.6))        # 40% of the bar taken
    paid = env.drain_reward_terms().get("damage", 0.0)
    assert abs(paid - 0.4 * EmeraldEnv.DAMAGE_REWARD) < 1e-6 and paid > 0.1, f"damage paid {paid}"

    env._compute_reward(_battle_state(s, True, 0.8))        # enemy healed 20%
    charged = env.drain_reward_terms().get("damage", 0.0)
    assert abs(charged + 0.2 * EmeraldEnv.DAMAGE_REWARD) < 1e-6, f"healing charged {charged}"
    env.close()
    print("test_damage_pays_on_enemy_hp_falling_and_charges_on_healing PASSED")


def test_damage_pays_nothing_across_battle_start_and_end():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = _battle_state(s, False, 0.0)
    env._compute_reward(_battle_state(s, False, 0.0))
    env.drain_reward_terms()
    env._compute_reward(_battle_state(s, True, 1.0))        # battle starts: full bar appears
    env._compute_reward(_battle_state(s, True, 0.3))        # damage inside the battle
    inside = env.drain_reward_terms().get("damage", 0.0)
    env._compute_reward(_battle_state(s, False, 0.0))       # enemy fainted, battle over, zeros
    assert "damage" not in env.drain_reward_terms(), "the battle ending was billed as damage"
    assert abs(inside - 0.7 * EmeraldEnv.DAMAGE_REWARD) < 1e-6, f"in-battle damage paid {inside}"
    env.close()
    print("test_damage_pays_nothing_across_battle_start_and_end PASSED")


def _frontier_env():
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    size = EmeraldEnv.SECTION_SIZE
    start = (3 * size, 3 * size)
    base = dict(s, pos=start)
    g, n = s["map"]
    # Everything within two sections of the start is known; the frontier is
    # three sections out in every direction.
    env._sections_ever = {(g, n, sx, sy) for sx in range(1, 6) for sy in range(1, 6)}
    env._prev = dict(base)
    env._compute_reward(base)                                # prime: d = 3
    env.drain_reward_terms()
    return env, base, size


def test_frontier_pays_toward_unvisited_and_charges_away_and_loops_net_zero():
    env, base, size = _frontier_env()
    east = dict(base, pos=(base["pos"][0] + size, base["pos"][1]))   # d 3 -> 2
    env._compute_reward(east)
    toward = env.drain_reward_terms().get("frontier", 0.0)
    assert abs(toward - EmeraldEnv.FRONTIER_REWARD) < 1e-6 and toward > 0.05, f"toward paid {toward}"
    env._compute_reward(base)                                         # d 2 -> 3
    away = env.drain_reward_terms().get("frontier", 0.0)
    assert abs(away + EmeraldEnv.FRONTIER_REWARD) < 1e-6, f"away charged {away}"
    env.close()
    print("test_frontier_pays_toward_unvisited_and_charges_away_and_loops_net_zero PASSED")


def test_frontier_pays_nothing_on_a_map_change():
    env, base, size = _frontier_env()
    other = dict(base, map=(base["map"][0], base["map"][1] + 1), pos=(0, 0))
    env._compute_reward(other)                                        # transition step
    env._compute_reward(other)                                        # first stable step on the new map
    assert "frontier" not in env.drain_reward_terms(), "a map change was billed as frontier progress"
    env.close()
    print("test_frontier_pays_nothing_on_a_map_change PASSED")


def test_observation_has_tiles_and_battle():
    env = EmeraldEnv()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "observation outside the declared space"
    assert obs["tiles"].shape == (9, 11, 15) and obs["tiles"].dtype == np.uint8
    assert obs["battle"].shape == (12,) and obs["battle"].dtype == np.float32
    obs, *_ = env.step(0)
    assert env.observation_space.contains(obs)
    env.close()
    print("test_observation_has_tiles_and_battle PASSED")


def test_tiles_reuse_previous_window_during_map_transition():
    """A mid-relocation read must not reach the policy as a window of garbage;
    the previous stable window is reused until the map id settles."""
    env = EmeraldEnv()
    obs, _ = env.reset()
    stable_tiles = obs["tiles"].copy()
    real_read, real_window = env.state_reader.read, env.state_reader.tiles.window
    s = real_read()
    bogus = dict(s, map=(s["map"][0], s["map"][1] + 1), pos=(2560, 255))
    garbage = np.zeros_like(stable_tiles); garbage[8] = 1          # an all-unknown window
    env.state_reader.read = lambda: bogus
    env.state_reader.tiles.window = lambda: garbage                # what a relocated read would yield
    try:
        obs, *_ = env.step(0)
    finally:
        env.state_reader.read, env.state_reader.tiles.window = real_read, real_window
    assert np.array_equal(obs["tiles"], stable_tiles), "a transition step produced a fresh window"
    env.close()
    print("test_tiles_reuse_previous_window_during_map_transition PASSED")


def test_damage_pays_nothing_when_the_enemy_changes():
    """Review finding: gBattleMons is stale, and a battle's first in-battle
    step still shows the previous enemy, so stale hp -> new enemy's 1.0 was
    booked as a charge at almost every battle start (and at every trainer
    switch-in), cancelling the KO just paid. Damage only counts on the same
    enemy, identified by species and personality."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    a = dict(_battle_state(s, True, 0.0), enemy_id=(309, 12345))     # stale: last foe at 0 hp
    b = dict(_battle_state(s, True, 1.0), enemy_id=(286, 99999))     # the new foe, full bar
    env._prev = a
    env._compute_reward(a)
    env.drain_reward_terms()
    env._compute_reward(b)
    assert "damage" not in env.drain_reward_terms(), "a new enemy's full bar was billed as healing"
    env._compute_reward(dict(b, battle=_battle_state(s, True, 0.4)["battle"]))   # same foe takes damage
    paid = env.drain_reward_terms().get("damage", 0.0)
    assert abs(paid - 0.6 * EmeraldEnv.DAMAGE_REWARD) < 1e-6, f"damage on the same enemy paid {paid}"
    env.close()
    print("test_damage_pays_nothing_when_the_enemy_changes PASSED")



def test_encounters_stop_paying_after_the_cap_on_one_map():
    """The 3.4M policy farmed Route 101's grass: 43 battles in 4,000 steps,
    encounter its largest income term, 97% of steps on one map. Encounters
    pay for the first ENCOUNTER_CAP_PER_MAP on a map per novelty window."""
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    base = dict(s, screen_text="")
    env._prev = dict(base); env._compute_reward(base); env.drain_reward_terms()
    for k in range(EmeraldEnv.ENCOUNTER_CAP_PER_MAP + 5):
        env._compute_reward(dict(base, screen_text=f"Wild POOCHYENA appeared! {k}"))
        env._compute_reward(base)
    paid = env.drain_reward_terms().get("encounter", 0.0)
    assert abs(paid - EmeraldEnv.ENCOUNTER_CAP_PER_MAP * EmeraldEnv.ENCOUNTER_REWARD) < 1e-6, f"encounters paid {paid}"
    env._refresh_novelty()
    env._compute_reward(dict(base, screen_text="Wild POOCHYENA appeared! again"))
    assert env.drain_reward_terms().get("encounter", 0.0) > 0, "the cap must reset with the novelty window"
    env.close()
    print("test_encounters_stop_paying_after_the_cap_on_one_map PASSED")


def test_b_cannot_cancel_an_evolution():
    """Gen 3: B during 'X is evolving!' cancels it. The env turns that B into
    a no-op, so an evolution can never be cancelled."""
    env = EmeraldEnv()
    env.reset()
    pressed = []
    real = env.gba.core.set_keys
    env.gba.core.set_keys = lambda *k: (pressed.append(k), real(*k))[1]
    env._prev = dict(env._prev, screen_text="What? TORCHIC is evolving!")
    env.step(EmeraldEnv.ACTIONS.index("B"))
    from pygba.utils import KEY_MAP
    assert all(KEY_MAP["B"] not in k for k in pressed), f"B was pressed during an evolution: {pressed}"
    pressed.clear()
    env._prev = dict(env._prev, screen_text="")
    env.step(EmeraldEnv.ACTIONS.index("B"))
    assert any(KEY_MAP["B"] in k for k in pressed), "B must still work outside an evolution"
    env.gba.core.set_keys = real
    env.close()
    print("test_b_cannot_cancel_an_evolution PASSED")


def test_completed_evolution_pays_once():
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), screen_text="")
    env._prev = dict(s); env._compute_reward(s); env.drain_reward_terms()
    done = dict(s, screen_text="Congratulations! Your TORCHIC evolved into COMBUSKEN!")
    env._compute_reward(done); env._compute_reward(done)
    paid = env.drain_reward_terms().get("evolution", 0.0)
    assert abs(paid - EmeraldEnv.EVOLUTION_REWARD) < 1e-6 and paid > 1, f"evolution paid {paid}"
    env.close()
    print("test_completed_evolution_pays_once PASSED")


def test_effectiveness_messages_pay_only_after_our_move():
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), screen_text="")
    env._prev = dict(s); env._compute_reward(s); env.drain_reward_terms()
    for t in ("TORCHIC used EMBER!", "It's super effective!",
              "Wild TREECKO used POUND!", "It's super effective!",
              "TORCHIC used SCRATCH!", "It's not very effective..."):
        env._compute_reward(dict(s, screen_text=t))
    terms = env.drain_reward_terms()
    assert abs(terms.get("super_effective", 0) - EmeraldEnv.SUPER_EFFECTIVE_REWARD) < 1e-6, terms
    assert abs(terms.get("not_effective", 0) - EmeraldEnv.NOT_EFFECTIVE_PENALTY) < 1e-6, terms
    assert EmeraldEnv.SUPER_EFFECTIVE_REWARD > 0 > EmeraldEnv.NOT_EFFECTIVE_PENALTY
    env.close()
    print("test_effectiveness_messages_pay_only_after_our_move PASSED")


def test_walkthrough_starts_at_the_rival_and_points_at_him():
    """Every save on disk has the Route 103 rival unbeaten and no Pokedex, so
    the footprints man in Oldale has blocked Route 102 in every run. The
    first story stage must be the rival, at (10, 3) on Route 103."""
    from knowledge import STORY
    env = EmeraldEnv()
    obs, _ = env.reset()                          # Route 103 at (9, 21)
    assert env._stage == 0 and STORY[0]["name"] == "rival", env._stage
    g = obs["guide"]
    assert g[1] == 1.0 and g[5] == 1.0, f"guide should point north at the rival: {g}"
    assert abs(g[6] - 19 / 64) < 1e-6, f"distance to (10, 3) from (9, 21) is 19 tiles, north-dominant: {g}"
    env.close()
    print("test_walkthrough_starts_at_the_rival_and_points_at_him PASSED")


def test_story_flag_pays_once_and_moves_to_the_next_stage():
    from knowledge import STORY
    env = EmeraldEnv()
    env.reset()
    s = env.state_reader.read()
    env._prev = dict(s); env._compute_reward(s); env.drain_reward_terms()
    beaten = dict(s, story=(True, False, False))
    env._compute_reward(beaten); env._compute_reward(beaten)
    terms = env.drain_reward_terms()
    assert terms.get("story", 0) > 10, f"finishing a story stage paid {terms.get('story', 0)}"
    assert abs(terms.get("story", 0) - EmeraldEnv.STORY_REWARD) < 1e-6, terms
    assert env._stage == 1 and STORY[1]["name"] == "pokedex"
    env.close()
    print("test_story_flag_pays_once_and_moves_to_the_next_stage PASSED")


def test_stage_route_pays_the_next_map_only_once():
    from knowledge import STORY
    env = EmeraldEnv()
    env.reset()
    s = dict(env.state_reader.read(), story=(True, False, False))
    env._prev = dict(s); env._compute_reward(s); env.drain_reward_terms()   # now stage 1, on Route 103
    route = STORY[1]["route"]
    def visit(m):
        a = dict(s, map=m, pos=(1, 1)); env._compute_reward(a); env._compute_reward(a)
    visit(route[-1])                              # skipping ahead pays nothing
    assert "waypoint" not in env.drain_reward_terms()
    visit(route[1])                               # Oldale: the next stop
    assert abs(env.drain_reward_terms().get("waypoint", 0) - EmeraldEnv.WAYPOINT_REWARD) < 1e-6
    visit(route[0]); visit(route[1])              # back and forth pays nothing again
    assert "waypoint" not in env.drain_reward_terms()
    env.close()
    print("test_stage_route_pays_the_next_map_only_once PASSED")


def test_observation_has_moves_and_guide():
    env = EmeraldEnv()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["moves"].shape == (8,) and obs["guide"].shape == (8,)
    assert obs["heal"].shape == (6,) and obs["map_info"].shape == (3,) and obs["evo"].shape == (1,)
    assert obs["map_info"][0] == 1.0, "Route 103 has grass encounters"
    assert abs(obs["evo"][0] - 9 / 50) < 1e-6, f"Torchic Lv7 evolves in 9 levels: {obs['evo']}"
    assert obs["guide"][5] == 1.0, f"the guide has a route from the anchor: {obs['guide']}"
    b = EmeraldEnv(state_path="battle.state"); ob, _ = b.reset()
    assert ob["moves"][4:].any(), "battle fixture: move powers missing"
    env.close(); b.close()
    print("test_observation_has_moves_and_guide PASSED")


def test_heal_guide_points_to_the_last_center_when_hurt():
    env = EmeraldEnv()
    env.reset()                                          # party HP 24%, last heal: Oldale, south of Route 103
    s = env.state_reader.read()
    h = env._heal(dict(s, party_hp_frac=0.24))
    assert h[0] == 1.0 and h[2] == 1.0, f"hurt on Route 103: should point south to Oldale: {h}"
    assert not env._heal(dict(s, party_hp_frac=0.9)).any(), "healthy: no heal guidance"
    env.close()
    print("test_heal_guide_points_to_the_last_center_when_hurt PASSED")


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
    test_experience_gain_is_rewarded()
    test_experience_uses_a_high_water_mark()
    test_running_away_is_penalised_once()
    test_fleeing_costs_more_than_a_battle_wins()
    test_opponent_fleeing_is_not_penalised()
    test_zero_max_steps_removes_the_limit()
    test_positive_max_steps_still_truncates()
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
    test_healing_is_rewarded()
    test_damage_then_heal_nets_zero()
    test_empty_party_does_not_move_hp_reward()
    test_starts_from_furthest_progress_state()
    test_changing_options_is_penalised()
    test_unchanged_options_cost_nothing()
    test_failed_options_read_is_not_billed()
    test_text_speed_is_pinned_to_fast()
    test_pinning_text_speed_leaves_other_options_alone()
    test_pinned_text_speed_is_not_billed_as_a_change()
    test_pacing_a_few_tiles_is_penalised()
    test_pacing_across_a_block_boundary_is_penalised()
    test_walking_somewhere_is_not_loitering()
    test_standing_still_is_never_billed_as_loitering()
    test_catching_outweighs_seeing()
    test_a_level_beats_a_handful_of_tiles()
    test_reward_ledger_matches_reward_paid()
    test_draining_the_ledger_clears_it()
    test_novelty_refresh_restores_income()
    test_refresh_fires_on_schedule_during_an_endless_episode()
    test_refresh_does_not_clear_the_coverage_record()
    test_pressing_start_costs_more_than_other_buttons()
    test_only_start_is_charged()
    test_standing_against_a_wall_is_charged()
    test_moving_clears_the_freeze_counter()
    test_a_battle_is_not_billed_as_a_freeze()
    test_observation_reports_the_last_action()
    test_observation_reports_the_penalty_counters()
    test_observation_reports_whether_the_step_moved()
    test_warp_door_pump_pays_nothing()
    test_flag_credit_waits_for_the_map_to_settle()
    test_flags_pay_only_for_a_new_high()
    test_counters_clear_on_income_not_net_reward()
    test_long_garbage_window_nets_zero()
    test_our_attack_pays_once_per_message()
    test_pressing_into_a_wall_is_charged()
    test_bump_exempts_battles_menus_and_moves()
    test_wild_encounter_pays_once_per_message()
    test_encounters_stop_paying_after_the_cap_on_one_map()
    test_b_cannot_cancel_an_evolution()
    test_completed_evolution_pays_once()
    test_effectiveness_messages_pay_only_after_our_move()
    test_walkthrough_starts_at_the_rival_and_points_at_him()
    test_story_flag_pays_once_and_moves_to_the_next_stage()
    test_stage_route_pays_the_next_map_only_once()
    test_observation_has_moves_and_guide()
    test_heal_guide_points_to_the_last_center_when_hurt()
    test_first_ever_section_pays_extra_and_survives_refresh()
    test_damage_pays_on_enemy_hp_falling_and_charges_on_healing()
    test_damage_pays_nothing_across_battle_start_and_end()
    test_damage_pays_nothing_when_the_enemy_changes()
    test_frontier_pays_toward_unvisited_and_charges_away_and_loops_net_zero()
    test_frontier_pays_nothing_on_a_map_change()
    test_observation_has_tiles_and_battle()
    test_tiles_reuse_previous_window_during_map_transition()
    print("\nall env tests passed")
