"""Measure how far a random agent gets. This is the go/no-go gate on the
exploration reward, per the spec's primary risk.

Corrections applied per project-lead ruling (see task-5-report.md):
1. maps_visited (env's running distinct-map count) drives left_start, not
   furthest_map (a running max that can't register a "lower" map). maps_reached
   still legitimately uses furthest_map, since that's the intended metric there.
2. Extra measurement: count step-to-step decreases in script_flag_count and
   trainer_flag_count, to check for a reward-leak precondition (a flag that
   clears and re-sets would pay NEW_TILE-style reward again via max(0, delta)).
   Decreases are only counted within an episode, not across a reset (a reset
   reloads boot.state, which is a boundary artifact, not in-game flag clearing).
"""

import json

from env import EmeraldEnv

EPISODES = 20
STEPS = 4096


def main():
    env = EmeraldEnv(max_steps=STEPS)
    maps_reached = set()
    tiles = []
    start_map = None
    left_start = 0
    script_flag_decreases = 0
    trainer_flag_decreases = 0

    for ep in range(EPISODES):
        obs, info = env.reset()
        state = env.state_reader.read()
        if start_map is None:
            start_map = state["map"]
        prev_script = state["script_flag_count"]
        prev_trainer = state["trainer_flag_count"]

        for _ in range(STEPS):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            state = env.state_reader.read()
            if state["script_flag_count"] < prev_script:
                script_flag_decreases += 1
            if state["trainer_flag_count"] < prev_trainer:
                trainer_flag_decreases += 1
            prev_script = state["script_flag_count"]
            prev_trainer = state["trainer_flag_count"]
            if term or trunc:
                break

        maps_this_episode = info["maps_visited"]
        maps_reached.add(info["furthest_map"])
        tiles.append(info["tiles_visited"])
        if maps_this_episode > 1:
            left_start += 1
        print(f"episode {ep+1}/{EPISODES}: {info['tiles_visited']} tiles, "
              f"{maps_this_episode} maps, furthest {info['furthest_map']}")

    result = {
        "episodes": EPISODES,
        "steps_per_episode": STEPS,
        "maps_reached": sorted([list(m) for m in maps_reached]),
        "mean_tiles": sum(tiles) / len(tiles),
        "max_tiles": max(tiles),
        "left_start_map_fraction": left_start / EPISODES,
        "script_flag_decreases": script_flag_decreases,
        "trainer_flag_decreases": trainer_flag_decreases,
    }
    with open("baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nleft the starting map in {left_start}/{EPISODES} episodes")
    print(f"mean tiles {result['mean_tiles']:.1f}, max {result['max_tiles']}")
    print(f"script_flag_count decreases: {script_flag_decreases}")
    print(f"trainer_flag_count decreases: {trainer_flag_decreases}")
    env.close()


if __name__ == "__main__":
    main()
