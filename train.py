"""PPO training for the Emerald agent.

8 workers, not 16: measured throughput barely improves past 8 (262 vs 360
steps/sec) and the machine has ~12GB free RAM, so 16 risks swapping.
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import EmeraldEnv

N_ENVS = 8
N_STEPS = 1024  # 8 x 1024 x (3x80x120 + 17) ~ 236MB rollout buffer


def make_env():
    def _init():
        return EmeraldEnv()
    return _init


class ProgressCallback(BaseCallback):
    """Logs progress metrics and dumps a savestate at each new furthest map.

    The savestates are the raw material for the spec's curriculum fallback.
    Workers live in separate processes, so the state is pulled back with
    env_method rather than read directly.
    """

    def __init__(self, save_dir: str = "checkpoints"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_map = (0, 0)

    def _on_step(self) -> bool:
        best_badges = 0
        best_tiles = 0
        for i, info in enumerate(self.locals.get("infos", [])):
            if "furthest_map" not in info:
                continue
            best_badges = max(best_badges, info["badges"])
            best_tiles = max(best_tiles, info["tiles_visited"])
            if info["furthest_map"] > self.best_map:
                self.best_map = info["furthest_map"]
                group, num = self.best_map
                blob = self.training_env.env_method("save_state", indices=[i])[0]
                path = self.save_dir / f"furthest_{group}_{num}.state"
                path.write_bytes(blob)
                self.logger.record("progress/new_furthest_map", float(num))
                print(f"new furthest map: group {group} num {num} "
                      f"at {self.num_timesteps} steps -> {path}")

        self.logger.record("progress/badges", best_badges)
        self.logger.record("progress/tiles_visited", best_tiles)
        self.logger.record("progress/furthest_map_num", float(self.best_map[1]))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log="runs")
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy",  # handles Dict obs: CNN for screen, MLP for state
            env,
            n_steps=N_STEPS,
            batch_size=512,
            n_epochs=3,
            gamma=0.999,       # long horizon: the badge is ~50k steps away
            ent_coef=0.01,     # keep exploring
            learning_rate=2.5e-4,
            tensorboard_log="runs",
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path="checkpoints",
            name_prefix="emerald",
        ),
        ProgressCallback(),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks,
                reset_num_timesteps=args.resume is None)
    model.save("checkpoints/emerald_final")


if __name__ == "__main__":
    main()
