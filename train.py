"""PPO training for the Emerald agent.

8 workers, not 16: measured throughput barely improves past 8 (262 vs 360
steps/sec) and the machine has ~12GB free RAM, so 16 risks swapping.
"""

import argparse
from pathlib import Path

# ponytail: importing torch before any mgba core exists makes libmgba's
# run_frame() spin forever. Creating one core first inoculates the process.
# Only the ORDER matters: verified that the inoculation survives this core
# being dereferenced and garbage collected, so the name is for clarity only.
# This block must stay above the torch/SB3 imports below.
# Ceiling: costs one extra 16MB ROM copy at startup.
from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Image as TBImage
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from coverage import render_coverage, render_to_array
from env import EmeraldEnv

N_ENVS = 8
N_STEPS = 1024  # 8 x 1024 x (3x80x120 + 17) ~ 236MB rollout buffer


def make_env():
    def _init():
        return EmeraldEnv()
    return _init


class ProgressCallback(BaseCallback):
    """Logs progress metrics and dumps a savestate at each new high in
    distinct maps visited.

    The savestates are the raw material for the spec's curriculum fallback.
    Workers live in separate processes, so the state is pulled back with
    env_method rather than read directly. maps_visited (a running distinct-
    map count) drives this rather than a lexicographic map-id comparison,
    which has nothing to do with route progress: Littleroot is (0,9) but
    Oldale is (0,10), Petalburg (0,0) and Rustboro (0,3), so pinning on
    "furthest" map id would never register most of the early-game towns.
    """

    def __init__(self, save_dir: str = "checkpoints"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_maps_visited = 0

    def _on_step(self) -> bool:
        best_badges = 0
        best_tiles = 0
        for i, info in enumerate(self.locals.get("infos", [])):
            if "maps_visited" not in info:
                continue
            best_badges = max(best_badges, info["badges"])
            best_tiles = max(best_tiles, info["tiles_visited"])
            if info["maps_visited"] > self.best_maps_visited:
                self.best_maps_visited = info["maps_visited"]
                n = self.best_maps_visited
                group, num = info["map"]
                blob = self.training_env.env_method("save_state", indices=[i])[0]
                path = self.save_dir / f"furthest_{n}maps_{group}_{num}.state"
                path.write_bytes(blob)
                self.logger.record("progress/new_maps_visited", float(n))
                print(f"new maps_visited high: {n} maps, now at group {group} num {num} "
                      f"at {self.num_timesteps} steps -> {path}")

        self.logger.record("progress/badges", best_badges)
        self.logger.record("progress/tiles_visited", best_tiles)
        self.logger.record("progress/maps_visited", float(self.best_maps_visited))
        return True


class CoverageCallback(BaseCallback):
    """Renders where the agent has been, as a contact sheet of per-map grids.

    Unions every worker's coverage set, writes media/coverage_<step>.png, and
    logs the same image to TensorBoard so exploration can be watched spreading
    live in the Images tab rather than by opening files.
    """

    def __init__(self, every: int = 50_000, media_dir: str = "media"):
        super().__init__()
        self.every = every
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(exist_ok=True)
        self._next_at = every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_at:
            return True
        self._next_at += self.every

        tiles: set[tuple[int, int, int, int]] = set()
        for worker_tiles in self.training_env.env_method("coverage"):
            tiles.update(worker_tiles)
        if not tiles:
            return True

        path = self.media_dir / f"coverage_{self.num_timesteps:09d}.png"
        render_coverage(sorted(tiles), path)
        # channels-last uint8; SB3's Image wants the dataformats spelled out
        self.logger.record(
            "coverage/map",
            TBImage(render_to_array(sorted(tiles)), "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record("coverage/unique_tiles", float(len(tiles)))
        print(f"coverage: {len(tiles)} tiles at {self.num_timesteps} steps -> {path}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--coverage-every", type=int, default=50_000,
                        help="steps between coverage renders")
    args = parser.parse_args()

    # start_method explicit: SB3 defaults to "forkserver", which forks workers
    # from a clean server process that never saw the torch-after-mgba warmup
    # above, so they'd hang the same way. "fork" makes workers inherit this
    # already-inoculated parent process. Do not "clean this up" to the default.
    # VecMonitor: without it there is no rollout/ep_rew_mean or ep_len_mean in
    # TensorBoard, the one number that answers "is this learning" on a
    # multi-day run.
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(N_ENVS)], start_method="fork"))

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
            gamma=0.999,       # ~1k-step effective horizon. The badge is far beyond it,
                               # but the reward is dense by design (tile novelty pays
                               # every few steps), so the agent follows a local gradient
                               # rather than needing to see the badge from the start.
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
        CoverageCallback(every=args.coverage_every),
    ]

    model.learn(total_timesteps=args.steps, callback=callbacks,
                reset_num_timesteps=args.resume is None)
    model.save("checkpoints/emerald_final")


if __name__ == "__main__":
    main()
