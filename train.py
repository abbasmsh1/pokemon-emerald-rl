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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper, VecMonitor

from coverage import render_coverage, render_to_array
from env import EmeraldEnv

N_ENVS = 8
N_STEPS = 1024  # 8 x 1024 x (3x80x120 + 17) ~ 236MB rollout buffer


def make_env(max_steps: int):
    def _init():
        return EmeraldEnv(max_steps=max_steps)
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
        best_sections = 0
        for i, info in enumerate(self.locals.get("infos", [])):
            if "maps_visited" not in info:
                continue
            best_badges = max(best_badges, info["badges"])
            best_tiles = max(best_tiles, info["tiles_visited"])
            best_sections = max(best_sections, info.get("sections_visited", 0))
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
        self.logger.record("progress/sections_visited", best_sections)
        self.logger.record("progress/maps_visited", float(self.best_maps_visited))
        return True


class RewardBreakdownCallback(BaseCallback):
    """Logs reward per 1,000 steps and which term paid it.

    SB3's ep_rew_mean is useless on this run. With --max-steps 0 an episode
    only ends on a badge, so it never updates, and what it does print is an
    ep_info_buffer restored from the checkpoint: 1.34e6 reward over 6.55e4-step
    episodes, from a run whose step cap was removed long ago. Every reward
    change since then has been flying blind.

    This is episode-independent: it drains the workers' ledgers once per
    rollout and reports a rate, so a term that starts farming itself shows up
    as a curve climbing away from the others.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        totals: dict[str, float] = {}
        for worker in self.training_env.env_method("drain_reward_terms"):
            for term, value in worker.items():
                totals[term] = totals.get(term, 0.0) + value

        drain = getattr(self.training_env, "drain_curiosity", None)
        if drain is not None:
            totals["curiosity"] = drain()
        steps = self.model.n_steps * self.training_env.num_envs
        for term, value in totals.items():
            self.logger.record(f"reward/{term}", value / steps * 1000)
        self.logger.record("reward/total", sum(totals.values()) / steps * 1000)


class CuriosityWrapper(VecEnvWrapper):
    """Pays the agent for reaching screens the forward model cannot predict.

    This is the "vision guides where to go" part. The forward model in
    extractor.py predicts the next screen's features from the current ones and
    the action; where it is wrong, the screen is doing something the agent has
    not seen enough of, and that is where exploration should go. Where it is
    right, the place is known and the bonus fades on its own. It is a per-step
    signal from pixels, so it works in places the RAM novelty terms cannot see:
    a battle that is going differently, a new menu, an animation.

    The error is divided by its running mean so BETA is the bonus per step on
    an average screen regardless of feature scale, and capped so one strange
    frame cannot outbid a badge. Zero on the step after an episode boundary,
    where the pair is not consecutive.

    Known limit (the "noisy TV"): screens that are unpredictable but useless,
    like battle animations, also pay. The cap and the decay bound the damage;
    if reward/curiosity ever dominates while new_tile is flat, lower BETA.
    """

    BETA = 0.005
    CAP = 5.0          # in units of BETA

    def __init__(self, venv):
        super().__init__(venv)
        self._fe = None
        self._device = None
        self._prev_screen = None
        self._actions = None
        self._running = None     # running mean error; seeded from the first batch
        self._booked = 0.0

    def attach(self, extractor, device) -> None:
        """Called once the model exists; the extractor is the policy's own."""
        self._fe, self._device = extractor, device

    def reset(self):
        obs = self.venv.reset()
        self._prev_screen = obs["screen"].copy()
        return obs

    def step_async(self, actions) -> None:
        self._actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        import numpy as np
        import torch

        obs, rewards, dones, infos = self.venv.step_wait()
        if self._fe is not None and self._prev_screen is not None:
            with torch.no_grad():
                s0 = torch.as_tensor(self._prev_screen, dtype=torch.float32, device=self._device) / 255.0
                s1 = torch.as_tensor(obs["screen"], dtype=torch.float32, device=self._device) / 255.0
                a = torch.as_tensor(np.asarray(self._actions).reshape(-1), device=self._device)
                err = self._fe.forward_error(s0, s1, a).cpu().numpy()
            if self._running is None:
                self._running = float(err.mean())
            bonus = self.BETA * np.minimum(err / (self._running + 1e-8), self.CAP)
            bonus[dones.astype(bool)] = 0.0
            self._running = 0.99 * self._running + 0.01 * float(err.mean())
            rewards = rewards + bonus.astype(rewards.dtype)
            self._booked += float(bonus.sum())
        self._prev_screen = obs["screen"].copy()
        return obs, rewards, dones, infos

    def drain_curiosity(self) -> float:
        total, self._booked = self._booked, 0.0
        return total


class InverseDynamicsCallback(BaseCallback):
    """Trains the shared CNN to predict the button pressed between consecutive
    screens, from pairs in the rollout buffer, after each PPO update.

    The policy's own gradient only reaches the CNN through reward, and 54
    million steps of that produced features that could not tell a step into
    a wall from a step that moved. The inverse-dynamics loss supervises exactly
    that distinction, with labels we already have for free: the action we took.
    Gradients flow into extractor.cnn, which PPO shares for pi and vf, so the
    policy sees the improved features on its next update.

    aux/idm_acc is the number to watch: chance is 1/8. It should climb well
    above that within the first few million steps, and if it does not, the
    features are not learning what the screen shows.
    """

    def __init__(self, pairs: int = 2048, steps: int = 4, lr: float = 2.5e-4, weight: float = 1.0):
        super().__init__()
        self.pairs, self.steps, self.lr, self.weight = pairs, steps, lr, weight
        self._opt = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import numpy as np
        import torch
        import torch.nn.functional as F

        fe = self.model.policy.features_extractor
        if not hasattr(fe, "idm_logits"):
            return
        if self._opt is None:
            params = list(fe.cnn.parameters()) + list(fe.idm.parameters()) + list(fe.fwd.parameters())
            self._opt = torch.optim.Adam(params, lr=self.lr)

        buf = self.model.rollout_buffer
        screens = buf.observations["screen"]            # (n_steps, n_envs, 3, 80, 120)
        actions = buf.actions.reshape(buf.buffer_size, buf.n_envs).astype(np.int64)
        # A pair (t, t+1) is valid when t+1 did not start a new episode.
        valid = np.argwhere(buf.episode_starts[1:] == 0)   # rows: (t, env)
        if len(valid) == 0:
            return
        rng = np.random.default_rng(self.num_timesteps)
        pick = valid[rng.choice(len(valid), size=min(self.pairs, len(valid)), replace=False)]
        t, e = pick[:, 0], pick[:, 1]
        dev = self.model.device
        s0 = torch.as_tensor(screens[t, e], dtype=torch.float32, device=dev) / 255.0
        s1 = torch.as_tensor(screens[t + 1, e], dtype=torch.float32, device=dev) / 255.0
        a = torch.as_tensor(actions[t, e], device=dev)

        fe.train()
        total_loss = fwd_total = 0.0
        for _ in range(self.steps):
            logits = fe.idm_logits(s0, s1)
            idm_loss = self.weight * F.cross_entropy(logits, a)
            fwd_loss = fe.forward_error(s0, s1, a).mean()   # features detached inside
            self._opt.zero_grad()
            (idm_loss + fwd_loss).backward()
            self._opt.step()
            total_loss += idm_loss.item()
            fwd_total += fwd_loss.item()
        with torch.no_grad():
            acc = (fe.idm_logits(s0, s1).argmax(1) == a).float().mean().item()
        self.logger.record("aux/idm_loss", total_loss / self.steps)
        self.logger.record("aux/idm_acc", acc)
        self.logger.record("aux/fwd_loss", fwd_total / self.steps)


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
        # Set on the first step, not here. Resuming starts num_timesteps in the
        # millions while this would start at `every`, so the callback fired,
        # bumped by `every`, was still behind, and fired again every step --
        # 1,027 renders before it caught up.
        self._next_at: int | None = None

    def _on_step(self) -> bool:
        if self._next_at is None:
            self._next_at = self.num_timesteps + self.every
            return True
        if self.num_timesteps < self._next_at:
            return True
        self._next_at = self.num_timesteps + self.every

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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="where the PPO zips are written. A change of "
                             "feature extractor cannot be resumed into, so a "
                             "new backbone needs its own directory rather than "
                             "the old zips being moved out of the way. "
                             "Savestates still go to checkpoints/, since they "
                             "are backbone-independent and are what the "
                             "curriculum anchor is chosen from")
    parser.add_argument("--coverage-every", type=int, default=50_000,
                        help="steps between coverage renders")
    parser.add_argument("--backbone", choices=("nature", "resnet18", "agent"), default="nature",
                        help="nature: small CNN trained from scratch (default). "
                             "resnet18: frozen ImageNet backbone. "
                             "agent: NatureCNN plus a learned text encoder over the "
                             "on-screen text and an inverse-dynamics head that "
                             "trains the CNN to predict the pressed button from "
                             "consecutive screens (see extractor.py)")
    parser.add_argument("--finetune", action="store_true",
                        help="unfreeze the resnet18 backbone; needs far more VRAM")
    parser.add_argument("--max-steps", type=int, default=16384,
                        help="steps per episode before truncation. The design doc "
                             "names this the first lever when training plateaus: "
                             "the badge is ~50k steps out, so a 16k episode cannot "
                             "reach it in one life. 0 removes the limit, which "
                             "also stops reset() ever running, so exploration "
                             "novelty is never replenished")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="PPO minibatch size. Smaller means more optimizer "
                             "steps per rollout: better sample efficiency, lower "
                             "throughput, less VRAM per step")
    args = parser.parse_args()

    # start_method explicit: SB3 defaults to "forkserver", which forks workers
    # from a clean server process that never saw the torch-after-mgba warmup
    # above, so they'd hang the same way. "fork" makes workers inherit this
    # already-inoculated parent process. Do not "clean this up" to the default.
    # VecMonitor: without it there is no rollout/ep_rew_mean or ep_len_mean in
    # TensorBoard, the one number that answers "is this learning" on a
    # multi-day run.
    curiosity = CuriosityWrapper(SubprocVecEnv([make_env(args.max_steps) for _ in range(N_ENVS)], start_method="fork"))
    env = VecMonitor(curiosity)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log="runs")
        print(f"resumed from {args.resume}")
    else:
        policy_kwargs = {}
        if args.backbone == "resnet18":
            # Imported here, not at module scope, so a `nature` run never pays for
            # torchvision or pulls the ImageNet weights off disk.
            from backbone import ResNetExtractor

            policy_kwargs = {
                "features_extractor_class": ResNetExtractor,
                "features_extractor_kwargs": {"finetune": args.finetune},
            }
            print(f"backbone: resnet18 (finetune={args.finetune})")
        elif args.backbone == "agent":
            from extractor import AgentExtractor

            policy_kwargs = {
                "features_extractor_class": AgentExtractor,
                "features_extractor_kwargs": {"n_actions": int(env.action_space.n)},
            }
            print("backbone: agent (NatureCNN + text encoder + inverse dynamics)")

        model = PPO(
            "MultiInputPolicy",  # handles Dict obs: CNN for screen, MLP for state
            env,
            n_steps=N_STEPS,
            batch_size=args.batch_size,
            n_epochs=3,
            gamma=0.999,       # ~1k-step effective horizon. The badge is far beyond it,
                               # but the reward is dense by design (tile novelty pays
                               # every few steps), so the agent follows a local gradient
                               # rather than needing to see the badge from the start.
            ent_coef=0.01,     # keep exploring
            learning_rate=2.5e-4,
            tensorboard_log="runs",
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=args.checkpoint_dir,
            name_prefix="emerald",
        ),
        ProgressCallback(),
        RewardBreakdownCallback(),
        CoverageCallback(every=args.coverage_every),
    ]
    # Harmless on other backbones: the callback checks for idm_logits and
    # returns, and the wrapper pays nothing until attached. Registered
    # unconditionally so a resume does not need the flag.
    callbacks.append(InverseDynamicsCallback())
    if hasattr(model.policy.features_extractor, "forward_error"):
        curiosity.attach(model.policy.features_extractor, model.device)

    model.learn(total_timesteps=args.steps, callback=callbacks,
                reset_num_timesteps=args.resume is None)
    model.save(str(Path(args.checkpoint_dir) / "emerald_final"))


if __name__ == "__main__":
    main()
