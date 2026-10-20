"""Play Emerald yourself and record it for the agent to learn from.

Same environment, same observation and same step as training (one button per
24 frames), so every recorded step is an exact (observation, action) pair the
policy can be fitted to; train.py's ImitationCallback reads them from
demos_human/. Usage:

    python play.py                 # starts where training starts (Route 103)
    python play.py --state X.state # or from any savestate

Keys: arrows walk (hold to keep walking), Z = A, X = B, Enter = Start (one
press each), nothing held = wait. Close the window or press Esc to stop;
every step is saved, including the last few. The game runs at its real speed,
about 2.5 agent steps a second.
"""

# A core must run a frame before torch is imported (see README); play.py
# imports no torch, but env does through nothing, so this is belt and braces.
from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from demos import write_shard  # noqa: E402
from env import EmeraldEnv  # noqa: E402

DEMO_DIR = Path("demos_human")
SHARD_STEPS = 512
SCALE = 3

# Buttons beat directions so pressing A while walking still talks; directions
# resolve in this fixed order.
KEY_PRIORITY = (("z", "A"), ("x", "B"), ("return", "start"),
                ("up", "up"), ("down", "down"), ("left", "left"), ("right", "right"))


def keys_to_action(pressed: set[str]) -> int:
    for key, action in KEY_PRIORITY:
        if key in pressed:
            return EmeraldEnv.ACTIONS.index(action)
    return EmeraldEnv.ACTIONS.index(None)


class Recorder:
    """Buffers (observation, action, reward, episode start) and writes a
    shard every `shard_steps`, and the remainder on close."""

    def __init__(self, demo_dir: Path = DEMO_DIR, shard_steps: int = SHARD_STEPS):
        self.dir, self.shard_steps = Path(demo_dir), shard_steps
        self.buf: dict[str, list] = {}
        self.steps = 0
        self.shards = 0

    def add(self, obs: dict, action: int, reward: float, episode_start: bool) -> None:
        for k, v in obs.items():
            self.buf.setdefault(k, []).append(np.asarray(v))
        self.buf.setdefault("action", []).append(action)
        self.buf.setdefault("reward", []).append(np.float32(reward))
        self.buf.setdefault("episode_start", []).append(bool(episode_start))
        self.steps += 1
        if len(self.buf["action"]) >= self.shard_steps:
            self.flush()

    def flush(self) -> None:
        if self.buf.get("action"):
            write_shard(self.dir, {k: np.asarray(v) for k, v in self.buf.items()})
            self.shards += 1
        self.buf = {}

    def close(self) -> None:
        self.flush()


def main() -> None:
    import pygame

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default=None, help="savestate to start from (default: where training starts)")
    parser.add_argument("--demo-dir", default=str(DEMO_DIR))
    parser.add_argument("--speed", type=float, default=1.25,
                        help="game speed relative to real time (1.25 = 25%% faster); "
                             "only the pace of play changes, each step is still one agent step")
    args = parser.parse_args()

    env = EmeraldEnv(render_mode="rgb_array", state_path=args.state, max_steps=0)
    obs, _ = env.reset()
    rec = Recorder(Path(args.demo_dir))
    names = {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
             pygame.K_z: "z", pygame.K_x: "x", pygame.K_RETURN: "return"}

    pygame.init()
    w, h = env.width * SCALE, env.height * SCALE
    screen = pygame.display.set_mode((w, h + 24))
    pygame.display.set_caption("Emerald: you are playing (recording)")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    start, total = True, 0.0
    # A, B and Start count once per key press, queued so a quick tap between
    # steps is not lost; arrows count every step they are held, i.e. walking.
    # Each step is paced to real game speed (frameskip frames at 60 fps), so
    # holding an arrow walks at the game's own pace rather than racing.
    press_keys = {pygame.K_z: "z", pygame.K_x: "x", pygame.K_RETURN: "return"}
    arrow_keys = {k: n for k, n in names.items() if k not in press_keys}
    pending: list[str] = []
    steps_per_second = 60.0 / env.frameskip * args.speed
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in press_keys:
                    pending.append(press_keys[event.key])
            if not running:
                break
            if pending:
                action = keys_to_action({pending.pop(0)})
            else:
                held = pygame.key.get_pressed()
                action = keys_to_action({n for k, n in arrow_keys.items() if held[k]})
            nxt, reward, terminated, truncated, info = env.step(action)
            rec.add(obs, action, reward, start)
            obs, start = nxt, False
            total += reward
            if terminated or truncated:
                obs, _ = env.reset()
                start = True

            frame = env.render()
            surf = pygame.transform.scale(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (w, h))
            screen.fill((0, 0, 0))
            screen.blit(surf, (0, 0))
            line = (f"recorded {rec.steps} steps   pressed {env.ACTIONS[action] or 'none'}   "
                    f"reward {total:.1f}   hold arrows to walk / Z=A / X=B / Enter=Start / Esc=stop")
            screen.blit(font.render(line, True, (230, 230, 230)), (6, h + 4))
            pygame.display.flip()
            clock.tick(steps_per_second)
    finally:
        rec.close()
        pygame.quit()
        print(f"saved {rec.steps} steps in {rec.shards} shard(s) to {args.demo_dir}/")


if __name__ == "__main__":
    main()
