"""Render a trained checkpoint playing, with a HUD showing what the text
encoder reads, what the inverse-dynamics head thinks was pressed, and a
running count of battles, flees, wins and whiteouts. Usage:
    python watch.py checkpoints/nature3/emerald_final.zip
"""

import argparse

# ponytail: importing torch before any mgba core exists makes libmgba's
# run_frame() spin forever. Creating one core first inoculates the process.
# Only the ORDER matters: verified that the inoculation survives this core
# being dereferenced and garbage collected, so the name is for clarity only.
# This block must stay above the torch/SB3 imports below.
# Ceiling: costs one extra 16MB ROM copy at startup.
from pygba import PyGBA as _PyGBA

_WARMUP = _PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
_WARMUP.core.run_frame()

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from env import EmeraldEnv
from text import player_fled

SCALE = 3
HUD_LINES = 4
LINE_H = 18


def _battle_started(text: str) -> bool:
    t = text.lower()
    return "appeared" in t or "like to battle" in t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=16384)
    args = parser.parse_args()

    env = EmeraldEnv(render_mode="rgb_array")
    model = PPO.load(args.checkpoint, device="cpu")
    fe = model.policy.features_extractor
    has_idm = hasattr(fe, "idm_logits")

    pygame.init()
    w, h = env.width * SCALE, env.height * SCALE
    screen = pygame.display.set_mode((w, h + HUD_LINES * LINE_H))
    pygame.display.set_caption("Emerald agent")
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    env.drain_reward_terms()
    s = env.state_reader.read()
    counts = {"battles": 0, "fled": 0, "flee_booked": 0, "wins": 0, "whiteouts": 0}
    prev_battle = _battle_started(s["screen_text"])
    prev_fled = player_fled(s["screen_text"])
    prev_exp, prev_wo = s["party_exp"], s["whiteout"]
    total = 0.0
    terms_total: dict[str, float] = {}
    idm_line = "idm: n/a"

    for i in range(1, args.steps + 1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        prev_screen = obs["screen"]
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        for k, v in env.drain_reward_terms().items():
            if v:
                terms_total[k] = terms_total.get(k, 0.0) + v
                if k == "flee":
                    counts["flee_booked"] += 1

        s = env.state_reader.read()
        text = s["screen_text"]
        battle = _battle_started(text)
        fled = player_fled(text)
        counts["battles"] += int(battle and not prev_battle)
        counts["fled"] += int(fled and not prev_fled)
        counts["wins"] += int(s["party_exp"] > prev_exp)
        counts["whiteouts"] += int(s["whiteout"] and not prev_wo)
        prev_battle, prev_fled, prev_exp, prev_wo = battle, fled, s["party_exp"], s["whiteout"]

        if has_idm:
            with torch.no_grad():
                s0 = torch.as_tensor(prev_screen[None], dtype=torch.float32) / 255.0
                s1 = torch.as_tensor(obs["screen"][None], dtype=torch.float32) / 255.0
                probs = torch.softmax(fe.idm_logits(s0, s1), dim=1)[0]
            guess = int(probs.argmax())
            idm_line = (f"idm thinks: {env.ACTIONS[guess] or 'none'} {probs[guess]:.2f}"
                        f"{'  ok' if guess == action else ''}")

        frame = env.render()
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (w, h))
        screen.fill((0, 0, 0))
        screen.blit(surf, (0, 0))
        top = sorted(terms_total.items(), key=lambda kv: -abs(kv[1]))[:4]
        lines = [
            f"text: {text[:62]}",
            f"battles {counts['battles']}  fled {counts['fled']} (penalty x{counts['flee_booked']})"
            f"  wins {counts['wins']}  whiteouts {counts['whiteouts']}",
            f"pressed: {env.ACTIONS[action] or 'none'}   {idm_line}   step {i}  reward {total:.1f}",
            "terms: " + "  ".join(f"{k} {v:+.1f}" for k, v in top),
        ]
        for n, line in enumerate(lines):
            screen.blit(font.render(line, True, (230, 230, 230)), (6, h + 2 + n * LINE_H))
        pygame.display.flip()
        clock.tick(30)

        if i % 1000 == 0:
            print(f"{i:>7} {counts}  reward {total:.1f}  text {text[:40]!r}")

        if terminated or truncated:
            print(f"episode end: badges {info['badges']}, tiles {info['tiles_visited']}, map {info['map']}")
            obs, info = env.reset()

    print("final:", counts, "reward", round(total, 1))
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
