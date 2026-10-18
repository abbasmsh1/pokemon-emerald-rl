"""Stage 0: the teacher alone, against a random policy and the newest nature4
checkpoint, 20,000 steps each from the same anchor. Prints one table and a
GATE line: the teacher must beat random on reward per 1k, distinct tiles,
and encounters + attacks.

Usage: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python tools/measure_teacher.py [steps]
"""
import sys
import tempfile
import time
from pathlib import Path

from env import EmeraldEnv   # creates no core yet; EmeraldEnv() below does, before torch

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000


def play(env, policy, steps):
    obs, info = env.reset()
    env.drain_reward_terms()
    exp0 = env.state_reader.read()["party_exp"]
    wins = whiteouts = 0
    tiles, maps = set(), set()
    prev_wo = False
    total = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        s = env.state_reader.read()
        tiles.add((s["map"], s["pos"])); maps.add(s["map"])
        if s["party_exp"] > exp0:
            wins += 1; exp0 = s["party_exp"]
        if s["whiteout"] and not prev_wo:
            whiteouts += 1
        prev_wo = s["whiteout"]
        if term or trunc:
            obs, info = env.reset()
    terms = env.drain_reward_terms()
    return {"reward_per_1k": total / steps * 1000, "terms": {k: v / steps * 1000 for k, v in terms.items()},
            "encounters": int(round(terms.get("encounter", 0.0) / EmeraldEnv.ENCOUNTER_REWARD)),
            "attacks": int(round(terms.get("attack", 0.0) / EmeraldEnv.ATTACK_REWARD)),
            "wins": wins, "whiteouts": whiteouts, "tiles": len(tiles), "maps": len(maps),
            "per_s": steps / (time.perf_counter() - t0)}


def main() -> None:
    import numpy as np

    env = EmeraldEnv()                            # the core, before torch
    from teacher import run_teacher
    from stable_baselines3 import PPO

    rows = {}
    rng = np.random.default_rng(0)
    rows["random"] = play(EmeraldEnv(), lambda obs: int(rng.integers(8)), STEPS)

    ckpts = sorted(Path("checkpoints/nature4").glob("*_steps.zip"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        model = PPO.load(ckpts[-1], device="cpu")
        rows[f"ppo {ckpts[-1].stem}"] = play(EmeraldEnv(), lambda obs: int(model.predict(obs, deterministic=False)[0]), STEPS)

    with tempfile.TemporaryDirectory() as d:
        out = run_teacher(STEPS, demo_dir=Path(d), log_every=2000, env=env)
    t = out["terms"]
    rows["teacher"] = {"reward_per_1k": out["reward"] / STEPS * 1000, "terms": {k: v / STEPS * 1000 for k, v in t.items()},
                       "encounters": int(round(t.get("encounter", 0.0) / EmeraldEnv.ENCOUNTER_REWARD)),
                       "attacks": int(round(t.get("attack", 0.0) / EmeraldEnv.ATTACK_REWARD)),
                       "wins": None, "whiteouts": int(round(-t.get("whiteout", 0.0) / -EmeraldEnv.WHITEOUT_PENALTY)) if t.get("whiteout") else 0,
                       "tiles": out["distinct_tiles"], "maps": out["maps"], "per_s": out["decisions_per_s"]}

    print(f"\n{'agent':<28} {'reward/1k':>10} {'encounters':>10} {'attacks':>8} {'wins':>6} {'whiteouts':>9} {'tiles':>6} {'maps':>5} {'steps/s':>8}")
    for name, r in rows.items():
        print(f"{name:<28} {r['reward_per_1k']:>10.2f} {r['encounters']:>10} {r['attacks']:>8} {str(r['wins']):>6} {r['whiteouts']:>9} {r['tiles']:>6} {r['maps']:>5} {r['per_s']:>8.1f}")
    print("\nper-1k terms:")
    for name, r in rows.items():
        print(f"  {name}: " + "  ".join(f"{k} {v:+.2f}" for k, v in sorted(r["terms"].items(), key=lambda kv: -abs(kv[1])) if v))
    tr, rd = rows["teacher"], rows["random"]
    ok = (tr["reward_per_1k"] > rd["reward_per_1k"] and tr["tiles"] > rd["tiles"]
          and tr["encounters"] + tr["attacks"] > rd["encounters"] + rd["attacks"])
    print("\nGATE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
