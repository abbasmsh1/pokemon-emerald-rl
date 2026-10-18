"""A small language model plays Pokemon Emerald from the exact game state.

The teacher is a demonstration source: it reads what the environment exposes
(screen text, the tile classes around the player, the battle state, its own
situation), scores the eight buttons in one forward pass, presses one, and
records every step to the demonstration ring in demos.py. The policy and the
vision model train on that record (train.py's ImitationCallback) while PPO
continues on top, so the student can leave the teacher's mistakes behind.

The prompt carries navigation the student never sees: the map's name, the
position, the direction of the nearest never-visited section and a one-line
sequence of towns to Rustboro. The student observes only the game state.
"""

import numpy as np

from battle import move_name
from tilemap import CLASSES, TILE_H, TILE_W

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ACTION_WORDS = ("none", "up", "down", "left", "right", "A", "B", "start")   # index-aligned with EmeraldEnv.ACTIONS
# 256, not the spec's "~200": the dialogue fixture measures 224 with the
# tightest wording, and the binding constraint is decisions per second, which
# 256 tokens leaves at ~25 on the GPU.
MAX_PROMPT_TOKENS = 256
SYSTEM = "You play Pokemon Emerald."      # replaces Qwen's 20-token default system line
MAX_TEXT_CHARS = 64

MAP_NAMES = {
    (0, 9): "Littleroot Town", (0, 16): "Route 101", (0, 10): "Oldale Town", (0, 17): "Route 102",
    (0, 18): "Route 103", (0, 0): "Petalburg City", (0, 19): "Route 104", (0, 3): "Rustboro City",
}

# Both lines are sent every step, so they are kept terse: the prompt budget
# is MAX_PROMPT_TOKENS and the dialogue fixture sat at 245 with longer wording.
GOAL = ("Goal: win the Rustboro gym badge. Route: Littleroot, Route 101 N, Oldale, Route 102 W, "
        "Petalburg, Route 104 N, Petalburg Woods, Rustboro.")
LEGEND = "Buttons: up down left right = walk or move cursor; A = talk, confirm, attack; B = cancel, run; start = menu; none = wait."
PROMPT_CLASS = {"warp": "door"}


def map_name(map_key) -> str:
    return MAP_NAMES.get(tuple(map_key), f"map {map_key[0]}-{map_key[1]}")


def _direction(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "here"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


PASSABLE = ("walkable", "grass", "warp")
ADJACENT = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}    # (dy, dx)
DIRECTION_NAME = {"up": "north", "down": "south", "left": "west", "right": "east"}


def possible_moves(tiles: np.ndarray) -> list[str]:
    """Directions whose adjacent tile the map says the player can step onto.
    A ledge is passable only along its jump direction, which the window does
    not encode, so it counts as blocked here."""
    cy, cx = TILE_H // 2, TILE_W // 2
    cls = np.asarray(tiles).argmax(axis=0)
    return [d for d, (dy, dx) in ADJACENT.items() if CLASSES[int(cls[cy + dy, cx + dx])] in PASSABLE]


def summarise_tiles(tiles: np.ndarray) -> str:
    """Adjacent tiles, the nearest grass/door/NPC/water, and off-map edges."""
    cy, cx = TILE_H // 2, TILE_W // 2
    cls = np.asarray(tiles).argmax(axis=0)
    def name(y, x): return CLASSES[int(cls[y, x])]
    parts = [f"north: {name(cy - 1, cx)}, south: {name(cy + 1, cx)}, "
             f"east: {name(cy, cx + 1)}, west: {name(cy, cx - 1)}"]
    for wanted in ("grass", "warp", "npc", "water"):
        ys, xs = np.where(cls == CLASSES.index(wanted))
        if len(ys) == 0:
            continue
        d = np.abs(ys - cy) + np.abs(xs - cx)
        i = int(d.argmin())
        label = PROMPT_CLASS.get(wanted, wanted)
        if d[i] == 0:
            parts.append(f"nearest {label}: here")
        else:
            parts.append(f"nearest {label}: {int(d[i])} tiles {_direction(int(xs[i] - cx), int(ys[i] - cy))}")
    ys, xs = np.where(cls == CLASSES.index("outside"))
    if len(ys):
        edges = sorted({_direction(int(x - cx), int(y - cy)) for y, x in zip(ys, xs)} - {"here"})
        parts.append("map edge: " + ", ".join(edges))
    return "; ".join(parts)


def _clean_text(text: str) -> str:
    return " ".join(text.replace('"', "'").split())[:MAX_TEXT_CHARS]


def build_prompt(s: dict, obs: dict, recent: list, moved: bool, frontier: str = "") -> str:
    """The user turn. Deterministic; under MAX_PROMPT_TOKENS on the fixtures."""
    lines = [GOAL]
    loc = f"location: {map_name(s['map'])} at ({s['pos'][0]}, {s['pos'][1]})"
    if frontier:
        loc += f"; unexplored: {frontier}"
    lines.append(loc)
    text = _clean_text(s.get("screen_text", ""))
    lines.append(f'text: "{text}"' if text else "text: none")
    if not s.get("in_battle"):
        # The ground around the player matters only when walking; in a battle
        # the block below replaces it, which also keeps the prompt in budget.
        lines.append("tiles: " + summarise_tiles(obs["tiles"]))
        moves_ok = possible_moves(obs["tiles"])
        lines.append("can move: " + (", ".join(DIRECTION_NAME[d] for d in moves_ok) if moves_ok else "nowhere"))
    if s.get("in_battle"):
        b = s["battle"]
        lines.append(f"battle: enemy HP {int(round(b[1] * 100))}%, our HP {int(round(b[2] * 100))}%")
        moves = [f"{move_name(mid)} ({'D' if b[8 + k] else 'S'}) PP {int(round(b[3 + k] * 100))}%"
                 for k, mid in enumerate(s.get("our_moves", (0, 0, 0, 0))) if mid]
        if moves:
            lines.append("moves: " + ", ".join(moves))
    levels = s.get("party_levels") or [0]
    last = ", ".join(recent[-3:]) if recent else "none"
    lines.append(f"situation: party level {max(levels)}, HP {int(round(s.get('party_hp_frac', 0.0) * 100))}%, "
                 f"money {s.get('money', 0)}, last actions: {last}, last press {'moved' if moved else 'blocked'}")
    lines.append(LEGEND)
    lines.append("Answer with one button only.")
    return "\n".join(lines)


def format_chat(tok, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ---------------------------------------------------------------------------
# Decision: one forward pass, the eight buttons scored at the next token.
# torch and transformers are imported lazily: any entry point must create an
# mGBA core before they load, and keeping them out of the module import makes
# that the caller's one job rather than this file's.


def load_model(device: str | None = None):
    """Qwen2.5-0.5B-Instruct from the local cache, fp16 on cuda if available.
    Raises RuntimeError with a clear message on any failure, so the service
    exits non-zero and systemd restarts it with backoff."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, local_files_only=True).to(device).eval()
        return model, tok
    except Exception as e:  # noqa: BLE001 - every failure here means "no teacher"
        raise RuntimeError(f"teacher model unavailable: {type(e).__name__}: {e}") from e


# The assistant turn is prefilled with this, so the next token must be the
# button. Without it a 0.5B model wants to open with "The" or "Button" and the
# restricted scores collapse to the same word on every screen (measured: "down"
# on all three fixtures). After a colon the button arrives as a space-prefixed
# token, so those are the ids scored.
PREFILL = "Button:"


def action_token_ids(tok) -> list[int]:
    ids = [tok.encode(" " + word, add_special_tokens=False)[0] for word in ACTION_WORDS]
    assert len(set(ids)) == len(ACTION_WORDS), f"action words share a first token: {ids}"
    return ids


def overworld_mask(s: dict, obs: dict) -> np.ndarray | None:
    """False for Start always, and for directions the tile window says cannot
    move the player when walking is what a direction does (no text on
    screen, not in a battle); in a menu or a battle directions move a cursor
    and stay free."""
    mask = np.ones(len(ACTION_WORDS), dtype=bool)
    # Start is never what the badge needs (a Center heals by talking, the
    # battle BAG is reached with A) and the reward taxes every press; the
    # round-1 teacher pressed it 30-80 times per 1,000 steps and the student
    # would have copied the habit. Masked everywhere.
    mask[ACTION_WORDS.index("start")] = False
    if s.get("in_battle") or s.get("screen_text"):
        return mask
    ok = set(possible_moves(obs["tiles"]))
    for d in ADJACENT:
        if d not in ok:
            mask[ACTION_WORDS.index(d)] = False
    return mask


def choose_action(model, tok, prompt: str, temperature: float = 0.0, rng=None, mask=None):
    """(action index, scores[8]) from the next-token logits at the action
    words. `mask`, when given, removes actions from the choice (their scores
    are still reported) so a blocked direction is never pressed in the
    overworld; the first measured teacher pressed one on every step."""
    import torch

    ids = getattr(tok, "_action_ids", None)
    if ids is None:
        ids = tok._action_ids = action_token_ids(tok)
    enc = tok(format_chat(tok, prompt) + PREFILL, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    scores = logits[ids].float().cpu().numpy()
    choosable = scores.copy()
    if mask is not None and mask.any():
        choosable[~mask] = -np.inf
    if temperature <= 0:
        return int(choosable.argmax()), scores
    z = choosable / temperature
    probs = np.exp(z - z.max()); probs /= probs.sum()
    rng = rng or np.random.default_rng()
    return int(rng.choice(len(scores), p=probs)), scores


# ---------------------------------------------------------------------------
# The play loop.

def frontier_direction(section: tuple[int, int], visited: set, cap: int = 8) -> str:
    """Compass direction of the nearest section not in `visited`, "" if none
    within `cap` rings. Same ring scan as frontier.nearest_unvisited_distance,
    but it returns where rather than how far."""
    sx, sy = section
    for d in range(1, cap + 1):
        for dx in range(-d, d + 1):
            dy = d - abs(dx)
            for yy in ({sy + dy, sy - dy}):
                if (sx + dx, yy) not in visited:
                    return _direction(dx, yy - sy)
    return ""


def run_teacher(steps: int, demo_dir="demos", shard_steps: int | None = None, keep: int | None = None,
                temperature: float = 0.7, log_every: int = 1000, env=None, model_tok=None) -> dict:
    """Play `steps` steps, write a shard every `shard_steps`, prune the ring,
    log a reward ledger every `log_every` steps; returns the counts."""
    import time
    from pathlib import Path

    from demos import FIELDS, KEEP_SHARDS, SHARD_STEPS, prune_ring, write_shard
    from env import EmeraldEnv

    shard_steps = shard_steps or SHARD_STEPS
    keep = keep or KEEP_SHARDS
    demo_dir = Path(demo_dir)
    env = env or EmeraldEnv()                 # the core exists before torch loads
    model, tok = model_tok or load_model()
    rng = np.random.default_rng()

    obs, info = env.reset()
    env.drain_reward_terms()
    s = env.state_reader.read()
    recent: list[str] = []
    moved = False
    episode_start = True
    buf = {k: [] for k in FIELDS}
    shards = 0
    total = 0.0
    tiles: set = set()
    maps: set = set()
    terms_total: dict[str, float] = {}
    t0 = time.perf_counter()

    for i in range(1, steps + 1):
        g, n = s["map"]
        section = (s["pos"][0] // 8, s["pos"][1] // 8)
        visited = {(sx, sy) for (gg, nn, sx, sy) in env._sections_ever if (gg, nn) == (g, n)}
        frontier = frontier_direction(section, visited)
        prompt = build_prompt(s, obs, recent, moved, frontier)
        action, _ = choose_action(model, tok, prompt, temperature, rng, overworld_mask(s, obs))

        for k in ("screen", "state", "text", "tiles", "battle"):
            buf[k].append(np.asarray(obs[k]))
        obs, reward, terminated, truncated, info = env.step(action)
        buf["action"].append(action)
        buf["reward"].append(np.float32(reward))
        buf["episode_start"].append(episode_start)
        episode_start = False

        total += reward
        s = env.state_reader.read()
        moved = env._moved
        recent = (recent + [ACTION_WORDS[action]])[-3:]
        tiles.add((s["map"], s["pos"]))
        maps.add(s["map"])

        if len(buf["action"]) >= shard_steps:
            write_shard(demo_dir, {k: np.asarray(v) for k, v in buf.items()})
            prune_ring(demo_dir, keep)
            shards += 1
            buf = {k: [] for k in FIELDS}
        if i % log_every == 0:
            terms = env.drain_reward_terms()
            for k, v in terms.items():
                terms_total[k] = terms_total.get(k, 0.0) + v
            rate = i / (time.perf_counter() - t0)
            shown = "  ".join(f"{k} {v / log_every * 1000:+.2f}" for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])) if v)
            print(f"teacher {i:>8} steps  {rate:5.1f}/s  per 1k: {shown}", flush=True)
        if terminated or truncated:
            obs, info = env.reset()
            s = env.state_reader.read()
            episode_start = True
            recent, moved = [], False

    for k, v in env.drain_reward_terms().items():
        terms_total[k] = terms_total.get(k, 0.0) + v
    elapsed = time.perf_counter() - t0
    return {"steps": steps, "shards": shards, "reward": total, "distinct_tiles": len(tiles),
            "maps": len(maps), "decisions_per_s": steps / elapsed, "terms": terms_total}


def _warm_up_libmgba() -> None:
    """Run one frame on a throwaway core before torch loads.

    Creating a core is not enough: a core created after torch is imported
    spins forever in run_frame() unless some core in this process has ALREADY
    RUN A FRAME before the import. EmeraldEnv() alone does not run one
    (reset() does), and the first version of this entry point hung on exactly
    that, as did tools/measure_teacher.py. test_teacher.py runs the real
    entry point as a subprocess to keep it that way.
    """
    from pygba import PyGBA
    warm = PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba")
    warm.core.run_frame()


def main() -> None:
    import argparse
    import sys

    _warm_up_libmgba()
    from env import EmeraldEnv

    parser = argparse.ArgumentParser(description="Let the SLM play and record demonstrations.")
    parser.add_argument("--steps", type=int, default=10**9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--demo-dir", default="demos")
    args = parser.parse_args()

    env = EmeraldEnv()
    try:
        model_tok = load_model()
    except RuntimeError as e:
        print(e, file=sys.stderr, flush=True)
        sys.exit(2)
    print(f"teacher: {MODEL_NAME} on {next(model_tok[0].parameters()).device}, temperature {args.temperature}", flush=True)
    out = run_teacher(args.steps, demo_dir=args.demo_dir, temperature=args.temperature, env=env, model_tok=model_tok)
    print("teacher done:", {k: v for k, v in out.items() if k != "terms"}, flush=True)


if __name__ == "__main__":
    main()
