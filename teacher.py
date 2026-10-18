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
    lines.append("tiles: " + summarise_tiles(obs["tiles"]))
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
