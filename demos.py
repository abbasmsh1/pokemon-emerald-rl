"""Demonstration shards: the teacher's record, as the trainer reads it.

One format for both sides. The teacher appends a shard every SHARD_STEPS
transitions and prunes the ring; the trainer concatenates the newest few
shards between rollouts and samples consecutive pairs. Writes go to a
temporary name and are renamed into place, so a reader never sees a partial
file, and readers only ever look at *.npz.
"""

import os
import time
from pathlib import Path

import numpy as np

SHARD_STEPS = 2048
KEEP_SHARDS = 50
FIELDS = ("screen", "state", "text", "tiles", "battle", "action", "reward", "episode_start")
OBS_FIELDS = ("screen", "state", "text", "tiles", "battle")
# Per-transition bookkeeping; every other key in a shard is an observation
# key. Shards keep whatever observation they are given, so new observation
# parts (moves, guide, heal, map_info, evo) travel without a format change.
META = ("action", "reward", "episode_start")


def write_shard(dir: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Atomically save one shard; returns its final path."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    n = len(arrays["action"])
    for k in META:
        assert k in arrays, f"shard field {k} missing"
    for k, v in arrays.items():
        assert len(v) == n, f"shard field {k} ragged"
    final = dir / f"demo_{int(time.time() * 1000)}_{n}.npz"
    tmp = final.with_name(final.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, final)
    return final


def _shards(dir: Path) -> list[Path]:
    return sorted(Path(dir).glob("demo_*.npz"))      # names sort by timestamp


def prune_ring(dir: Path, keep: int = KEEP_SHARDS) -> int:
    old = _shards(dir)[:-keep] if keep > 0 else _shards(dir)
    for p in old:
        p.unlink()
    return len(old)


def load_recent(dir: Path, max_shards: int = 10) -> dict[str, np.ndarray] | None:
    """The newest max_shards shards concatenated in time order, plus a
    shard_id per transition so pairs never cross a shard; None if empty."""
    paths = _shards(dir)[-max_shards:] if Path(dir).is_dir() else []
    if not paths:
        return None
    parts = []
    for p in paths:
        with np.load(p) as z:
            parts.append({k: z[k] for k in z.files})
    keys = set.intersection(*(set(d) for d in parts))
    out = {k: np.concatenate([d[k] for d in parts]) for k in sorted(keys)}
    out["shard_id"] = np.concatenate([np.full(len(d["action"]), i, dtype=np.int64) for i, d in enumerate(parts)])
    return out


def _obs_keys(data: dict) -> list[str]:
    return [k for k in data if k not in META and k != "shard_id"]


def sample_pairs(data: dict, n: int, rng: np.random.Generator):
    """n consecutive (t, t+1) pairs that stay inside one shard and one
    episode. Returns (obs_t, action_t, obs_t1), each obs a dict of OBS_FIELDS."""
    total = len(data["action"])
    t = np.arange(total - 1)
    ok = (data["shard_id"][t] == data["shard_id"][t + 1]) & ~data["episode_start"][t + 1]
    valid = t[ok]
    if len(valid) == 0:
        empty = {k: data[k][:0] for k in _obs_keys(data)}
        return empty, data["action"][:0], empty
    pick = rng.choice(valid, size=min(n, len(valid)), replace=False)
    obs_t = {k: data[k][pick] for k in _obs_keys(data)}
    obs_t1 = {k: data[k][pick + 1] for k in _obs_keys(data)}
    return obs_t, data["action"][pick], obs_t1
