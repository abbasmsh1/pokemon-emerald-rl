"""Checks for the demonstration shard store. No emulator, no framework."""

import tempfile
from pathlib import Path

import numpy as np

from demos import FIELDS, load_recent, prune_ring, sample_pairs, write_shard


def _fake(n: int, seed: int = 0, episode_starts=(0,)) -> dict:
    rng = np.random.default_rng(seed)
    a = {
        "screen": rng.integers(0, 256, (n, 3, 80, 120), dtype=np.uint8),
        "state": rng.random((n, 29), dtype=np.float32),
        "text": rng.integers(0, 1024, (n, 16), dtype=np.int64),
        "tiles": rng.integers(0, 2, (n, 9, 11, 15), dtype=np.uint8),
        "battle": rng.random((n, 12), dtype=np.float32),
        "action": rng.integers(0, 8, (n,), dtype=np.int64),
        "reward": rng.random((n,), dtype=np.float32),
        "episode_start": np.zeros((n,), dtype=bool),
    }
    for i in episode_starts:
        a["episode_start"][i] = True
    assert set(a) == set(FIELDS)
    return a


def test_shard_round_trip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        arrays = _fake(64)
        p = write_shard(tmp, arrays)
        assert p.suffix == ".npz" and p.exists() and not list(tmp.glob("*.tmp"))
        back = load_recent(tmp)
        for k in FIELDS:
            assert np.array_equal(back[k], arrays[k]), k
        assert back["shard_id"].shape == (64,)
    print("test_shard_round_trip PASSED")


def test_partial_writes_are_ignored():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_shard(tmp, _fake(8))
        (tmp / "demo_1_8.npz.tmp").write_bytes(b"not a shard")
        back = load_recent(tmp)
        assert len(back["action"]) == 8 and len(set(back["shard_id"].tolist())) == 1
    print("test_partial_writes_are_ignored PASSED")


def test_ring_keeps_the_newest():
    import time
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        names = []
        for i in range(6):
            names.append(write_shard(tmp, _fake(4, seed=i)).name)
            time.sleep(0.002)                       # distinct millisecond names
        deleted = prune_ring(tmp, keep=4)
        left = sorted(p.name for p in tmp.glob("*.npz"))
        assert deleted == 2 and len(left) == 4
        assert names[0] not in left and names[1] not in left, "the oldest two should be gone"
    print("test_ring_keeps_the_newest PASSED")


def test_sample_pairs_never_cross_episode_or_shard_boundaries():
    import time
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_shard(tmp, _fake(40, seed=1, episode_starts=(0, 10)))
        time.sleep(0.002)
        write_shard(tmp, _fake(40, seed=2, episode_starts=(0, 10)))
        data = load_recent(tmp)
        rng = np.random.default_rng(0)
        obs_t, actions, obs_t1 = sample_pairs(data, 500, rng)
        n = len(actions)
        assert 1 <= n <= 76, f"{n} pairs from 80 transitions with 4 forbidden boundaries"
        assert set(obs_t) == set(obs_t1) == {"screen", "state", "text", "tiles", "battle"}
        # Recover the sampled indices from the screens and check every boundary rule.
        for i in range(n):
            t = next(j for j in range(80) if np.array_equal(data["screen"][j], obs_t["screen"][i]))
            assert data["shard_id"][t] == data["shard_id"][t + 1], "pair crossed a shard"
            assert not data["episode_start"][t + 1], "pair crossed an episode start"
            assert np.array_equal(data["screen"][t + 1], obs_t1["screen"][i])
            assert actions[i] == data["action"][t]
    print("test_sample_pairs_never_cross_episode_or_shard_boundaries PASSED")


def test_load_recent_on_empty_dir_is_none():
    with tempfile.TemporaryDirectory() as d:
        assert load_recent(Path(d)) is None
        assert load_recent(Path(d) / "missing") is None
    print("test_load_recent_on_empty_dir_is_none PASSED")


if __name__ == "__main__":
    test_shard_round_trip()
    test_partial_writes_are_ignored()
    test_ring_keeps_the_newest()
    test_sample_pairs_never_cross_episode_or_shard_boundaries()
    test_load_recent_on_empty_dir_is_none()
    print("\nall demos tests passed")
