"""Assertion-based checks for the advice cache. No framework."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from advice_cache import NO_ADVICE, AdviceCache


def test_miss_returns_no_advice():
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        assert c.lookup("never seen this") == NO_ADVICE
        print("test_miss_returns_no_advice PASSED")


def test_put_then_lookup_returns_action():
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        c.put("Do you want to use the PC?", 6)
        assert c.lookup("Do you want to use the PC?") == 6
        print("test_put_then_lookup_returns_action PASSED")


def test_lookup_is_normalised():
    """Whitespace and accent differences must not create a second entry."""
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        c.put("A wild POKeMON appeared!", 5)
        assert c.lookup("A  wild   POKeMON appeared!") == 5
        assert c.lookup("A wild POKéMON appeared!") == 5
        print("test_lookup_is_normalised PASSED")


def test_enqueue_then_pending_then_put_clears():
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        c.enqueue("Go check it out, dear!")
        assert "Go check it out, dear!" in c.pending()
        c.put("Go check it out, dear!", 1)
        assert "Go check it out, dear!" not in c.pending()
        print("test_enqueue_then_pending_then_put_clears PASSED")


def _cache_worker(args):
    """Module level so multiprocessing can pickle it.

    Shortens the busy timeout so a rollback-journal lock raises instead of
    being waited out. Without this the test passes even with WAL disabled.
    """
    import advice_cache

    advice_cache.BUSY_TIMEOUT = 0.05

    path, index = args
    if index == 0:
        c = advice_cache.AdviceCache(path)
        for i in range(200):
            c.put(f"line {i}", i % 8)
        return True
    c = advice_cache.AdviceCache(path, readonly=True)
    return all(c.lookup("shared line") == 5 for _ in range(200))


def test_concurrent_processes_with_a_writer():
    """The real case: 8 worker processes reading while the advisor writes.

    The default rollback journal raises 'database is locked' here; WAL is
    what makes it survive. Threads in one process do not exercise this.
    """
    import multiprocessing as mp

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.db")
        AdviceCache(path).put("shared line", 5)

        ctx = mp.get_context("fork")
        with ctx.Pool(9) as pool:
            results = pool.map(_cache_worker, [(path, i) for i in range(9)])

        assert all(results), "a process saw a wrong value or hit a lock"
    print("test_concurrent_processes_with_a_writer PASSED")


def test_broken_cache_recovers_after_backoff():
    """A transient failure must not disable advice for the rest of the run."""
    import time as _time

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.db")
        c = AdviceCache(path)
        c.put("recoverable line", 4)
        assert c.lookup("recoverable line") == 4

        c._mark_broken()
        assert c.lookup("recoverable line") == NO_ADVICE, "backoff not in effect"

        # Expire the backoff rather than sleeping for it
        c._broken_until = _time.monotonic() - 1
        assert c.lookup("recoverable line") == 4, "cache never reconnected"
    print("test_broken_cache_recovers_after_backoff PASSED")


def test_corrupt_db_degrades_to_no_advice():
    """Review Focus 4. A damaged cache must never raise into the step loop."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "corrupt.db"
        path.write_bytes(b"this is not a database")
        c = AdviceCache(str(path))
        assert c.lookup("anything") == NO_ADVICE
        c.enqueue("anything")  # must not raise either
        print("test_corrupt_db_degrades_to_no_advice PASSED")


def test_unwritable_path_degrades_to_no_advice():
    """A read-only filesystem or missing directory must not kill training."""
    c = AdviceCache("/proc/nonexistent/advice.db")
    assert c.lookup("anything") == NO_ADVICE
    c.enqueue("anything")
    c.put("anything", 3)
    print("test_unwritable_path_degrades_to_no_advice PASSED")


def test_non_str_input_never_raises():
    """Review found put(123, 1) raising TypeError out of a public method."""
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        for bad in (123, b"x", ["a"], None, 3.5):
            assert c.lookup(bad) == NO_ADVICE
            c.enqueue(bad)
            c.put(bad, 1)
        assert c.lookup("still works") == NO_ADVICE
        c.put("still works", 2)
        assert c.lookup("still works") == 2, "bad input broke a healthy cache"
    print("test_non_str_input_never_raises PASSED")


def test_put_deletes_by_normalised_key():
    """The old fixture was normalise-identity, so it could not catch put
    deleting by the raw text instead of the key."""
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        raw = "Hello   there,  TRAINER!"
        c.enqueue(raw)
        assert c.pending(), "enqueue stored nothing"
        c.put(raw, 5)
        assert not c.pending(), "put did not clear the queue row"
    print("test_put_deletes_by_normalised_key PASSED")


def test_enqueue_is_memoised_per_process():
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        for _ in range(50):
            c.enqueue("same line over and over")
        assert len(c.pending(1000)) == 1
        assert len(c._enqueued) == 1
    print("test_enqueue_is_memoised_per_process PASSED")


def test_queue_is_bounded():
    import advice_cache

    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        original = advice_cache.MAX_QUEUE_ROWS
        advice_cache.MAX_QUEUE_ROWS = 10
        try:
            for i in range(40):
                c.enqueue(f"line number {i}")
            assert len(c.pending(1000)) <= 10, "queue grew past its cap"
        finally:
            advice_cache.MAX_QUEUE_ROWS = original
    print("test_queue_is_bounded PASSED")


def test_overlong_text_is_not_queued():
    with tempfile.TemporaryDirectory() as d:
        c = AdviceCache(str(Path(d) / "a.db"))
        c.enqueue("x" * 1000)
        assert not c.pending(), "a 1000-character 'message' was queued"
    print("test_overlong_text_is_not_queued PASSED")


def test_already_answered_text_is_not_queued():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.db")
        AdviceCache(path).put("known line", 6)
        c = AdviceCache(path)  # fresh process-equivalent, empty memo
        c.enqueue("known line")
        assert not c.pending(), "queued text the advisor had already answered"
    print("test_already_answered_text_is_not_queued PASSED")


if __name__ == "__main__":
    test_miss_returns_no_advice()
    test_put_then_lookup_returns_action()
    test_lookup_is_normalised()
    test_enqueue_then_pending_then_put_clears()
    test_concurrent_processes_with_a_writer()
    test_broken_cache_recovers_after_backoff()
    test_corrupt_db_degrades_to_no_advice()
    test_unwritable_path_degrades_to_no_advice()
    test_non_str_input_never_raises()
    test_put_deletes_by_normalised_key()
    test_enqueue_is_memoised_per_process()
    test_queue_is_bounded()
    test_overlong_text_is_not_queued()
    test_already_answered_text_is_not_queued()
    print("\nall advice cache tests passed")
