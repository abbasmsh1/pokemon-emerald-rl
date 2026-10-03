"""Assertion-based checks for the advice cache. No framework."""

import os
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


def test_concurrent_readers_do_not_lock():
    """Review Focus 3. Eight workers share one file; the default journal mode
    raises 'database is locked' under this load."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.db")
        writer = AdviceCache(path)
        writer.put("shared line", 5)

        def hammer(_):
            c = AdviceCache(path, readonly=True)
            return [c.lookup("shared line") for _ in range(50)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(hammer, range(8)))

        assert all(v == 5 for batch in results for v in batch), (
            "concurrent readers saw wrong or missing values"
        )
        print("test_concurrent_readers_do_not_lock PASSED")


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


if __name__ == "__main__":
    test_miss_returns_no_advice()
    test_put_then_lookup_returns_action()
    test_lookup_is_normalised()
    test_enqueue_then_pending_then_put_clears()
    test_concurrent_readers_do_not_lock()
    test_corrupt_db_degrades_to_no_advice()
    test_unwritable_path_degrades_to_no_advice()
    print("\nall advice cache tests passed")
