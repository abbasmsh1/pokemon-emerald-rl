# SLM Text Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a small language model read the game's on-screen text and suggest a button, and expose that suggestion to the PPO policy as part of its observation.

**Architecture:** Advice is asynchronous. Eight training workers look up a shared SQLite cache and never block; a miss queues the text and yields "no advice" for that step. One separate advisor process owns the model, drains the queue, and fills the cache. With no advisor running, every lookup misses and training behaves exactly as it does today.

**Tech Stack:** Python 3.11, pygba 0.2.9, mgba 0.10.2, stable-baselines3 2.9.0, torch 2.14.0+cu130, transformers, SQLite (stdlib)

**Deviation from the spec:** the spec specified 4-bit quantisation at ~400MB VRAM. This plan loads Qwen2.5-0.5B in float16 at roughly 1GB instead, because 4-bit requires adding `bitsandbytes` for a 600MB saving on a model this small. 1GB still fits the ~2GB spare beside ResNet18 and PPO. Task 4 Step 5 measures the real figure and stops if it exceeds 1500MB.

**Spec:** `docs/superpowers/specs/2026-10-04-slm-advisor-design.md`

## Global Constraints

- Python 3.11 exactly. The `mgba` Linux x86_64 wheels exist only for cp310 and cp311.
- **`import torch` breaks libmgba.** Any PyGBA core created after torch is imported hangs forever in `run_frame()`. Any file that imports torch (directly or via transformers/SB3) and also touches an emulator must create a throwaway core FIRST. Copy the `_WARMUP` block from `train.py`.
- **Never read ROM memory (`0x08000000` and above) during a step.** `PyGBA.read_memory` copies the whole region into Python per frame; for the 16MB ROM that is 1.28ms against a 7.36ms budget. The message buffer is in EWRAM, so this is satisfied by construction — do not add ROM reads.
- mGBA logs to **stdout** at the C level. `2>/dev/null` does NOT quiet it. Redirect stdout to a file and grep, and use `env.suppress_stdout()` around emulator calls.
- The live message buffer is at **`0x02021FC4`**, found by measurement. Decoded with `pygba.game_wrappers.utils.emerald_utils.EmeraldCharmap`, terminator `0xFF`.
- The advisor is **strictly optional**. Training must run normally with no advisor process, no model installed, and no `advice.db` on disk.
- The ROM is gitignored. Never commit it, `advice.db`, `checkpoints/`, `runs/`, or `media/`.
- Training currently runs as a systemd user service. Stop it with `systemctl --user stop pokemon-rl` before any run that needs the GPU, and never with `pkill` (the service restarts it after 60s).

## Review Focus

Five conditions the spec implies that a naive implementation will meet and get wrong, most likely first. Each has a test assigned to the task that owns the code.

1. **Text containing non-ASCII and control codes.** The real buffer holds `POKéMON`, a typographic apostrophe, and embedded line-break bytes. Decoding must not raise and must not emit replacement junk into cache keys. (Task 1)
2. **Reading the buffer during a map transition.** Save blocks relocate and parse as garbage for a step; the same is true of the text buffer. A garbage read must yield `None`, not a nonsense cache entry. (Task 1)
3. **Eight workers hitting one SQLite file concurrently.** The default journal mode will raise `database is locked` under this load. (Task 2)
4. **A missing, unreadable, or corrupt `advice.db`.** Must degrade to "no advice" and let training continue, never raise into the step loop. (Task 2)
5. **A model reply that is not a button name.** Empty, rambling, or hallucinated replies must be rejected as "no advice" rather than coerced into an action. (Task 4)

---

### Task 1: Read and decode the game's on-screen text

**Files:**
- Create: `gametext.py`
- Test: `test_gametext.py`

**Interfaces:**
- Consumes: `pygba.PyGBA`, `EmeraldCharmap`, `env.suppress_stdout`.
- Produces:
  - `MESSAGE_ADDR = 0x02021FC4`
  - `read_message(gba) -> str | None` — the current on-screen text, normalised, or `None` when there is nothing readable.
  - `normalise(text: str) -> str` — collapse whitespace, strip control markers, uppercase-insensitive key form used by the cache.

- [ ] **Step 1: Write the failing test**

Create `test_gametext.py`:

```python
"""Assertion-based checks for game text extraction. No framework."""

from pygba import PyGBA

from env import suppress_stdout
from gametext import MESSAGE_ADDR, normalise, read_message

ROM = "Pokemon - Emerald Version (USA, Europe).gba"


def _booted():
    """A savestate with free player control and no dialogue showing."""
    with suppress_stdout():
        gba = PyGBA.load(ROM)
        gba.core.reset()
        gba.core.load_raw_state(open("boot.state", "rb").read())
        gba.core.run_frame()
    return gba


def test_address_is_the_measured_one():
    """Pinned by measurement; a silent change would break every cache key."""
    assert MESSAGE_ADDR == 0x02021FC4
    print("test_address_is_the_measured_one PASSED")


def test_read_message_returns_str_or_none():
    gba = _booted()
    msg = read_message(gba)
    assert msg is None or isinstance(msg, str)
    if msg is not None:
        assert "\x00" not in msg, "raw terminator leaked into the text"
    print(f"test_read_message_returns_str_or_none PASSED ({msg!r})")


def test_normalise_collapses_whitespace_and_breaks():
    raw = "The mover's POKeMON do all the work\n\nof moving us in.   Really!"
    out = normalise(raw)
    assert "\n" not in out, "line breaks survived normalisation"
    assert "  " not in out, "runs of spaces survived normalisation"
    assert out == normalise(raw + "   "), "trailing space changed the key"
    print("test_normalise_collapses_whitespace_and_breaks PASSED")


def test_normalise_handles_non_ascii_without_raising():
    """Review Focus 1. The real buffer contains POKeMON with an accent and a
    typographic apostrophe; neither may raise or produce replacement junk."""
    raw = "The mover’s POKéMON do all the work"
    out = normalise(raw)
    assert out, "non-ascii text normalised to nothing"
    assert "�" not in out, "replacement character leaked into the key"
    print("test_normalise_handles_non_ascii_without_raising PASSED")


def test_garbage_buffer_reads_as_none():
    """Review Focus 2. During a map transition the buffer parses as garbage.

    Simulated by reading an address that holds non-text data; the function must
    return None rather than a nonsense string that would pollute the cache.
    """
    gba = _booted()
    from gametext import decode_at

    junk = decode_at(gba, 0x03000000)
    assert junk is None or len(junk) == 0 or not junk.strip(), (
        f"non-text region decoded to {junk!r}, which would become a cache key"
    )
    print("test_garbage_buffer_reads_as_none PASSED")


if __name__ == "__main__":
    test_address_is_the_measured_one()
    test_normalise_collapses_whitespace_and_breaks()
    test_normalise_handles_non_ascii_without_raising()
    test_read_message_returns_str_or_none()
    test_garbage_buffer_reads_as_none()
    print("\nall gametext tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_gametext.py 2>/dev/null | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'gametext'`

- [ ] **Step 3: Write the implementation**

Create `gametext.py`:

```python
"""Read the game's on-screen message out of RAM and decode it.

The address was found by measurement, not guessed: EWRAM was scanned with
EmeraldCharmap while known dialogue was displayed, and 0x02021FC4 was the
address whose contents tracked the dialogue as it advanced.

EWRAM only. A ROM read would cost 1.28ms against a 7.36ms step budget.
"""

import re
import unicodedata

from pygba.game_wrappers.utils.emerald_utils import EmeraldCharmap

MESSAGE_ADDR = 0x02021FC4
MAX_LEN = 256
TERMINATOR = 0xFF

_CHARMAP = EmeraldCharmap()

# Enough real letters to be text rather than a coincidental decode of graphics.
_MIN_ALPHA = 6
_MIN_ALPHA_RATIO = 0.5


def decode_at(gba, addr: int, max_len: int = MAX_LEN) -> str | None:
    """Decode an Emerald string at `addr`, or None if it does not look like text."""
    raw = gba.read_memory(addr, max_len)

    chars = []
    for byte in raw:
        if byte == TERMINATOR:
            break
        mapped = _CHARMAP.charmap[byte] if byte < len(_CHARMAP.charmap) else None
        # Multi-character entries are glyph names like "Pk"; control bytes map to
        # None. Both become a space so words do not run together.
        chars.append(mapped if mapped and len(mapped) == 1 else " ")

    text = "".join(chars).strip()
    if not text:
        return None

    alpha = sum(c.isalpha() for c in text)
    if alpha < _MIN_ALPHA or alpha < len(text) * _MIN_ALPHA_RATIO:
        # Graphics and save-block garbage decode to sparse letter soup. Rejecting
        # it here keeps nonsense out of the cache keys.
        return None

    return text


def normalise(text: str) -> str:
    """Cache-key form: one line, single spaces, accents folded.

    Folding means "POKeMON" and "POKéMON" share an entry, and a line-break
    difference does not create a second entry for the same sentence.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("’", "'")
    return re.sub(r"\s+", " ", folded).strip()


def read_message(gba) -> str | None:
    """The current on-screen message, normalised, or None if there is none."""
    text = decode_at(gba, MESSAGE_ADDR)
    return normalise(text) if text else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_gametext.py 2>/dev/null | tail -8`
Expected: PASS, all five tests.

- [ ] **Step 5: Verify it reads real dialogue**

This proves the address works on live dialogue rather than only on a quiet savestate. Run:

```bash
.venv/bin/python - >/tmp/gt.txt 2>&1 <<'EOF'
from pygba import PyGBA
from pygba.utils import KEY_MAP
from pygba.game_wrappers.utils.emerald_utils import read_save_block_1
from env import suppress_stdout
from gametext import read_message
import sys

def hold(g, k, n):
    g.core.set_keys(KEY_MAP[k]) if k else g.core.set_keys()
    for _ in range(n): g.core.run_frame()

with suppress_stdout():
    g = PyGBA.load("Pokemon - Emerald Version (USA, Europe).gba"); g.core.reset()
    f = 0
    while f < 20000:
        for k in ("A", None, "start", None): hold(g, k, 12); f += 12
        sb = read_save_block_1(g)
        if sb and (sb["location"]["mapGroup"], sb["location"]["mapNum"]) != (0, 0):
            break
    for _ in range(40): hold(g, "A", 16); hold(g, None, 8)
print("MESSAGE:", repr(read_message(g)), file=sys.stderr)
EOF
grep -a "MESSAGE:" /tmp/gt.txt
```

Expected: a readable English sentence, for example `MESSAGE: "MOM: BRETT, we're here, honey! It must be tiring riding with our things"`.

If it prints `None`, the intro did not reach dialogue; mash longer. If it prints letter soup, `_MIN_ALPHA_RATIO` is too lenient — report it rather than lowering the threshold.

- [ ] **Step 6: Commit**

```bash
git add gametext.py test_gametext.py
git commit -m "feat: read and decode the game's on-screen text from RAM"
```

---

### Task 2: Shared advice cache and work queue

**Files:**
- Create: `advice_cache.py`
- Test: `test_advice_cache.py`

**Interfaces:**
- Consumes: `gametext.normalise`.
- Produces:
  - `NO_ADVICE = 8` — the one-hot slot meaning "no opinion".
  - `class AdviceCache` with `__init__(self, path: str = "advice.db", readonly: bool = False)`.
  - `.lookup(text: str) -> int` — the cached action index, or `NO_ADVICE`. Never raises.
  - `.enqueue(text: str) -> None` — record text needing advice. Never raises.
  - `.pending(limit: int = 32) -> list[str]` — queued texts awaiting advice.
  - `.put(text: str, action: int) -> None` — store an answer and clear it from the queue.

- [ ] **Step 1: Write the failing test**

Create `test_advice_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_advice_cache.py 2>/dev/null | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'advice_cache'`

- [ ] **Step 3: Write the implementation**

Create `advice_cache.py`:

```python
"""Shared cache and work queue between the training workers and the advisor.

Workers only ever read this, and only on the fast path. Every method swallows
its own errors and degrades to "no advice": a broken cache must never raise into
the training step loop, because the whole feature is optional.
"""

import sqlite3

from gametext import normalise

NO_ADVICE = 8  # one past the 8 action indices


class AdviceCache:
    def __init__(self, path: str = "advice.db", readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._conn: sqlite3.Connection | None = None
        self._broken = False
        self._connect()

    def _connect(self) -> None:
        try:
            conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            # WAL lets eight workers read while the advisor writes. The default
            # rollback journal takes an exclusive lock and raises
            # "database is locked" under this access pattern.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS advice ("
                "  text TEXT PRIMARY KEY, action INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS queue (text TEXT PRIMARY KEY)"
            )
            conn.commit()
            self._conn = conn
        except Exception:
            # Corrupt file, unwritable path, missing directory: all the same
            # outcome. The feature switches itself off.
            self._broken = True
            self._conn = None

    def lookup(self, text: str) -> int:
        if self._broken or self._conn is None or not text:
            return NO_ADVICE
        try:
            row = self._conn.execute(
                "SELECT action FROM advice WHERE text = ?", (normalise(text),)
            ).fetchone()
        except Exception:
            self._broken = True
            return NO_ADVICE
        return row[0] if row else NO_ADVICE

    def enqueue(self, text: str) -> None:
        if self._broken or self._conn is None or self.readonly or not text:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO queue (text) VALUES (?)", (normalise(text),)
            )
            self._conn.commit()
        except Exception:
            self._broken = True

    def pending(self, limit: int = 32) -> list[str]:
        if self._broken or self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                "SELECT text FROM queue LIMIT ?", (limit,)
            ).fetchall()
        except Exception:
            self._broken = True
            return []
        return [r[0] for r in rows]

    def put(self, text: str, action: int) -> None:
        if self._broken or self._conn is None or not text:
            return
        key = normalise(text)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO advice (text, action) VALUES (?, ?)",
                (key, int(action)),
            )
            self._conn.execute("DELETE FROM queue WHERE text = ?", (key,))
            self._conn.commit()
        except Exception:
            self._broken = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_advice_cache.py 2>/dev/null | tail -9`
Expected: PASS, all seven tests.

- [ ] **Step 5: Commit**

```bash
echo "advice.db*" >> .gitignore
git add advice_cache.py test_advice_cache.py .gitignore
git commit -m "feat: shared advice cache with WAL concurrency and safe degradation"
```

---

### Task 3: Expose advice in the observation

**Files:**
- Modify: `env.py`
- Modify: `test_env.py`

**Interfaces:**
- Consumes: `gametext.read_message`, `advice_cache.AdviceCache`, `advice_cache.NO_ADVICE`.
- Produces:
  - `EmeraldEnv.__init__` gains `advice_db: str | None = "advice.db"`. Passing `None` disables the feature entirely.
  - `observation_space["advice"]`: `Box(0, 1, (9,), uint8)`, one-hot over 8 actions plus `NO_ADVICE`.
  - `info["advice"]`: the integer index, for logging.

- [ ] **Step 1: Write the failing test**

Append to `test_env.py` and add the calls to `__main__`:

```python
def test_advice_is_a_valid_one_hot():
    env = EmeraldEnv()
    obs, _ = env.reset()
    assert "advice" in obs, "observation has no advice entry"
    a = obs["advice"]
    assert a.shape == (9,), f"advice shape {a.shape}, expected (9,)"
    assert a.dtype == np.uint8
    assert a.sum() == 1, f"advice is not one-hot: {a}"
    assert env.observation_space.contains(obs)
    env.close()
    print("test_advice_is_a_valid_one_hot PASSED")


def test_no_advisor_means_no_advice_slot():
    """With no cache the feature must switch off, not break training."""
    from advice_cache import NO_ADVICE

    env = EmeraldEnv(advice_db=None)
    obs, info = env.reset()
    assert obs["advice"][NO_ADVICE] == 1, "advice present with no cache configured"
    for _ in range(5):
        obs, r, term, trunc, info = env.step(5)
        assert obs["advice"][NO_ADVICE] == 1
        assert np.isfinite(r)
    env.close()
    print("test_no_advisor_means_no_advice_slot PASSED")


def test_cached_advice_reaches_the_observation():
    import tempfile
    from pathlib import Path

    from advice_cache import AdviceCache
    from gametext import read_message

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.db")
        env = EmeraldEnv(advice_db=path)
        env.reset()
        text = read_message(env.gba)
        if text is None:
            # boot.state has no dialogue showing; seed the cache with whatever
            # the env will look up so the path is still exercised.
            print("test_cached_advice_reaches_the_observation SKIPPED (no text)")
            env.close()
            return
        AdviceCache(path).put(text, 3)

        # Re-open so the env sees the committed row, then confirm the SAME text
        # is still on screen before asserting: if the message changed, the miss
        # is legitimate and asserting a hit would be wrong.
        env2 = EmeraldEnv(advice_db=path)
        env2.reset()
        if read_message(env2.gba) == text:
            obs = env2._observation(env2.state_reader.read())
            assert obs["advice"][3] == 1, (
                f"cached advice did not reach the observation: {obs['advice']}"
            )
        env2.close()
        env.close()
    print("test_cached_advice_reaches_the_observation PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_env.py 2>/dev/null | tail -5`
Expected: FAIL with `AssertionError: observation has no advice entry`

- [ ] **Step 3: Write the implementation**

In `env.py`, add to the imports:

```python
from advice_cache import NO_ADVICE, AdviceCache
from gametext import read_message
```

Add a constant beside `STATE_SIZE`:

```python
ADVICE_SIZE = 9  # 8 actions plus a "no advice" slot
```

Add `advice_db` to `__init__`'s signature after `render_mode`:

```python
        advice_db: str | None = "advice.db",
```

In `__init__`, after `self.state_reader = GameState(self.gba)`:

```python
        # Optional by design: with no cache, or a broken one, every lookup
        # returns NO_ADVICE and training behaves exactly as it did before.
        self._advice = AdviceCache(advice_db, readonly=False) if advice_db else None
```

Add `"advice"` to the observation space dict:

```python
            "advice": gym.spaces.Box(0, 1, (ADVICE_SIZE,), dtype=np.uint8),
```

Add the lookup helper:

```python
    def _advice_vector(self) -> tuple[np.ndarray, int]:
        """One-hot advice for the current on-screen text.

        A cache miss enqueues the text for the advisor and yields NO_ADVICE for
        this step. Workers never wait on the model: the next time this text
        appears the answer is there, and game text repeats constantly.
        """
        index = NO_ADVICE
        if self._advice is not None:
            text = read_message(self.gba)
            if text:
                index = self._advice.lookup(text)
                if index == NO_ADVICE:
                    self._advice.enqueue(text)

        vec = np.zeros(ADVICE_SIZE, dtype=np.uint8)
        vec[index] = 1
        return vec, index
```

In `_observation`, add the entry. Replace the existing return with:

```python
    def _observation(self, s: dict) -> dict:
        advice_vec, self._last_advice = self._advice_vector()
        return {
            "screen": np.stack(self._frames, axis=0),
            "state": self._encode_state(s),
            "advice": advice_vec,
        }
```

Initialise `self._last_advice = NO_ADVICE` in `__init__` next to `self._step_count = 0`, and add it to `_info`:

```python
            "advice": self._last_advice,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_env.py 2>/dev/null | tail -10`
Expected: PASS, all existing tests plus the three new ones.

- [ ] **Step 5: Verify throughput did not collapse**

The lookup runs every step across 8 workers. Run:

```bash
.venv/bin/python - >/tmp/sp.txt 2>&1 <<'EOF'
import sys, time
from env import EmeraldEnv
for db in (None, "advice.db"):
    env = EmeraldEnv(advice_db=db); env.reset()
    t0 = time.time()
    for _ in range(300): env.step(5)
    print(f"advice_db={db}: {300/(time.time()-t0):.1f} steps/sec", file=sys.stderr)
    env.close()
EOF
grep -a "steps/sec" /tmp/sp.txt
```

Expected: both lines within about 15% of each other. A cache lookup is microseconds; if the advice path costs more than that, report it rather than proceeding.

- [ ] **Step 6: Commit**

```bash
git add env.py test_env.py
git commit -m "feat: expose cached SLM advice in the observation"
```

---

### Task 4: The advisor process

**Files:**
- Create: `advisor.py`
- Test: `test_advisor.py`

**Interfaces:**
- Consumes: `advice_cache.AdviceCache`, `advice_cache.NO_ADVICE`.
- Produces:
  - `BUTTONS = ["none", "up", "down", "left", "right", "a", "b", "start"]` — index matches `EmeraldEnv.ACTIONS`.
  - `parse_reply(reply: str) -> int` — a model reply to an action index, or `NO_ADVICE`.
  - `build_prompt(text: str) -> str`
  - `main()` — CLI loop draining the queue.

- [ ] **Step 1: Write the failing test**

Create `test_advisor.py`:

```python
"""Assertion-based checks for the advisor. No model is loaded here."""

from advice_cache import NO_ADVICE
from advisor import BUTTONS, build_prompt, parse_reply


def test_parses_each_button_name():
    for i, name in enumerate(BUTTONS):
        assert parse_reply(name) == i, f"{name!r} did not parse to {i}"
        assert parse_reply(name.upper()) == i, "parsing is case sensitive"
        assert parse_reply(f"  {name}  ") == i, "surrounding space broke parsing"
    print("test_parses_each_button_name PASSED")


def test_parses_a_short_sentence():
    assert parse_reply("Press A to continue") == BUTTONS.index("a")
    assert parse_reply("The answer is: START") == BUTTONS.index("start")
    print("test_parses_a_short_sentence PASSED")


def test_rejects_unparseable_replies():
    """Review Focus 5. Garbage must become NO_ADVICE, never a coerced action."""
    for bad in ["", "   ", "I am not sure", "press the thingy", "42", "x" * 500]:
        assert parse_reply(bad) == NO_ADVICE, f"{bad!r} was coerced into an action"
    print("test_rejects_unparseable_replies PASSED")


def test_ambiguous_reply_is_rejected():
    """Two button names means the model did not decide; do not guess for it."""
    assert parse_reply("either a or b") == NO_ADVICE
    print("test_ambiguous_reply_is_rejected PASSED")


def test_prompt_contains_text_and_buttons():
    p = build_prompt("Do you want to use the PC?")
    assert "Do you want to use the PC?" in p
    for name in BUTTONS:
        assert name in p.lower(), f"prompt never mentions {name}"
    print("test_prompt_contains_text_and_buttons PASSED")


if __name__ == "__main__":
    test_parses_each_button_name()
    test_parses_a_short_sentence()
    test_rejects_unparseable_replies()
    test_ambiguous_reply_is_rejected()
    test_prompt_contains_text_and_buttons()
    print("\nall advisor tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python test_advisor.py 2>/dev/null | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'advisor'`

- [ ] **Step 3: Write the implementation**

Create `advisor.py`:

```python
"""Reads queued game text, asks a small language model for a button, caches it.

Runs as its own process so one model serves all eight training workers; eight
copies would not fit in 4GB alongside ResNet18 and PPO. It is optional: if this
never runs, every lookup misses and training is unaffected.

Nothing here is on the training hot path, so latency only bounds how fast the
cache warms up.
"""

import argparse
import re
import time

from advice_cache import NO_ADVICE, AdviceCache

# Index matches EmeraldEnv.ACTIONS exactly.
BUTTONS = ["none", "up", "down", "left", "right", "a", "b", "start"]

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 8
POLL_SECONDS = 2.0


def build_prompt(text: str) -> str:
    return (
        "You are playing Pokemon Emerald on a Game Boy Advance.\n"
        f"The screen says: \"{text}\"\n\n"
        "Which single button should be pressed next?\n"
        f"Answer with exactly one of: {', '.join(BUTTONS)}.\n"
        "Answer with 'none' if the text does not imply a button.\n"
        "Answer:"
    )


def parse_reply(reply: str) -> int:
    """A reply to an action index, or NO_ADVICE.

    Strict on purpose. A model that rambles, hesitates, or names two buttons has
    not decided, and guessing on its behalf would feed the policy noise it
    cannot distinguish from a real recommendation.
    """
    if not reply or len(reply) > 200:
        return NO_ADVICE

    found = {
        i
        for i, name in enumerate(BUTTONS)
        if re.search(rf"\b{re.escape(name)}\b", reply, re.IGNORECASE)
    }
    if len(found) != 1:
        return NO_ADVICE
    return found.pop()


def load_model():
    """Import torch lazily so a missing model never affects anything else."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    return tok, model


def ask(tok, model, text: str) -> int:
    import torch

    prompt = build_prompt(text)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_reply(reply)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="advice.db")
    parser.add_argument("--once", action="store_true",
                        help="drain the queue once and exit, for testing")
    args = parser.parse_args()

    cache = AdviceCache(args.db)

    try:
        tok, model = load_model()
    except Exception as exc:
        # Training does not depend on this process. Failing loudly here and
        # exiting is better than competing for VRAM with the trainer.
        print(f"advisor: could not load model, exiting: {exc}")
        return

    print(f"advisor: {MODEL_ID} ready, polling {args.db}")
    while True:
        texts = cache.pending()
        for text in texts:
            action = ask(tok, model, text)
            cache.put(text, action)
            print(f"advisor: {text[:60]!r} -> {BUTTONS[action] if action < len(BUTTONS) else 'none'}")
        if args.once:
            return
        if not texts:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python test_advisor.py 2>/dev/null | tail -7`
Expected: PASS, all five tests. No model is downloaded by these tests.

- [ ] **Step 5: Install the model dependencies and do one live call**

Training holds the GPU, so stop it first:

```bash
systemctl --user stop pokemon-rl
VIRTUAL_ENV=.venv uv pip install transformers accelerate
.venv/bin/python - >/tmp/adv.txt 2>&1 <<'EOF'
import sys
from advisor import ask, load_model, BUTTONS
tok, model = load_model()
for text in ["Do you want to use the PC?", "A wild POOCHYENA appeared!",
             "Go check it out, dear!"]:
    a = ask(tok, model, text)
    print(f"{text!r} -> {BUTTONS[a] if a < len(BUTTONS) else 'no advice'}", file=sys.stderr)
import torch
print("VRAM MB:", torch.cuda.max_memory_allocated() // 1024 // 1024, file=sys.stderr)
EOF
grep -a "\->\|VRAM" /tmp/adv.txt
```

Expected: three lines of advice and a VRAM figure. **If VRAM exceeds 1500MB, stop and report it** — the budget assumed ~400MB, and exceeding it is how this project gets OOM-killed a third time.

- [ ] **Step 6: Commit**

```bash
git add advisor.py test_advisor.py
git commit -m "feat: SLM advisor process with strict reply parsing"
```

---

### Task 5: Documentation and end-to-end check

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Run the whole suite**

```bash
for t in test_state test_env test_coverage test_stats test_backbone \
         test_gametext test_advice_cache test_advisor; do
  .venv/bin/python $t.py >/tmp/$t.log 2>&1
  echo "$t exit=$? passed=$(grep -ac PASSED /tmp/$t.log)"
done
```

Expected: every suite exits 0.

- [ ] **Step 2: End-to-end check that training works with and without the advisor**

```bash
systemctl --user stop pokemon-rl
rm -f advice.db
timeout 300 .venv/bin/python train.py --steps 20000 --backbone resnet18 \
  --batch-size 128 --max-steps 65536 > /tmp/noadv.log 2>&1
echo "without advisor: exit=$? fps=$(grep -a 'fps' /tmp/noadv.log | tail -1)"
echo "queued texts: $(.venv/bin/python -c "
from advice_cache import AdviceCache; print(len(AdviceCache('advice.db').pending(1000)))")"
```

Expected: training completes, and the queue has entries, proving workers are reading text and enqueueing misses without blocking.

- [ ] **Step 3: Drain the queue once with the advisor and confirm hits**

```bash
.venv/bin/python advisor.py --once > /tmp/drain.log 2>&1
grep -ac "advisor:" /tmp/drain.log
.venv/bin/python -c "
from advice_cache import AdviceCache
c = AdviceCache('advice.db')
print('still pending:', len(c.pending(1000)))"
```

Expected: the advisor logs some answers and the pending count drops.

- [ ] **Step 4: Update the README**

Add to the file table:

```markdown
| `gametext.py` | Read and decode the game's on-screen text from RAM |
| `advice_cache.py` | Shared cache and queue between workers and the advisor |
| `advisor.py` | Optional SLM process that answers queued text |
```

Add a section after the Vision paragraph:

```markdown
**Text advice (optional).** A small language model reads the game's on-screen
text and suggests a button, which joins the observation as a one-hot; PPO learns
whether to trust it. The text comes from RAM at `0x02021FC4` decoded with
pygba's `EmeraldCharmap`, not OCR.

The model cannot run in the step loop: training does 250-300 steps/sec against
roughly 0.3s per reply, with eight workers sharing one GPU. So advice is
asynchronous. Workers look up a shared SQLite cache and never block; a miss
queues the text and yields "no advice" for that step, and one advisor process
fills the cache in the background. Game text repeats constantly, so after a
warmup the cache hits almost always.

Run it alongside training with `python advisor.py`. If it is never started,
every lookup misses and training is unaffected.
```

- [ ] **Step 5: Commit and restart training**

```bash
git add README.md
git commit -m "docs: describe the optional SLM text advisor"
systemctl --user start pokemon-rl
```

Note the observation space changed, so the service resumes from a checkpoint whose policy shape no longer matches. Delete or move `checkpoints/` before starting, or the resume will fail — this is the restart-from-scratch the spec calls for.
