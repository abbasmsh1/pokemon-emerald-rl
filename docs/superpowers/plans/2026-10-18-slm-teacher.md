# SLM Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small language model plays its own environment from the exact game state, its play is recorded to a ring of demonstration shards, and the PPO policy and the vision model train on that record continuously alongside PPO.

**Architecture:** `demos.py` owns the shard format and ring (shared by writer and reader). `teacher.py` owns the prompt, the one-pass action scoring, and the play loop; it runs as its own service. `train.py` gains an `ImitationCallback` that samples shards after each PPO update and applies an imitation loss plus the existing inverse-dynamics and forward losses. No observation or network-shape change, so the `nature4` run resumes with the callback simply switched on.

**Tech Stack:** Python 3.11, transformers (installed today) with Qwen/Qwen2.5-0.5B-Instruct from the HF cache, torch, numpy, stable-baselines3 2.9, gymnasium 1.3, pygba. Assertion-based test scripts, no framework.

**Spec:** `docs/superpowers/specs/2026-10-18-slm-teacher-design.md`

## Global Constraints

- Every new entry point and test file creates a throwaway `PyGBA` core before any torch/transformers/SB3 import (see the top of `test_backbone.py`).
- The teacher loads `Qwen/Qwen2.5-0.5B-Instruct` in fp16 on `cuda` when available, from the local HF cache (no download at runtime); peak VRAM must stay under 1.3 GB (measured 1.01 GB) and it uses one environment, i.e. one CPU core.
- The teacher's decision is a single forward pass scored over the 8 action words; it never calls `generate` in the loop.
- Shards are written to a temporary name and renamed into place; readers ignore anything not ending in `.npz`.
- The demonstration store is `demos/` at the repo root, git-ignored, ring-pruned to `KEEP_SHARDS = 50` shards of `SHARD_STEPS = 2048` transitions.
- `IMITATION_WEIGHT = 0.1`; the callback is a no-op when `demos/` holds no shard.
- No change to the observation space, `AgentExtractor`'s shape, or PPO hyperparameters; `run_training.sh` keeps `CKPT_DIR=checkpoints/nature4`.
- Stage 0 gate before Task 5: over 20,000 steps the teacher beats a random policy on reward per 1,000 steps and on distinct tiles; the numbers go to Abbas in the task's report. If it fails, rework the prompt (Task 2) and re-measure before building Task 5.
- Each guard is revert-checked with `PYTHONDONTWRITEBYTECODE=1` and `rm -rf __pycache__` after any sed-and-restore.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Builder-check answers (2026-10-18): the teacher's environment is `EmeraldEnv()` with defaults (the furthest outdoor savestate at construction, `--max-steps 0` semantics, whiteouts non-terminal, novelty refreshed as the env does); `episode_start` is True only on the first transition after a reset. Stage 0's three agents share that anchor. The teacher's prompt carries navigation the student never sees: map name, position, the direction of the nearest never-visited section (the env's frontier computation) and a one-line town sequence to Rustboro in the goal; the student observes only the game state, so no route is scripted into the policy. Stage 0 bar: teacher beats random on reward per 1k, distinct tiles, AND encounters plus attacks; at most two prompt rework rounds, then stop and report. The kill switch for imitation is stopping the teacher service and emptying `demos/`: the callback is a no-op without shards, no training restart needed; the "PPO reward must not fall" check is a manual read of `reward/*` against the window before the callback started.

## Review Focus

1. The trainer reads `demos/` while the teacher is mid-write: it must never load a partial shard. Test in Task 1 (`.tmp` files ignored, rename is atomic).
2. A shard whose first transition starts an episode, or whose last is an episode end: `sample_pairs` must never pair across an episode start or across shard ends. Test in Task 1.
3. Screen text with characters that could break the chat template or blow the token budget (quotes, braces, a 160-char line): the prompt stays under 220 tokens and the decision still returns 0..7. Test in Task 2 and Task 3.
4. The teacher cannot load the model (no GPU memory, cache missing): it must exit non-zero with a clear message, the service restarts with a 60 s backoff, and training is unaffected. Verified in Task 4 by unit settings and by running the trainer with the teacher down.
5. The callback starts while training is already running and `demos/` is still empty, then shards appear: it must begin contributing without a restart. Test in Task 5 (empty dir: no parameter changes; shard appears: loss logged).

---

### Task 1: Demonstration shards

**Files:**
- Create: `demos.py`
- Create: `test_demos.py`
- Modify: `.gitignore` (add `demos/`)

**Interfaces:**
- Produces in `demos.py`:
  - `SHARD_STEPS = 2048`, `KEEP_SHARDS = 50`, `FIELDS = ("screen", "state", "text", "tiles", "battle", "action", "reward", "episode_start")`
  - `write_shard(dir: Path, arrays: dict[str, np.ndarray]) -> Path`: saves with `np.savez_compressed` to `dir / f"demo_{int(time.time()*1000)}_{len(arrays['action'])}.npz.tmp"` then `os.replace` to the `.npz` name; returns the final path. Asserts all `FIELDS` are present with equal first dimension.
  - `prune_ring(dir: Path, keep: int = KEEP_SHARDS) -> int`: deletes the oldest `.npz` files beyond `keep` (by name, which sorts by timestamp); returns the number deleted.
  - `load_recent(dir: Path, max_shards: int = 10) -> dict[str, np.ndarray] | None`: concatenates the newest `max_shards` shards in time order; `None` if there are none. Adds `"shard_id": np.ndarray[int]` so pairs never cross shards.
  - `sample_pairs(data: dict, n: int, rng: np.random.Generator) -> tuple[dict, np.ndarray, dict]`: picks `n` indices `t` (without replacement, fewer if not enough) such that `t + 1` exists, `shard_id[t] == shard_id[t + 1]` and `episode_start[t + 1] == False`; returns `(obs_t, action_t, obs_t1)` where each obs is a dict of the five observation fields.

- [ ] **Step 1: Write the failing tests in `test_demos.py`**

```python
def test_shard_round_trip(tmp):
    arrays = _fake(64)                       # all FIELDS, 64 transitions
    p = write_shard(tmp, arrays)
    assert p.suffix == ".npz" and not list(tmp.glob("*.tmp"))
    back = load_recent(tmp)
    for k in FIELDS: assert np.array_equal(back[k], arrays[k])

def test_partial_writes_are_ignored():
    # a stray demo_1_8.npz.tmp beside a good shard: load_recent sees one shard
def test_ring_keeps_the_newest():
    # write 6 shards, prune_ring(keep=4) deletes 2, the oldest two names are gone
def test_sample_pairs_never_cross_episode_or_shard_boundaries():
    # two shards; episode_start True at index 0 and at index 10 of each;
    # 500 samples: for every t, shard_id[t]==shard_id[t+1] and not episode_start[t+1]
def test_load_recent_on_empty_dir_is_none():
```
`_fake(n)` builds arrays with the exact shapes: screen (n,3,80,120) uint8, state (n,29) float32, text (n,16) int64, tiles (n,9,11,15) uint8, battle (n,12) float32, action (n,) int64, reward (n,) float32, episode_start (n,) bool. Use `tempfile.TemporaryDirectory`.

- [ ] **Step 2: Run to verify they fail** — `ModuleNotFoundError: demos`.
- [ ] **Step 3: Implement `demos.py` as specified.**
- [ ] **Step 4: Run `test_demos.py` to verify they pass.**
- [ ] **Step 5: Revert-check** — make `write_shard` save straight to the `.npz` name and make `sample_pairs` ignore `episode_start`; the partial-write and boundary tests must fail. Restore.
- [ ] **Step 6: Commit** — `git add demos.py test_demos.py .gitignore && git commit -m "feat: demonstration shards with an atomic write and a ring"`

---

### Task 2: Prompt

**Files:**
- Modify: `battle.py` (add `MOVE_NAMES`)
- Create: `teacher.py` (prompt half)
- Create: `test_teacher.py`

**Interfaces:**
- `battle.py`: `MOVE_NAMES: dict[int, str]` for move ids 1-70 in the standard Gen 3 numbering (1 POUND ... 10 SCRATCH ... 33 TACKLE ... 45 GROWL ... 52 EMBER, 55 WATER GUN ... 70 STRENGTH) plus 98 QUICK ATTACK, 116 FOCUS ENERGY, 150 SPLASH; and `move_name(move_id: int) -> str` returning `MOVE_NAMES.get(id, f"move {id}")`. `BattleReader.our_move_ids()` already exists.
- `teacher.py`:
  - `ACTION_WORDS = ("none", "up", "down", "left", "right", "A", "B", "start")`, index-aligned with `EmeraldEnv.ACTIONS`.
  - `summarise_tiles(tiles: np.ndarray) -> str`: from the (9,11,15) one-hot window: `"north: <class>, south: <class>, east: <class>, west: <class>"` for the four adjacent tiles, then for each of grass, warp ("door"), npc, water that occurs in the window: `"nearest <name>: <n> tiles <direction>"` with Manhattan distance and the dominant direction (or "here" at distance 0), then `"map edge: <directions>"` if any `outside` tile lies in the window. Classes come from `tilemap.CLASSES`.
  - `MAP_NAMES = {(0, 9): "Littleroot Town", (0, 16): "Route 101", (0, 10): "Oldale Town", (0, 17): "Route 102", (0, 18): "Route 103", (0, 0): "Petalburg City", (0, 19): "Route 104", (0, 3): "Rustboro City"}`; `map_name(map_key) -> str` with `"map <g>-<n>"` as the fallback (interiors included).
  - `build_prompt(s: dict, obs: dict, recent: list[str], moved: bool, frontier: str = "") -> str`: the user-turn text, deterministic, containing in this order: goal line (`Goal: reach Rustboro City and win its gym badge. Way: Littleroot, Route 101 north, Oldale, Route 102 west, Petalburg, Route 104 north through Petalburg Woods, Rustboro.`); location line (`location: <map name> at (<x>, <y>)`, plus `unexplored: <direction>` when `frontier` is non-empty; the caller computes it from `nearest_unvisited_distance` over the env's `_sections_ever` for the current map and passes the compass direction of the nearest unvisited section, or "" when none within the cap); screen text line (`text: "<screen_text>"` or `text: none`); tile summary; battle block only when `s["in_battle"]` (`enemy HP 82%  our HP 24%`, then one line per non-empty move: `<name> (D|S) PP <pp%>`); situation line (`party level <n>, HP <pct>%, money <n>, last actions: <a, b, c>, last press <moved|blocked>`); the button legend (8 words, one clause each); and the instruction `Answer with one button only.`.
  - `format_chat(tok, prompt: str) -> str`: `tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)`.
  - `MAX_PROMPT_TOKENS = 220`.

- [ ] **Step 1: Write the failing tests in `test_teacher.py`** (warmup core first; the tokenizer loads from the cache with `local_files_only=True`)

```python
def test_move_names():
    assert move_name(10) == "SCRATCH" and move_name(45) == "GROWL" and move_name(33) == "TACKLE"
    assert move_name(55) == "WATER GUN" and move_name(116) == "FOCUS ENERGY" and move_name(999) == "move 999"

def test_tile_summary_names_adjacent_and_nearest():
    tiles = np.zeros((9, 11, 15), np.uint8); tiles[0] = 1                 # all walkable
    tiles[:, 5, 8] = 0; tiles[CLASSES.index("blocked"), 5, 8] = 1       # east: blocked
    tiles[:, 2, 7] = 0; tiles[CLASSES.index("grass"), 2, 7] = 1         # grass 3 north
    s = summarise_tiles(tiles)
    assert "east: blocked" in s and "nearest grass: 3 tiles north" in s and "door" not in s

def test_prompt_is_deterministic_and_bounded_on_fixtures():
    for state in (None, "dialogue.state", "battle.state"):
        env = EmeraldEnv(state_path=state) ... obs, _ = env.reset(); s = env.state_reader.read()
        p1 = build_prompt(s, obs, ["up", "A", "none"], True); p2 = build_prompt(...)
        assert p1 == p2
        assert len(tok(format_chat(tok, p1))["input_ids"]) <= MAX_PROMPT_TOKENS

def test_battle_prompt_marks_move_natures():
    # battle.state: prompt contains "SCRATCH (D)" and "GROWL (S)" and "enemy HP"

def test_prompt_survives_hostile_text():
    s2 = dict(s, screen_text='{"}' + "A" * 160 + "\n\n")
    p = build_prompt(s2, obs, [], False)
    assert len(tok(format_chat(tok, p))["input_ids"]) <= MAX_PROMPT_TOKENS
```
(Truncate the screen text to 80 characters and strip newlines inside `build_prompt`; that is what keeps the last test green.)

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `MOVE_NAMES`, `move_name`, `summarise_tiles`, `build_prompt`, `format_chat`.**
- [ ] **Step 4: Run `test_teacher.py` and `test_battle.py` to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: teacher prompt from the exact game state"`

---

### Task 3: Decision

**Files:**
- Modify: `teacher.py` (decision half)
- Modify: `test_teacher.py`

**Interfaces:**
- `load_model(device: str | None = None) -> tuple[model, tok]`: Qwen2.5-0.5B-Instruct from the cache (`local_files_only=True`), fp16 on cuda if available else fp32 on cpu, `eval()`; raises `RuntimeError("teacher model unavailable: ...")` wrapping any load failure.
- `action_token_ids(tok) -> list[int]`: first token id of each word in `ACTION_WORDS`, via `tok.encode(word, add_special_tokens=False)[0]`; asserts the 8 ids are distinct.
- `choose_action(model, tok, prompt: str, temperature: float = 0.0, rng=None) -> tuple[int, np.ndarray]`: one forward pass on `format_chat(tok, prompt)`; `scores` = the 8 next-token logits at the action ids as float32; action = argmax if `temperature == 0` else a sample from `softmax(scores / temperature)`. Returns `(action_index, scores)`.

- [ ] **Step 1: Write the failing tests** (skip the model tests with a printed SKIPPED if `load_model` raises because no GPU and no cache, but the cache exists on this machine so they run)

```python
def test_action_token_ids_are_distinct_and_eight():
def test_choose_action_is_valid_on_every_fixture():
    # anchor, dialogue.state, battle.state, and the hostile text prompt:
    # 0 <= action < 8, scores.shape == (8,), all finite
def test_battle_fixture_prefers_a_fight_button_over_start():
    # on battle.state, scores[ACTION_WORDS.index("A")] > scores[ACTION_WORDS.index("start")]
def test_decision_is_fast_on_gpu():
    # 20 decisions after a warm-up: mean under 40 ms on cuda; SKIPPED on cpu
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** The forward pass uses `model(**tok(text, return_tensors="pt").to(device)).logits[0, -1]`; no `generate`.
- [ ] **Step 4: Run `test_teacher.py` to verify they pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat: teacher decides by scoring the eight buttons in one pass"`

---

### Task 4: Teacher loop, service, Stage 0 measurement

**Files:**
- Modify: `teacher.py` (loop + `main`)
- Create: `run_teacher.sh`, `deploy/pokemon-teacher.service`
- Create: `tools/measure_teacher.py`
- Modify: `test_teacher.py` (loop test)

**Interfaces:**
- `run_teacher(steps: int, demo_dir: Path = Path("demos"), shard_steps: int = SHARD_STEPS, keep: int = KEEP_SHARDS, temperature: float = 0.0, log_every: int = 1000, env: EmeraldEnv | None = None, model_tok=None) -> dict`: plays, buffers transitions, writes a shard every `shard_steps` via `demos.write_shard` then `prune_ring`; prints a reward ledger per `log_every` steps in the training log's shape (`teacher reward/<term> <per 1k>`); returns the final counts (`steps`, `shards`, `reward`, `distinct_tiles`, `maps`). `episode_start` is True on the first transition and after any reset.
- `main()`: argparse `--steps` (default 10**9), `--temperature` (0.0), `--demo-dir` (`demos`); loads the model via `load_model` and exits 2 with the error message on failure.
- `run_teacher.sh`: `cd` to the project, `export PYTHONDONTWRITEBYTECODE=1`, `rm -rf __pycache__`, `exec .venv/bin/python teacher.py`.
- `deploy/pokemon-teacher.service`: copied from the training unit with `ExecStart=/home/abbas/pokemon-rl/run_teacher.sh`, `Restart=on-failure`, `RestartSec=60`, `Nice=10`, logs appended to `/home/abbas/pokemon-rl/teacher.log`. Installed with `cp deploy/pokemon-teacher.service ~/.config/systemd/user/ && systemctl --user daemon-reload`. Not enabled until Task 7.
- `tools/measure_teacher.py`: runs 20,000 steps each of (a) the teacher (`run_teacher` with a throwaway `demo_dir` under the scratchpad), (b) a uniform-random policy, (c) the newest `checkpoints/nature4` checkpoint, all from the default anchor, and prints one table: reward per 1,000 steps by term, encounters, wins (exp increases), whiteouts, distinct tiles, distinct maps, decisions per second. Ends with `GATE: PASS` or `GATE: FAIL` on the rule: teacher beats random on reward per 1k AND distinct tiles AND (encounters + attacks).

- [ ] **Step 1: Write the failing loop test**

```python
def test_run_teacher_writes_shards_and_prunes():
    # steps=300, shard_steps=100, keep=2 into a temp dir with the real model:
    # exactly 2 shards remain, each with 100 transitions, episode_start[0] is True,
    # actions all in 0..7, and the returned dict has steps == 300
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `run_teacher`, `main`, the two service files and the measurement tool.**
- [ ] **Step 4: Run `test_teacher.py` to verify it passes.**
- [ ] **Step 5: Install the unit (not enabled) and verify the failure path**: `systemctl --user start pokemon-teacher.service` with `HF_HUB_OFFLINE=1 TRANSFORMERS_CACHE=/nonexistent` injected via a drop-in, confirm `teacher.log` shows `teacher model unavailable` and `systemctl --user show -p NRestarts` increments after 60 s; remove the drop-in and stop it. Training's `time/fps` in `training.log` must not change across this.
- [ ] **Step 6: Run Stage 0**: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python tools/measure_teacher.py > scratchpad/stage0.log`. Expected: the table and a `GATE:` line. Report the table to Abbas. If `GATE: FAIL`, stop after this task's commit: rework the prompt (Task 2), re-measure; at most two rework rounds, then stop and report rather than build Task 5 on a teacher that is not better than random.
- [ ] **Step 7: Commit** — `git commit -m "feat: teacher loop, service unit and Stage 0 measurement"`

---

### Task 5: Imitation callback

**Files:**
- Modify: `train.py`
- Modify: `test_train.py`

**Interfaces:**
- `class ImitationCallback(BaseCallback)` with `__init__(self, demo_dir="demos", batch=256, steps=4, lr=2.5e-4, weight=IMITATION_WEIGHT)`; `IMITATION_WEIGHT = 0.1` module constant. `_on_rollout_end`: `data = load_recent(Path(demo_dir))`; return if `None`; `obs_t, actions, obs_t1 = sample_pairs(data, batch, rng)`; build torch dicts via `self.model.policy.obs_to_tensor(obs_t)[0]` (and `obs_t1`); for `steps` optimiser steps over `policy.parameters()` (a separate Adam created lazily): `logits = policy.get_distribution(obs_t_tensor).distribution.logits`; `imitation = weight * F.cross_entropy(logits, actions)`; `idm = F.cross_entropy(fe.idm_logits(s0, s1), actions)` and `fwd = fe.forward_error(s0, s1, actions).mean()` with `s0, s1` the screens as float /255 (these two only when the extractor has `idm_logits`); total = imitation + idm + fwd. Logs `aux/imitation_loss` and `aux/imitation_acc` (argmax agreement on the batch after the steps).
- Registered unconditionally in `main()` after the inverse-dynamics callback.

- [ ] **Step 1: Write the failing tests in `test_train.py`**

```python
def test_imitation_callback_is_a_noop_without_shards():
    # fresh PPO on a DummyVecEnv of 1 EmeraldEnv, empty temp demo dir; snapshot all
    # policy parameters; call cb._on_rollout_end(); no parameter changed, no aux/ key logged

def test_imitation_callback_learns_a_constant_teacher():
    # temp demo dir with one synthetic shard (512 transitions from the real env's
    # reset obs repeated, action always ACTION index 4 "right", episode_start[0] only);
    # fresh PPO; call _on_rollout_end 25 times; imitation_acc rises above 0.9 and at
    # least one extractor parameter changed
```
Use `PPO("MultiInputPolicy", env, n_steps=64, batch_size=32, policy_kwargs={"features_extractor_class": AgentExtractor, "features_extractor_kwargs": {"n_actions": 8}}, device="cpu")` and set `cb.model = model; cb.training_env = env; cb.logger = model.logger` directly (SB3's `init_callback`).

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement the callback and register it.**
- [ ] **Step 4: Run `test_train.py` to verify they pass**, then `test_extractor.py`.
- [ ] **Step 5: Revert-check** — set `IMITATION_WEIGHT = 0.0` and drop the idm/fwd terms: the constant-teacher test must fail. Restore.
- [ ] **Step 6: Smoke** — with one real shard in a scratch `demos` dir, `train.py --steps 16384 --backbone agent --checkpoint-dir <scratch>` logs `aux/imitation_loss` and `aux/imitation_acc` and no traceback.
- [ ] **Step 7: README** — one paragraph under the vision section: what the teacher is, where its record lives, what `aux/imitation_*` mean; a row for `teacher.py` and `demos.py` in the layout table.
- [ ] **Step 8: Commit** — `git commit -m "feat: policy and vision model train on the teacher's demonstrations"`

---

### Task 6: Viewer

**Files:**
- Modify: `watch.py`

**Interfaces:**
- `--teacher`: loads the model via `teacher.load_model`, drives the environment with `choose_action` (`--temperature` honoured), and shows on the HUD the tile summary's first line, the 8 scores rounded to one decimal with the chosen button marked, and the screen text line as today. No checkpoint required with `--teacher`.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify by eye** — `watch.py --teacher --steps 200 --snapshot /tmp/hud_teacher.png`; the snapshot shows the scores and a sensible button for what is on screen.
- [ ] **Step 3: Commit** — `git commit -m "feat: viewer drives with the teacher and shows its scores"`

---

### Task 7: Enable and verify

- [ ] **Step 1: Enable the teacher service** — `systemctl --user enable --now pokemon-teacher.service`; `teacher.log` shows reward ledgers; `demos/` gains a shard within ~2 minutes; `nvidia-smi` shows the teacher under 1.3 GB.
- [ ] **Step 2: Restart training** — `systemctl --user restart pokemon-rl.service` (resumes nature4); within two rollouts `training.log` shows `aux/imitation_loss` and `aux/imitation_acc`; `time/fps` within 15% of before.
- [ ] **Step 3: Full suites** — every `test_*.py` passes.
- [ ] **Step 4: Push** — `git push origin master`.
