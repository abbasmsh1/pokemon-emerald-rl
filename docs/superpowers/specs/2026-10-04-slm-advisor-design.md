# SLM Text Advisor - Design

Date: 2026-10-04

## Goal

Let a small language model read the game's on-screen text and suggest an action,
and expose that suggestion to the PPO policy as part of its observation. The
policy learns on its own when to trust the suggestion; the advisor never acts.

Success is not "the agent follows the advice". Success is the agent reaching
maps it currently cannot, with the advice available as a signal it may use.

## Why

Training has plateaued. A 42M-step run ended with the same 11 map milestones it
had at 5.5M and `ep_rew_mean` flat between 45.7 and 46.7. Removing whiteout
termination lifted that to 12 maps and produced the first Pokemon above level 5,
but the agent still cannot progress toward Rustboro.

A plausible reason is that the game constantly states, in English, what it wants:
"Go check it out, dear!", "Do you want to use the PC?", "A wild POOCHYENA
appeared!". The agent sees these as 80x120 grayscale pixels and must infer the
required button from reward alone. An SLM reads that text natively.

## Feasibility, established before this spec

Two things were measured rather than assumed.

**The text is readable from RAM.** `pygba` ships `EmeraldCharmap`, a decoder for
Emerald's custom character encoding. Scanning EWRAM while dialogue was on screen
produced clean text at `0x02021C00`-`0x02022100`:

```
0x02021fe8  'of moving us in and cleaning up after.'
0x0202200f  'This is so convenient!'
0x02022026  'BRETT, your room is upstairs.'
0x0202205b  'DAD bought you a new clock to mark'
```

No OCR is required. The exact buffer holding the *currently displayed* message
still needs pinning within that region; that is implementation work, not a risk
to the approach.

**The model cannot run in the step loop.** Training runs 250-300 steps/sec. A
0.5B model in 4-bit on the remaining ~2GB of a 4GB card produces maybe 40
tokens/sec, so a short answer costs ~0.3s. That is two orders of magnitude too
slow to consult every step, and there are 8 workers sharing one GPU.

## Architecture

The latency finding dictates the shape: **advice is asynchronous and cached**.

```
worker (x8)                    advisor process (x1)
-----------                    --------------------
read text from RAM
  |
  look up in SQLite cache  <-------- writes answers
  |                                      ^
  +-- hit  -> advice into observation    |
  +-- miss -> enqueue text, advice=none -+
             (never blocks)
```

**Workers never wait.** A cache hit costs microseconds. A miss puts "no advice"
in the observation for that step and queues the text. The next time that text
appears it is a hit, and Pokemon dialogue repeats relentlessly.

**One advisor process** owns the model, drains the queue, and writes results
back. It is optional: if it is not running, has crashed, or failed to load the
model, every lookup misses, every observation carries "no advice", and training
proceeds exactly as it does today.

This also solves the 8-worker problem. Eight copies of the model will not fit in
4GB; one copy behind a shared cache does.

## Components

| File | Responsibility |
| --- | --- |
| `gametext.py` | Read the message buffer from RAM, decode via `EmeraldCharmap`, return the current text or `None` |
| `advice_cache.py` | SQLite cache and work queue. The only thing workers touch |
| `advisor.py` | Long-running process: drains the queue, prompts the model, parses a reply to an action, writes the cache |
| `env.py` | New `"advice"` entry in the observation |

### Observation change

`"advice"`: `Box(0, 1, (9,), uint8)`, a one-hot over the 8 actions plus a "no
advice" slot at index 8. "No advice" covers three distinct cases that the policy
need not distinguish: no text on screen, a cache miss, and a model reply that did
not parse.

This changes the policy's input shape, so **training restarts from scratch**. At
43M steps with a flat reward curve, that is an acceptable loss.

### Model

Qwen2.5-0.5B-Instruct, 4-bit, roughly 400MB of VRAM. Chosen to leave headroom
beside ResNet18 and PPO on a 4GB card that this project has already OOM-killed
twice. The advisor enforces a hard VRAM budget and exits cleanly rather than
competing with training if the budget cannot be met.

The prompt gives the text, the 8 available buttons, and asks for exactly one
button name. Replies are parsed strictly: anything not matching a known button
is recorded as "no advice" rather than guessed at.

### Cache

SQLite at `advice.db`, keyed on normalised text (whitespace collapsed, player
name replaced with a placeholder so "BRETT, your room is upstairs" and any other
name share an entry). Persists across restarts, so the warmup cost is paid once
rather than per run.

## Testing

- `gametext.py` returns known strings from a savestate captured mid-dialogue, and
  `None` when no message box is open.
- Normalisation maps differing player names to one key.
- A cache hit returns the stored action; a miss returns "no advice" and enqueues.
- The observation is a valid one-hot and matches the declared space.
- **Training runs unchanged with no advisor process at all.** This is the
  important one: the feature must be strictly optional.
- The advisor parses a valid reply, and rejects an invalid one as "no advice".

## Risks

**The SLM may add nothing.** Pokemon dialogue is often ambiguous about which
button to press: "Go check it out, dear!" does not name a button. The honest
expectation is that it helps on explicit prompts (yes/no questions, menus) and is
silent elsewhere. The policy can learn to ignore it, which bounds the downside to
wasted compute rather than worse behaviour.

**GPU contention.** Adding a model to a 4GB card already running ResNet18 and PPO
carries real OOM risk. This was raised and accepted; it is mitigated by the small
model, the hard budget, and the advisor exiting rather than fighting for memory.
If training OOMs after this lands, the advisor is the first thing to stop.

**Advice lags by one sighting** on genuinely novel text. Acceptable given how
much Pokemon text repeats.

**The exact message buffer is not yet pinned.** The region is known and decodes
correctly; identifying which address within it holds the live message is the
first implementation task. If it turns out there is no single stable address,
the fallback is to scan the known region each step, which is a RAM-only read and
therefore cheap.
