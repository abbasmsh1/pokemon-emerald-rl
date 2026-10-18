"""Gymnasium environment for Pokemon Emerald on mGBA."""

import contextlib
import ctypes
import os
import re
import zlib
from collections import deque
from pathlib import Path

import gymnasium as gym
import mgba.image
import numpy as np
from mgba._pylib import ffi
from pygba import PyGBA
from pygba.utils import KEY_MAP

from state import (
    POKEDEX_CAPACITY,
    TEXT_SPEED_FAST,
    TEXT_SPEED_MASK,
    GameState,
    force_fast_text,
)
from frontier import nearest_unvisited_distance
from text import our_attack, player_fled, wild_encounter

SCREEN_SHAPE = (3, 80, 120)
# 17 game values, then 12 that describe the agent's own situation: the last
# action it took, whether that action moved it, and how far along each of the
# three penalty counters it is.
#
# Those last twelve exist because the penalties were being charged on state the
# agent could not perceive. _frozen_steps, _recent_moves and _steps_since_reward
# live in the environment and appeared nowhere in the observation, so "standing
# at (13,14) having pressed up 500 times" and "arriving at (13,14)" were the
# same input, and only one of them is about to be charged -5. Neither could the
# agent see which button it had just pressed, so a blocked step and a step that
# moved look alike for several frames: Gen-3 animates walking on the spot when
# the way is blocked.
STATE_SIZE = 29
N_ACTION_SLOTS = 8

# On-screen text as token ids, for the learned text encoder in extractor.py.
# Words are hashed with crc32 into a fixed vocabulary rather than looked up in
# one: there is no training corpus to build a vocabulary from, the game's
# strings are finite, and a hash is stable across processes where Python's
# own hash() is salted per run. Id 0 is padding. 16 tokens covers the 64-char
# text window with room to spare.
TEXT_TOKENS = 16
TEXT_VOCAB = 1024
_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize_text(text: str) -> np.ndarray:
    """Lowercased words -> stable hashed ids in 1..TEXT_VOCAB-1, zero padded."""
    ids = [1 + zlib.crc32(w.lower().encode()) % (TEXT_VOCAB - 1) for w in _WORD.findall(text)]
    ids = ids[:TEXT_TOKENS]
    return np.array(ids + [0] * (TEXT_TOKENS - len(ids)), dtype=np.int64)


_LIBC = ctypes.CDLL(None)


def latest_progress_state(save_dir: str = "checkpoints") -> str | None:
    """Furthest OUTDOOR savestate the agent has reached, or None.

    ProgressCallback dumps one of these every time a worker reaches a new high
    in distinct maps visited, so the highest-numbered file is the furthest the
    agent has ever got. Starting episodes there instead of from boot.state is
    the curriculum the design doc names as the fallback for a plateau: with
    --max-steps 0 an episode only ends on a badge, so reset() runs about once
    per process, and every service restart was throwing away a dozen maps of
    progress and walking out of Littleroot again.

    Indoor states are skipped, which is not a detail. The highest count on disk
    was furthest_14maps_2_4, the Oldale Pokemon Center interior, and a measured
    6,000-step rollout from it never left that one room: 3.5 reward collected in
    the first moments, then two stall charges, net -2.5. The policy learned its
    routes outdoors and does not know how to walk out of a building, so the
    deepest state is the worst possible anchor. Map group 0 is Hoenn's towns
    and routes; every other group is an interior.

    Ties on the map count are broken by mtime. The choice is made once at
    construction, so a new high written mid-run does not move the target
    underneath a live worker.
    """
    best = None
    for path in Path(save_dir).glob("furthest_*maps_*.state"):
        # furthest_<n>maps_<mapGroup>_<mapNum>.state
        try:
            count, group = path.name.removeprefix("furthest_").split("_")[:2]
            n = int(count.removesuffix("maps"))
            if int(group) != 0:
                continue
        except ValueError:
            continue
        key = (n, path.stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, path)
    return str(best[1]) if best else None


@contextlib.contextmanager
def suppress_stdout():
    """mGBA logs from C at fd 1. Python-level redirection does not catch it.

    The fflush is load-bearing, not tidiness. mGBA writes into libc's stdout
    FILE buffer, and dup2 swaps the descriptor but not the buffer, so anything
    still buffered flushes after fd 1 is restored and lands in the caller's
    output. Flushing while fd 1 still points at /dev/null sends it there.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        _LIBC.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class EmeraldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    # No in-battle flag: pygba exposes no battle address, and a full-screen
    # battle is trivially visible to the CNN given the 3-frame stack.
    # start is appended rather than replacing the no-op so indices 0-6 keep their
    # meaning. It unlocks the menu, so the agent can use items and heal instead of
    # only walking, talking and fighting with whatever move the cursor lands on.
    ACTIONS = [None, "up", "down", "left", "right", "A", "B", "start"]

    TILE_CAP_PER_MAP = 400

    # ponytail: real play sets at most a handful of event flags per 24-frame step.
    # A jump far above that is a mid-relocation save-block parse, not progress.
    # Ceiling: a legitimate burst above this cap is silently dropped.
    MAX_FLAG_DELTA = 16
    # A flag count must be read IDENTICALLY this many times in a row, on a map
    # that has not changed, before it can raise the high-water mark. A door
    # warp's save-block relocation parses as garbage for a step or two, and the
    # 54M-step nature2 run learned to stand on Birch's lab door warping in and
    # out: each warp read trainer 0->7 on an identical bogus map id for two
    # steps, passed the stability gate as "settled", and paid +14. Over the run
    # that term booked 301,873 against 23 for experience, with the party never
    # gaining a point. A map-id settle alone did not close it: garbage arrives
    # at many different levels (7, 11, 20, ...) and each step of 16 or less
    # above the last one is a "new high", so the mark ratcheted up through them
    # at 1.5-10 per 1,000 steps. A real flag increment persists forever; garbage
    # reverts within two reads. So the VALUE has to hold still, not just the map.
    #
    # And the credit is potential-based on top of that, the same trick as the HP
    # term: a rise above the mark is paid, a fall below it is charged at the
    # same rate, and the mark follows. A garbage excursion of ANY length then
    # nets exactly zero once the real value comes back, and real trainer flags
    # are never cleared in-game so real play is never charged. The settle
    # windows are kept because they stop single-step spikes from paying even
    # transiently, which would add variance for nothing.
    FLAG_SETTLE_STEPS = 3

    BADGE_REWARD = 100.0
    NEW_MAP_REWARD = 2.0
    NEW_TILE_REWARD = 0.05
    # Each map is diced into SECTION_SIZE-square blocks. Sits between the tile and
    # map terms: pacing inside one block earns nothing after the first step into
    # it, so this rewards direction rather than shuffling between adjacent tiles,
    # and unlike tile novelty it is not exhausted by the per-map cap.
    SECTION_SIZE = 8
    NEW_SECTION_REWARD = 0.5
    # On top of NEW_SECTION_REWARD, once per section for the life of the
    # process. The per-episode set is refreshed every 50,000 steps so that
    # novelty never runs dry, but that re-pays a section walked a hundred times
    # exactly like one never seen. This is the difference between the two.
    # Cumulative and never cleared, so it cannot be farmed.
    FIRST_SECTION_REWARD = 1.0
    SCRIPT_FLAG_REWARD = 1.0
    TRAINER_REWARD = 2.0
    # Raised from 0.2. A level is many battles' worth of experience and the
    # only thing that makes the gym winnable, so it should not be worth four
    # new tiles. Safe to make large: it is paid on a high-water mark of the
    # party level sum, so a PC deposit and withdraw cannot re-pay it.
    LEVEL_REWARD = 1.0
    # Experience is the battle signal. A wild win is worth 20-40 points, so at
    # 0.02 each a win pays about as much as finding a new section, while
    # fleeing pays nothing at all. Fleeing was close to optimal before this:
    # exploration paid continuously outside battle and a battle paid only on a
    # rare level-up, so running away got back to the paying activity fastest.
    EXP_REWARD = 0.02
    # Charged when the player chooses to run. A wild Pokemon fleeing on its own
    # is not the agent's doing and is deliberately not penalised.
    # Harsher than the -1 it started at. At -1 a single wild win (~+0.5) nearly
    # covered the cost of running, so fleeing stayed close to break-even. At -5
    # running costs ten wins, which is meant to be unambiguous.
    FLEE_PENALTY = -5.0
    # Our Pokemon using a move: "TORCHIC used SCRATCH!" on screen. Paid once per
    # message so that choosing to fight is rewarded on the turn it happens,
    # not only when the experience lands at the end of a battle it may lose.
    # Small against EXP: a wild win is 0.4-0.8 of experience, an attack 0.2.
    ATTACK_REWARD = 0.2
    # A wild battle starting, paid once per "Wild X appeared!" message. This is
    # the grass reward: encounters happen only where wild Pokemon are, so it
    # pays roaming tall grass without resolving the tile under the player, and
    # it pays for seeking the fights that leveling needs. Running from the
    # encounter costs 5x this, so encounter-and-flee stays clearly negative.
    ENCOUNTER_REWARD = 1.0
    # Damage dealt, potential-based on the enemy's HP fraction: a fall is paid,
    # a rise (the enemy healing) is charged at the same rate, and nothing is
    # booked across a battle's start or end, so a new enemy's full bar is not
    # a loss and a faint pays its final decrease once. A whole bar is 0.5,
    # about one wild win's experience.
    DAMAGE_REWARD = 0.5
    # Potential-based shaping on the Manhattan distance, in sections, to the
    # nearest section never visited on this map: a step that shortens it pays,
    # one that lengthens it charges, so a loop nets zero and nothing can be
    # farmed. Small by design: it tilts ties toward the edge of what the agent
    # knows when nothing else is paying. Not a route; there is no goal in it.
    FRONTIER_REWARD = 0.1
    FRONTIER_CAP = 8
    # Pokedex progress, raised from 0.1/0.5. Both are popcounts of the dex's
    # seen/owned bit arrays, which only ever gain bits, so neither can be
    # farmed and both can be paid properly. Catching is the rarer and more
    # useful act: it needs Poke Balls, money, and a battle taken to low HP
    # rather than won outright, and it is how the party grows past the starter.
    # There is no separate bonus for a complete dex. It is 386 species, several
    # of them version exclusives and one of them a Mew; against a run that has
    # yet to earn one badge it would be a reward term that never pays once.
    SEEN_REWARD = 0.5
    CAUGHT_REWARD = 5.0
    WHITEOUT_PENALTY = -5.0
    # Paid on the CHANGE in party HP fraction, both directions. Symmetric on
    # purpose: a one-way heal bonus is farmable, since taking damage would be
    # free and healing at a Pokemon Center is free and unlimited, so the agent
    # could loop "get hurt, walk back, heal" forever. Charging the damage too
    # makes a full cycle net zero, and what is left is the useful part: at any
    # given moment a hurt party is worth less than a healthy one, so walking to
    # a Center pays and the menu has a reason to exist.
    HEAL_REWARD = 3.0
    # Charged when the OPTION screen's settings change. Nothing else in the
    # game writes them, so this fires exactly when the agent has gone into
    # OPTION and fiddled, which it has: text speed was found set to SLOW,
    # making every mandatory dialogue take several times as many steps. The
    # values are only written back on leaving the screen, so the charge lands
    # on exit rather than on entry, and entering and leaving without touching
    # anything is billed by the stall penalty instead.
    OPTIONS_PENALTY = -10.0
    # Charged on the Start press itself, not on time spent in the menu.
    # Sampling the 100k policy every 1,000 steps put 7 of 20 frames in a menu:
    # the start menu twice, the party screen twice, the trainer card, an item
    # submenu, and a transition. A third of its life, with a Torchic sitting at
    # 8/25 HP it never healed.
    #
    # Detecting "a menu is open" would be the direct fix and three RAM scans
    # failed to find a marker that survived a repeat run, so the press is taxed
    # instead. Every one of those screens is reached through this one button, so
    # the entry point is the whole surface. Nothing the badge needs is behind
    # it: a Pokemon Center heals by talking to the nurse, and the battle BAG is
    # reached with A.
    #
    # The button is deliberately kept rather than removed from ACTIONS. It was
    # added on purpose and taxing it is reversible by one constant, where
    # dropping it would change the action space and discard the policy.
    #
    # Sized against the measured press rate, not guessed. The first rollout at
    # -2.0 booked -103 per 1,000 steps: 51 presses per 1,000 steps, 5% of all
    # actions, which is 100x every other term put together and would drown the
    # reward signal while the agent unlearns it. A press costs a few dozen steps
    # of menu time, worth 1-2 of forgone exploration, so the charge is set near
    # that. It still prices a press at ten new tiles.
    START_PENALTY = -0.5
    # Pressing a direction and not moving: a wall, a ledge from below, water,
    # the map's edge. The 400k nature3 policy spent 25% of its actions pressing
    # up into a blocked tile on Route 103, and nothing billed it: the loiter and
    # freeze terms exempt stationary steps so that battles are not taxed. This
    # is the targeted version. It is charged only in the overworld with no text
    # on screen, since in a battle, a menu or a dialogue a direction press moves
    # a cursor and never the player, and the text reader now tells those apart.
    # Sized at one new tile: a bump costs what a step somewhere new would earn.
    BUMP_PENALTY = -0.05
    _DIRECTIONS = frozenset({"up", "down", "left", "right"})

    # Charged every STALL_LIMIT steps that earn nothing at all. At 16,384 steps an
    # episode can absorb at most three of these, so it nudges away from idling
    # without overwhelming a +100 badge or the dense novelty signal.
    # Charged sooner and harder: 5,000 steps of nothing was ~33 minutes of game
    # time for -1, which barely registered against exploration income.
    # 600, down from 2,500. The same policy dodged the freeze penalty (600 steps
    # without moving) by moving one tile every four steps, so stall is now the
    # catch-all at the same horizon: 600 steps earning nothing is four minutes
    # of game time, pathological in a battle, a menu or the overworld alike.
    STALL_LIMIT = 600
    STALL_PENALTY = -3.0

    # Exploration novelty is exhaustible, and with --max-steps 0 it was being
    # exhausted permanently. reset() only runs when an episode ends, an episode
    # only ends on a badge, and no badge has ever been earned, so the visited
    # sets filled once and never cleared. Measured over 9 million steps from
    # 73.1M: tiles_visited frozen at 486, maps_visited at 8, and every one of
    # new_tile, new_section, new_map, exp, hp and caught reporting exactly 0.
    # Reward per 1,000 steps ran between -1 and -5.6, all of it stall penalty.
    # The agent was being trained on a landscape where nothing it could do paid
    # anything at all.
    #
    # So the sets are refreshed on a fixed schedule instead of only on reset.
    # Fixed, not triggered by behaviour: tying it to the stall penalty would
    # have let the agent stall deliberately to make its old routes pay again.
    # What it can collect per refresh is bounded by the map it has opened up
    # (486 tiles is roughly 35 reward, under 1 per 1,000 steps at this
    # interval), and collecting it means walking rather than standing still.
    NOVELTY_REFRESH_STEPS = 50_000

    # Pacing: the last LOITER_LIMIT moves covered no more than LOITER_DISTINCT
    # tiles. Distinct from the stall penalty, which only fires when nothing at
    # all is earned: shuffling over a few fresh tiles draws tile novelty while
    # going nowhere.
    #
    # Grid-free on purpose. Two earlier versions keyed this on a fixed 16x16
    # then 8x8 block, and the 400k nature3 policy found the loophole in both
    # within a day: stand on Route 103's row 13 pressing up into a blocked
    # tile, step one tile sideways every few steps, and drift back and forth
    # across x=8, the boundary between two blocks. The block counter reset on
    # every crossing and never charged. A window of recent moves has no
    # boundary to straddle: 30 moves over 6 or fewer tiles is shuffling
    # wherever it happens, and 30 moves over 7 or more is going somewhere.
    #
    # Only steps where the player actually MOVED count. A battle, a dialogue
    # and an open menu hold the position frozen, and they are the things worth
    # doing stationary; billing them would undo the battle rewards.
    LOITER_LIMIT = 30
    LOITER_DISTINCT = 6
    LOITER_PENALTY = -1.0

    # Standing still, which LOITER_PENALTY deliberately exempts so that battles
    # and dialogue are not billed. That exemption made walking into a wall free:
    # measured on the 300k policy over 5,000 steps, the position was frozen on
    # 94.7% of them, the longest single freeze ran 1,170 steps against one wall,
    # only 53 distinct tiles were touched, and the whole 1,170-step freeze cost
    # about -1.4 because the stall penalty charges only once per 2,500 steps.
    # The action mix was 28% up, into that wall.
    #
    # 600 steps is chosen to sit well above a battle and well below the observed
    # freeze. A wild battle runs 60-150 steps and a trainer battle a few hundred,
    # and all of them hold the overworld position frozen, so a tighter threshold
    # would tax exactly the behaviour the experience reward is there to buy.
    # Earning anything also clears the counter, which exempts a battle outright.
    FREEZE_LIMIT = 600
    FREEZE_PENALTY = -5.0

    def __init__(
        self,
        rom_path: str = "Pokemon - Emerald Version (USA, Europe).gba",
        state_path: str | None = None,
        frameskip: int = 24,
        max_steps: int = 16384,   # 0 or less means no step limit
        render_mode: str | None = None,
    ):
        super().__init__()
        self.frameskip = frameskip
        self.max_steps = max_steps
        self.render_mode = render_mode

        # None means "furthest progress so far", falling back to the intro
        # savestate on a fresh clone with no progress recorded yet.
        if state_path is None:
            state_path = latest_progress_state() or "boot.state"
        self.state_path = state_path
        with open(state_path, "rb") as f:
            self._boot_state = f.read()

        with suppress_stdout():
            self.gba = PyGBA.load(rom_path)
            self.width, self.height = self.gba.core.desired_video_dimensions()
            self._framebuffer = mgba.image.Image(self.width, self.height)
            self.gba.core.set_video_buffer(self._framebuffer)
            self.gba.core.reset()

        self.state_reader = GameState(self.gba)
        self._frames = deque(maxlen=3)
        self._step_count = 0
        self._visited_tiles: set[tuple[int, int, int, int]] = set()
        self._visited_maps: set[tuple[int, int]] = set()
        self._visited_sections: set[tuple[int, int, int, int]] = set()
        self._tiles_per_map: dict[tuple[int, int], int] = {}
        self._prev: dict | None = None
        # High-water marks, like levels and experience. Trainer flags are never
        # cleared in-game and story flags almost never are, so paying only for a
        # new all-time high is correct, and it turns the warp-door pump above from
        # an income into a one-off bounded by MAX_FLAG_DELTA.
        self._max_flags: dict[str, int] = {"script": 0, "trainer": 0}
        # (value, consecutive identical reads) per flag kind; see FLAG_SETTLE_STEPS.
        self._flag_run: dict[str, tuple[int, int]] = {"script": (0, 0), "trainer": (0, 0)}
        self._steps_since_map_change = 0
        # Positive income booked this step. The freeze and stall counters clear on
        # this rather than on the net reward: the symmetric HP term and the loiter
        # charge routinely drag a productive step to zero or below, so a wild win
        # that cost 20% HP (+0.6 exp, -0.6 hp) read as "earned nothing" and the
        # battle kept the freeze counter running.
        self._earned = 0.0
        self._max_level_sum = 0
        self._max_party_exp = 0
        self._was_fleeing = False
        self._steps_since_reward = 0
        self._recent_moves: deque = deque(maxlen=self.LOITER_LIMIT)
        self._frozen_steps = 0
        self._frontier_prev: int | None = None
        self._last_action: int | None = None
        self._moved = False
        self._reward_terms: dict[str, float] = {}
        # Reporting only, never used in reward. Deliberately uncapped and NOT
        # cleared on reset: _visited_tiles is capped at TILE_CAP_PER_MAP to stop
        # reward farming and resets each episode, so it would draw a truncated,
        # single-episode picture. This accumulates the real footprint.
        self._coverage: set[tuple[int, int, int, int]] = set()
        # Sections visited at least once in this process; see FIRST_SECTION_REWARD.
        self._sections_ever: set[tuple[int, int, int, int]] = set()

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, SCREEN_SHAPE, dtype=np.uint8),
            "state": gym.spaces.Box(-1.0, 1.0, (STATE_SIZE,), dtype=np.float32),
            "text": gym.spaces.Box(0, TEXT_VOCAB - 1, (TEXT_TOKENS,), dtype=np.int64),
        })

    def _raw_frame(self) -> np.ndarray:
        """Read the framebuffer directly. Byte-identical to the PIL path and
        76x faster (0.001ms vs 0.076ms)."""
        arr = np.frombuffer(ffi.buffer(self._framebuffer.buffer), dtype=np.uint8)
        return arr.reshape(self.height, self._framebuffer.stride, 4)[:, :self.width, :3]

    def _gray_downscaled(self) -> np.ndarray:
        """240x160 RGB -> 120x80 grayscale. Green channel with 2x stride: one
        slice, no copy, no extra dependency. Green is a fine luminance proxy."""
        return np.ascontiguousarray(self._raw_frame()[::2, ::2, 1])

    def _encode_state(self, s: dict) -> np.ndarray:
        """17 values read from the game, then 12 about the agent's own
        situation. See STATE_SIZE for why the second group exists."""
        levels = s["party_levels"]
        vec = np.zeros(STATE_SIZE, dtype=np.float32)
        for i in range(8):
            vec[i] = 1.0 if i < s["badges"] else 0.0
        vec[8] = s["party_hp_frac"]
        vec[9] = (sum(levels) / len(levels) / 100.0) if levels else 0.0
        # 127.0 is exact for signed-byte map identifiers (mapGroup/mapNum are
        # "b", -128..127) and cannot saturate; 255.0 gives roughly 2x headroom
        # over the largest Hoenn maps for tile coordinates (x/y are "H").
        vec[10] = s["map"][0] / 127.0
        vec[11] = s["map"][1] / 127.0
        vec[12] = s["pos"][0] / 255.0
        vec[13] = s["pos"][1] / 255.0
        vec[14] = np.log1p(s["money"]) / np.log1p(999_999)
        vec[15] = s["seen"] / POKEDEX_CAPACITY
        vec[16] = s["caught"] / POKEDEX_CAPACITY

        if self._last_action is not None:
            vec[17 + self._last_action] = 1.0
        vec[25] = 1.0 if self._moved else 0.0
        # Each counter as a fraction of the limit that charges it, so the agent
        # can see a penalty coming instead of being surprised by it.
        vec[26] = min(self._frozen_steps / self.FREEZE_LIMIT, 1.0)
        vec[27] = len(self._recent_moves) / self.LOITER_LIMIT
        vec[28] = min(self._steps_since_reward / self.STALL_LIMIT, 1.0)
        return np.clip(vec, -1.0, 1.0)

    def _enforce_fast_text(self, s: dict) -> dict:
        """Hold TEXT SPEED at FAST and report the corrected reading.

        Returning a corrected copy rather than mutating matters twice over:
        state.py hands back its own cached dict on a bad read, and the reward
        compares this step's options against the previous step's. Without the
        correction the agent's change would be billed once when it lands and
        again when this reverts it.
        """
        if not force_fast_text(self.gba):
            return s
        button_mode, packed = s["options"]
        fixed = (packed & ~TEXT_SPEED_MASK) | TEXT_SPEED_FAST
        return dict(s, options=(button_mode, fixed))

    def _bump(self, key, s: dict, stable: bool, had_prev: bool) -> float:
        """BUMP_PENALTY if a direction was pressed in the overworld and the
        player did not move; otherwise 0. Factored out so it can be tested on
        plain dicts."""
        if (
            key in self._DIRECTIONS
            and had_prev
            and stable
            and not self._moved
            and not s["screen_text"]
            and not self._text_box_visible()
        ):
            return self._pay("bump", self.BUMP_PENALTY)
        return 0.0

    def _text_box_visible(self) -> bool:
        """Dialogue boxes are white and fill the bottom 48 rows."""
        return float(self._raw_frame()[112:160].mean()) > 200.0

    def _observation(self, s: dict) -> dict:
        return {
            "screen": np.stack(self._frames, axis=0),
            "state": self._encode_state(s),
            "text": tokenize_text(s["screen_text"]),
        }

    def _refresh_novelty(self) -> None:
        """Clear the exploration sets so there is something to find again.

        _coverage is deliberately untouched: it is the cumulative record of
        everywhere this worker has ever stood, and the coverage overlay would
        be wrong if it reset with the reward bookkeeping.
        """
        self._visited_tiles.clear()
        self._visited_maps.clear()
        self._visited_sections.clear()
        self._tiles_per_map.clear()

    def _pay(self, term: str, amount: float) -> float:
        """Book `amount` against a named term and hand it back to the caller.

        Every reward line goes through this so the breakdown cannot drift from
        the reward actually paid. Without it there was no live signal at all:
        with --max-steps 0 an episode only ends on a badge, so SB3's
        ep_rew_mean never updates, and the numbers it does report are an
        ep_info_buffer restored from the checkpoint -- for weeks it was showing
        6.55e4-step episodes from a run whose step cap no longer exists.
        """
        self._reward_terms[term] = self._reward_terms.get(term, 0.0) + amount
        self._earned += max(0.0, amount)
        return amount

    def drain_reward_terms(self) -> dict[str, float]:
        """Totals since the last drain, and reset. Pulled across the process
        boundary by the training callback with env_method."""
        terms = self._reward_terms
        self._reward_terms = {}
        return terms

    def _compute_reward(self, s: dict) -> float:
        # ponytail: the save block relocates during a map transition and parses as
        # garbage for a step or two, with a bogus map id like (0,0) or (-1,-1) and a
        # flag array read at a shifted offset. Any step where the map id is still
        # changing is in-flight: score nothing and record nothing. The genuine
        # arrival is credited on the first stable step instead.
        # Ceiling: new-map credit is delayed one step, and flags set exactly during
        # a warp fold into the next stable step's delta.
        map_key = s["map"]
        if self._prev is not None and map_key != self._prev["map"]:
            self._prev = s
            self._steps_since_map_change = 0
            self._frontier_prev = None
            return 0.0
        self._steps_since_map_change += 1

        reward = 0.0
        self._earned = 0.0
        prev = self._prev

        if map_key not in self._visited_maps:
            self._visited_maps.add(map_key)
            reward += self._pay("new_map", self.NEW_MAP_REWARD)

        tile_key = (map_key[0], map_key[1], s["pos"][0], s["pos"][1])
        if tile_key not in self._visited_tiles:
            count = self._tiles_per_map.get(map_key, 0)
            # Cap stops the agent farming reward by pacing a large open route.
            if count < self.TILE_CAP_PER_MAP:
                self._visited_tiles.add(tile_key)
                self._tiles_per_map[map_key] = count + 1
                reward += self._pay("new_tile", self.NEW_TILE_REWARD)

        section_key = (
            map_key[0],
            map_key[1],
            s["pos"][0] // self.SECTION_SIZE,
            s["pos"][1] // self.SECTION_SIZE,
        )
        if section_key not in self._visited_sections:
            self._visited_sections.add(section_key)
            reward += self._pay("new_section", self.NEW_SECTION_REWARD)
            if section_key not in self._sections_ever:
                self._sections_ever.add(section_key)
                reward += self._pay("first_section", self.FIRST_SECTION_REWARD)

        # Frontier shaping; see FRONTIER_REWARD. The current section was added
        # to _sections_ever just above, so the distance is never 0 here.
        self._sections_ever.add(section_key)
        visited = {(sx, sy) for (g, n, sx, sy) in self._sections_ever if (g, n) == map_key}
        d = nearest_unvisited_distance(section_key[2:], visited, self.FRONTIER_CAP)
        if self._frontier_prev is not None:
            reward += self._pay("frontier", self.FRONTIER_REWARD * (self._frontier_prev - d))
        self._frontier_prev = d

        if prev is not None:
            reward += self._pay("badge", self.BADGE_REWARD * max(0, s["badges"] - prev["badges"]))

            # Pay only for a new all-time high, only once the map has been stable
            # for FLAG_SETTLE_STEPS, and never for a jump above MAX_FLAG_DELTA,
            # which real play cannot produce in one step. See FLAG_SETTLE_STEPS for
            # the pump this replaces.
            # Ceiling: a legitimate burst above MAX_FLAG_DELTA is dropped, not deferred.
            for name, key, term, weight in (
                ("script", "script_flag_count", "script_flag", self.SCRIPT_FLAG_REWARD),
                ("trainer", "trainer_flag_count", "trainer", self.TRAINER_REWARD),
            ):
                value, run = self._flag_run[name]
                run = run + 1 if s[key] == value else 1
                self._flag_run[name] = (s[key], run)
                delta = s[key] - self._max_flags[name]
                if (
                    run >= self.FLAG_SETTLE_STEPS
                    and self._steps_since_map_change >= self.FLAG_SETTLE_STEPS
                    and 0 < abs(delta) <= self.MAX_FLAG_DELTA
                ):
                    reward += self._pay(term, weight * delta)
                    self._max_flags[name] = s[key]

            # ponytail: pay only for a new all-time-high party level sum this episode.
            # Diffing against the previous step lets a PC deposit/withdraw cycle re-pay
            # the same levels indefinitely.
            level_sum = sum(s["party_levels"])
            if level_sum > self._max_level_sum:
                reward += self._pay("level", self.LEVEL_REWARD * (level_sum - self._max_level_sum))
                self._max_level_sum = level_sum

            # Same high-water mark as levels: a PC deposit then withdraw must
            # not re-pay experience the agent already banked.
            if s["party_exp"] > self._max_party_exp:
                reward += self._pay("exp", self.EXP_REWARD * (s["party_exp"] - self._max_party_exp))
                self._max_party_exp = s["party_exp"]

            # ponytail: skipped when either party is empty. A failed save-block read
            # reports hp_frac 0.0 with an empty party, which would otherwise bill a
            # full-health party -3 and refund it the next step.
            # Ceiling: catching or depositing a Pokemon shifts the fraction without
            # any healing, paying or charging up to HEAL_REWARD once. Bounded by the
            # six-slot party and rare enough not to be worth tracking per-slot HP.
            if s["party_levels"] and prev["party_levels"]:
                reward += self._pay("hp", self.HEAL_REWARD * (
                    s["party_hp_frac"] - prev["party_hp_frac"]
                ))

            # (0, 0) is state.py's failed-read sentinel, not a real setting, so
            # ignore transitions into or out of it rather than billing twice.
            if (
                s["options"] != prev["options"]
                and s["options"] != (0, 0)
                and prev["options"] != (0, 0)
            ):
                reward += self._pay("options", self.OPTIONS_PENALTY)

            # Edge-triggered: the message lingers for many frames, so charge the
            # decision to run once rather than every step it stays on screen.
            fleeing = player_fled(s["screen_text"])
            if fleeing and not self._was_fleeing:
                reward += self._pay("flee", self.FLEE_PENALTY)
            self._was_fleeing = fleeing

            # Edge on the message text, so one attack pays once however many
            # steps the line stays on screen.
            attacking = our_attack(s["screen_text"])
            if attacking and s["screen_text"] != prev["screen_text"]:
                reward += self._pay("attack", self.ATTACK_REWARD)
            if wild_encounter(s["screen_text"]) and s["screen_text"] != prev["screen_text"]:
                reward += self._pay("encounter", self.ENCOUNTER_REWARD)

            # Damage dealt; see DAMAGE_REWARD. Only when both steps are in a
            # battle, so the start (zeros -> full bar) and the end (bar ->
            # zeros) book nothing.
            if s["in_battle"] and prev["in_battle"]:
                delta = float(prev["battle"][1]) - float(s["battle"][1])
                if delta:
                    reward += self._pay("damage", self.DAMAGE_REWARD * delta)

            # Counted only while walking; see LOITER_PENALTY.
            if s["pos"] != prev["pos"]:
                self._recent_moves.append((map_key, s["pos"]))
                if (len(self._recent_moves) == self.LOITER_LIMIT
                        and len(set(self._recent_moves)) <= self.LOITER_DISTINCT):
                    reward += self._pay("loiter", self.LOITER_PENALTY)
                    self._recent_moves.clear()

            reward += self._pay("seen", self.SEEN_REWARD * max(0, s["seen"] - prev["seen"]))
            reward += self._pay("caught", self.CAUGHT_REWARD * max(0, s["caught"] - prev["caught"]))
            if s["whiteout"] and not prev["whiteout"]:
                reward += self._pay("whiteout", self.WHITEOUT_PENALTY)

        # Frozen in place: same tile, same map, nothing earned. Checked before
        # the stall counter because it is the tighter and more specific of the
        # two, and the one that catches a wall.
        if prev is not None and s["pos"] == prev["pos"]:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0
        if self._earned > 0.0:
            self._frozen_steps = 0
        elif self._frozen_steps >= self.FREEZE_LIMIT:
            reward += self._pay("freeze", self.FREEZE_PENALTY)
            self._frozen_steps = 0

        # Stall penalty: charge for going STALL_LIMIT steps without earning
        # anything. Novelty runs out once a map is fully walked, leaving the agent
        # free to idle at zero reward forever; this makes standing still cost
        # something. The counter resets on the charge as well as on progress, so a
        # long stall is billed repeatedly rather than only once.
        if self._earned > 0.0:
            self._steps_since_reward = 0
        else:
            self._steps_since_reward += 1
            if self._steps_since_reward >= self.STALL_LIMIT:
                reward += self._pay("stall", self.STALL_PENALTY)
                self._steps_since_reward = 0

        self._prev = s
        return reward

    def _should_terminate(self, s: dict, stable: bool) -> bool:
        """Only the badge ends an episode.

        A whiteout is not death in this game: it warps the player to the last
        Pokemon Center with a healed party and play continues. Ending the episode
        on it was our invention, and an expensive one. Measured at 42.5M steps,
        ep_len_mean sat at 18,300 against a 65,536 cap with no badges earned, so
        every episode was ending this way, discarding all accumulated exploration
        and restarting from Littleroot. The agent therefore never experienced
        recovery and could not learn it.

        The -5 WHITEOUT_PENALTY still applies, so whiting out is costly; it is
        just no longer fatal.
        """
        return bool(stable and s["badges"] >= 1)

    def _info(self, s: dict) -> dict:
        return {
            "badges": s["badges"],
            "maps_visited": len(self._visited_maps),
            "tiles_visited": len(self._visited_tiles),
            "sections_visited": len(self._visited_sections),
            "map": s["map"],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        with suppress_stdout():
            self.gba.core.reset()
            self.gba.core.load_raw_state(self._boot_state)
            self.gba.core.run_frame()
            # Pin the text speed before the first reading is taken, then step a
            # frame so that reading is genuinely FAST. Correcting it after the
            # fact instead would leave _prev disagreeing with a raw state read
            # for one step, which is both a lie and a -10 waiting to happen.
            force_fast_text(self.gba)
            self.gba.core.run_frame()

        frame = self._gray_downscaled()
        self._frames.clear()
        for _ in range(3):
            self._frames.append(frame)

        self._refresh_novelty()

        self.state_reader.reset()
        s = self.state_reader.read()
        self._prev = s
        self._max_flags = {
            "script": s["script_flag_count"],
            "trainer": s["trainer_flag_count"],
        }
        # A loaded savestate is settled, not mid-warp.
        self._steps_since_map_change = self.FLAG_SETTLE_STEPS
        self._flag_run = {
            "script": (s["script_flag_count"], self.FLAG_SETTLE_STEPS),
            "trainer": (s["trainer_flag_count"], self.FLAG_SETTLE_STEPS),
        }
        self._earned = 0.0
        self._max_level_sum = sum(s["party_levels"])
        self._max_party_exp = s["party_exp"]
        self._was_fleeing = False
        self._steps_since_reward = 0
        self._recent_moves.clear()
        self._frozen_steps = 0
        self._frontier_prev = None
        self._last_action = None
        self._moved = False
        return self._observation(s), self._info(s)

    def step(self, action: int):
        key = self.ACTIONS[action]
        self._last_action = action
        prev_pos = self._prev["pos"] if self._prev is not None else None
        # ponytail: Gen-3 gates dialogue, menus and battle moves on a fresh press
        # edge. Holding a key across steps reads as ONE press, so two consecutive
        # A actions would advance one textbox, not two. Release for one frame first.
        # Ceiling: costs 1 of the 24 frames per step.
        with suppress_stdout():
            self.gba.core.set_keys()
            self.gba.core.run_frame()
            if key is not None:
                self.gba.core.set_keys(KEY_MAP[key])
            for _ in range(self.frameskip - 1):
                self.gba.core.run_frame()

        self._frames.append(self._gray_downscaled())
        self._step_count += 1
        if self._step_count % self.NOVELTY_REFRESH_STEPS == 0:
            self._refresh_novelty()

        s = self._enforce_fast_text(self.state_reader.read())
        # The text reader lets a string go stale after a few steps; a dialogue
        # box waiting for A stays on screen far longer. If the bottom of the
        # frame is a bright text box, keep the last string alive.
        if not s["screen_text"] and self._text_box_visible():
            s = dict(s, screen_text=self.state_reader.text.keep_alive())
        self._moved = prev_pos is not None and s["pos"] != prev_pos

        # ponytail: badges and whiteout are parsed from the same save-block read that
        # relocates during a map transition, so an in-flight read can fabricate a
        # badge bit or an all-fainted party. Terminate only on a stable reading.
        # Ceiling: a genuine badge or whiteout landing exactly on a transition step
        # is recognised one step late.
        stable = self._prev is None or s["map"] == self._prev["map"]
        if stable:
            # ponytail: same in-flight read that fabricates badges also fabricates
            # coordinates, and pos is uint16, so a garbage value up to 65535 lands
            # in the coverage set. One bad x and one bad y on the same map made
            # render_coverage allocate a 65528x65526 bounding box (12.9GB) and the
            # kernel OOM-killed training twice. Only record stable readings.
            self._coverage.add((s["map"][0], s["map"][1], s["pos"][0], s["pos"][1]))
        reward = self._compute_reward(s)
        # After the reward, so the counters the observation reports are this
        # step's and not the previous one's.
        obs = self._observation(s)
        # Outside _compute_reward because that gates everything on map
        # stability, and a Start press costs the same mid-transition as anywhere
        # else. Booked through _pay so the breakdown stays exact.
        if key == "start":
            reward += self._pay("start_press", self.START_PENALTY)
        reward += self._bump(key, s, stable, prev_pos is not None)
        terminated = self._should_terminate(s, stable)
        # max_steps <= 0 removes the limit entirely. Combined with whiteouts no
        # longer terminating, only a badge then ends an episode, so reset() may
        # never run: the per-episode exploration sets never clear and tile,
        # section and map novelty are permanently exhausted once walked.
        truncated = self.max_steps > 0 and self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, self._info(s)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._raw_frame().copy()
        return None

    def coverage(self) -> list[tuple[int, int, int, int]]:
        """Every (mapGroup, mapNum, x, y) this worker has stood on, across all
        episodes. Retrieved across the process boundary with env_method."""
        return list(self._coverage)

    def save_state(self) -> bytes:
        """Dump the emulator state. Used across process boundaries by the
        training callback via SubprocVecEnv.env_method."""
        with suppress_stdout():
            return bytes(ffi.buffer(self.gba.core.save_raw_state()))

    def close(self):
        pass
