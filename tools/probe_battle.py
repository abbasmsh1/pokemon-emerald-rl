"""Verify the battle Pokemon array and the ROM move table, and locate a byte
that is set only during a battle.

Usage: PYTHONPATH=. .venv/bin/python tools/probe_battle.py
"""
import struct

import numpy as np

from env import EmeraldEnv

BATTLE_MONS = 0x02024084
ENTRY = 0x58
MOVES_TABLE = 0x0831C898      # candidate: pokeemerald gBattleMoves, 12 bytes each
KNOWN = {10: ("SCRATCH", 40, 35), 45: ("GROWL", 0, 40)}   # id: (name, power, max pp)
REGIONS = [(0x02000000, 0x40000), (0x03000000, 0x8000)]


def mons(g, i):
    e = BATTLE_MONS + i * ENTRY
    return dict(species=g.read_u16(e), hp=g.read_u16(e + 0x28), maxHP=g.read_u16(e + 0x2C),
                level=g.read_u8(e + 0x2A), pp=[g.read_u8(e + 0x24 + k) for k in range(4)],
                moves=[g.read_u16(e + 0x0C + 2 * k) for k in range(4)])


def snap(env):
    return [np.frombuffer(bytes(env.gba.read_memory(a, n)), dtype=np.uint8) for a, n in REGIONS]


def main() -> None:
    battle = EmeraldEnv(state_path="battle.state"); battle.reset()
    dialogue = EmeraldEnv(state_path="dialogue.state"); dialogue.reset()
    anchor = EmeraldEnv(); anchor.reset()
    for _ in range(5):
        battle.step(0); dialogue.step(0); anchor.step(0)

    for name, env in (("battle", battle), ("dialogue", dialogue), ("anchor", anchor)):
        print(name, "ours", mons(env.gba, 0), "enemy", mons(env.gba, 1))

    ok = True
    for mid, (nm, power, pp) in KNOWN.items():
        raw = bytes(battle.gba.core.memory.u8[MOVES_TABLE + 12 * mid:MOVES_TABLE + 12 * mid + 12])
        got_power, got_pp = raw[1], raw[4]
        print(f"move {mid} {nm}: table power {got_power} pp {got_pp} (expected {power}, {pp})")
        ok &= (got_power, got_pp) == (power, pp)
    print("MOVES_TABLE confirmed" if ok else "MOVES_TABLE WRONG: scan needed")

    b, d, a = snap(battle), snap(dialogue), snap(anchor)
    survivors = []
    for r, (base, _) in enumerate(REGIONS):
        hits = np.where((b[r] != 0) & (d[r] == 0) & (a[r] == 0))[0]
        for i in hits:
            v = int(b[r][i])
            if v & (v - 1) == 0:            # single bit set
                survivors.append((base + int(i), v))
    print(f"{len(survivors)} single-bit bytes set only in battle; first 20:", [(hex(a), v) for a, v in survivors[:20]])

    # Play the battle out with A presses and see which survivors clear.
    for k in range(200):
        battle.step(5 if k % 2 == 0 else 0)
    s = battle.state_reader.read()
    after = snap(battle)
    print("after 200 presses: map", s["map"], "pos", s["pos"], "exp", s["party_exp"], "text", s["screen_text"][:40] if s["screen_text"] else battle.state_reader.text.last[:40])
    cleared = [(hex(a), v) for a, v in survivors if after[0 if a < 0x03000000 else 1][a - (0x02000000 if a < 0x03000000 else 0x03000000)] == 0]
    print(f"{len(cleared)} of them cleared after the battle; first 20:", cleared[:20])


if __name__ == "__main__":
    main()
