"""Report what the agent actually achieved in a savestate.

Reads the progress savestates train.py drops into checkpoints/ and prints party,
levels, badges, Pokedex counts, money, cities visited and gyms beaten.

Unlike state.py, this is free to read ROM (species names live there). state.py
avoids ROM because a read costs 1.28ms inside a 7.36ms training step; nothing
here runs in a training loop, so the cost is irrelevant.

Usage:
    python stats.py                      # every savestate in checkpoints/
    python stats.py checkpoints/foo.state
    python stats.py --json               # machine-readable
"""

import argparse
import json
import re
from pathlib import Path

from pygba import PyGBA, PokemonEmerald
from pygba.game_wrappers.utils.emerald_utils import read_species_names

from env import suppress_stdout

ROM = "Pokemon - Emerald Version (USA, Europe).gba"

# Map ids are not ordered by route progress, so name the ones seen early rather
# than trying to derive an ordering. Anything unlisted prints as its raw id.
KNOWN_MAPS = {
    (0, 9): "Littleroot Town",
    (0, 10): "Oldale Town",
    (0, 0): "Petalburg City",
    (0, 3): "Rustboro City",
    (0, 16): "Route 101",
    (1, 0): "player house 1F",
    (1, 1): "player house 2F",
    (1, 2): "rival house 1F",
    (1, 3): "rival house 2F",
    (1, 4): "Birch's lab",
    (25, 40): "inside the truck",
}


def map_name(group: int, num: int) -> str:
    return KNOWN_MAPS.get((group, num), f"map {group}-{num}")


def read_stats(state_path: Path) -> dict:
    """Load one savestate and decode everything worth reporting."""
    # mGBA logs from C at fd 1 and would shred the report it is printed beside.
    with suppress_stdout():
        gba = PyGBA.load(ROM)
        gba.core.reset()
        gba.core.load_raw_state(state_path.read_bytes())
        gba.core.run_frame()

        s = PokemonEmerald().game_state(gba)
        names = read_species_names(gba)
        # Free the core inside the suppressed block: mGBA logs during teardown
        # too, and a core collected at interpreter exit would print after the
        # report. s and names are plain Python values by now, holding no ref.
        del gba

    party = []
    for mon in s.get("party", []):
        species_id = mon["box"]["substructs"][0]["species"]
        species = (
            names[species_id].lower()
            if names and species_id < len(names)
            else f"#{species_id}"
        )
        party.append(
            {
                "species": species,
                "level": mon["level"],
                "hp": mon["hp"],
                "max_hp": mon["maxHp"],
            }
        )

    loc = s["location"]
    return {
        "file": state_path.name,
        "map": (loc["mapGroup"], loc["mapNum"]),
        "map_name": map_name(loc["mapGroup"], loc["mapNum"]),
        "pos": (s["pos"]["x"], s["pos"]["y"]),
        "badges": s["num_badges"],
        "money": s.get("money", 0),
        "seen": s.get("num_seen_pokemon", 0),
        "caught": s.get("num_caught_pokemon", 0),
        "party": party,
        "cities": sorted(k for k, v in s["visited_cities"].items() if v),
        "gyms": sorted(k for k, v in s["defeated_gyms"].items() if v),
        "champion": s.get("is_champion", False),
    }


def milestone_of(path: Path) -> int:
    """Sort key from train.py's furthest_<n>maps_<group>_<num>.state naming."""
    m = re.search(r"furthest_(\d+)maps", path.name)
    return int(m.group(1)) if m else 0


def print_report(rows: list[dict]) -> None:
    if not rows:
        print("No savestates found.")
        return

    print(f"{'savestate':<30} {'where':<20} {'badges':>6} {'party':>5} "
          f"{'seen':>5} {'caught':>6} {'money':>7}")
    print("-" * 86)
    for r in rows:
        print(f"{r['file']:<30} {r['map_name']:<20} {r['badges']:>6} "
              f"{len(r['party']):>5} {r['seen']:>5} {r['caught']:>6} {r['money']:>7}")

    for r in rows:
        if not r["party"] and not r["gyms"]:
            continue
        print(f"\n{r['file']}  at {r['map_name']} {r['pos']}")
        for mon in r["party"]:
            print(f"    {mon['species']:<12} lv{mon['level']:<3} "
                  f"{mon['hp']}/{mon['max_hp']} HP")
        if r["gyms"]:
            print(f"    gyms beaten: {', '.join(r['gyms'])}")
        if r["champion"]:
            print("    CHAMPION")

    if len(rows) > 1:
        print("\nEach savestate comes from whichever worker hit that milestone, so these")
        print("are separate trajectories rather than one run. A later milestone with an")
        print("empty party means that worker got further without stopping for a starter.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="checkpoints",
                        help="a .state file, or a directory of them")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        states = sorted(target.glob("*.state"), key=milestone_of)
    elif target.is_file():
        states = [target]
    else:
        raise SystemExit(f"No such path: {target}")

    rows = [read_stats(p) for p in states]

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_report(rows)


if __name__ == "__main__":
    main()
