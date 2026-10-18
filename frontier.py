"""Distance to the nearest section the agent has never visited.

Pure function, so the reward term built on it (FRONTIER_REWARD in env.py) is
a potential over something testable without an emulator. Sections are the
8x8 tile blocks the section reward already uses; "visited" is the cumulative
set for the current map.
"""


def nearest_unvisited_distance(section: tuple[int, int], visited: set[tuple[int, int]], cap: int = 8) -> int:
    """Manhattan distance, in sections, from `section` to the nearest section
    not in `visited`, searching rings of increasing distance; `cap` if none is
    found within it."""
    sx, sy = section
    for d in range(cap + 1):
        if d == 0:
            if (sx, sy) not in visited:
                return 0
            continue
        for dx in range(-d, d + 1):
            dy = d - abs(dx)
            if (sx + dx, sy + dy) not in visited or (dy and (sx + dx, sy - dy) not in visited):
                return d
    return cap
