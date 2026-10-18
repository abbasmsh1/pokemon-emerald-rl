"""Checks for the frontier distance. Pure function, no emulator."""

from frontier import nearest_unvisited_distance


def test_unvisited_neighbour_is_distance_one():
    assert nearest_unvisited_distance((5, 5), {(5, 5)}) == 1
    print("test_unvisited_neighbour_is_distance_one PASSED")


def test_distance_counts_rings():
    visited = {(x, y) for x in range(3, 8) for y in range(3, 8)}   # 5x5 block around (5, 5)
    assert nearest_unvisited_distance((5, 5), visited) == 3
    print("test_distance_counts_rings PASSED")


def test_cap_when_everything_nearby_is_visited():
    visited = {(x, y) for x in range(-10, 21) for y in range(-10, 21)}
    assert nearest_unvisited_distance((5, 5), visited, cap=8) == 8
    print("test_cap_when_everything_nearby_is_visited PASSED")


def test_current_section_unvisited_is_zero():
    assert nearest_unvisited_distance((0, 0), set()) == 0
    print("test_current_section_unvisited_is_zero PASSED")


if __name__ == "__main__":
    test_unvisited_neighbour_is_distance_one()
    test_distance_counts_rings()
    test_cap_when_everything_nearby_is_visited()
    test_current_section_unvisited_is_zero()
    print("\nall frontier tests passed")
