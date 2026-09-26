"""Render where the agent has been, as a contact sheet of per-map grids.

One panel per map the agent has entered, each grid sized to the bounding box of
the tiles it actually stood on. Bounding boxes rather than true map dimensions:
the map header holding width/height is not in pygba's address table, and this
project has already lost time to guessed addresses. A bounding box needs no new
data source and still shows exploration spreading outward.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CELL = 4           # pixels per tile
PANEL_GAP = 10     # pixels between panels
LABEL_H = 12       # pixels reserved above each panel for its caption
LABEL_CHAR_W = 6   # approx width of PIL's default font, to size columns
COLS = 4           # panels per row on the contact sheet
# ponytail: a single garbage coordinate from an in-flight save-block read can be
# up to 65535 (pos is uint16). One bad x and one bad y on the same map produced a
# 65528x65526 bounding box, a 12.9GB allocation, and two OOM-killed training runs.
# env.py now only records stable reads; this is the backstop so no input can do
# that again. Ceiling: a genuinely enormous map is cropped rather than drawn whole.
MAX_PANEL_SPAN = 512

BACKGROUND = (18, 18, 22)
UNVISITED = (44, 46, 54)
VISITED = (86, 168, 118)
CURRENT = (240, 196, 92)
LABEL = (188, 192, 200)


def group_by_map(
    tiles: list[tuple[int, int, int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Split flat (mapGroup, mapNum, x, y) records into per-map coordinate lists."""
    by_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for group, num, x, y in tiles:
        by_map.setdefault((group, num), []).append((x, y))
    return by_map


def panel_bounds(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Bounding box as (min_x, min_y, width, height), inclusive of both edges.

    Spans are capped at MAX_PANEL_SPAN so one outlying coordinate cannot turn a
    panel into a multi-gigabyte allocation. See the constant for why.
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = min(max_x - min_x + 1, MAX_PANEL_SPAN)
    h = min(max_y - min_y + 1, MAX_PANEL_SPAN)
    return min_x, min_y, w, h


def render_coverage(
    tiles: list[tuple[int, int, int, int]],
    out_path: str | Path,
    current: tuple[int, int, int, int] | None = None,
) -> Path:
    """Write a PNG contact sheet of per-map coverage. Returns the path written.

    `current` is an optional (mapGroup, mapNum, x, y) marking where the agent
    stands now, drawn in a contrasting colour on its own panel.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_map = group_by_map(tiles)
    if not by_map:
        Image.new("RGB", (CELL * 8, CELL * 8), BACKGROUND).save(out_path)
        return out_path

    panels = []
    labels = {}
    for map_key in sorted(by_map):
        coords = by_map[map_key]
        labels[map_key] = f"{map_key[0]}-{map_key[1]}  {len(coords)} tiles"
        min_x, min_y, w, h = panel_bounds(coords)
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        grid[:, :] = UNVISITED
        for x, y in coords:
            gx, gy = x - min_x, y - min_y
            if 0 <= gy < h and 0 <= gx < w:
                grid[gy, gx] = VISITED
        if current is not None and (current[0], current[1]) == map_key:
            cx, cy = current[2] - min_x, current[3] - min_y
            if 0 <= cy < h and 0 <= cx < w:
                grid[cy, cx] = CURRENT
        panels.append((map_key, grid, len(coords)))

    cols = min(COLS, len(panels))
    rows = (len(panels) + cols - 1) // cols
    col_w = [0] * cols
    row_h = [0] * rows
    for i, (map_key, grid, _) in enumerate(panels):
        r, c = divmod(i, cols)
        # Columns must fit the caption too, or labels bleed into the next panel.
        label_w = len(labels[map_key]) * LABEL_CHAR_W
        col_w[c] = max(col_w[c], grid.shape[1] * CELL, label_w)
        row_h[r] = max(row_h[r], grid.shape[0] * CELL + LABEL_H)

    sheet_w = sum(col_w) + PANEL_GAP * (cols + 1)
    sheet_h = sum(row_h) + PANEL_GAP * (rows + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for i, (map_key, grid, count) in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = PANEL_GAP + sum(col_w[:c]) + PANEL_GAP * c
        y0 = PANEL_GAP + sum(row_h[:r]) + PANEL_GAP * r
        draw.text((x0, y0), labels[map_key], fill=LABEL)
        img = Image.fromarray(grid, "RGB").resize(
            (grid.shape[1] * CELL, grid.shape[0] * CELL), Image.NEAREST
        )
        sheet.paste(img, (x0, y0 + LABEL_H))

    sheet.save(out_path)
    return out_path


def render_to_array(
    tiles: list[tuple[int, int, int, int]],
    current: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Same sheet as an (H, W, 3) uint8 array, for logging to TensorBoard.

    Goes through render_coverage and reads the result back so the PNG on disk
    and the TensorBoard image can never drift apart.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        render_coverage(tiles, f.name, current)
        with Image.open(f.name) as im:
            return np.array(im.convert("RGB"))
