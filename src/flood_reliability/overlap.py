"""Assign one owner to identical projected pixel centers in overlapping tiles."""

import numpy as np


def unique_pixel_mask(current: dict, preceding: list[dict], shape=(256, 256)) -> np.ndarray:
    """Pixel centers already covered by a lower-ID tile in the process are False."""
    height, width = shape
    left, bottom = float(current["left"]), float(current["bottom"])
    right, top = float(current["right"]), float(current["top"])
    dx, dy = (right - left) / width, (top - bottom) / height
    if dx <= 0 or dy <= 0:
        raise ValueError("Expected north-up grid with positive pixel spacing")
    xs = left + (np.arange(width) + .5) * dx
    ys = top - (np.arange(height) + .5) * dy
    owned = np.ones(shape, dtype=bool)
    for old in preceding:
        if old["grid_crs"] != current["grid_crs"]:
            continue
        old_left, old_bottom = float(old["left"]), float(old["bottom"])
        old_right, old_top = float(old["right"]), float(old["top"])
        old_dx, old_dy = (old_right - old_left) / width, (old_top - old_bottom) / height
        if not (np.isclose(dx, old_dx, atol=max(abs(dx) * 1e-9, 1e-12), rtol=0)
                and np.isclose(dy, old_dy, atol=max(abs(dy) * 1e-9, 1e-12), rtol=0)):
            continue
        # Identical center coordinates require aligned lattice phases.
        shift_x, shift_y = (left - old_left) / dx, (top - old_top) / dy
        if (abs(shift_x - round(shift_x)) > 1e-6
                or abs(shift_y - round(shift_y)) > 1e-6):
            continue
        old_x = (xs >= old_left) & (xs < old_right)
        old_y = (ys > old_bottom) & (ys <= old_top)
        owned[np.ix_(old_y, old_x)] = False
    return owned
