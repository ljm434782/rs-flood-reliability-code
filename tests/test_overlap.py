import numpy as np

from flood_reliability.overlap import unique_pixel_mask


def grid(left, bottom, crs="EPSG:32632"):
    return {"grid_crs": crs, "left": left, "bottom": bottom,
            "right": left + 40, "top": bottom + 40}


def test_overlapping_lattice_counts_each_center_once():
    first = grid(0, 0)
    second = grid(20, 10)
    mask = unique_pixel_mask(second, [first], shape=(4, 4))
    assert mask.sum() == 10
    assert np.array_equal(mask[:, :2], np.array([
        [True, True], [False, False], [False, False], [False, False]]))


def test_crs_or_half_pixel_shift_does_not_claim_identical_centers():
    target = grid(20, 10)
    assert unique_pixel_mask(target, [grid(0, 0, "EPSG:32633")], shape=(4, 4)).all()
    assert unique_pixel_mask(target, [grid(5, 0)], shape=(4, 4)).all()


def test_geographic_non_square_degree_pixels_can_be_deduplicated():
    first = {"grid_crs": "EPSG:4326", "left": 10., "right": 10.04,
             "bottom": 1., "top": 1.08}
    second = {"grid_crs": "EPSG:4326", "left": 10.02, "right": 10.06,
              "bottom": 1.02, "top": 1.10}
    mask = unique_pixel_mask(second, [first], shape=(4, 4))
    assert mask.sum() == 10
