import numpy as np
import pytest
from scipy.ndimage import label

from flood_reliability.upgrade_masks import (
    IMAGE_SHAPE,
    apply_evaluation_mask,
    apply_train_local_exposure,
    block_retention_mask,
    irregular_retention_mask,
    train_local_plan,
    whole_dropout_decision,
)


def first_applied_plan():
    for index in range(10_000):
        plan = train_local_plan(20261013, 7, f"synthetic_{index:05d}", whole_dropped=False)
        if plan.apply:
            return f"synthetic_{index:05d}", plan
    raise AssertionError("deterministic synthetic stream did not apply a local mask")


def test_block_masks_are_deterministic_nested_and_respect_validity():
    valid = np.ones(IMAGE_SHAPE, dtype=bool)
    valid[:13, :] = False
    valid[:, :7] = False
    masks = [block_retention_mask(valid, "synthetic_tile", level, 20261013,
                                  namespace="eval-block-v1")
             for level in (0, 25, 50, 75, 100)]
    assert not masks[0].any()
    assert all(np.all(~left | right) for left, right in zip(masks, masks[1:]))
    assert all(np.all(~mask | valid) for mask in masks)
    np.testing.assert_array_equal(
        masks[3], block_retention_mask(valid, "synthetic_tile", 75, 20261013,
                                       namespace="eval-block-v1"),
    )
    full_valid = np.ones(IMAGE_SHAPE, dtype=bool)
    assert block_retention_mask(full_valid, "synthetic_tile", 25, 20261013,
                                namespace="cal-state-v1").sum() == 16 * 32 * 32
    assert not np.array_equal(
        block_retention_mask(full_valid, "synthetic_tile", 25, 20261013,
                             namespace="cal-state-v1"),
        block_retention_mask(full_valid, "synthetic_tile", 25, 20261013,
                             namespace="eval-block-v1"),
    )


def test_train_local_rectangle_is_connected_exact_and_deterministic():
    sample_id, plan = first_applied_plan()
    repeat = train_local_plan(20261013, 7, sample_id, whole_dropped=False)
    assert plan.apply and repeat.apply
    np.testing.assert_array_equal(plan.block_mask, repeat.block_mask)
    assert plan.target_blocks in (16, 32, 48)
    assert plan.block_mask.sum() == plan.target_blocks * 32 * 32
    components, count = label(plan.block_mask)
    assert count == 1 and components.max() == 1
    rows, cols = np.where(plan.block_mask)
    assert (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) == plan.block_mask.sum()
    assert plan.grid_shape[0] * plan.grid_shape[1] == plan.target_blocks


def test_train_local_named_stream_has_declared_balanced_draws_on_synthetic_ids():
    plans = [train_local_plan(20261013, 8, f"stream_{index:05d}", whole_dropped=False)
             for index in range(6_000)]
    applied = [plan for plan in plans if plan.apply]
    assert 0.45 <= len(applied) / len(plans) <= 0.55
    frequencies = {area: sum(plan.target_blocks == area for plan in applied) / len(applied)
                   for area in (16, 32, 48)}
    assert all(0.28 <= value <= 0.39 for value in frequencies.values())
    assert not train_local_plan(20261013, 8, "stream_00000", whole_dropped=True).apply


def test_whole_dropout_is_a_shared_named_p025_decision_and_preempts_local_draw():
    decisions = [whole_dropout_decision(20261013, 8, f"whole_{index:05d}")
                 for index in range(6_000)]
    assert 0.22 <= sum(decisions) / len(decisions) <= 0.28
    shared = [whole_dropout_decision(20261013, 8, "shared_sample")
              for _candidate in ("B5_A", "B5_C", "P", "SMAGNet")]
    assert shared == [shared[0]] * 4
    sample_id = next(f"whole_{index:05d}" for index in range(6_000)
                     if whole_dropout_decision(20261013, 8, f"whole_{index:05d}"))
    plan = train_local_plan(20261013, 8, sample_id, whole_dropped=True)
    assert not plan.apply and plan.target_blocks is None and not plan.block_mask.any()


def test_train_local_application_only_removes_validity_and_zeroes_rectangle():
    sample_id, plan = first_applied_plan()
    optical = np.arange(12 * 256 * 256, dtype=np.float32).reshape(12, 256, 256) + 1
    valid = np.ones(IMAGE_SHAPE, dtype=bool)
    valid[:5, :] = False
    result = apply_train_local_exposure(
        optical, valid, present=True, time_offset=-0.25, seed=20261013, epoch=7,
        sample_id=sample_id, whole_dropped=False,
    )
    assert result.present and result.time_offset == -0.25
    assert np.all(result.optical[:, plan.block_mask] == 0)
    assert not result.valid[plan.block_mask].any()
    unaffected = ~plan.block_mask
    np.testing.assert_array_equal(result.optical[:, unaffected], optical[:, unaffected])
    np.testing.assert_array_equal(result.valid, valid & ~plan.block_mask)
    np.testing.assert_array_equal(optical, np.arange(12 * 256 * 256, dtype=np.float32).reshape(12, 256, 256) + 1)


def test_whole_dropout_preempts_local_and_zeroes_everything():
    optical = np.ones((12, 256, 256), dtype=np.float32)
    valid = np.ones(IMAGE_SHAPE, dtype=bool)
    result = apply_train_local_exposure(
        optical, valid, present=True, time_offset=-0.25, seed=20261013, epoch=4,
        sample_id="whole_case", whole_dropped=True,
    )
    assert not result.present and result.time_offset == 0.0
    assert not result.valid.any() and not result.optical.any()
    assert not result.local_plan.apply


def test_evaluation_application_zeroes_only_unretained_and_handles_zero_percent():
    optical = np.ones((12, 256, 256), dtype=np.float32)
    valid = np.ones(IMAGE_SHAPE, dtype=bool)
    retained = block_retention_mask(valid, "eval_case", 50, 20261013, namespace="eval-block-v1")
    result = apply_evaluation_mask(optical, retained, time_offset=-0.5)
    assert result.present and result.time_offset == -0.5
    assert np.all(result.optical[:, retained] == 1)
    assert not result.optical[:, ~retained].any()
    zero = apply_evaluation_mask(optical, np.zeros(IMAGE_SHAPE, dtype=bool), time_offset=-0.5)
    assert not zero.present and zero.time_offset == 0.0 and not zero.optical.any()


def test_nonzero_nominal_block_level_with_empty_sparse_intersection_is_absent():
    optical = np.ones((12, 256, 256), dtype=np.float32)
    valid = np.zeros(IMAGE_SHAPE, dtype=bool)
    valid[0, 0] = True
    sample_id = next(
        f"sparse_{index:05d}" for index in range(1_000)
        if not block_retention_mask(valid, f"sparse_{index:05d}", 75, 20261013,
                                    namespace="eval-block-v1").any()
    )
    retained = block_retention_mask(valid, sample_id, 75, 20261013, namespace="eval-block-v1")
    assert not retained.any()  # nominal 75%, but its selected blocks miss the sole valid pixel
    result = apply_evaluation_mask(optical, retained, time_offset=-0.5)
    assert not result.present and result.time_offset == 0.0 and not result.optical.any()


def test_irregular_mask_is_deterministic_nested_floor_count_and_never_revives_invalid_pixels():
    valid = np.ones(IMAGE_SHAPE, dtype=bool)
    valid[:16, :16] = False  # remaining valid count stays divisible by four
    retained = irregular_retention_mask(valid, "irregular_case", 0.5)
    repeated = irregular_retention_mask(valid, "irregular_case", 0.5)
    np.testing.assert_array_equal(retained, repeated)
    assert retained.sum() == valid.sum() // 2
    assert np.all(~retained | valid)
    ambiguous = np.ones(IMAGE_SHAPE, dtype=bool)
    ambiguous[0, 0] = False
    masks = [irregular_retention_mask(ambiguous, "ambiguous_count", level)
             for level in (0.25, 0.5, 0.75, 1.0)]
    expected = [ambiguous.sum() - int(np.floor((1 - level) * ambiguous.sum()))
                for level in (0.25, 0.5, 0.75, 1.0)]
    assert [int(mask.sum()) for mask in masks] == expected
    assert all(np.all(~left | right) for left, right in zip(masks, masks[1:]))
