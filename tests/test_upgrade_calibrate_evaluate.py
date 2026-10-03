"""Synthetic-only lifecycle tests; no project data readers are constructed."""
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upgrade_calibrate_evaluate.py"
_SPEC = importlib.util.spec_from_file_location("upgrade_calibrate_evaluate", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

(
    STATES, TilePrediction, calibration_cells, evaluate_matrix,
    fit_and_record_temperature, manifest_owner_masks,
    primary_training_calibration_interaction, run_real_after_freeze,
    combine_evaluation_results, deterministic_block_mask,
    classify_holdout_strata, fit_global_temperature_streaming,
    evaluation_output_path,
    write_matrix_summaries,
) = tuple(getattr(_MODULE, name) for name in (
    "STATES", "TilePrediction", "calibration_cells", "evaluate_matrix",
    "fit_and_record_temperature", "manifest_owner_masks",
    "primary_training_calibration_interaction", "run_real_after_freeze",
    "combine_evaluation_results", "deterministic_block_mask",
    "classify_holdout_strata", "fit_global_temperature_streaming", "evaluation_output_path",
    "write_matrix_summaries"))
irregular_retention_mask = _MODULE.irregular_retention_mask


def _tiles():
    result = []
    # p2 intentionally has no flood truth so its AUPRC must stay undefined.
    for process, labels in (("p1", np.array([[0, 2], [255, 1]], dtype=np.uint8)),
                            ("p2", np.array([[0, 0], [255, 0]], dtype=np.uint8))):
        owner = np.array([[True, True], [True, False]])
        for state_index, state in enumerate(STATES):
            logits = np.array([
                [[2.0, -1.0], [0.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0]],
                [[-1.0, 2.0], [0.0, -1.0]],
            ], dtype=np.float64)
            # State shift changes confidence while preserving a nontrivial fit.
            logits *= 0.8 + state_index * 0.1
            result.append(TilePrediction(process, f"{process}_{state}", state,
                                         logits, labels.copy(), owner.copy()))
    return result


def test_synthetic_calibration_to_fixed_evaluation_end_to_end():
    all_tiles = _tiles()
    calibration = {}
    for strategy in ("original", "multistate"):
        selected = [tile for tile in all_tiles if strategy == "multistate" or tile.state_id == "100"]
        record = fit_and_record_temperature(selected, model="B5_whole", seed=20261013,
                                            strategy=strategy, hashes={"fixture": "synthetic"})
        assert record["temperature"] > 0
        assert record["optimizer_success"]
        assert record["process_ids"] == ["p1", "p2"]
        assert record["state_ids"] == (list(STATES) if strategy == "multistate" else ["100"])
        calibration[("B5_whole", 20261013, strategy)] = record["temperature"]

    # One fixed model/seed is evaluated under each independently-fitted strategy.
    matrix = [("B5_whole", 20261013, strategy, tile)
              for strategy in ("original", "multistate") for tile in all_tiles]
    result = evaluate_matrix(matrix, calibration, evaluation_role="internal_retro",
                             bootstrap_repetitions=100, bootstrap_seed=91)
    assert result["temperature_argmax_invariant"] is True
    assert result["interpretation"].startswith("retrospective descriptive")
    raw = result["per_process"]["B5_whole|20261013|original|p1|100|raw"]
    calibrated = result["per_process"]["B5_whole|20261013|original|p1|100|calibrated"]
    assert raw["pixels"] == calibrated["pixels"] == 2  # Ignore and unowned excluded
    np.testing.assert_array_equal(raw["confusion"], calibrated["confusion"])
    assert result["per_process"]["B5_whole|20261013|original|p2|100|calibrated"]["flood_auprc"] is None
    summary = result["summary"]["B5_whole|20261013|original|100|raw|binary_flood_brier"]
    assert summary["process_ids"] == ["p1", "p2"]
    assert summary["n_processes"] == 2
    assert summary["ci95"][0] <= summary["mean"] <= summary["ci95"][1]


def test_calibration_requires_complete_process_state_grid_and_owner_pixels():
    tiles = _tiles()
    with pytest.raises(ValueError, match="missing calibration process×state"):
        calibration_cells([t for t in tiles if not (t.process_id == "p2" and t.state_id == "0")],
                          strategy="multistate")
    bad = TilePrediction("p", "s", "100", np.zeros((3, 1, 1)),
                         np.array([[0]], dtype=np.uint8), np.array([[False]]))
    with pytest.raises(ValueError, match="empty owned"):
        fit_and_record_temperature([bad], model="B5", seed=1, strategy="original")


def test_manifest_owner_masks_are_process_local_and_nonduplicating(tmp_path):
    manifest = tmp_path / "role.csv"
    grids = tmp_path / "grid.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "derived_role", "conservative_unit_id"])
        writer.writeheader()
        writer.writerow({"sample_id": "a", "derived_role": "calibration", "conservative_unit_id": "p"})
        writer.writerow({"sample_id": "b", "derived_role": "calibration", "conservative_unit_id": "p"})
    with grids.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "id_zone_matches_grid", "grid_crs", "left", "bottom", "right", "top"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for sid in ("a", "b"):
            writer.writerow({"sample_id": sid, "id_zone_matches_grid": "1", "grid_crs": "EPSG:32631",
                             "left": 0, "bottom": 0, "right": 256, "top": 256})
    owners = manifest_owner_masks(manifest, grids, expected_role="calibration")
    assert owners["a"][0] == owners["b"][0] == "p"
    assert owners["a"][1].all()
    assert not owners["b"][1].any()


def test_invalid_label_class_and_nonpositive_temperature_fail():
    bad = TilePrediction("p", "s", "100", np.zeros((3, 1, 1)),
                         np.array([[3]], dtype=np.uint8), np.array([[True]]))
    with pytest.raises(ValueError, match="only 0, 1, 2"):
        calibration_cells([bad], strategy="original")


def test_real_reader_factory_is_never_called_before_dual_freeze(tmp_path):
    constructed = []

    def reader_factory(_hashes):
        constructed.append(True)
        return object()

    with pytest.raises(RuntimeError, match="not FROZEN"):
        run_real_after_freeze(tmp_path, {"status": "DRAFT_NOT_FROZEN"}, reader_factory,
                              lambda *_: None)
    assert constructed == []


def test_nested_fixed_masks_and_role_safe_result_combining():
    valid = np.ones((256, 256), dtype=np.bool_)
    keep100 = deterministic_block_mask(valid, "tile", "100", 7, purpose="calibration")
    keep75 = deterministic_block_mask(valid, "tile", "75", 7, purpose="calibration")
    keep0 = deterministic_block_mask(valid, "tile", "0", 7, purpose="evaluation")
    assert keep100.all() and not keep0.any()
    assert np.all(~keep75 | keep100)
    assert keep75.sum() == 48 * 32 * 32
    assert np.array_equal(keep75, deterministic_block_mask(valid, "tile", "75", 7,
                                                            purpose="calibration"))
    assert len(combine_evaluation_results([{"role": "internal_retro", "per_process": {"a": 1}},
                                           {"role": "internal_retro", "per_process": {"b": 2}}])["per_process"]) == 2
    with pytest.raises(ValueError, match="roles must remain separate"):
        combine_evaluation_results([{"role": "internal_retro", "per_process": {}},
                                   {"role": "holdout_retro", "per_process": {}}])


def test_holdout_metadata_strata_are_joined_before_prediction_and_separate():
    candidates = [
        {"sample_id": "pair", "derived_role": "holdout_retro"},
        {"sample_id": "low", "derived_role": "holdout_retro"},
        {"sample_id": "absent", "derived_role": "holdout_retro"},
    ]
    eligibility = [
        {"sample_id": "pair", "paired_eligible": "1", "selected_time_index": "2",
         "selected_s2_utc": "2020-01-01T00:00:00Z", "selected_age_days": "2.0",
         "selected_joint_visible_fraction": "0.8", "failure_reasons": ""},
        {"sample_id": "low", "paired_eligible": "0", "selected_time_index": "1",
         "selected_s2_utc": "2020-01-01T00:00:00Z", "selected_age_days": "4.0",
         "selected_joint_visible_fraction": "0.3", "failure_reasons": "visible_below_0.5"},
        {"sample_id": "absent", "paired_eligible": "0", "selected_time_index": "",
         "selected_s2_utc": "", "selected_age_days": "",
         "selected_joint_visible_fraction": "", "failure_reasons": "no_pre_s1_s2_within_30d"},
    ]
    anchors = [{"sample_id": "absent", "s1_event_utc_from_zattrs": "2021-12-21T09:48:59Z"}]
    groups = classify_holdout_strata(candidates, eligibility, anchors,
                                     expected_counts={"paired_eligible": 1, "low_visible": 1,
                                                      "no_qualifying_s2": 1})
    assert groups["paired_eligible"][0]["selected_time_index"] == "2"
    assert groups["low_visible"][0]["selected_joint_visible_fraction"] == "0.3"
    assert groups["no_qualifying_s2"][0]["selected_time_index"] == ""


def test_streaming_temperature_matches_reference_without_float64_cache():
    cells = []
    for process, count in (("p1", 2), ("p2", 5)):
        for state_index, state in enumerate(STATES):
            logits = np.tile(np.array([[1.0 + state_index * .1, 0.0, -1.0]], np.float32),
                             (count, 1))
            labels = np.zeros(count, np.uint8)
            cells.append((process, state, logits, labels))
    streamed = fit_global_temperature_streaming(
        cells, model="B5", seed=7, strategy="multistate",
        pixel_counts={f"{p}|{s}": len(y) for p, s, _, y in cells}, hashes={})
    metric_cells = [_MODULE.CalibrationCell(p, s, z, y, np.ones(len(y), bool))
                    for p, s, z, y in cells]
    reference = _MODULE.fit_global_temperature(metric_cells, strategy="multistate")
    assert streamed["temperature"] == pytest.approx(reference.temperature, abs=1e-8)
    assert streamed["cache_dtype"] == "float32 logits + uint8 labels; owner/Ignore filtered during tile generation"
    assert "one float64 cell temporary" in streamed["memory_policy"]


def test_strategy_specific_evaluation_paths_do_not_overwrite(tmp_path):
    original = evaluation_output_path(tmp_path, "internal_retro", "original")
    multistate = evaluation_output_path(tmp_path, "internal_retro", "multistate")
    assert original.name == "evaluation_internal_retro_original.json"
    assert multistate.name == "evaluation_internal_retro_multistate.json"
    assert original != multistate
    original.write_text("{}", encoding="utf-8")
    multistate.write_text("{}", encoding="utf-8")
    assert original.is_file() and multistate.is_file()


def test_irregular_masks_are_deterministic_nested_synthetic_masks():
    valid = np.zeros((256, 256), dtype=np.bool_)
    valid[20:230, 17:241] = True
    keep75 = irregular_retention_mask(valid, "synthetic_tile", .75)
    keep50 = irregular_retention_mask(valid, "synthetic_tile", .5)
    keep25 = irregular_retention_mask(valid, "synthetic_tile", .25)
    assert np.all(~keep50 | keep75)
    assert np.all(~keep25 | keep50)
    assert keep75.sum() == int(valid.sum() - np.floor(valid.sum() * .25))
    assert keep50.sum() == int(valid.sum() - np.floor(valid.sum() * .5))
    assert keep25.sum() == int(valid.sum() - np.floor(valid.sum() * .75))


def test_synthetic_matrix_summary_writes_internal_primary_and_separate_holdout(tmp_path, monkeypatch):
    import hashlib
    import json

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "upgrade_20260929.json"
    config_path.write_text('{"fixture":true}\n', encoding="utf-8")
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    manifest = manifests / "internal.csv"
    manifest.write_text("sample_id,conservative_unit_id\ntile,p1\n", encoding="utf-8")
    thresholds = tmp_path / "outputs" / "upgrade_20260929" / "B1" / "thresholds.json"
    thresholds.parent.mkdir(parents=True)
    thresholds.write_text(json.dumps({"model": "B1", "status": "FIT_ROLE_ONLY_COMPLETE",
        "fit_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "fit_script_sha256": "synthetic-fit-script-hash",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}), encoding="utf-8")
    threshold_hash = hashlib.sha256(thresholds.read_bytes()).hexdigest()
    config = {
        "seeds": [1],
        "models": {
            "B5_whole": {"calibration": ["original", "multistate"]},
            "B5_local": {"calibration": ["original", "multistate"]},
        },
        "manifests": {"fit": {"path": "data/manifests/internal.csv",
                               "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                      "internal_retro": {"path": "data/manifests/internal.csv",
                                         "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}},
        "fixed_baselines": {"B1": {"thresholds_path": "outputs/upgrade_20260929/B1/thresholds.json",
                                     "fit_script_sha256": "synthetic-fit-script-hash"}},
    }
    locked = {"fixture-manifest": "locked"}
    monkeypatch.setattr(_MODULE, "require_real_access", lambda *_: locked)
    all_metrics = ("binary_flood_brier", "flood_iou", "flood_precision", "flood_recall",
                   "flood_f1", "flood_auprc", "macro_iou", "multiclass_nll",
                   "multiclass_brier", "ece15")
    for model in config["models"]:
        run_dir = tmp_path / "outputs" / "upgrade_20260929" / f"{model}_seed_1"
        run_dir.mkdir(parents=True)
        weight = run_dir / "model_only.pt"
        weight.write_bytes(b"synthetic-model-weight")
        weight_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
        (run_dir / "run.json").write_text(json.dumps({"status": "FIT_COMPLETE", "model": model,
            "seed": 1, "model_only_sha256": weight_hash}), encoding="utf-8")
        for strategy in ("original", "multistate"):
            temp_path = run_dir / f"temperature_{strategy}.json"
            temp_path.write_text(json.dumps({"model": model, "seed": 1,
                "calibration_strategy": strategy, "checkpoint_sha256": weight_hash}), encoding="utf-8")
            temp_hash = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            per_process = {}
            internal_summary, holdout_summary = {}, {}
            for state in (*STATES, "irregular_75", "irregular_50", "irregular_25"):
                for tag in ("raw", "calibrated"):
                    key = f"{model}|1|{strategy}|p1|{state}|{tag}"
                    per_process[key] = {"binary_flood_brier": .2, "pixels": 3}
                    group = "paired_eligible"
                    hkey = f"{group}|{state}|{tag}"
                    for metric in all_metrics:
                        holdout_summary[f"{hkey}|{metric}"] = {"n_processes": 1,
                            "process_ids": ["p1"], "mean": .2, "ci95": [.1, .3]}
                    for metric in all_metrics:
                        internal_summary[f"{model}|1|{strategy}|{state}|{tag}|{metric}"] = {
                            "n_processes": 1, "process_ids": ["p1"], "mean": .2, "ci95": [.1, .3]}
            for group, state in (("low_visible", "observed"),
                                 ("no_qualifying_s2", "observed_absent")):
                for tag in ("raw", "calibrated"):
                    for metric in all_metrics:
                        holdout_summary[f"{group}|{state}|{tag}|{metric}"] = {
                            "n_processes": 1, "process_ids": ["p1"], "mean": .3, "ci95": [.2, .4]}
            for role in ("internal_retro", "holdout_retro"):
                record = {"role": role, "model": model, "seed": 1,
                          "calibration_strategy": strategy, "checkpoint_sha256": weight_hash,
                          "temperature_sha256": temp_hash, "locked_input_hashes": locked,
                          "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                          "per_process": per_process,
                          "summary": internal_summary if role == "internal_retro" else holdout_summary}
                path = evaluation_output_path(run_dir, role, strategy)
                path.write_text(json.dumps(record), encoding="utf-8")
    b1_dir = tmp_path / "outputs" / "upgrade_20260929" / "B1"
    b1_dir.mkdir(parents=True, exist_ok=True)
    for role, counts in (("internal_retro", {"internal_retro": 1906}),
                         ("holdout_retro", {"paired_eligible": 432, "low_visible": 49,
                                             "no_qualifying_s2": 11})):
        strata = {name: {"processes": {"p1": {"flood_iou": .5}},
                         "summary": {"flood_iou": {"n_processes": 1,
                             "process_ids": ["p1"], "mean": .5, "ci95": [.2, .8]}}}
                  for name in counts}
        path = b1_dir / f"evaluation_{role}.json"
        path.write_text(json.dumps({"model": "B1", "role": role,
            "interpretation": "retrospective descriptive only",
            "thresholds_sha256": threshold_hash,
            "fit_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "strata_counts": counts,
            "results": strata}), encoding="utf-8")
    audit = write_matrix_summaries(tmp_path, config)
    out_dir = tmp_path / "outputs" / "upgrade_20260929" / "summaries"
    primary = json.loads((out_dir / "primary_interaction_internal_retro.json").read_text(encoding="utf-8"))
    assert primary["role"] == "internal_retro"
    assert primary["process_ids"] == ["p1"]
    assert (out_dir / "internal_metric_summary.csv").is_file()
    assert (out_dir / "holdout_stratified_summary.csv").is_file()
    assert (out_dir / "b1_hard_metrics.csv").is_file()
    assert audit["roles_separate"] is True
    assert "evaluation_holdout_retro_original.json" in "\n".join(audit["input_file_hashes"])


def test_primary_interaction_uses_common_process_support_and_paired_bootstrap():
    tiles = _tiles()
    temperatures = {}
    matrix = []
    arms = {"A": ("B5_whole", "original"), "B": ("B5_whole", "multistate"),
            "C": ("B5_local", "original"), "D": ("B5_local", "multistate")}
    for seed in (101, 102, 103):
        for arm, (model, strategy) in arms.items():
            temperatures[(model, seed, strategy)] = 1.2 if strategy == "multistate" else 1.0
            for tile in tiles:
                # Use arm-specific deterministic shifts to create measurable
                # (but synthetic) differences without changing truth/support.
                logits = tile.logits.copy()
                if arm in ("C", "D"):
                    logits[2] += 0.15
                if arm in ("B", "D"):
                    logits *= 0.9
                changed = TilePrediction(tile.process_id, f"{seed}_{tile.sample_id}",
                                         tile.state_id, logits, tile.labels, tile.owner_mask)
                matrix.append((model, seed, strategy, changed))
    scored = evaluate_matrix(matrix, temperatures, evaluation_role="internal_retro",
                             bootstrap_repetitions=80, bootstrap_seed=20261029)
    with pytest.raises(ValueError, match="only for internal_retro"):
        primary_training_calibration_interaction(
            {**scored, "role": "holdout_retro"}, arm_keys=arms,
            seeds=(101, 102, 103), bootstrap_repetitions=30)
    result = primary_training_calibration_interaction(
        scored, arm_keys=arms, seeds=(101, 102, 103),
        bootstrap_repetitions=80, bootstrap_seed=20261029)
    assert result["status"] == "defined"
    assert result["process_ids"] == ["p1", "p2"]
    assert set(result["per_seed"]) == {"101", "102", "103"}
    assert result["bootstrap"]["paired_arms_and_seeds"] is True
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]
    # A missing cell in any arm/state/seed removes that process from every arm.
    one_missing = dict(scored)
    one_missing["per_process"] = dict(scored["per_process"])
    del one_missing["per_process"]["B5_local|102|multistate|p2|0|calibrated"]
    excluded = primary_training_calibration_interaction(
        one_missing, arm_keys=arms, seeds=(101, 102, 103), bootstrap_repetitions=30)
    assert excluded["process_ids"] == ["p1"]
    assert "p2" in excluded["excluded_processes"]
