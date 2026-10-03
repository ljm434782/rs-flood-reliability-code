import pytest

import json
import os

import pytest

from scripts import upgrade_20260929
from scripts.upgrade_20260929 import calibration_jobs, load_config, planned_runs, require_frozen, validate_manifest_lock


def test_draft_upgrade_matrix_has_18_runs_and_locked_nonoverlap_manifests():
    config = load_config()
    assert len(planned_runs(config)) == 18
    assert len(calibration_jobs(config)) == 24
    audit = validate_manifest_lock(config)
    assert audit["counts"] == {"fit": 3525, "calibration": 2117, "internal_retro": 1906, "holdout_retro": 492}
    assert not any(audit["overlap"].values())


@pytest.mark.parametrize("command", ["train", "calibrate", "evaluate", "summarize"])
def test_draft_blocks_all_data_bearing_commands(command):
    config = load_config()
    config["status"] = "DRAFT_NOT_FROZEN"
    with pytest.raises(RuntimeError, match="DRAFT_NOT_FROZEN"):
        require_frozen(config, command)


def test_xpu_budget_sums_serial_device_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrade_20260929, "ROOT", tmp_path)
    registry = tmp_path / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text("\n".join(json.dumps(row) for row in (
        {"event": "START", "pid": 41, "job_id": "SMAGNet_seed_1", "xpu_task": True,
         "smagnet_task": True, "started_utc": "2026-01-01T00:00:00+00:00"},
        {"event": "FINISH", "pid": 41, "job_id": "SMAGNet_seed_1", "xpu_task": True,
         "smagnet_task": True, "elapsed_xpu_hours": 2.5},
        {"event": "START", "pid": 42, "job_id": "B2_seed_1", "xpu_task": True,
         "smagnet_task": False, "started_utc": "2026-01-01T00:00:00+00:00"},
        {"event": "FINISH", "pid": 42, "job_id": "B2_seed_1", "xpu_task": True,
         "smagnet_task": False, "elapsed_xpu_hours": 1.25},
        {"event": "FINISH", "pid": 43, "job_id": "evaluate_B1_internal_retro", "xpu_task": False,
         "elapsed_xpu_hours": 0.0},
    )), encoding="utf-8")
    assert upgrade_20260929._xpu_usage_hours() == {"total": 3.75, "smagnet": 2.5}


def test_xpu_budget_blocks_unresolved_live_child(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrade_20260929, "ROOT", tmp_path)
    registry = tmp_path / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"event": "START", "pid": os.getpid(),
        "job_id": "B2_seed_1", "xpu_task": True, "started_utc": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="still active"):
        upgrade_20260929._xpu_usage_hours()
