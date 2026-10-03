import json
from scripts import upgrade_20260929 as runner


def test_accelerator_budget_counts_cuda_and_legacy_xpu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "outputs/upgrade_20260929/orchestrator/runs.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"event": "FINISH", "pid": 1, "xpu_task": True, "elapsed_xpu_hours": 1.0},
        {"event": "FINISH", "pid": 2, "xpu_task": False, "accelerator_task": True,
         "device": "cuda", "smagnet_task": True, "elapsed_accelerator_hours": 2.0},
        {"event": "FINISH", "pid": 3, "accelerator_task": True,
         "device": "cuda", "elapsed_accelerator_hours": 0.5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert runner._accelerator_usage_hours() == {"total": 3.5, "smagnet": 2.0}
