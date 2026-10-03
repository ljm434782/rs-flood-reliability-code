"""Fail-closed coordinator for the 2026-09-29 upgrade matrix.

All data-bearing commands require the immutable G3 freeze manifest and a
matching independent PASS attestation before constructing project readers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(ROOT_SRC.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC.parent))
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from flood_reliability.upgrade_execution import require_dual_freeze


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "upgrade_20260929.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def planned_runs(config: dict) -> list[dict]:
    return [{"model": model, "seed": seed, "run_id": f"{model}_seed_{seed}"}
            for model in config["models"] for seed in config["seeds"]]


def _ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def validate_manifest_lock(config: dict) -> dict:
    ids = {}
    hashes = {}
    for role, locked in config["manifests"].items():
        path = ROOT / locked["path"]
        actual = sha256(path)
        if actual != locked["sha256"]:
            raise RuntimeError(f"Manifest hash mismatch for {role}: {actual}")
        ids[role] = _ids(path)
        hashes[role] = actual
    roles = list(ids)
    overlap = {f"{left}__{right}": len(ids[left] & ids[right])
               for index, left in enumerate(roles) for right in roles[index + 1:]}
    if any(overlap.values()):
        raise RuntimeError(f"Cross-role sample-ID overlap: {overlap}")
    fit_rows = _ids(ROOT / config["manifests"]["fit"]["path"])
    if len(fit_rows) != 3525:
        raise RuntimeError(f"Expected 3525 fit IDs, got {len(fit_rows)}")
    return {"hashes": hashes, "counts": {role: len(value) for role, value in ids.items()},
            "overlap": overlap}


def validate_input_hashes(config: dict) -> dict:
    """Verify every metadata-only input declared by the candidate freeze config."""
    entries = {}
    for section in ("manifests", "role_grids", "auxiliary_metadata"):
        for item in config.get(section, {}).values():
            entries[item["path"]] = item["sha256"]
            if item.get("source") and item.get("source_sha256"):
                entries[item["source"]] = item["source_sha256"]
    data_source = config.get("data_source", {})
    for item in (data_source.get("publisher_asset_manifest", {}), data_source.get("archive_audit", {})):
        if item.get("path") and item.get("sha256"):
            entries[item["path"]] = item["sha256"]
    for item in config.get("fixed_baselines", {}).values():
        if item.get("fit_script_path") and item.get("fit_script_sha256"):
            entries[item["fit_script_path"]] = item["fit_script_sha256"]
    observed = {}
    for relative, expected in sorted(entries.items()):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Frozen metadata/code input is missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen metadata/code input hash mismatch for {relative}: {actual}")
        observed[relative] = actual
    return observed


def require_frozen(config: dict, command: str) -> None:
    if config["status"] != "FROZEN":
        raise RuntimeError(f"{command} is blocked: config status is {config['status']}")
    missing = [name for name in config["freeze_required_files"] if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"{command} is blocked: missing freeze evidence {missing}")


def calibration_jobs(config: dict) -> list[dict]:
    return [{"model": model, "seed": seed, "strategy": strategy,
             "job_id": f"{model}_seed_{seed}_{strategy}"}
            for model, settings in config["models"].items()
            for seed in config["seeds"]
            for strategy in settings["calibration"]]


def _run_child(command: list[str], *, job_id: str, log_path: Path) -> dict:
    """Run one serial child, logging the PID and exit code for restart audit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    accelerator_device = command[command.index("--device") + 1] if "--device" in command else None
    xpu_task = accelerator_device in {"xpu", "cuda"}
    smagnet_task = xpu_task and ("SMAGNet" in job_id)
    usage = _accelerator_usage_hours()
    total_cap = float(load_config()["resources"]["total_stop_xpu_hours"])
    smagnet_cap = float(load_config()["resources"]["smagnet_stop_xpu_hours"])
    other_cap = float(load_config()["resources"]["other_stop_xpu_hours"])
    registry = ROOT / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl"
    attempts = sum(1 for line in registry.read_text(encoding="utf-8").splitlines()
                   if line.strip() and (entry := json.loads(line)).get("event") == "START"
                   and entry.get("job_id") == job_id) if registry.exists() else 0
    if attempts >= 2:
        raise RuntimeError(f"Frozen one-retry limit reached for {job_id}")
    if xpu_task and usage["total"] >= total_cap:
        raise RuntimeError(f"Total accelerator-hour stop cap already reached ({usage['total']:.3f}/{total_cap})")
    if smagnet_task and usage["smagnet"] >= smagnet_cap:
        raise RuntimeError(f"SMAGNet accelerator-hour stop cap already reached ({usage['smagnet']:.3f}/{smagnet_cap})")
    if xpu_task and not smagnet_task and usage["total"] - usage["smagnet"] >= other_cap:
        raise RuntimeError("Non-SMAGNet accelerator-hour stop cap already reached")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"event": "START", "job_id": job_id,
                              "started_utc": started, "command": command}) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=creationflags)
        atomic_append_jsonl(ROOT / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl",
                            {"event": "START", "job_id": job_id, "pid": process.pid,
                             "started_utc": started, "command": command,
                             "xpu_task": accelerator_device == "xpu", "accelerator_task": xpu_task,
                             "device": accelerator_device, "smagnet_task": smagnet_task,
                             "log": str(log_path.relative_to(ROOT))})
        stop_reason = None
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds() / 3600
                if xpu_task and usage["total"] + elapsed >= total_cap:
                    stop_reason = "total accelerator-hour cap"
                elif smagnet_task and usage["smagnet"] + elapsed >= smagnet_cap:
                    stop_reason = "SMAGNet accelerator-hour cap"
                elif xpu_task and not smagnet_task and usage["total"] - usage["smagnet"] + elapsed >= other_cap:
                    stop_reason = "non-SMAGNet accelerator-hour cap"
                if stop_reason:
                    process.terminate()
                    code = process.wait(timeout=30)
                    break
        finished = datetime.now(timezone.utc).isoformat()
        result = {"event": "FINISH", "job_id": job_id, "pid": process.pid,
                  "started_utc": started, "finished_utc": finished,
                  "exit_code": code, "xpu_task": accelerator_device == "xpu",
                  "accelerator_task": xpu_task, "device": accelerator_device,
                  "stop_reason": stop_reason,
                  "elapsed_xpu_hours": ((datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds() / 3600
                                         if accelerator_device == "xpu" else 0.0),
                  "elapsed_accelerator_hours": ((datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds() / 3600
                                         if xpu_task else 0.0),
                  "log": str(log_path.relative_to(ROOT))}
        log.write(json.dumps(result) + "\n")
    atomic_append_jsonl(ROOT / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl", result)
    if stop_reason:
        raise RuntimeError(f"Stopped {job_id} at the frozen {stop_reason}")
    if code:
        raise RuntimeError(f"Job {job_id} failed with exit code {code}; see {log_path}")
    return result


def _accelerator_usage_hours() -> dict[str, float]:
    """Accumulate serial XPU/CUDA wall time; unresolved children block recovery."""
    path = ROOT / "outputs" / "upgrade_20260929" / "orchestrator" / "runs.jsonl"
    if not path.is_file():
        return {"total": 0.0, "smagnet": 0.0}
    starts = {}
    totals = {"total": 0.0, "smagnet": 0.0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("pid")
        if row.get("event") == "START" and row.get("accelerator_task", row.get("xpu_task")):
            starts[key] = row
        elif row.get("event") == "FINISH" and row.get("accelerator_task", row.get("xpu_task")):
            start = starts.pop(key, None)
            hours = float(row.get("elapsed_accelerator_hours", row.get("elapsed_xpu_hours", 0.0)))
            totals["total"] += hours
            if row.get("smagnet_task") or (start and "SMAGNet" in start.get("job_id", "")):
                totals["smagnet"] += hours
    now = datetime.now(timezone.utc)
    for pid, row in starts.items():
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            hours = max(0.0, (now - datetime.fromisoformat(row["started_utc"])).total_seconds() / 3600)
            totals["total"] += hours
            if row.get("smagnet_task") or "SMAGNet" in row.get("job_id", ""):
                totals["smagnet"] += hours
            atomic_append_jsonl(path, {"event": "FINISH", "job_id": row["job_id"], "pid": pid,
                                       "finished_utc": now.isoformat(), "exit_code": None,
                                       "status": "INTERRUPTED_RECOVERED_CONSERVATIVE_TIME",
                                       "xpu_task": bool(row.get("xpu_task")), "accelerator_task": True,
                                       "device": row.get("device", "xpu"),
                                       "smagnet_task": bool(row.get("smagnet_task")),
                                       "elapsed_accelerator_hours": hours})
        else:
            raise RuntimeError(f"Unresolved accelerator child PID {pid} ({row.get('job_id')}) is still active")
    return totals


def _xpu_usage_hours() -> dict[str, float]:
    """Compatibility alias; historical XPU and migrated CUDA share the same caps."""
    return _accelerator_usage_hours()


def atomic_append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _python() -> str:
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise RuntimeError(f"Python executable not found: {executable}")
    return str(executable)


def _load_completed_artifact(path: Path, expected: dict) -> bool:
    """Skip only an existing artifact whose identity still matches the frozen run."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing artifact is unreadable; refusing to overwrite: {path}") from exc
    mismatches = {key: (payload.get(key), value) for key, value in expected.items()
                  if payload.get(key) != value}
    if mismatches:
        raise RuntimeError(f"existing artifact does not match frozen run; refusing overwrite: {path}: {mismatches}")
    return True


def _run_training_matrix(config: dict) -> None:
    require_dual_freeze(ROOT, config)
    # The hard-label B1 thresholds are fitted first, using only the frozen fit
    # role. This is a deterministic baseline fit, not a neural training run.
    b1_spec = config["fixed_baselines"]["B1"]
    b1_output = ROOT / b1_spec["thresholds_path"]
    b1_script = ROOT / b1_spec["fit_script_path"]
    b1_expected = {"model": "B1", "status": "FIT_ROLE_ONLY_COMPLETE",
                   "fit_manifest_sha256": config["manifests"]["fit"]["sha256"],
                   "fit_script_sha256": sha256(b1_script),
                   "config_sha256": sha256(CONFIG)}
    if not _load_completed_artifact(b1_output, b1_expected):
        _run_child([_python(), str(b1_script)], job_id="fit_B1_fit_role_only",
                   log_path=b1_output.parent / "fit.log")
    script = ROOT / "scripts" / "upgrade_train.py"
    output = ROOT / "outputs" / "upgrade_20260929"
    for run in planned_runs(config):
        run_dir = output / run["run_id"]
        completed = run_dir / "run.json"
        if completed.is_file():
            record = json.loads(completed.read_text(encoding="utf-8"))
            if record.get("status") == "FIT_COMPLETE":
                model_file = run_dir / "model_only.pt"
                if (not model_file.is_file() or sha256(model_file) != record.get("model_only_sha256")
                        or record.get("hashes", {}).get("config") != sha256(CONFIG)):
                    raise RuntimeError(f"Completed training artifact mismatch: {run['run_id']}")
                continue
        resume = (run_dir / "run.lock.json").exists()
        command = [_python(), str(script), run["model"], str(run["seed"]), "--device", config.get("runtime", {}).get("device", "xpu")]
        if resume:
            command.append("--resume")
        _run_child(command, job_id=run["run_id"], log_path=run_dir / "runner.log")


def _run_calibration_matrix(config: dict) -> None:
    require_dual_freeze(ROOT, config)
    script = ROOT / "scripts" / "upgrade_calibrate_evaluate.py"
    for job in calibration_jobs(config):
        run_dir = ROOT / "outputs" / "upgrade_20260929" / f"{job['model']}_seed_{job['seed']}"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        temp_path = run_dir / f"temperature_{job['strategy']}.json"
        if _load_completed_artifact(temp_path, {
                "model": job["model"], "seed": job["seed"],
                "calibration_strategy": job["strategy"],
                "checkpoint_sha256": run.get("model_only_sha256"),
                "config_sha256": sha256(CONFIG)}):
            continue
        command = [_python(), str(script), "calibrate", job["model"], str(job["seed"]),
                   "--strategy", job["strategy"], "--device", config.get("runtime", {}).get("device", "xpu")]
        _run_child(command, job_id=f"calibrate_{job['job_id']}", log_path=run_dir / f"calibrate_{job['strategy']}.log")


def _run_evaluation_matrix(config: dict) -> None:
    require_dual_freeze(ROOT, config)
    script = ROOT / "scripts" / "upgrade_calibrate_evaluate.py"
    for job in calibration_jobs(config):
        run_dir = ROOT / "outputs" / "upgrade_20260929" / f"{job['model']}_seed_{job['seed']}"
        for role in ("internal_retro", "holdout_retro"):
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            evaluation = run_dir / (f"evaluation_{role}_{job['strategy']}.json")
            if _load_completed_artifact(evaluation, {
                    "model": job["model"], "seed": job["seed"], "role": role,
                    "calibration_strategy": job["strategy"],
                    "checkpoint_sha256": run.get("model_only_sha256"),
                    "temperature_sha256": sha256(run_dir / f"temperature_{job['strategy']}.json"),
                    "config_sha256": sha256(CONFIG)}):
                continue
            command = [_python(), str(script), "evaluate", job["model"], str(job["seed"]),
                       "--strategy", job["strategy"], "--device", config.get("runtime", {}).get("device", "xpu"), "--role", role]
            _run_child(command, job_id=f"evaluate_{job['job_id']}_{role}",
                       log_path=run_dir / f"evaluate_{job['strategy']}_{role}.log")
    # B1 is a frozen deterministic hard-label SAR baseline. It is evaluated
    # once per retrospective role and does not receive a temperature or enter
    # the probability-based primary estimand.
    b1_script = ROOT / "scripts" / "upgrade_b1_eval.py"
    b1_dir = ROOT / "outputs" / "upgrade_20260929" / "B1"
    for role in ("internal_retro", "holdout_retro"):
        b1_spec = config["fixed_baselines"]["B1"]
        b1_output = b1_dir / f"evaluation_{role}.json"
        thresholds_path = ROOT / b1_spec["thresholds_path"]
        if _load_completed_artifact(b1_output, {
                "model": "B1", "role": role,
                "thresholds_sha256": sha256(thresholds_path),
                "config_sha256": sha256(CONFIG)}):
            continue
        command = [_python(), str(b1_script), role]
        _run_child(command, job_id=f"evaluate_B1_{role}",
                   log_path=b1_dir / f"evaluate_{role}.log")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dry-run", "train-one", "matrix-train",
                                             "matrix-calibrate", "matrix-evaluate", "run-all"))
    parser.add_argument("model", nargs="?", choices=("B2", "B4", "P", "B5_whole", "B5_local", "SMAGNet"))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    if args.command == "dry-run":
        lock = validate_manifest_lock(config)
        input_hashes = validate_input_hashes(config)
        runs = planned_runs(config)
        if len(runs) != 18 or len(calibration_jobs(config)) != 24:
            raise RuntimeError("Frozen matrix dimensions differ from protocol (18 fits, 24 temperatures)")
        expected_updates = config["epochs"] * config["units_per_epoch"] * config["tiles_per_unit_per_epoch"] // config["batch_size"]
        print(json.dumps({"status": config["status"], "planned_training_runs": runs,
                          "training_run_count": len(runs), "calibration_jobs": len(calibration_jobs(config)),
                          "model_evaluation_jobs": len(calibration_jobs(config)) * 2,
                          "fixed_B1_hard_evaluation_jobs": 2,
                          "updates_per_run": expected_updates, "manifest_lock": lock,
                          "metadata_and_code_hashes": input_hashes,
                          "data_access": "metadata-only"}, indent=2))
        return
    require_frozen(config, args.command)
    require_dual_freeze(ROOT, config)
    if args.command == "train-one":
        if args.model is None or args.seed is None:
            parser.error("train-one requires MODEL SEED")
        from scripts.upgrade_train import execute_fit
        run_dir = ROOT / "outputs" / "upgrade_20260929" / f"{args.model}_seed_{args.seed}"
        run_exists = (run_dir / "run.lock.json").exists()
        execute_fit(ROOT, config, args.model, args.seed, config.get("runtime", {}).get("device", "xpu"), resume=args.resume or run_exists)
        return
    from scripts.upgrade_train import exclusive_process_lock
    with exclusive_process_lock(ROOT / "outputs/upgrade_20260929/orchestrator/matrix.process.lock"):
        if args.command in ("matrix-train", "run-all"):
            _run_training_matrix(config)
        if args.command in ("matrix-calibrate", "run-all"):
            _run_calibration_matrix(config)
        if args.command in ("matrix-evaluate", "run-all"):
            _run_evaluation_matrix(config)
            from scripts.upgrade_calibrate_evaluate import write_matrix_summaries
            write_matrix_summaries(ROOT, config)


if __name__ == "__main__":
    main()
