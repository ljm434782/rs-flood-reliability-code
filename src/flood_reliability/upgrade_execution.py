"""Pure persistence and freeze-gate primitives for the upgrade runner."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def learning_rate(epoch: int) -> float:
    if not 1 <= epoch <= 24:
        raise ValueError("epoch must be in 1..24")
    return 0.001 if epoch <= 6 else 0.0003

def epoch_sample_ids(fit_rows: list[dict], epoch: int) -> list[str]:
    """Fixed 60-unit × 8 cyclic schedule from frozen global/unit order rows."""
    if len({row["conservative_unit_id"] for row in fit_rows}) != 60:
        raise ValueError("fit schedule requires exactly 60 conservative units")
    chosen=[]
    for unit in sorted({row["conservative_unit_id"] for row in fit_rows}, key=lambda u: min(int(r["unit_order"]) for r in fit_rows if r["conservative_unit_id"]==u)):
        rows=sorted((r for r in fit_rows if r["conservative_unit_id"]==unit), key=lambda r:int(r["within_unit_order"]))
        chosen.extend(r["sample_id"] for r in (rows[((epoch - 1) * 8 + k) % len(rows)] for k in range(8)))
    return chosen

def schedule_audit(fit_rows: list[dict], epoch: int) -> dict:
    """Report presentations and within-epoch repeats; never assume eight are unique."""
    selected=epoch_sample_ids(fit_rows, epoch)
    return {"presentations": len(selected), "unique_sample_ids": len(set(selected)),
            "within_epoch_repeated_presentations": len(selected)-len(set(selected))}

def synthetic_train_epoch(model, batches, optimizer, epoch: int, checkpoint: Path) -> dict:
    """Generic training body for injected synthetic fixtures or frozen real adapters."""
    losses=[]
    for x, label in batches:
        if not bool((label != 255).any()): raise RuntimeError("all-Ignore training batch")
        optimizer.zero_grad(set_to_none=True); logits=model(x)
        loss=F.cross_entropy(logits,label,ignore_index=255)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    temporary=checkpoint.with_suffix(".tmp")
    torch.save({"epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict()},temporary)
    os.replace(temporary,checkpoint)
    return {"epoch":epoch,"mean_loss":float(np.mean(losses)),"checkpoint_sha256":sha256(checkpoint)}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_data_source_record(root: Path, config: dict) -> dict:
    """Validate the frozen raw-archive provenance and current byte lengths."""
    source=config.get("data_source")
    if not source:
        raise RuntimeError("Freeze config lacks a data-source provenance record")
    assets=source["publisher_asset_manifest"]; audit=source["archive_audit"]
    for entry in (assets,audit):
        path=root/entry["path"]
        if not path.is_file() or sha256(path)!=entry["sha256"]:
            raise RuntimeError(f"Frozen data provenance record hash mismatch: {entry['path']}")
    report=json.loads((root/audit["path"]).read_text(encoding="utf-8"))
    if (report.get("status")!="PASS" or report.get("source_repository")!=source["repository"]
            or report.get("source_version")!=f"ImpactMesh-Flood {source['version']}"
            or report.get("license")!=source["license"]
            or report.get("publisher_asset_manifest_sha256")!=assets["sha256"]):
        raise RuntimeError("Raw archive audit provenance does not match frozen source config")
    recorded={row["path"]:(row["size_bytes"],row["sha256"],row["publisher_lfs_sha256"])
              for row in report.get("archives",[])}
    if len(recorded)!=len(source.get("raw_archives",[])):
        raise RuntimeError("Raw archive audit has incomplete source coverage")
    for entry in source["raw_archives"]:
        path=root/entry["path"]
        archive_record=recorded.get(entry["path"])
        expected=(entry["size_bytes"],entry["sha256"],entry["sha256"])
        if (archive_record!=expected or not path.is_file() or path.stat().st_size!=entry["size_bytes"]):
            raise RuntimeError(f"Raw archive no longer matches frozen audit metadata: {entry['path']}")
    return {"repository":source["repository"],"version":source["version"],
            "license":source["license"],"archive_count":len(recorded),
            "archive_audit_sha256":audit["sha256"]}

def require_dual_freeze(root: Path, config: dict) -> None:
    """Public distribution: historical execution-control evidence is excluded.

    This is a release-only fail-closed replacement, not a historical approval
    or a new scientific rule. Cached-result rendering does not invoke it.
    """
    raise RuntimeError(
        "Historical full execution is unavailable in this public distribution: "
        "execution-control metadata was excluded. See docs/CODE_REPRODUCTION.md."
    )

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def acquire_run_lock(run_dir: Path, run_id: str, lock_hashes: dict) -> Path:
    lock = run_dir / "run.lock.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        if old != {"run_id": run_id, "hashes": lock_hashes}:
            raise RuntimeError("Existing run lock differs from requested run")
        return lock
    atomic_json(lock, {"run_id": run_id, "hashes": lock_hashes})
    return lock

def validate_resume(lock: Path, expected: dict, checkpoint: Path) -> dict:
    if not lock.exists() or not checkpoint.exists():
        raise RuntimeError("Cannot resume without matching lock and checkpoint")
    actual = json.loads(lock.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("Resume lock mismatch")
    return {"checkpoint_sha256": sha256(checkpoint), "lock_sha256": sha256(lock)}

def owned_logits_and_labels(logits: np.ndarray, labels: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select only externally frozen owned, non-Ignore pixels for cal/evaluation.

    Ownership is deliberately an argument: role and cross-role spatial policy is
    supplied by the future freeze package, never reconstructed by a scorer.
    """
    if logits.ndim != 3 or logits.shape[0] != 3:
        raise ValueError("logits must have shape [3,H,W]")
    if labels.shape != logits.shape[1:] or owner.shape != labels.shape or owner.dtype != np.bool_:
        raise ValueError("labels and boolean owner must match logits spatial shape")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    selected = owner & (labels != 255)
    if not selected.any():
        raise ValueError("no owned non-Ignore pixels")
    return logits[:, selected].T, labels[selected]
