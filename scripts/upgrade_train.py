"""Fit-only, freeze-gated trainer for the 2026-09-29 upgrade matrix.

The production CLI fails closed unless both the coordinator's immutable G3
freeze binder and the independent PASS attestation are present.  Tests use the
same epoch/checkpoint engine with synthetic tensors and never open project data.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from flood_reliability.availability import apply_modality_dropout
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet, SmallUNet
from flood_reliability.smagnet_adapter import (
    AUTHOR_COMMIT, AUTHOR_SOURCE_SHA256, flood_soft_dice, make_smagnet_resnet50,
    two_head_loss, validate_batch,
)
from flood_reliability.upgrade_execution import require_dual_freeze, sha256
from flood_reliability.upgrade_masks import apply_train_local_exposure, whole_dropout_decision

CONFIG_PATH = ROOT / "configs" / "upgrade_20260929.json"
FIT_RELATIVE = "data/manifests/g3_rule_a_derived_candidate_fit.csv"
EXPECTED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
FIT_BAND_RECORD_SHA256 = "c5af4a1b5eb9eba769ec205f2573e131d12c1b1479a6378aff97ae24b2fd1a0d"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


def atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _jsonl_atomic(path: Path, rows: list[dict]) -> None:
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, data.encode("utf-8"))


@contextlib.contextmanager
def exclusive_process_lock(path: Path):
    """OS advisory lock: stable across resume and released after process exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Run is already active (process lock: {path})") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Run is already active (process lock: {path})") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": _utc()}).encode("utf-8"))
        handle.flush()
        yield
    finally:
        if not handle.closed:
            if os.name == "nt":
                import msvcrt
                with contextlib.suppress(OSError):
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_fit_rows(root: Path, config: dict) -> tuple[list[dict], str]:
    """Read and hash only the fit manifest; never open other-role manifests."""
    locked = config["manifests"]["fit"]
    path = root / locked["path"]
    actual = sha256(path)
    if actual != locked["sha256"]:
        raise RuntimeError(f"Fit manifest SHA256 mismatch: {actual}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "conservative_unit_id", "derived_unit_order",
                "derived_within_unit_order", "selected_time_index", "selected_age_days",
                "selected_joint_visible_fraction", "derived_role"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Fit manifest is empty or lacks required columns")
    if any(row["derived_role"] != "fit" for row in rows):
        raise RuntimeError("Fit manifest contains a non-fit role row")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sample ID in fit manifest")
    units = {r["conservative_unit_id"] for r in rows}
    expected = int(config["units_per_epoch"])
    if len(units) != expected:
        raise RuntimeError(f"Expected {expected} fit units, found {len(units)}")
    for row in rows:
        if not 0 <= int(row["selected_time_index"]) <= 3:
            raise RuntimeError(f"Invalid selected S2 time index: {row['sample_id']}")
        if not np.isfinite(float(row["selected_age_days"])):
            raise RuntimeError(f"Invalid S2 age: {row['sample_id']}")
        if not 0.5 <= float(row["selected_joint_visible_fraction"]) <= 1:
            raise RuntimeError(f"Invalid frozen visible fraction: {row['sample_id']}")
    return rows, actual


def validate_fit_band_attestation(root: Path, freeze: dict, fit_count: int) -> None:
    """Require the metadata-only all-fit named-band check in the freeze binder."""
    audit = freeze.get("fit_band_audit", {})
    if (audit.get("sample_count") != fit_count
            or audit.get("ordered_bands") != EXPECTED_BANDS
            or audit.get("record_sha256") != FIT_BAND_RECORD_SHA256
            or audit.get("s2_indices_b04_b03_b02_b08") != [3, 2, 1, 7]):
        raise RuntimeError("Freeze binder lacks the verified all-fit S2 band-order attestation")


def fixed_epoch_rows(rows: list[dict], config: dict, epoch: int, seed: int) -> list[dict]:
    """Frozen cyclic unit schedule followed by deterministic common permutation."""
    unit_order: dict[str, int] = {}
    unit_rows: dict[str, list[dict]] = {}
    for row in rows:
        unit = row["conservative_unit_id"]
        unit_order[unit] = int(row["derived_unit_order"])
        unit_rows.setdefault(unit, []).append(row)
    chosen: list[dict] = []
    k = int(config["tiles_per_unit_per_epoch"])
    for unit in sorted(unit_rows, key=lambda value: (unit_order[value], value)):
        group = sorted(unit_rows[unit], key=lambda r: int(r["derived_within_unit_order"]))
        start = (epoch - 1) * k
        chosen.extend(group[(start + i) % len(group)] for i in range(k))
    # Identical batch presentation order for every architecture at a common seed.
    rng = np.random.Generator(np.random.PCG64(seed + epoch * 1_000_003))
    return [chosen[i] for i in rng.permutation(len(chosen))]


def _stats(reader: TrainArchiveReader, rows: list[dict], config: dict) -> dict:
    per_unit: dict[str, list[dict]] = {}
    for row in rows:
        per_unit.setdefault(row["conservative_unit_id"], []).append(row)
    selected = []
    for unit in sorted(per_unit, key=lambda u: int(per_unit[u][0]["derived_unit_order"])):
        ordered = sorted(per_unit[unit], key=lambda r: int(r["derived_within_unit_order"]))
        selected.extend(ordered[:int(config["normalization_chips_per_process"])])
    sar_sum = np.zeros(2, np.float64); sar_sq = np.zeros(2, np.float64)
    opt_sum = np.zeros(12, np.float64); opt_sq = np.zeros(12, np.float64)
    n_sar = 0; n_opt = 0; counts = np.zeros(3, np.int64)
    for row in selected:
        s1, s2, valid, label = reader.load(row["sample_id"], int(row["selected_time_index"]))
        if abs(float(valid.mean()) - float(row["selected_joint_visible_fraction"])) > 1e-6:
            raise RuntimeError(f"Fit optical validity drift: {row['sample_id']}")
        sar_sum += s1.sum(axis=(1, 2), dtype=np.float64)
        sar_sq += np.square(s1.astype(np.float64)).sum(axis=(1, 2))
        n_sar += s1.shape[1] * s1.shape[2]
        values = s2[:, valid].astype(np.float64)
        opt_sum += values.sum(axis=1); opt_sq += np.square(values).sum(axis=1)
        n_opt += int(valid.sum())
        counts += np.bincount(label[label != 255], minlength=3)
    sar_mean = sar_sum / n_sar; sar_std = np.sqrt(np.maximum(sar_sq / n_sar - sar_mean**2, 1e-12))
    opt_mean = opt_sum / n_opt; opt_std = np.sqrt(np.maximum(opt_sq / n_opt - opt_mean**2, 1e-12))
    if not np.isfinite(sar_mean).all() or not np.isfinite(opt_mean).all() or (counts == 0).any():
        raise RuntimeError("Nonfinite statistics or missing class in fit-only statistic sample")
    weights = np.clip(np.sqrt(counts[0] / counts), 1., float(config["maximum_class_weight"]))
    return {"sar_mean": sar_mean.tolist(), "sar_std": sar_std.tolist(),
            "optical_mean": opt_mean.tolist(), "optical_std": opt_std.tolist(),
            "class_count": counts.tolist(), "class_weights": weights.tolist(),
            "sample_ids": [r["sample_id"] for r in selected], "tile_count": len(selected),
            "fit_manifest_sha256": config["manifests"]["fit"]["sha256"],
            "computed_utc": _utc()}


def _encode(reader, batch: list[dict], stats: dict, device: torch.device):
    sar_arr=[]; opt_arr=[]; valid_arr=[]; label_arr=[]; age_arr=[]
    sm=np.asarray(stats["sar_mean"], np.float32)[:, None, None]
    ss=np.asarray(stats["sar_std"], np.float32)[:, None, None]
    om=np.asarray(stats["optical_mean"], np.float32)[:, None, None]
    osd=np.asarray(stats["optical_std"], np.float32)[:, None, None]
    for row in batch:
        s1, s2, valid, label=reader.load(row["sample_id"], int(row["selected_time_index"]))
        if abs(float(valid.mean())-float(row["selected_joint_visible_fraction"])) > 1e-6:
            raise RuntimeError(f"Optical validity drift: {row['sample_id']}")
        sar_arr.append(((s1-sm)/ss).astype(np.float32))
        opt_arr.append(np.where(valid[None], (s2-om)/osd, 0).astype(np.float32))
        valid_arr.append(valid); label_arr.append(label)
        age_arr.append(-float(row["selected_age_days"])/30.0)
    return (torch.from_numpy(np.stack(sar_arr)).to(device),
            torch.from_numpy(np.stack(opt_arr)).to(device),
            torch.from_numpy(np.stack(valid_arr)[:, None]).bool().to(device),
            torch.from_numpy(np.stack(label_arr)).long().to(device),
            torch.tensor(age_arr, dtype=torch.float32, device=device))


def make_model(name: str, root: Path, config: dict) -> torch.nn.Module:
    if name == "B2": return SmallUNet(2, 3, 16)
    if name in {"B4", "B5_whole", "B5_local", "P"}:
        return DualUNet(2, 12, 3, 16, gated=(name == "P"))
    if name == "SMAGNet":
        return make_smagnet_resnet50(root / "third_party" / "smagnet" / "smagnet.py")
    raise ValueError(f"Unknown model candidate: {name}")


def _batch_loss(model, name, batch, stats, config, seed, epoch, device):
    sar, optical, valid, label, time_offset = batch
    present = torch.ones(label.shape[0], dtype=torch.bool, device=device)
    rows = batch[-1] if False else None
    # Presentation-specific deterministic interventions are applied before this helper.
    weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    if name == "B2":
        logits = model(sar)
        loss = F.cross_entropy(logits, label, weight=weights, ignore_index=255) + \
            float(config["loss"]["flood_dice_weight"]) * flood_soft_dice(logits, label)
    elif name == "SMAGNet":
        picked = optical[:, (3, 2, 1, 7)]
        invalid = (~valid).to(picked.dtype)
        fused, sar_logits, _ = model(sar, picked, invalid)
        loss, _ = two_head_loss(fused, sar_logits, label, weights,
                                dice_weight=float(config["loss"]["flood_dice_weight"]))
    else:
        if name == "P" and not bool(present.any()):
            time_offset = torch.zeros_like(time_offset)
        logits = model(sar, optical, valid, present,
                       normalized_time_offset=time_offset if name == "P" else None)
        loss = F.cross_entropy(logits, label, weight=weights, ignore_index=255) + \
            float(config["loss"]["flood_dice_weight"]) * flood_soft_dice(logits, label)
    if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
    return loss


def _train_epoch(model, name, batches, optimizer, stats, config, device) -> dict:
    model.train(); losses=[]; pixels=0; start=time.monotonic()
    for (sar, optical, valid, label, time_offset), row_ids in batches:
        if not bool((label != 255).any()): raise RuntimeError("All-Ignore fit batch")
        # Deterministic shared whole-modality mask and local-mask plan derive from
        # common seed/epoch/sample identity; whole dropout takes precedence.
        epoch = config["_active_epoch"]; seed = config["_active_seed"]
        drop_np = [whole_dropout_decision(seed, epoch, sample_id)
                   for sample_id in row_ids] if name in {"P", "B5_whole", "B5_local", "SMAGNet"} else [False]*len(row_ids)
        drop = torch.tensor(drop_np, dtype=torch.bool, device=device)
        if name in {"P", "B5_whole", "B5_local", "SMAGNet"}:
            optical, valid, present = apply_modality_dropout(optical, valid,
                torch.ones(len(row_ids), dtype=torch.bool, device=device), drop)
            time_offset = torch.where(present, time_offset, torch.zeros_like(time_offset))
        if name == "B5_local":
            # Only fit mask application requires matching per-sample IDs. Arrays
            # are already loaded; the helper uses no labels or non-fit data.
            changed=[]; changed_valid=[]
            for i, sample_id in enumerate(row_ids):
                exposure = apply_train_local_exposure(
                    optical[i].detach().cpu().numpy(), valid[i, 0].detach().cpu().numpy(),
                    present=bool(present[i]), time_offset=float(time_offset[i]), seed=seed,
                    epoch=epoch, sample_id=sample_id, whole_dropped=bool(drop_np[i]))
                changed.append(exposure.optical); changed_valid.append(exposure.valid)
            optical = torch.from_numpy(np.stack(changed)).to(device)
            valid = torch.from_numpy(np.stack(changed_valid)[:, None]).bool().to(device)
        optimizer.zero_grad(set_to_none=True)
        batch = (sar, optical, valid, label, time_offset)
        loss = _batch_loss(model, name, batch, stats, config, config["_active_seed"],
                           config["_active_epoch"], device)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
            raise RuntimeError("Nonfinite or absent training gradient; stop the frozen matrix")
        optimizer.step()
        losses.append(float(loss.detach().cpu())); pixels += int((label != 255).sum().cpu())
    return {"mean_batch_loss": float(np.mean(losses)), "batches": len(losses),
            "non_ignore_pixels": pixels, "duration_seconds": time.monotonic()-start}


def _save_checkpoint(path: Path, *, epoch: int, model, optimizer, history: list[dict],
                     hashes: dict, seed: int, run_id: str) -> str:
    payload = {"schema": 1, "run_id": run_id, "seed": seed, "completed_epoch": epoch,
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "history": history, "hashes": hashes,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(),
               "xpu_rng": torch.xpu.get_rng_state_all() if torch.xpu.is_available() else None}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        with open(temp_name, "r+b") as stream: os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError): os.unlink(temp_name)
    return sha256(path)


def _write_model_only(path: Path, *, model, hashes: dict, seed: int, run_id: str) -> str:
    """Persist and verify a resumable-free final artifact before retiring Adam state."""
    fd, temp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save({"schema": 1, "run_id": run_id, "seed": seed,
                    "model_state": model.state_dict(), "hashes": hashes}, temp_name)
        with open(temp_name, "r+b") as stream: os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError): os.unlink(temp_name)
    digest = sha256(path)
    check = torch.load(path, map_location="cpu", weights_only=False)
    if check.get("hashes") != hashes or check.get("run_id") != run_id:
        raise RuntimeError("Final model-only artifact verification failed")
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), check["model_state"][key].detach().cpu()):
            raise RuntimeError(f"Final model-only tensor mismatch: {key}")
    return digest


def _enforce_incremental_disk_cap(root: Path, run_dir: Path, model: torch.nn.Module,
                                  config: dict) -> int:
    cap = int(config["resources"]["durable_incremental_disk_bytes"])
    output_root = root / "outputs" / "upgrade_20260929"
    existing = sum(p.stat().st_size for p in output_root.rglob("*") if p.is_file()) if output_root.exists() else 0
    # One live checkpoint + atomic replacement temp + final model-only + logs/stats.
    reserve = sum(p.numel() for p in model.parameters()) * 32 + 64 * 1024 * 1024
    if existing + reserve > cap:
        raise RuntimeError(f"Upgrade output cap would be exceeded: existing={existing}, reserve={reserve}, cap={cap}")
    free = shutil.disk_usage(run_dir).free
    if free < reserve:
        raise RuntimeError(f"Insufficient free disk for bounded run reserve: free={free}, reserve={reserve}")
    return reserve


def train_synthetic(model, batches_by_epoch: list[list[tuple[torch.Tensor, torch.Tensor]]],
                    *, run_dir: Path, run_id: str, seed: int, max_epochs: int,
                    hashes: dict, stop_after_epoch: int | None = None) -> list[dict]:
    """Small CPU-capable training engine used only with synthetic fixtures in tests."""
    if max_epochs < 1 or len(batches_by_epoch) < max_epochs: raise ValueError("invalid epoch plan")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    ckpt = run_dir / "latest.pt"; logs=run_dir / "epochs.jsonl"
    history=[]; first=0
    if ckpt.exists():
        state=torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["hashes"] != hashes or state["run_id"] != run_id or state["seed"] != seed:
            raise RuntimeError("Synthetic resume checkpoint hash/run mismatch")
        model.load_state_dict(state["model_state"]); opt.load_state_dict(state["optimizer_state"])
        history=state["history"]; first=state["completed_epoch"]
    for epoch in range(first+1, max_epochs+1):
        model.train(); losses=[]
        for x, label in batches_by_epoch[epoch-1]:
            if not bool((label != 255).any()): raise RuntimeError("all-Ignore synthetic batch")
            opt.zero_grad(set_to_none=True); logits=model(x)
            loss=F.cross_entropy(logits, label, ignore_index=255)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite synthetic loss")
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        entry={"epoch":epoch,"mean_loss":float(np.mean(losses)),"finished_utc":_utc()}
        history.append(entry)
        digest=_save_checkpoint(ckpt, epoch=epoch, model=model, optimizer=opt,
                                history=history, hashes=hashes, seed=seed, run_id=run_id)
        entry["checkpoint_sha256"]=digest
        # Keep checkpoint and log history consistent if interruption occurs.
        history[-1]=entry; _jsonl_atomic(logs, history)
        if stop_after_epoch == epoch: break
    return history


def execute_fit(root: Path, config: dict, model_name: str, seed: int, device_name: str,
                *, resume: bool = False) -> Path:
    """Execute one frozen fit run; this is the only production entry point."""
    # This is deliberately the first gate before opening a dataset archive/model.
    require_dual_freeze(root, config)
    if device_name != "xpu":
        raise RuntimeError("Production fit is XPU-only; CPU is reserved for synthetic tests")
    freeze = json.loads((root / "G3_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    rows, fit_hash = load_fit_rows(root, config)
    validate_fit_band_attestation(root, freeze, len(rows))
    if model_name not in config["models"] or seed not in config["seeds"]:
        raise ValueError("model or seed is outside the frozen candidate set")
    if device_name == "xpu" and not torch.xpu.is_available(): raise RuntimeError("XPU unavailable")
    device=torch.device(device_name)
    run_id=f"{model_name}_seed_{seed}"
    run_dir=root/"outputs"/"upgrade_20260929"/run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(run_dir/"process.lock"):
        config_path=root/"configs"/"upgrade_20260929.json"
        script=Path(__file__).resolve()
        code_paths=[script, root/"src/flood_reliability/models.py",
                    root/"src/flood_reliability/impactmesh_io.py",
                    root/"src/flood_reliability/availability.py",
                    root/"src/flood_reliability/upgrade_masks.py",
                    root/"src/flood_reliability/smagnet_adapter.py",
                    root/"third_party/smagnet/smagnet.py"]
        hashes={"config":sha256(config_path),"fit_manifest":fit_hash,
                "freeze":sha256(root/"G3_FREEZE_MANIFEST.json"),
                "independent_review":sha256(root/"G3_INDEPENDENT_REVIEW.json"),
                "code":{p.relative_to(root).as_posix():sha256(p) for p in code_paths}}
        lock_path=run_dir/"run.lock.json"
        expected_lock={"run_id":run_id,"seed":seed,"hashes":hashes}
        if lock_path.exists():
            prior=json.loads(lock_path.read_text(encoding="utf-8"))
            if prior != expected_lock: raise RuntimeError("Existing run registration does not match frozen inputs")
            if not resume: raise RuntimeError("Run already registered; pass --resume to continue")
        else:
            if resume: raise RuntimeError("Cannot resume an unregistered run")
            atomic_json(lock_path, expected_lock)
        run_record_path=run_dir/"run.json"
        if resume and run_record_path.is_file():
            completed=json.loads(run_record_path.read_text(encoding="utf-8"))
            if completed.get("status")=="FIT_COMPLETE":
                for key,value in hashes.items():
                    if completed.get("hashes",{}).get(key)!=value:
                        raise RuntimeError("Completed run record no longer matches frozen inputs")
                stats_path=run_dir/"fit_only_statistics.json"
                if (not stats_path.is_file()
                        or sha256(stats_path)!=completed.get("hashes",{}).get("statistics")):
                    raise RuntimeError("Completed run fit-statistics artifact failed hash validation")
                final_path=run_dir/"model_only.pt"
                if (not final_path.is_file() or sha256(final_path)!=completed.get("model_only_sha256")):
                    raise RuntimeError("Completed run model-only artifact failed hash validation")
                model_payload=torch.load(final_path,map_location="cpu",weights_only=False)
                if (model_payload.get("run_id")!=run_id
                        or model_payload.get("hashes")!=completed.get("hashes")):
                    raise RuntimeError("Completed model-only artifact metadata mismatch")
                (run_dir/"latest.pt").unlink(missing_ok=True)
                if not completed.get("full_resume_checkpoint_retired"):
                    completed["full_resume_checkpoint_retired"]=True
                    atomic_json(run_record_path,completed)
                return run_dir
        checkpoint=run_dir/"latest.pt"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device_name == "xpu": torch.xpu.manual_seed_all(seed)
        # XPU weighted NLL kernels do not offer a deterministic implementation.
        # Persist RNG and masks exactly; disclose non-bitwise device arithmetic.
        torch.use_deterministic_algorithms(True, warn_only=bool(config["reproducibility"]["warn_only"]))
        model=make_model(model_name, root, config).to(device)
        reserve_bytes = _enforce_incremental_disk_cap(root, run_dir, model, config)
        optimizer=torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr_epochs_1_6"]),
                                   foreach=False)
        history=[]; first_epoch=1
        stats_path=run_dir/"fit_only_statistics.json"
        if stats_path.exists():
            stats=json.loads(stats_path.read_text(encoding="utf-8"))
            if stats.get("fit_manifest_sha256") != fit_hash: raise RuntimeError("Fit-statistics manifest mismatch")
        else:
            with TrainArchiveReader(root/"data/raw/impactmesh/train") as reader:
                stats=_stats(reader,rows,config)
            atomic_json(stats_path,stats)
        stats_hash=sha256(stats_path); hashes["statistics"]=stats_hash
        if resume:
            if not checkpoint.is_file(): raise RuntimeError("Resume requires latest.pt")
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)
            if state.get("hashes") != hashes or state.get("run_id") != run_id or state.get("seed") != seed:
                raise RuntimeError("Checkpoint does not match locked run/config/data/code/statistics hashes")
            model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
            history=state["history"]; first_epoch=int(state["completed_epoch"])+1
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            if state.get("xpu_rng") is not None: torch.xpu.set_rng_state_all(state["xpu_rng"])
        with TrainArchiveReader(root/"data/raw/impactmesh/train") as reader:
            for epoch in range(first_epoch, int(config["epochs"])+1):
                lr=float(config["optimizer"]["lr_epochs_1_6"] if epoch<=6 else config["optimizer"]["lr_epochs_7_24"])
                for group in optimizer.param_groups: group["lr"]=lr
                chosen=fixed_epoch_rows(rows,config,epoch,seed)
                def encoded_batches():
                    # Keep only one minibatch of 12-band rasters in memory.
                    for start in range(0,len(chosen),int(config["batch_size"])):
                        group=chosen[start:start+int(config["batch_size"])]
                        yield _encode(reader,group,stats,device), [r["sample_id"] for r in group]
                run_config=dict(config); run_config["_active_epoch"]=epoch; run_config["_active_seed"]=seed
                result=_train_epoch(model,model_name,encoded_batches(),optimizer,stats,run_config,device)
                row={"run_id":run_id,"model":model_name,"seed":seed,"epoch":epoch,
                     "learning_rate":lr,"train_tiles":len(chosen),**result,"time_utc":_utc()}
                history.append(row)
                digest=_save_checkpoint(checkpoint,epoch=epoch,model=model,optimizer=optimizer,
                                        history=history,hashes=hashes,seed=seed,run_id=run_id)
                history[-1]["checkpoint_sha256"]=digest
                _jsonl_atomic(run_dir/"epochs.jsonl",history)
                print(json.dumps(history[-1],sort_keys=True),flush=True)
        if len(history) != int(config["epochs"]):
            raise RuntimeError("Training ended without the frozen epoch budget")
        final_path=run_dir/"model_only.pt"
        final_hash=_write_model_only(final_path,model=model,hashes=hashes,seed=seed,run_id=run_id)
        if sha256(final_path) != final_hash: raise RuntimeError("Final model-only hash changed")
        run_record={"run_id":run_id,"model":model_name,"seed":seed,
            "finished_utc":_utc(),"device":device_name,"torch":torch.__version__,
            "hashes":hashes,"fit_tile_count":len(rows),"completed_epochs":len(history),
            "model_only_sha256":final_hash,"reserved_incremental_bytes":reserve_bytes,
            "status":"FIT_COMPLETE","evaluation_roles_read":[],
            "full_resume_checkpoint_retired":False}
        # Record a verified final weight before removing optimizer moments. If a
        # crash occurs between these writes, --resume sees the complete record,
        # verifies the model-only artifact, and finishes retirement safely.
        atomic_json(run_dir/"run.json",run_record)
        if checkpoint.exists(): checkpoint.unlink()
        run_record["full_resume_checkpoint_retired"]=True
        atomic_json(run_dir/"run.json",run_record)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B2","B4","P","B5_whole","B5_local","SMAGNet"))
    parser.add_argument("seed",type=int)
    parser.add_argument("--device",choices=("xpu",),default="xpu")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    execute_fit(ROOT,config,args.model,args.seed,args.device,resume=args.resume)


if __name__ == "__main__": main()
