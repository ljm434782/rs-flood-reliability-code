"""One-tile fit-only input/model smoke before the G3 independent review.

This narrowly scoped utility is not a training/evaluation entry point: it reads
one deterministic fit tile, runs no optimizer step, and never constructs any
calibration or evaluation reader. Run once after the protocol config is marked
FROZEN and before generating the immutable G3 review package.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet, SmallUNet


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_fit_row(root: Path, config: dict) -> tuple[dict, str]:
    item = config["manifests"]["fit"]
    manifest = root / item["path"]
    actual = sha256(manifest)
    if actual != item["sha256"]:
        raise RuntimeError("fit-only smoke blocked: candidate fit manifest hash mismatch")
    with manifest.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None or row.get("derived_role") != "fit":
        raise RuntimeError("fit-only smoke blocked: no first-row fit sample")
    return row, actual


def _fit_s1_time(root: Path, row: dict, config: dict) -> str:
    item = config["auxiliary_metadata"]["fit_s1_event_times"]
    path = root / item["path"]
    if sha256(path) != item["sha256"]:
        raise RuntimeError("fit-only smoke blocked: S1 timestamp metadata hash mismatch")
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        matches = [record for record in csv.DictReader(stream)
                   if record["sample_id"] == row["sample_id"]]
    if (len(matches) != 1 or matches[0]["role"] != "train"
            or matches[0]["activation"] != row["activation"]):
        raise RuntimeError("fit-only smoke blocked: sample does not have one train S1 timestamp")
    timestamp = matches[0]["s1_event_utc"]
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return timestamp


def run_smoke(root: Path = ROOT, device_name: str = "xpu") -> dict:
    config_path = root / "configs" / "upgrade_20260929.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN":
        raise RuntimeError("fit-only smoke requires the protocol config status FROZEN")
    if device_name != "xpu" or not torch.xpu.is_available():
        raise RuntimeError("fit-only smoke requires the configured XPU; no CPU fallback")
    row, fit_hash = _one_fit_row(root, config)
    s1_time = _fit_s1_time(root, row, config)
    age = float(row["selected_age_days"])
    if not np.isfinite(age) or not 0 <= int(row["selected_time_index"]) <= 3:
        raise RuntimeError("fit-only smoke blocked: invalid selected S2 metadata")

    # This is the only project-array read in this pre-review utility. The reader
    # checks S1/S2/MASK CRS, transform/coordinates, dimensions, nodata and codes.
    with TrainArchiveReader(root / "data" / "raw" / "impactmesh" / "train") as reader:
        s1, s2, valid, label = reader.load(row["sample_id"], int(row["selected_time_index"]))

    if (s1.shape != (2, 256, 256) or s2.shape != (12, 256, 256)
            or valid.shape != (256, 256) or valid.dtype != np.bool_
            or label.shape != (256, 256) or not np.isfinite(s1).all()
            or not np.isfinite(s2).all()):
        raise RuntimeError("fit-only smoke blocked: unexpected real input tensor contract")
    if abs(float(valid.mean()) - float(row["selected_joint_visible_fraction"])) > 1e-6:
        raise RuntimeError("fit-only smoke blocked: optical validity differs from locked manifest")
    label_codes = sorted(int(value) for value in np.unique(label))
    if not set(label_codes) <= {0, 1, 2, 255}:
        raise RuntimeError("fit-only smoke blocked: unexpected fit label code")

    device = torch.device("xpu")
    s1_tensor = torch.from_numpy(s1[None]).to(device)
    s2_tensor = torch.from_numpy(s2[None]).to(device)
    valid_tensor = torch.from_numpy(valid[None, None]).to(device)
    with torch.inference_mode():
        b2_logits = SmallUNet(2, 3, 16).to(device).eval()(s1_tensor)
        b5_logits = DualUNet(2, 12, 3, 16, gated=False).to(device).eval()(
            s1_tensor, s2_tensor, valid_tensor,
            torch.ones(1, dtype=torch.bool, device=device),
        )
        for name, logits in (("B2", b2_logits), ("B5", b5_logits)):
            if logits.shape != (1, 3, 256, 256) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"fit-only smoke {name} forward produced invalid logits")
    del s1_tensor, s2_tensor, valid_tensor, b2_logits, b5_logits
    torch.xpu.synchronize()

    code_paths = (Path(__file__), root / "src/flood_reliability/impactmesh_io.py",
                  root / "src/flood_reliability/models.py")
    result = {
        "status": "PASS_FIT_ONLY_INPUT_SMOKE",
        "purpose": "one fit tile I/O and B2/B5 forward compatibility only",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name,
        "sample_id": row["sample_id"], "activation": row["activation"],
        "s1_event_utc": s1_time,
        "s2_time_index": int(row["selected_time_index"]),
        "s2_utc": row["selected_s2_utc"], "s2_age_days": age,
        "shapes": {"s1": list(s1.shape), "s2": list(s2.shape),
                   "optical_valid": list(valid.shape), "label": list(label.shape)},
        "label_codes": label_codes,
        "optical_valid_fraction": float(valid.mean()),
        "forward_models": ["B2", "B5"], "optimizer_steps": 0,
        "evaluation_roles_opened": [], "calibration_labels_read": False,
        "internal_or_holdout_labels_read": False, "scores_read_or_written": False,
        "fit_manifest_sha256": fit_hash,
        "config_sha256": sha256(config_path),
        "code_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in code_paths},
        "label_product_reference_time": "UNVERIFIED; this smoke checks array contract only",
    }
    out = root / "outputs" / "upgrade_20260929" / "preflight" / "fit_input_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(out)
    result["output_path"] = str(out)
    result["output_sha256"] = sha256(out)
    return result


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
