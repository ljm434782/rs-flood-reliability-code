"""Measure v1 P gate factors on validation inputs only; do not read test data."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet
from scripts.calibrate_impactmesh_formal import STATS
from scripts.train_impactmesh_exploratory_v2 import BASE, ROOT
from scripts.train_impactmesh_formal import encode_batch, sha256


def main():
    torch.set_num_threads(2)
    frame_path = BASE / "validation_frame.json"
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    checkpoint = ROOT / "outputs" / "formal_v1" / "P" / "latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DualUNet(2, 12, 3, 16, gated=True).eval()
    model.load_state_dict(payload["model_state"])
    sar_means = {level: [] for level in range(3)}
    optical_means = {level: [] for level in range(3)}
    with torch.inference_mode(), TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for rows in frame["paired"].values():
            sar, optical, valid, _, offset = encode_batch(reader, rows, stats, torch.device("cpu"))
            sar_features = model._encode(model.sar_encoder, sar)
            optical_features = model._encode(model.optical_encoder, optical)
            for level, (sar_feature, optical_feature) in enumerate(zip(sar_features, optical_features)):
                optical_valid = F.adaptive_max_pool2d(valid.float(), optical_feature.shape[-2:])
                presence = torch.ones_like(optical_valid)
                age = offset[:, None, None, None].expand_as(optical_valid)
                sar_factor = torch.sigmoid(model.sar_gates[level](sar_feature))
                optical_factor = torch.sigmoid(model.optical_gates[level](
                    torch.cat((optical_feature, optical_valid, presence, age), dim=1)))
                sar_means[level].extend(sar_factor.mean(dim=(1, 2, 3)).numpy().tolist())
                optical_means[level].extend(optical_factor.mean(dim=(1, 2, 3)).numpy().tolist())
    summary = {}
    for level in range(3):
        for kind, values in (("sar", sar_means[level]), ("optical", optical_means[level])):
            arr = np.asarray(values)
            if len(arr) != 40 or not np.isfinite(arr).all():
                raise RuntimeError("Gate diagnostic frame changed or is nonfinite")
            summary[f"{kind}_level_{level}"] = {
                "mean_tile_factor": float(arr.mean()), "min_tile_factor": float(arr.min()),
                "max_tile_factor": float(arr.max()), "tile_count": len(arr)}
    output = BASE / "v1_p_validation_gate_factors.json"
    output.write_text(json.dumps({
        "purpose": "descriptive v1 P gate factors on frozen v2 validation input frame",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint), "frame_sha256": sha256(frame_path),
        "stats_sha256": sha256(STATS), "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
