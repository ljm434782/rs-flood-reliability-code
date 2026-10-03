"""Measure learned SAR gate factors on fit-only chips; no evaluation labels."""

import json
from datetime import datetime, timezone

import torch

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.train_impactmesh_exploratory_v3 import BASE, CONFIG, STATS, model_for, prepare
from scripts.train_impactmesh_formal import ROOT, encode_batch, sha256

OUT = BASE / "gate_mechanism_fit_only.json"
CANDIDATES = ("P_sigmoid_v3", "P_residual_v3")


def main():
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite fit-only mechanism diagnostic")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fit, _, _ = prepare(config)
    sample = [rows[0] for _, rows in sorted(fit.items())]
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    results = {}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        for candidate in CANDIDATES:
            checkpoint = BASE / candidate / "selected.pt"
            run = json.loads((BASE / candidate / "run.json").read_text(encoding="utf-8"))
            if run["selected_sha256"] != sha256(checkpoint):
                raise RuntimeError("Selected gate checkpoint changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = model_for(candidate, config, device).eval()
            model.load_state_dict(saved["model_state"])
            accum = [{"count": 0, "sum": 0., "sum_sq": 0., "above_one": 0,
                      "minimum": float("inf"), "maximum": float("-inf")}
                     for _ in model.sar_gates]
            hooks = []
            for level, gate in enumerate(model.sar_gates):
                def collect(_module, _inputs, output, index=level):
                    factor = (torch.sigmoid(output) if candidate == "P_sigmoid_v3"
                              else 1 + .5 * torch.tanh(output))
                    record = accum[index]
                    record["count"] += factor.numel()
                    record["sum"] += float(factor.sum().cpu())
                    record["sum_sq"] += float(factor.square().sum().cpu())
                    record["above_one"] += int((factor > 1).sum().cpu())
                    record["minimum"] = min(record["minimum"], float(factor.min().cpu()))
                    record["maximum"] = max(record["maximum"], float(factor.max().cpu()))
                hooks.append(gate.register_forward_hook(collect))
            try:
                with torch.inference_mode():
                    for start in range(0, len(sample), config["batch_size"]):
                        batch = sample[start:start + config["batch_size"]]
                        sar, optical, valid, _, offset = encode_batch(reader, batch, stats, device)
                        present = torch.ones(len(batch), dtype=torch.bool, device=device)
                        model(sar, optical, valid, present, normalized_time_offset=offset)
            finally:
                for hook in hooks:
                    hook.remove()
            levels = []
            for record in accum:
                count = record["count"]
                mean = record["sum"] / count
                levels.append({"mean": mean,
                               "std": max(0., record["sum_sq"] / count - mean * mean) ** .5,
                               "minimum": record["minimum"], "maximum": record["maximum"],
                               "fraction_above_one": record["above_one"] / count,
                               "feature_values": count})
            results[candidate] = {"checkpoint_sha256": sha256(checkpoint),
                                  "selected_epoch": run["selected_epoch"], "levels": levels}
            print(candidate, levels, flush=True)
    OUT.write_text(json.dumps({"purpose": "training-fit-only gate factor diagnostic",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rule": "first hash-ordered paired chip in each of 65 fit processes",
        "sample_ids": [row["sample_id"] for row in sample],
        "config_sha256": sha256(CONFIG), "stats_sha256": sha256(STATS),
        "label_values_not_used": True, "validation_or_test_not_used": True,
        "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
