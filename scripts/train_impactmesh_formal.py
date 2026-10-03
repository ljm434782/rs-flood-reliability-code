"""Fit a preregistered ImpactMesh model on process-disjoint official-train roles."""

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flood_reliability.availability import apply_modality_dropout
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet, SmallUNet
from flood_reliability.splits import require_process_disjoint


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SPLIT = MANIFESTS / "impactmesh_process_split_grouped.csv"
ELIGIBILITY = MANIFESTS / "impactmesh_all_train_frozen_eligibility.csv.gz"
CONFIG = ROOT / "configs" / "impactmesh_formal_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_order(values: list[dict], key: str) -> list[dict]:
    return sorted(values, key=lambda row: (hashlib.sha256(
        f"{key}|{row['sample_id']}".encode("ascii")).hexdigest(), row["sample_id"]))


def prepare_rows(config: dict):
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    roles = defaultdict(set)
    for row in activations.values():
        roles[row["role"]].add(row["process_id"])
    require_process_disjoint({role: sorted(roles[role]) for role in (
        "train", "validation", "calibration", "internal_test", "official_test_holdout")})
    pool = defaultdict(lambda: defaultdict(list))
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["paired_eligible"] != "1":
                continue
            activation = activations[row["activation"]]
            role = activation["role"]
            if role in ("train", "validation"):
                pool[role][activation["process_id"]].append(row)
    if len(pool["validation"]) != 20 or len(pool["train"]) < 40:
        raise RuntimeError("Expected frozen independent train/validation process roles")
    for role, groups in pool.items():
        for process_id, rows in groups.items():
            if len(rows) < 5:
                raise RuntimeError(f"Eligible process has fewer than five tiles: {process_id}")
            groups[process_id] = hash_order(rows, config["sample_order_seed"])
    return pool


def training_stats(reader: TrainArchiveReader, groups: dict, config: dict):
    # Fixed label-blind sample selection from the *training* process pool only.
    selected = [row for process in sorted(groups)
                for row in groups[process][:config["normalization_chips_per_process"]]]
    sar_sum = np.zeros(2, np.float64)
    sar_sq = np.zeros(2, np.float64)
    optical_sum = np.zeros(12, np.float64)
    optical_sq = np.zeros(12, np.float64)
    optical_count = 0
    class_count = np.zeros(3, np.int64)
    for index, row in enumerate(selected, 1):
        s1, s2, valid, label = reader.load(row["sample_id"], int(row["selected_time_index"]))
        sar_sum += s1.sum(axis=(1, 2), dtype=np.float64)
        sar_sq += np.square(s1.astype(np.float64)).sum(axis=(1, 2))
        values = s2[:, valid].astype(np.float64)
        optical_sum += values.sum(axis=1)
        optical_sq += np.square(values).sum(axis=1)
        optical_count += int(valid.sum())
        class_count += np.bincount(label[label != 255].ravel(), minlength=3)
        if index % 100 == 0:
            print(f"train-only stats: {index}/{len(selected)}", flush=True)
    sar_count = len(selected) * 256 * 256
    sar_mean = sar_sum / sar_count
    sar_std = np.sqrt(np.maximum(sar_sq / sar_count - sar_mean ** 2, 1e-12))
    optical_mean = optical_sum / optical_count
    optical_std = np.sqrt(np.maximum(optical_sq / optical_count - optical_mean ** 2, 1e-12))
    if not all(np.isfinite(a).all() and (a > 0).all() for a in (sar_std, optical_std)):
        raise RuntimeError("Invalid train-only normalization statistics")
    # Inverse square-root frequency relative to non-water, capped in advance.
    weights = np.sqrt(class_count[0] / np.maximum(class_count, 1))
    weights = np.clip(weights, 1., config["maximum_class_weight"])
    return {"sar_mean": sar_mean.tolist(), "sar_std": sar_std.tolist(),
            "optical_mean": optical_mean.tolist(), "optical_std": optical_std.tolist(),
            "class_count": class_count.tolist(), "class_weights": weights.tolist(),
            "training_stat_tile_count": len(selected), "training_stat_optical_pixels": optical_count,
            "training_stat_sample_ids": [row["sample_id"] for row in selected]}


def encode_batch(reader, rows, stats, device):
    s1_list, s2_list, valid_list, label_list, age_list = [], [], [], [], []
    s1_mean = np.array(stats["sar_mean"], dtype=np.float32)[:, None, None]
    s1_std = np.array(stats["sar_std"], dtype=np.float32)[:, None, None]
    s2_mean = np.array(stats["optical_mean"], dtype=np.float32)[:, None, None]
    s2_std = np.array(stats["optical_std"], dtype=np.float32)[:, None, None]
    for row in rows:
        s1, s2, valid, label = reader.load(row["sample_id"], int(row["selected_time_index"]))
        if abs(float(row["selected_joint_visible_fraction"]) - valid.mean()) > 1e-6:
            raise RuntimeError(f"Optical validity changed since frozen audit: {row['sample_id']}")
        s1_list.append(((s1 - s1_mean) / s1_std).astype(np.float32))
        s2_list.append(np.where(valid[None], (s2 - s2_mean) / s2_std, 0).astype(np.float32))
        valid_list.append(valid[None])
        label_list.append(label)
        age_list.append(-float(row["selected_age_days"]) / 30.0)
    return (torch.from_numpy(np.stack(s1_list)).to(device),
            torch.from_numpy(np.stack(s2_list)).to(device),
            torch.from_numpy(np.stack(valid_list)).to(device),
            torch.from_numpy(np.stack(label_list)).to(device),
            torch.tensor(age_list, dtype=torch.float32, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Configured XPU unavailable")
    if (ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8").find(
            "状态：**正式训练规则已冻结") < 0:
        raise RuntimeError("Formal preregistration has not been frozen")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "xpu":
        torch.xpu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    groups = prepare_rows(config)
    device = torch.device(args.device)
    model = (SmallUNet(2, 3, config["base_width"]) if args.model == "B2" else
             DualUNet(2, 12, 3, config["base_width"], gated=args.model == "P")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], foreach=False)
    output = ROOT / "outputs" / "formal_v1" / args.model
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    config_hash = sha256(CONFIG)
    split_hash = sha256(SPLIT)
    eligibility_hash = sha256(ELIGIBILITY)
    code_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (
        Path(__file__), ROOT / "src/flood_reliability/models.py",
        ROOT / "src/flood_reliability/impactmesh_io.py",
        ROOT / "src/flood_reliability/availability.py")}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        stats_path = ROOT / "outputs" / "formal_v1" / "train_only_statistics.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if (stats["split_sha256"] != split_hash
                    or stats["eligibility_sha256"] != eligibility_hash
                    or stats["config_sha256"] != config_hash):
                raise RuntimeError("Train-only statistics do not match frozen data")
        else:
            stats = training_stats(reader, groups["train"], config)
            stats.update({"split_sha256": split_hash, "eligibility_sha256": eligibility_hash,
                          "config_sha256": config_hash})
            stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
        rng = np.random.default_rng(seed + {"B2": 2, "B4": 4, "B5": 5, "P": 7}[args.model])
        val_rows = [row for process in sorted(groups["validation"])
                    for row in groups["validation"][process][:config["validation_chips_per_process"]]]
        log_path = output / "epochs.jsonl"
        with log_path.open("w", encoding="utf-8") as log:
            for epoch in range(config["epochs"]):
                chosen = []
                for process in sorted(groups["train"]):
                    values = groups["train"][process]
                    for k in range(config["training_chips_per_process_per_epoch"]):
                        chosen.append(values[(epoch * config["training_chips_per_process_per_epoch"] + k)
                                             % len(values)])
                rng.shuffle(chosen)
                model.train()
                loss_sum = 0.
                for start in range(0, len(chosen), config["batch_size"]):
                    batch = chosen[start:start + config["batch_size"]]
                    sar, optical, valid, label, time_offset = encode_batch(reader, batch, stats, device)
                    present = torch.ones(len(batch), dtype=torch.bool, device=device)
                    if args.model in ("B5", "P"):
                        drop = torch.tensor(rng.random(len(batch)) < config["optical_modality_dropout"],
                                            dtype=torch.bool, device=device)
                        optical, valid, present = apply_modality_dropout(optical, valid, present, drop)
                        if args.model == "P":
                            time_offset = torch.where(present, time_offset, torch.zeros_like(time_offset))
                    optimizer.zero_grad(set_to_none=True)
                    logits = (model(sar) if args.model == "B2" else
                              model(sar, optical, valid, present, normalized_time_offset=time_offset))
                    loss = torch.nn.functional.cross_entropy(logits, label, weight=weights, ignore_index=255)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss in {args.model} epoch {epoch + 1}")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(batch)
                    if (start // config["batch_size"] + 1) % 25 == 0:
                        print(f"{args.model} epoch {epoch + 1}/{config['epochs']}: "
                              f"{min(start + len(batch), len(chosen))}/{len(chosen)} train chips", flush=True)
                model.eval()
                val_loss, val_pixels = 0., 0
                with torch.inference_mode():
                    for start in range(0, len(val_rows), config["batch_size"]):
                        batch = val_rows[start:start + config["batch_size"]]
                        sar, optical, valid, label, time_offset = encode_batch(reader, batch, stats, device)
                        present = torch.ones(len(batch), dtype=torch.bool, device=device)
                        logits = (model(sar) if args.model == "B2" else
                                  model(sar, optical, valid, present, normalized_time_offset=time_offset))
                        per_pixel = torch.nn.functional.cross_entropy(
                            logits, label, ignore_index=255, reduction="none")
                        count = int((label != 255).sum().cpu())
                        val_loss += float(per_pixel.sum().cpu())
                        val_pixels += count
                entry = {"model": args.model, "epoch": epoch + 1,
                         "train_weighted_mean_chip_loss": loss_sum / len(chosen),
                         "validation_nll": val_loss / val_pixels,
                         "validation_pixels": val_pixels,
                         "train_chips": len(chosen), "validation_chips": len(val_rows),
                         "time_utc": datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                torch.save({"model": args.model, "epoch": epoch + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "stats_sha256": sha256(stats_path), "config_sha256": config_hash,
                            "split_sha256": split_hash, "eligibility_sha256": eligibility_hash,
                            "seed": seed, "code_sha256": code_hashes,
                            "purpose": "formal v1; no internal/external test used in fitting"}, checkpoint)
                print(json.dumps(entry), flush=True)
    (output / "run.json").write_text(json.dumps({
        "model": args.model, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device, "torch": torch.__version__,
        "config": config, "config_sha256": config_hash, "split_sha256": split_hash,
        "eligibility_sha256": eligibility_hash, "statistics_sha256": sha256(stats_path),
        "code_sha256": code_hashes, "checkpoint_sha256": sha256(checkpoint),
        "training_processes": len(groups["train"]), "validation_processes": len(groups["validation"]),
        "selection": "fixed final epoch, no validation-based model selection",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
