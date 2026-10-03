"""Frozen training-only v3 comparison; old tests never enter selection."""

import argparse
import csv
import gzip
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flood_reliability.availability import apply_modality_dropout
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet
from flood_reliability.models_v3 import ResidualSARGatedDualUNet
from scripts.calibrate_impactmesh_formal import GRID
from scripts.train_impactmesh_exploratory_v2 import REASONS, validate
from scripts.train_impactmesh_formal import (
    ELIGIBILITY, ROOT, SPLIT, encode_batch, hash_order, sha256, training_stats,
)

CONFIG = ROOT / "configs" / "impactmesh_exploratory_v3.json"
SPLIT_V3 = ROOT / "data" / "manifests" / "impactmesh_v3_training_dev_processes.csv"
BASE = ROOT / "outputs" / "exploratory_v3"
STATS = BASE / "fit_only_statistics.json"
FRAME = BASE / "development_frame.json"


def prepare(config):
    meta = json.loads(SPLIT_V3.with_suffix(".json").read_text(encoding="utf-8"))
    if (meta["config_sha256"] != sha256(CONFIG) or meta["output_sha256"] != sha256(SPLIT_V3)
            or meta["original_split_sha256"] != sha256(SPLIT)
            or meta["eligibility_sha256"] != sha256(ELIGIBILITY)):
        raise RuntimeError("v3 input manifests changed")
    with SPLIT_V3.open(newline="", encoding="utf-8") as handle:
        roles_v3 = {row["process_id"]: row["v3_role"] for row in csv.DictReader(handle)}
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    groups = {role: defaultdict(list) for role in ("fit", "dev")}
    missing = defaultdict(list)
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = activations[row["activation"]]
            if group["role"] != "train":
                continue
            process = group["process_id"]
            role = roles_v3[process]
            if row["paired_eligible"] == "1":
                groups[role][process].append(row)
            elif role == "dev" and row["failure_reasons"] in REASONS:
                missing[process].append(row)
    if len(groups["fit"]) != 65 or len(groups["dev"]) != 16 or len(missing) != 10:
        raise RuntimeError("v3 fit/development source process counts changed")
    for role in groups:
        groups[role] = {process: hash_order(rows, config["sample_order_seed"])
                        for process, rows in sorted(groups[role].items())}
    paired = {process: rows[:config["development_paired_chips_per_process"]]
              for process, rows in groups["dev"].items()}
    real_missing = {process: hash_order(rows, config["sample_order_seed"] + "|v3_missing")[
        :config["development_real_missing_chips_per_process"]]
        for process, rows in sorted(missing.items())}
    if any(len(rows) != 2 for rows in list(paired.values()) + list(real_missing.values())):
        raise RuntimeError("v3 development frame is incomplete")
    frame = {"paired": paired, "real_missing": real_missing,
             "selection_states": config["selection_states"], "seed": config["seed"],
             "split_sha256": sha256(SPLIT), "v3_split_sha256": sha256(SPLIT_V3),
             "eligibility_sha256": sha256(ELIGIBILITY), "grid_sha256": sha256(GRID)}
    BASE.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(frame, indent=2) + "\n"
    if FRAME.exists() and FRAME.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("v3 development frame changed")
    if not FRAME.exists():
        FRAME.write_text(serialized, encoding="utf-8")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "train"}
    if any(row["sample_id"] not in grids for rows in list(paired.values()) + list(real_missing.values())
           for row in rows):
        raise RuntimeError("Development grid index incomplete")
    return groups["fit"], frame, grids


def fit_stats(reader, fit, config):
    if STATS.exists():
        stats = json.loads(STATS.read_text(encoding="utf-8"))
        if (stats["config_sha256"] != sha256(CONFIG) or
                stats["v3_split_sha256"] != sha256(SPLIT_V3)):
            raise RuntimeError("v3 fit-only statistics source mismatch")
        return stats
    stats = training_stats(reader, fit, config)
    stats.update({"config_sha256": sha256(CONFIG), "v3_split_sha256": sha256(SPLIT_V3),
                  "original_split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
                  "purpose": "fit65 process normalization and class weights only"})
    STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def model_for(candidate, config, device):
    if candidate == "P_residual_v3":
        model = ResidualSARGatedDualUNet(base_width=config["base_width"])
    else:
        model = DualUNet(2, 12, 3, config["base_width"],
                         gated=config["candidates"][candidate]["model"] == "P")
    return model.to(device)


def flood_soft_dice(logits, label):
    active = label != 255
    target = (label == 2).to(logits.dtype)
    probability = torch.softmax(logits, dim=1)[:, 2] * active
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1 - (2 * intersection + 1) / (denominator + 1)


def score_development(evaluation, config):
    states = config["selection_states"]
    if any(evaluation["summary"][state]["flood_iou_defined_processes"] == 0 for state in states):
        raise RuntimeError("A development state has no defined flood IoU")
    return float(np.mean([evaluation["summary"][s]["nll"] for s in states]) -
                 np.mean([evaluation["summary"][s]["flood_iou"] for s in states]))


def train(candidate, config, fit, frame, grids, device, attempt_tag=""):
    output = BASE / (candidate if not attempt_tag else f"{candidate}_{attempt_tag}")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite candidate {output}")
    output.mkdir()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = model_for(candidate, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], foreach=False)
    rng_order = np.random.default_rng(seed + 11)
    rng_dropout = np.random.default_rng(seed + 29)
    code_files = [Path(__file__), ROOT / "src/flood_reliability/models_v3.py",
                  ROOT / "src/flood_reliability/models.py",
                  ROOT / "scripts/train_impactmesh_exploratory_v2.py",
                  ROOT / "scripts/train_impactmesh_formal.py"]
    code_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_files}
    with TrainArchiveReader(ROOT / "data/raw/impactmesh/train") as reader:
        stats = fit_stats(reader, fit, config)
        weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
        best_score, best_epoch = float("inf"), None
        with (output / "epochs.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(config["epochs"]):
                chosen = []
                for process in sorted(fit):
                    rows = fit[process]
                    for k in range(config["training_chips_per_process_per_epoch"]):
                        chosen.append(rows[(epoch * config["training_chips_per_process_per_epoch"] + k) % len(rows)])
                rng_order.shuffle(chosen)
                model.train()
                loss_sum = 0.
                for start in range(0, len(chosen), config["batch_size"]):
                    batch = chosen[start:start + config["batch_size"]]
                    sar, optical, valid, label, offset = encode_batch(reader, batch, stats, device)
                    present = torch.ones(len(batch), dtype=torch.bool, device=device)
                    if config["candidates"][candidate]["dropout"]:
                        drop = torch.tensor(rng_dropout.random(len(batch)) < config["candidates"][candidate]["dropout"],
                                            dtype=torch.bool, device=device)
                        optical, valid, present = apply_modality_dropout(optical, valid, present, drop)
                        offset = torch.where(present, offset, torch.zeros_like(offset))
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(sar, optical, valid, present, normalized_time_offset=offset)
                    ce = F.cross_entropy(logits, label, weight=weights, ignore_index=255)
                    loss = ce + config["flood_dice_weight"] * flood_soft_dice(logits, label)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite v3 loss {candidate} epoch {epoch + 1}")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(batch)
                    if (start // config["batch_size"] + 1) % 25 == 0:
                        print(f"{candidate} epoch {epoch + 1}/{config['epochs']}: "
                              f"{min(start + len(batch), len(chosen))}/{len(chosen)}", flush=True)
                evaluation = validate(model, reader, frame, grids, stats, config, device)
                score = score_development(evaluation, config)
                record = {"candidate": candidate, "epoch": epoch + 1,
                          "train_mean_chip_loss": loss_sum / len(chosen),
                          "development_selection_score": score, "development": evaluation["summary"],
                          "time_utc": datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(record) + "\n")
                log.flush()
                checkpoint = output / f"epoch_{epoch + 1:02d}.pt"
                torch.save({"candidate": candidate, "epoch": epoch + 1,
                            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "config_sha256": sha256(CONFIG), "v3_split_sha256": sha256(SPLIT_V3),
                            "frame_sha256": sha256(FRAME), "stats_sha256": sha256(STATS),
                            "code_sha256": code_hashes, "purpose": "v3 training-only exploratory selection"}, checkpoint)
                if score < best_score:
                    best_score, best_epoch = score, epoch + 1
                    shutil.copyfile(checkpoint, output / "selected.pt")
                print(json.dumps(record), flush=True)
    (output / "run.json").write_text(json.dumps({
        "candidate": candidate, "seed": seed, "spec": config["candidates"][candidate],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection": config["selection_metric"], "selected_epoch": best_epoch,
        "selected_score": best_score, "selected_sha256": sha256(output / "selected.pt"),
        "config_sha256": sha256(CONFIG), "v3_split_sha256": sha256(SPLIT_V3),
        "frame_sha256": sha256(FRAME), "stats_sha256": sha256(STATS),
        "code_sha256": code_hashes, "fit_processes": len(fit),
        "development_processes": len(frame["paired"]),
        "real_missing_development_processes": len(frame["real_missing"]),
        "no_formal_test_or_previous_validation_used_for_selection": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {candidate} epoch {best_epoch}, score {best_score:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("B4_v3", "B5_v3", "P_sigmoid_v3", "P_residual_v3"), required=True)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--attempt-tag", default="")
    args = parser.parse_args()
    if args.attempt_tag and (not args.attempt_tag.isascii() or not args.attempt_tag.replace("_", "").isalnum()):
        raise ValueError("Attempt tag must contain ASCII letters, digits, or underscores")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["selection_states"] != ["full", "half", "zero", "real_missing"]:
        raise RuntimeError("v3 states changed")
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    fit, frame, grids = prepare(config)
    train(args.candidate, config, fit, frame, grids, torch.device(args.device), args.attempt_tag)


if __name__ == "__main__":
    main()
