"""Validation-only, post-v1-test exploratory training; never touches v1 outputs."""

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

from flood_reliability.availability import apply_modality_dropout
from flood_reliability.evaluation_counts import confusion_counts
from flood_reliability.exploratory_models import OpticalOnlyGatedDualUNet
from flood_reliability.formal_metrics import class_iou
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.models import DualUNet
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import GRID, STATS
from scripts.evaluate_impactmesh_formal import block_valid
from scripts.train_impactmesh_formal import (
    ELIGIBILITY, ROOT, SPLIT, encode_batch, hash_order, prepare_rows, sha256,
)


CONFIG = ROOT / "configs" / "impactmesh_exploratory_v2.json"
BASE = ROOT / "outputs" / "exploratory_v2"
STATES = ("full", "half", "zero", "real_missing")
REASONS = {"visible_below_0.5", "no_pre_s1_s2_within_30d"}


def make_model(candidate: str, config: dict, device):
    spec = config["candidates"][candidate]
    if candidate == "P_no_sar_gate":
        return OpticalOnlyGatedDualUNet(base_width=config["base_width"]).to(device)
    return DualUNet(2, 12, 3, config["base_width"], gated=spec["model"] == "P").to(device)


def fixed_frame(config: dict):
    pool = prepare_rows(config)
    paired = {process: values[:config["validation_paired_chips_per_process"]]
              for process, values in sorted(pool["validation"].items())}
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        roles = {row["activation"]: row for row in csv.DictReader(handle)}
    missing = defaultdict(list)
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            info = roles[row["activation"]]
            if (info["role"] == "validation" and row["paired_eligible"] == "0"
                    and row["failure_reasons"] in REASONS):
                missing[info["process_id"]].append(row)
    missing = {process: hash_order(values, config["sample_order_seed"] + "|v2_missing")[
        :config["validation_real_missing_chips_per_process"]]
        for process, values in sorted(missing.items())}
    if len(paired) != 20 or len(missing) != 13:
        raise RuntimeError("Frozen validation frame count changed")
    frame = {"paired": paired, "real_missing": missing,
             "selection_states": list(STATES), "seed": config["seed"],
             "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
             "grid_sha256": sha256(GRID), "stats_sha256": sha256(STATS)}
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / "validation_frame.json"
    serialized = json.dumps(frame, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Validation frame differs from earlier v2 run")
    if not path.exists():
        path.write_text(serialized, encoding="utf-8")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "train"}
    for rows in list(paired.values()) + list(missing.values()):
        if not all(row["sample_id"] in grids for row in rows):
            raise RuntimeError("Validation grid index is incomplete")
    return pool, frame, grids, path


def summarize_process(logits, labels, owners):
    confusion = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.
    pixels = 0
    for prediction_logits, target, owner in zip(logits, labels, owners):
        active = owner & (target != 255)
        if not active.any():
            continue
        prediction = np.argmax(prediction_logits, axis=0)
        confusion += confusion_counts(target, prediction, num_classes=3,
                                      evaluation_mask=owner)
        shifted = prediction_logits - prediction_logits.max(axis=0, keepdims=True)
        log_denominator = np.log(np.exp(shifted, dtype=np.float64).sum(axis=0))
        rows, cols = np.nonzero(active)
        truth = target[rows, cols].astype(np.int64)
        loss_sum += float(np.sum(log_denominator[rows, cols] - shifted[truth, rows, cols]))
        pixels += len(rows)
    if not pixels:
        raise RuntimeError("Validation process has no usable label pixels")
    return {"nll": loss_sum / pixels, "flood_iou": class_iou(confusion, 2),
            "pixels": pixels, "confusion": confusion.tolist()}


def infer(model, sar, optical, valid, offset):
    present = valid.flatten(1).any(dim=1)
    safe_optical = torch.where(valid, optical, torch.zeros_like(optical))
    safe_offset = torch.where(present, offset, torch.zeros_like(offset))
    with torch.inference_mode():
        return model(sar, safe_optical, valid, present,
                     normalized_time_offset=safe_offset).cpu().numpy()


def validate(model, reader, frame, grids, stats, config, device):
    model.eval()
    records = {state: {} for state in STATES}
    for process, rows in frame["paired"].items():
        sar, optical, valid, target, offset = encode_batch(reader, rows, stats, device)
        original = valid.cpu().numpy()[:, 0]
        labels = target.cpu().numpy()
        preceding = []
        owners = []
        for row in rows:
            owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding))
            preceding.append(grids[row["sample_id"]])
        for state, level in (("full", 100), ("half", 50), ("zero", 0)):
            kept = np.stack([block_valid(original[i], row["sample_id"],
                                         level, config["seed"])
                             for i, row in enumerate(rows)])
            kept_t = torch.from_numpy(kept[:, None]).to(device)
            logits = infer(model, sar, optical, kept_t, offset)
            records[state][process] = summarize_process(logits, labels, owners)
    means = np.asarray(stats["sar_mean"], dtype=np.float32)[:, None, None]
    stds = np.asarray(stats["sar_std"], dtype=np.float32)[:, None, None]
    for process, rows in frame["real_missing"].items():
        arrays = [reader.load_s1_label(row["sample_id"]) for row in rows]
        sar = torch.from_numpy(np.stack([((s1 - means) / stds).astype(np.float32)
                                         for s1, _ in arrays])).to(device)
        labels = [target for _, target in arrays]
        optical = torch.zeros((len(rows), 12, 256, 256), device=device)
        valid = torch.zeros((len(rows), 1, 256, 256), dtype=torch.bool, device=device)
        offset = torch.zeros(len(rows), device=device)
        preceding = []
        owners = []
        for row in rows:
            owners.append(unique_pixel_mask(grids[row["sample_id"]], preceding))
            preceding.append(grids[row["sample_id"]])
        logits = infer(model, sar, optical, valid, offset)
        records["real_missing"][process] = summarize_process(logits, labels, owners)
    summary = {}
    for state in STATES:
        values = list(records[state].values())
        nll = np.asarray([value["nll"] for value in values])
        iou = np.asarray([np.nan if value["flood_iou"] is None else
                          value["flood_iou"] for value in values])
        summary[state] = {"processes": len(values), "nll": float(np.mean(nll)),
                          "flood_iou": float(np.nanmean(iou)),
                          "flood_iou_defined_processes": int(np.isfinite(iou).sum())}
    selection_score = float(np.mean([summary[state]["nll"] for state in STATES]))
    return {"selection_score": selection_score, "summary": summary, "processes": records}


def check_inputs(config):
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if stats["split_sha256"] != sha256(SPLIT) or stats["eligibility_sha256"] != sha256(ELIGIBILITY):
        raise RuntimeError("v1 train-only statistics do not match frozen source manifests")
    if config["selection_states"] != list(STATES):
        raise RuntimeError("v2 selection states changed")
    return stats


def diagnose_v1(name, config, stats, frame, grids, frame_path, device):
    model = (DualUNet(2, 12, 3, config["base_width"], gated=name == "P")).to(device)
    checkpoint = ROOT / "outputs" / "formal_v1" / name / "latest.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["model"] != name or payload["epoch"] != 8:
        raise RuntimeError("Unexpected v1 checkpoint")
    model.load_state_dict(payload["model_state"])
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        evaluation = validate(model, reader, frame, grids, stats, config, device)
    output = BASE / f"v1_validation_diagnostic_{name}.json"
    output.write_text(json.dumps({"purpose": "v1 final checkpoint on v2 validation frame only",
                                  "model": name, "checkpoint_sha256": sha256(checkpoint),
                                  "frame_sha256": sha256(frame_path),
                                  "evaluation": evaluation}, indent=2) + "\n", encoding="utf-8")
    print(f"v1 validation diagnostic {name}: {evaluation['summary']}; "
          f"score={evaluation['selection_score']:.6f}", flush=True)


def train(candidate, config, stats, pool, frame, grids, frame_path, device, attempt_tag=""):
    output_name = candidate if not attempt_tag else f"{candidate}_{attempt_tag}"
    output = BASE / output_name
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite exploratory candidate: {output}")
    output.mkdir(parents=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = make_model(candidate, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], foreach=False)
    weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    rng_order = np.random.default_rng(seed + 11)
    rng_dropout = np.random.default_rng(seed + 29)
    spec = config["candidates"][candidate]
    best_score = float("inf")
    best_epoch = None
    code_files = [Path(__file__), ROOT / "src/flood_reliability/exploratory_models.py",
                  ROOT / "src/flood_reliability/models.py"]
    code_hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_files}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        with (output / "epochs.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(config["epochs"]):
                chosen = []
                for process in sorted(pool["train"]):
                    values = pool["train"][process]
                    for k in range(config["training_chips_per_process_per_epoch"]):
                        chosen.append(values[(epoch * config["training_chips_per_process_per_epoch"] + k)
                                             % len(values)])
                rng_order.shuffle(chosen)
                model.train()
                loss_sum = 0.
                for start in range(0, len(chosen), config["batch_size"]):
                    batch = chosen[start:start + config["batch_size"]]
                    sar, optical, valid, label, offset = encode_batch(reader, batch, stats, device)
                    present = torch.ones(len(batch), dtype=torch.bool, device=device)
                    if spec["dropout"]:
                        drop = torch.tensor(rng_dropout.random(len(batch)) < spec["dropout"],
                                            dtype=torch.bool, device=device)
                        optical, valid, present = apply_modality_dropout(optical, valid, present, drop)
                        offset = torch.where(present, offset, torch.zeros_like(offset))
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(sar, optical, valid, present, normalized_time_offset=offset)
                    loss = torch.nn.functional.cross_entropy(logits, label, weight=weights,
                                                             ignore_index=255)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite v2 loss: {candidate} epoch {epoch + 1}")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach().cpu()) * len(batch)
                    if (start // config["batch_size"] + 1) % 25 == 0:
                        print(f"{candidate} epoch {epoch + 1}/{config['epochs']}: "
                              f"{min(start + len(batch), len(chosen))}/{len(chosen)}", flush=True)
                evaluation = validate(model, reader, frame, grids, stats, config, device)
                score = evaluation["selection_score"]
                record = {"candidate": candidate, "epoch": epoch + 1,
                          "train_weighted_mean_chip_loss": loss_sum / len(chosen),
                          "validation_selection_score": score,
                          "validation": evaluation["summary"],
                          "time_utc": datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(record) + "\n")
                log.flush()
                checkpoint = output / f"epoch_{epoch + 1:02d}.pt"
                torch.save({"candidate": candidate, "epoch": epoch + 1,
                            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "config_sha256": sha256(CONFIG), "frame_sha256": sha256(frame_path),
                            "stats_sha256": sha256(STATS), "code_sha256": code_hashes,
                            "purpose": "post-v1-test exploratory; validation-only selection"}, checkpoint)
                if score < best_score:
                    best_score, best_epoch = score, epoch + 1
                    shutil.copyfile(checkpoint, output / "selected.pt")
                print(json.dumps(record), flush=True)
    (output / "run.json").write_text(json.dumps({
        "candidate": candidate, "output_attempt": output_name, "spec": spec, "seed": seed,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection": config["selection_metric"], "selected_epoch": best_epoch,
        "selected_score": best_score, "selected_sha256": sha256(output / "selected.pt"),
        "config_sha256": sha256(CONFIG), "frame_sha256": sha256(frame_path),
        "stats_sha256": sha256(STATS), "split_sha256": sha256(SPLIT),
        "eligibility_sha256": sha256(ELIGIBILITY), "code_sha256": code_hashes,
        "train_processes": len(pool["train"]), "validation_processes": len(frame["paired"]),
        "real_missing_validation_processes": len(frame["real_missing"]),
        "no_test_labels_or_metrics_used_for_selection": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {candidate} epoch {best_epoch}, validation score {best_score:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", choices=("B4_bestval", "B5_drop050", "B5_drop025",
                                                "P_gate_both", "P_no_sar_gate"))
    group.add_argument("--diagnose-v1", choices=("B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--attempt-tag", default="", help="New safe output directory suffix after interruption")
    args = parser.parse_args()
    if args.attempt_tag and (not args.candidate or not args.attempt_tag.isascii()
                             or not args.attempt_tag.replace("_", "").isalnum()):
        raise ValueError("Attempt tag requires a candidate and ASCII letters/digits/underscores")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = check_inputs(config)
    pool, frame, grids, frame_path = fixed_frame(config)
    if args.device == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    device = torch.device(args.device)
    if args.diagnose_v1:
        diagnose_v1(args.diagnose_v1, config, stats, frame, grids, frame_path, device)
    else:
        train(args.candidate, config, stats, pool, frame, grids, frame_path, device,
              args.attempt_tag)


if __name__ == "__main__":
    main()
