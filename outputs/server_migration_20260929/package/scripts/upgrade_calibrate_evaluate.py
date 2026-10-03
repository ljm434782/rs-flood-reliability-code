"""Calibration and fixed retrospective evaluation helpers for G3.

The array lifecycle is reader-agnostic for synthetic tests. Real-data callers
verify the immutable G3 freeze and independent PASS attestation before opening
any project archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from flood_reliability.overlap import unique_pixel_mask
from flood_reliability.upgrade_masks import irregular_retention_mask
from flood_reliability.upgrade_metrics import (
    CalibrationCell, grouped_process_bootstrap, score_process_state,
    fit_global_temperature,
)

CONFIG_PATH = ROOT / "configs" / "upgrade_20260929.json"
STATES = ("100", "75", "50", "25", "0")
IRREGULAR_STATES = ("irregular_75", "irregular_50", "irregular_25")
METRICS = ("binary_flood_brier", "flood_iou", "flood_precision", "flood_recall",
           "flood_f1", "flood_auprc", "macro_iou", "multiclass_nll",
           "multiclass_brier", "ece15")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dual_freeze(root: Path, config: dict) -> None:
    """Use the shared attestation gate before constructing any project reader."""
    if config.get("status") != "FROZEN":
        raise RuntimeError("calibration/evaluation blocked: config is not FROZEN")
    freeze_path = root / "G3_FREEZE_MANIFEST.json"
    from flood_reliability.upgrade_execution import require_dual_freeze as shared_gate
    shared_gate(root, config)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    locked_by_path = {str(item.get("path", "")).replace("\\", "/"): item.get("sha256")
                      for item in freeze.get("locked_files", [])}
    required_code = (
        "scripts/upgrade_calibrate_evaluate.py",
        "tests/test_upgrade_calibrate_evaluate.py",
        "scripts/upgrade_train.py",
        "scripts/upgrade_b1_fit.py",
        "scripts/upgrade_b1_eval.py",
        "scripts/fit_impactmesh_b1.py",
        "src/flood_reliability/upgrade_execution.py",
        "src/flood_reliability/upgrade_metrics.py",
        "src/flood_reliability/upgrade_masks.py",
        "src/flood_reliability/impactmesh_io.py",
        "src/flood_reliability/overlap.py",
        "src/flood_reliability/models.py",
        "src/flood_reliability/smagnet_adapter.py",
        "third_party/smagnet/smagnet.py",
    )
    for relative in required_code:
        target = root / relative
        if not target.is_file() or locked_by_path.get(relative) != sha256(target):
            raise RuntimeError(f"calibration/evaluation blocked: required executable hash is not frozen: {relative}")


@dataclass(frozen=True)
class TilePrediction:
    process_id: str
    sample_id: str
    state_id: str
    logits: np.ndarray  # [3,H,W] or [H,W,3]
    labels: np.ndarray  # [H,W], values 0/1/2/255
    owner_mask: np.ndarray  # [H,W], bool


def _flat_tile(tile: TilePrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(tile.logits)
    labels = np.asarray(tile.labels)
    owner = np.asarray(tile.owner_mask)
    if logits.ndim != 3 or labels.ndim != 2 or owner.shape != labels.shape or owner.dtype != np.bool_:
        raise ValueError("tile requires [3,H,W] or [H,W,3] logits and matching labels/boolean owner")
    if logits.shape == (3, *labels.shape):
        flat = np.moveaxis(logits, 0, -1).reshape(-1, 3)
    elif logits.shape == (*labels.shape, 3):
        flat = logits.reshape(-1, 3)
    else:
        raise ValueError("logit tile shape does not match labels")
    truth = labels.reshape(-1)
    owner_flat = owner.reshape(-1)
    if not np.issubdtype(truth.dtype, np.integer) or not np.all(np.isin(truth, (0, 1, 2, 255))):
        raise ValueError("labels must contain only 0, 1, 2, or Ignore=255")
    # Ignore pixels are excluded even if geometrically owned; invalid logits at
    # ignored/nonowned locations never enter metrics or calibration.
    return flat, truth, owner_flat


def manifest_owner_masks(manifest_path: Path, grid_path: Path, *, expected_role: str) -> dict[str, tuple[str, np.ndarray]]:
    """Construct deterministic per-process unique-pixel owners from frozen manifests."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with grid_path.open(newline="", encoding="utf-8") as f:
        grids = {r["sample_id"]: r for r in csv.DictReader(f)}
    by_process: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"duplicate sample in role manifest: {sid}")
        seen.add(sid)
        if row.get("derived_role") != expected_role:
            raise ValueError(f"manifest role mismatch for {sid}")
        process = row.get("conservative_unit_id") or row.get("process_id")
        if not process or sid not in grids:
            raise ValueError(f"missing process/grid metadata for {sid}")
        grid = grids[sid]
        # The dataset's filename zone token can differ from its actual raster
        # CRS. Ownership uses audited TIFF headers, never filename coordinates.
        if not grid.get("grid_crs"):
            raise ValueError(f"actual raster CRS is missing for {sid}")
        by_process[process].append(row)
    result = {}
    for process, members in by_process.items():
        preceding = []
        for row in sorted(members, key=lambda r: r["sample_id"]):
            sid = row["sample_id"]
            grid = grids[sid]
            owner = unique_pixel_mask(grid, preceding)
            result[sid] = (process, owner)
            preceding.append(grid)
    if len(result) != len(rows):
        raise RuntimeError("owner construction omitted manifest tiles")
    return result


def calibration_cells(tiles: Iterable[TilePrediction], *, strategy: str) -> list[CalibrationCell]:
    """Build complete calibration process×state cells from one model/seed cache."""
    grouped: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    allowed = {"100"} if strategy == "original" else set(STATES) if strategy == "multistate" else set()
    if not allowed:
        raise ValueError("strategy must be original or multistate")
    tile_ids = set()
    for tile in tiles:
        if not tile.process_id or not tile.sample_id or tile.state_id not in allowed:
            raise ValueError("invalid calibration tile identity/state")
        key = (tile.process_id, tile.sample_id, tile.state_id)
        if key in tile_ids:
            raise ValueError(f"duplicate calibration tile/state {key}")
        tile_ids.add(key)
        grouped[(tile.process_id, tile.state_id)].append(_flat_tile(tile))
    if not grouped:
        raise ValueError("calibration cache is empty")
    processes = sorted({key[0] for key in grouped})
    cells = []
    for process in processes:
        for state in sorted(allowed):
            parts = grouped.get((process, state))
            if not parts:
                raise ValueError(f"missing calibration process×state cell {(process, state)}")
            logits = np.concatenate([x[0] for x in parts])
            labels = np.concatenate([x[1] for x in parts])
            owners = np.concatenate([x[2] for x in parts])
            cells.append(CalibrationCell(process, state, logits, labels, owners))
    return cells


def tiles_from_frozen_manifest(manifest_path: Path, grid_path: Path, *, role: str,
                               states: Sequence[str],
                               predict_tile: Callable[[str, str, str, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> list[TilePrediction]:
    """Call an injected reader/model on manifest IDs with manifest-derived owner masks.

    ``predict_tile(sample_id, process_id, state_id, owner_mask)`` returns the
    tile logits and labels. This helper itself does not open data and is thus
    synthetic-fixture friendly; real callers must use ``run_real_after_freeze``.
    """
    owners = manifest_owner_masks(manifest_path, grid_path, expected_role=role)
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["sample_id"])
    result = []
    for row in rows:
        sid = row["sample_id"]
        process, owner = owners[sid]
        for state in states:
            logits, labels = predict_tile(sid, process, state, owner.copy())
            result.append(TilePrediction(process, sid, state, logits, labels, owner.copy()))
    return result


def fit_and_record_temperature(tiles: Iterable[TilePrediction], *, model: str, seed: int,
                               strategy: str, hashes: Mapping[str, str] | None = None) -> dict:
    """Fit one positive global T; for multistate, the same T applies to every state."""
    cells = calibration_cells(tiles, strategy=strategy)
    fit = fit_global_temperature(cells, strategy=strategy)
    return {
        "model": model, "seed": int(seed), "calibration_strategy": strategy,
        "temperature": fit.temperature, "log_temperature": fit.log_temperature,
        "objective_nll": fit.objective_nll, "boundary_hit": fit.boundary_hit,
        "optimizer_success": fit.optimizer_success, "evaluations": fit.evaluations,
        "process_ids": list(fit.process_ids),
        "state_ids": list(STATES if strategy == "multistate" else ("100",)),
        "application": "single fixed global T across all evaluation availability states",
        "calibration_hashes": dict(hashes or {}),
    }


def _metric_value(record: dict, metric: str):
    if metric == "ece15":
        return record["reliability"]["ece15"]
    return record[metric]


def evaluate_matrix(cells: Iterable[tuple[str, int, str, TilePrediction]],
                    temperatures: Mapping[tuple[str, int, str], float], *,
                    evaluation_role: str, bootstrap_repetitions: int = 10_000,
                    bootstrap_seed: int = 20261029) -> dict:
    """Fixed evaluation over (model, seed, calibration strategy, tile) rows."""
    if evaluation_role not in {"internal_retro", "holdout_retro"}:
        raise ValueError("evaluation_role must be an explicitly frozen retrospective role")
    grouped: dict[tuple[str, int, str, str, str], list[TilePrediction]] = defaultdict(list)
    seen = set()
    for model, seed, strategy, tile in cells:
        if strategy not in {"original", "multistate"}:
            raise ValueError("unknown calibration strategy")
        if tile.state_id not in STATES:
            raise ValueError("evaluation state must be one of the five frozen states")
        key_tile = (model, int(seed), strategy, tile.process_id, tile.sample_id, tile.state_id)
        if key_tile in seen:
            raise ValueError(f"duplicate evaluation tile {key_tile}")
        seen.add(key_tile)
        grouped[(model, int(seed), strategy, tile.process_id, tile.state_id)].append(tile)
    if not grouped:
        raise ValueError("evaluation matrix is empty")
    per_process: dict[str, dict] = {}
    # Persist mergeable primitive metrics and full reliability bin counts.
    for (model, seed, strategy, process, state), tiles_here in sorted(grouped.items()):
        key = (model, seed, strategy)
        if key not in temperatures or not np.isfinite(temperatures[key]) or temperatures[key] <= 0:
            raise ValueError(f"missing/nonpositive frozen temperature for {key}")
        flat_logits, flat_labels, flat_owner = [], [], []
        for tile in tiles_here:
            z, y, owner = _flat_tile(tile)
            flat_logits.append(z); flat_labels.append(y); flat_owner.append(owner)
        logits = np.concatenate(flat_logits)
        labels = np.concatenate(flat_labels)
        owner = np.concatenate(flat_owner)
        for tag, temperature in (("raw", 1.0), ("calibrated", float(temperatures[key]))):
            score = score_process_state(logits, labels, owner, temperature=temperature)
            per_process[f"{model}|{seed}|{strategy}|{process}|{state}|{tag}"] = score

    # Each summary is stratified by one candidate/seed/strategy/metric. Common
    # process support is intersection over the five states for that stratum.
    summaries = {}
    strata = sorted({key[:3] for key in grouped})
    tags = ("raw", "calibrated")
    for model, seed, strategy in strata:
        processes = sorted({key[3] for key in grouped if key[:3] == (model, seed, strategy)})
        for state in STATES:
            state_processes = [p for p in processes if (model, seed, strategy, p, state) in grouped]
            for tag in tags:
                for metric in METRICS:
                    vals = {p: _metric_value(per_process[f"{model}|{seed}|{strategy}|{p}|{state}|{tag}"], metric)
                            for p in state_processes}
                    defined = [p for p in state_processes if vals[p] is not None]
                    summary_key = f"{model}|{seed}|{strategy}|{state}|{tag}|{metric}"
                    if not defined:
                        summaries[summary_key] = {"n_processes": 0, "process_ids": [], "mean": None, "ci95": None}
                    else:
                        # Never replace missing event outcomes with zeros. The
                        # support is explicit and the bootstrap operates on events.
                        arr = np.asarray([vals[p] for p in defined], dtype=np.float64)
                        boot = grouped_process_bootstrap(arr[:, None], defined,
                                                         repetitions=bootstrap_repetitions,
                                                         seed=bootstrap_seed)
                        summaries[summary_key] = {
                            "n_processes": len(defined), "process_ids": defined,
                            "mean": float(boot["mean"][0]),
                            "ci95": [float(boot["ci95"][0, 0]), float(boot["ci95"][1, 0])],
                        }
    return {
        "role": evaluation_role, "interpretation": "retrospective descriptive; not blind or confirmatory",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_argmax_invariant": True,
        "undefined_metric_policy": "null; no flood positives implies undefined AUPRC",
        "per_process": per_process, "summary": summaries,
    }


def primary_training_calibration_interaction(
    evaluation: dict, *, arm_keys: Mapping[str, tuple[str, str]], seeds: Sequence[int],
    expected_process_ids: Sequence[str] | None = None,
    bootstrap_repetitions: int = 10_000, bootstrap_seed: int = 20261029,
) -> dict:
    """Compute frozen (D-C)-(B-A) using common process×state support.

    ``arm_keys`` maps A/B/C/D to ``(model_name, calibration_strategy)``. Only
    calibrated predictions enter. Each process×state Brier is first pixel
    averaged by ``score_process_state``; five states are equally averaged
    within process; processes are equally weighted within seed. One shared
    process resample preserves all arms and seeds in every replicate.
    """
    if set(arm_keys) != {"A", "B", "C", "D"} or not seeds:
        raise ValueError("arm_keys must define A/B/C/D and seeds must be nonempty")
    if evaluation.get("role") != "internal_retro":
        raise ValueError("the primary interaction is defined only for internal_retro; holdout is separate")
    lookup = {}
    process_ids = set()
    for key, row in evaluation["per_process"].items():
        parts = key.split("|")
        if len(parts) != 6 or parts[-1] != "calibrated":
            continue
        model, seed_text, strategy, process, state, _ = parts
        if state not in STATES:
            continue
        lookup[(model, int(seed_text), strategy, process, state)] = row.get("binary_flood_brier")
        process_ids.add(process)
    if not lookup:
        raise ValueError("evaluation contains no calibrated process/state Brier cells")
    candidate_processes = set(expected_process_ids) if expected_process_ids is not None else process_ids
    if not candidate_processes:
        raise ValueError("expected process support must be nonempty")
    common = []
    excluded = {}
    for process in sorted(candidate_processes):
        missing = []
        for seed in seeds:
            for arm, (model, strategy) in arm_keys.items():
                for state in STATES:
                    value = lookup.get((model, int(seed), strategy, process, state))
                    if value is None:
                        missing.append(f"seed={seed}:{arm}:{state}")
        if missing:
            excluded[process] = missing
        else:
            common.append(process)
    if not common:
        return {"estimand": "(D-C)-(B-A)", "status": "undefined_no_common_process_support",
                "process_ids": [], "excluded_processes": excluded,
                "per_seed": {}, "mean": None, "ci95": None}
    # tensor [process, seed, arm]; seed and arm positions are preserved jointly.
    arm_order = ("A", "B", "C", "D")
    tensor = np.empty((len(common), len(seeds), 4), dtype=np.float64)
    for pi, process in enumerate(common):
        for si, seed in enumerate(seeds):
            for ai, arm in enumerate(arm_order):
                model, strategy = arm_keys[arm]
                state_values = [lookup[(model, int(seed), strategy, process, state)] for state in STATES]
                tensor[pi, si, ai] = float(np.mean(state_values))
    # Process-level paired contrast, then average the three fixed seeds. This
    # permits one clustered process bootstrap while retaining seed pairing.
    interaction_by_process_seed = ((tensor[:, :, 3] - tensor[:, :, 2])
                                   - (tensor[:, :, 1] - tensor[:, :, 0]))
    interaction_by_process = np.mean(interaction_by_process_seed, axis=1)
    boot = grouped_process_bootstrap(interaction_by_process[:, None], common,
                                     repetitions=bootstrap_repetitions, seed=bootstrap_seed)
    seed_values = np.mean(interaction_by_process_seed, axis=0)
    return {
        "estimand": "(D-C)-(B-A)",
        "construction": "pixel Brier per process×state; five-state equal mean per process; paired process effects per seed; seeds equally weighted",
        "status": "defined",
        "process_ids": common, "n_processes": len(common),
        "excluded_processes": excluded,
        "per_seed": {str(seed): float(value) for seed, value in zip(seeds, seed_values)},
        "mean": float(np.mean(seed_values)),
        "seed_min": float(np.min(seed_values)), "seed_max": float(np.max(seed_values)),
        "seed_sd": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None,
        "ci95": [float(boot["ci95"][0, 0]), float(boot["ci95"][1, 0])],
        "bootstrap": {"unit": "process", "repetitions": bootstrap_repetitions, "seed": bootstrap_seed,
                      "paired_arms_and_seeds": True},
    }


def combine_evaluation_results(results: Sequence[dict]) -> dict:
    """Combine frozen per-run outputs within one role without pooling roles."""
    if not results:
        raise ValueError("at least one evaluation result is required")
    roles = {result.get("role") for result in results}
    if len(roles) != 1:
        raise ValueError("evaluation roles must remain separate")
    merged = {}
    for result in results:
        for key, value in result.get("per_process", {}).items():
            if key in merged:
                raise ValueError(f"duplicate model/seed/process/state score: {key}")
            merged[key] = value
    return {"role": next(iter(roles)),
            "interpretation": "retrospective descriptive; not blind or confirmatory",
            "per_process": merged,
            "source_evaluation_count": len(results)}


def write_json(path: Path, payload: dict) -> None:
    """Write stable JSON, converting NumPy scalars/arrays without dropping nulls."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n", encoding="utf-8")
    temp.replace(path)


def deterministic_block_mask(valid: np.ndarray, sample_id: str, level: str, seed: int,
                             *, purpose: str) -> np.ndarray:
    """Frozen nested 8×8 block mask used by calibration or evaluation."""
    valid = np.asarray(valid)
    if valid.shape != (256, 256) or valid.dtype != np.bool_ or level not in STATES:
        raise ValueError("expected boolean 256×256 valid mask and a frozen state")
    if level == "100":
        return valid.copy()
    if purpose == "calibration":
        prefix = "cal-state-v1"
    elif purpose == "evaluation":
        prefix = "eval-block-v1"
    else:
        raise ValueError("purpose must be calibration or evaluation")
    keep_count = int(level) * 64 // 100
    order = sorted(range(64), key=lambda block: (
        hashlib.sha256(f"{prefix}|{seed}|{sample_id}|{block}".encode("ascii")).hexdigest(), block))
    blocks = np.zeros(64, dtype=np.bool_)
    blocks[order[:keep_count]] = True
    return valid & np.repeat(np.repeat(blocks.reshape(8, 8), 32, axis=0), 32, axis=1)


class FrozenInferenceAdapter:
    """Read-only model/reader adapter; construction is only exposed behind dual freeze."""
    def __init__(self, root: Path, config: dict, model_name: str, seed: int,
                 archive_dir: Path, device_name: str):
        # This is repeated here by design: callers cannot construct model or
        # reader by bypassing the high-level factory.
        self.locked_hashes = require_real_access(root, config)
        import sys
        import torch
        source = root / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from flood_reliability.impactmesh_io import TrainArchiveReader
        from scripts import upgrade_train

        self.torch = torch
        self.device = torch.device(device_name)
        if device_name == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("XPU unavailable")
        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self.model_name = model_name
        self.seed = int(seed)
        self.run_id = f"{model_name}_seed_{seed}"
        self.run_dir = root / "outputs" / "upgrade_20260929" / self.run_id
        run_path = self.run_dir / "run.json"
        model_path = self.run_dir / "model_only.pt"
        stats_path = self.run_dir / "fit_only_statistics.json"
        for path in (run_path, model_path, stats_path):
            if not path.is_file():
                raise RuntimeError(f"frozen fit artifact missing: {path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (run.get("status") != "FIT_COMPLETE" or run.get("run_id") != self.run_id
                or run.get("model") != model_name or run.get("seed") != int(seed)
                or run.get("completed_epochs") != int(config["epochs"])
                or sha256(model_path) != run.get("model_only_sha256")):
            raise RuntimeError("model-only artifact does not match its completed frozen run record")
        self.model_path = model_path
        self.model_sha256 = sha256(model_path)
        if run.get("hashes", {}).get("config") != sha256(root / "configs" / "upgrade_20260929.json"):
            raise RuntimeError("model was trained with another config hash")
        if run.get("hashes", {}).get("fit_manifest") != config["manifests"]["fit"]["sha256"]:
            raise RuntimeError("model was trained with another fit manifest")
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if self.stats.get("fit_manifest_sha256") != config["manifests"]["fit"]["sha256"]:
            raise RuntimeError("normalization statistics do not match frozen fit role")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_id") != self.run_id or checkpoint.get("hashes") != run.get("hashes"):
            raise RuntimeError("model-only checkpoint and run record disagree")
        for relative, expected_hash in run.get("hashes", {}).get("code", {}).items():
            dependency = root / relative
            if not dependency.is_file() or sha256(dependency) != expected_hash:
                raise RuntimeError(f"training code dependency hash mismatch: {relative}")
        self.model = upgrade_train.make_model(model_name, root, config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        # Constructed last, after every freeze/hash/model check above.
        self.reader = TrainArchiveReader(archive_dir)
        self.upgrade_train = upgrade_train

    def close(self):
        self.reader.close()

    def predict_tile(self, row: dict, state: str, *, purpose: str,
                     optical_mode: str = "paired") -> tuple[np.ndarray, np.ndarray]:
        """Read exactly one frozen-role label tile and predict one availability state."""
        torch = self.torch
        if optical_mode == "paired":
            encoded = self.upgrade_train._encode(self.reader, [row], self.stats, self.device)
            sar, optical, valid, labels, offset = encoded
        elif optical_mode == "low_visible":
            s1, s2, valid_np, label_np = load_holdout_tile_allow_low_visibility(
                self.reader, row["sample_id"], int(row["selected_time_index"]))
            if abs(float(valid_np.mean()) - float(row["selected_joint_visible_fraction"])) > 1e-6:
                raise RuntimeError(f"holdout low-visible fraction drift: {row['sample_id']}")
            sar, optical, valid, labels, offset = _encode_observed_arrays(
                s1, s2, valid_np, label_np, row, self.stats, self.device)
        elif optical_mode == "absent":
            s1, label_np = self.reader.load_s1_label(row["sample_id"])
            sar_norm = ((s1 - np.asarray(self.stats["sar_mean"], np.float32)[:, None, None])
                        / np.asarray(self.stats["sar_std"], np.float32)[:, None, None])
            sar = torch.from_numpy(sar_norm[None].astype(np.float32)).to(self.device)
            optical = torch.zeros((1, 12, 256, 256), dtype=torch.float32, device=self.device)
            valid = torch.zeros((1, 1, 256, 256), dtype=torch.bool, device=self.device)
            labels = torch.from_numpy(label_np[None]).long().to(self.device)
            offset = torch.zeros((1,), dtype=torch.float32, device=self.device)
        else:
            raise ValueError("optical_mode must be paired, low_visible, or absent")
        original_valid = valid[0, 0].detach().cpu().numpy()
        if optical_mode != "paired":
            kept = original_valid.copy()
        elif state.startswith("irregular_"):
            level = float(state.removeprefix("irregular_")) / 100.
            kept = irregular_retention_mask(original_valid, row["sample_id"], level)
        else:
            kept = deterministic_block_mask(original_valid, row["sample_id"], state,
                                            self.seed, purpose=purpose)
        kept_t = torch.from_numpy(kept[None, None]).to(self.device)
        optical = torch.where(kept_t, optical, torch.zeros_like(optical))
        present = kept_t.flatten(1).any(dim=1)
        offset = torch.where(present, offset, torch.zeros_like(offset))
        with torch.inference_mode():
            if self.model_name == "B2":
                logits = self.model(sar)
            elif self.model_name == "SMAGNet":
                picked = optical[:, (3, 2, 1, 7)]
                fused, _, _ = self.model(sar, picked, (~kept_t).to(picked.dtype))
                logits = fused
            else:
                logits = self.model(sar, optical, kept_t, present,
                                    normalized_time_offset=offset if self.model_name == "P" else None)
        return logits[0].detach().cpu().numpy(), labels[0].detach().cpu().numpy()


def _encode_observed_arrays(s1: np.ndarray, s2: np.ndarray, valid_np: np.ndarray,
                            label_np: np.ndarray, row: dict, stats: dict, device):
    """Apply fit-only normalization to a measured holdout optical observation."""
    import torch
    sm = np.asarray(stats["sar_mean"], np.float32)[:, None, None]
    ss = np.asarray(stats["sar_std"], np.float32)[:, None, None]
    om = np.asarray(stats["optical_mean"], np.float32)[:, None, None]
    osd = np.asarray(stats["optical_std"], np.float32)[:, None, None]
    sar = torch.from_numpy(((s1 - sm) / ss)[None].astype(np.float32)).to(device)
    optical = np.where(valid_np[None], (s2 - om) / osd, 0).astype(np.float32)
    optical_t = torch.from_numpy(optical[None]).to(device)
    valid_t = torch.from_numpy(valid_np[None, None]).bool().to(device)
    labels_t = torch.from_numpy(label_np[None]).long().to(device)
    age = float(row["selected_age_days"])
    offset = torch.tensor([-age / 30.0], dtype=torch.float32, device=device)
    return sar, optical_t, valid_t, labels_t, offset


def load_holdout_tile_allow_low_visibility(reader, sample_id: str, s2_index: int):
    """Load paired test arrays with full grid checks but without the >=50% gate.

    The frozen candidate group decides eligibility before scores are read;
    this loader is only for the metadata-audited 49-tile low-visible stratum.
    """
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from flood_reliability.impactmesh_io import _group

    if s2_index not in range(4):
        raise ValueError("selected S2 time index must be 0..3")
    s1 = _group(reader.blob("S1RTC", sample_id))
    s2 = _group(reader.blob("S2L2A", sample_id))
    with MemoryFile(reader.blob("MASK", sample_id)) as mem:
        with mem.open() as src:
            if src.shape != (256, 256) or src.count != 1:
                raise RuntimeError("unexpected holdout label shape")
            crs = CRS.from_wkt(s1["spatial_ref"].attrs["crs_wkt"])
            if src.crs != crs or CRS.from_wkt(s2["spatial_ref"].attrs["crs_wkt"]) != crs:
                raise RuntimeError("holdout S1/S2/MASK CRS mismatch")
            expected_x = src.transform.c + (np.arange(256) + .5) * src.transform.a
            expected_y = src.transform.f + (np.arange(256) + .5) * src.transform.e
            if not (np.allclose(s1["x"][:], expected_x, atol=1e-6, rtol=0)
                    and np.allclose(s1["y"][:], expected_y, atol=1e-6, rtol=0)
                    and np.allclose(s2["x"][:], expected_x, atol=1e-6, rtol=0)
                    and np.allclose(s2["y"][:], expected_y, atol=1e-6, rtol=0)):
                raise RuntimeError("holdout S1/S2/MASK grid mismatch")
            labels = src.read(1).astype(np.int64)
    sar = np.asarray(s1["bands"][2], dtype=np.float32)
    optical = np.asarray(s2["bands"][s2_index], dtype=np.float32)
    valid = (np.isin(np.asarray(s2["SCL"][s2_index]), (4, 5, 6))
             & (np.asarray(s2["cloud_mask"][s2_index]) == 0)
             & (np.asarray(s2["nan_mask"][s2_index]) == 0)
             & np.all(np.isfinite(optical) & (optical != -9999), axis=0)
             & (int(s2["data_mask"][s2_index]) == 1))
    if (sar.shape != (2, 256, 256) or optical.shape != (12, 256, 256)
            or int(s1["data_mask"][2]) != 1 or np.any(s1["nan_mask"][2] != 0)
            or not np.isfinite(sar).all()):
        raise RuntimeError("invalid holdout S1/S2 shapes or S1 pixels")
    if not set(np.unique(labels)) <= {-1, 0, 1, 2}:
        raise RuntimeError("unexpected holdout label codes")
    labels[labels == -1] = 255
    optical = np.where(valid[None], optical, 0).astype(np.float32, copy=False)
    return sar, optical, valid, labels


def open_frozen_inference_adapter(root: Path, config: dict, model_name: str, seed: int,
                                  archive_dir: Path, device_name: str = "xpu") -> FrozenInferenceAdapter:
    """Gate before adapter import/model construction/real archive reader creation."""
    require_real_access(root, config)
    return FrozenInferenceAdapter(root, config, model_name, seed, archive_dir, device_name)


def _manifest_inputs(root: Path, config: dict, role: str) -> tuple[list[dict], dict]:
    manifest_item = config["manifests"][role]
    grid_key = role
    manifest_path = root / manifest_item["path"]
    grid_path = root / config["role_grids"][grid_key]["path"]
    owners = manifest_owner_masks(manifest_path, grid_path, expected_role=role)
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: row["sample_id"])
    return rows, owners


def classify_holdout_strata(candidate_rows: Sequence[dict], eligibility_rows: Sequence[dict],
                            anchor_rows: Sequence[dict], *,
                            expected_counts: Mapping[str, int] | None = None) -> dict[str, list[dict]]:
    """Pre-score metadata partition: paired, low-visible, and no qualifying S2."""
    expected = dict(expected_counts or {"paired_eligible": 432, "low_visible": 49,
                                        "no_qualifying_s2": 11})
    eligibility = {row["sample_id"]: row for row in eligibility_rows}
    anchors = {row["sample_id"]: row for row in anchor_rows}
    if len(eligibility) != len(eligibility_rows) or len(anchors) != len(anchor_rows):
        raise ValueError("duplicate sample ID in frozen holdout eligibility or anchor metadata")
    groups = {name: [] for name in expected}
    seen = set()
    for candidate in candidate_rows:
        sid = candidate["sample_id"]
        if sid in seen:
            raise ValueError(f"duplicate holdout candidate ID: {sid}")
        seen.add(sid)
        if candidate.get("derived_role") != "holdout_retro":
            raise ValueError(f"non-holdout candidate row in holdout manifest: {sid}")
        item = eligibility.get(sid)
        if item is None:
            raise ValueError(f"holdout eligibility metadata missing candidate {sid}")
        if item.get("paired_eligible") == "1":
            group = "paired_eligible"
        elif item.get("failure_reasons") == "visible_below_0.5" and item.get("selected_time_index"):
            group = "low_visible"
        elif item.get("failure_reasons") == "no_pre_s1_s2_within_30d" and not item.get("selected_time_index"):
            anchor = anchors.get(sid)
            if not anchor or not anchor.get("s1_event_utc_from_zattrs"):
                raise ValueError(f"missing direct S1 anchor audit for no-S2 candidate {sid}")
            group = "no_qualifying_s2"
        else:
            raise ValueError(f"unrecognized frozen holdout eligibility for {sid}: {item}")
        # Candidate rows have deliberately blank holdout selection fields;
        # attach the frozen eligibility row before any model/archive read.
        enriched = dict(candidate)
        for field in ("selected_time_index", "selected_s2_utc", "selected_age_days",
                      "selected_joint_visible_fraction"):
            enriched[field] = item.get(field, "")
        enriched["holdout_stratum"] = group
        groups[group].append(enriched)
    counts = {name: len(rows) for name, rows in groups.items()}
    if counts != expected:
        raise ValueError(f"holdout metadata strata counts differ from frozen expectation: {counts}")
    return groups


def _require_frozen_extra_asset(root: Path, relative: str, expected_sha256: str) -> str:
    """Require auxiliary holdout metadata to be explicitly covered by G3 binder."""
    freeze_path = root / "G3_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    locked = {str(item.get("path", "")).replace("\\", "/"): item.get("sha256")
              for item in freeze.get("locked_files", [])}
    relative = relative.replace("\\", "/")
    actual = sha256(root / relative)
    if actual != expected_sha256 or locked.get(relative) != actual:
        raise RuntimeError(f"auxiliary holdout metadata is not frozen/hash-matched: {relative}")
    return actual


def calibrate_frozen_model(root: Path, config: dict, model: str, seed: int,
                           strategy: str, *, device_name: str = "xpu") -> dict:
    """Freeze-gated real calibration entry; writes one temperature record per strategy."""
    _validate_model_strategy(config, model, seed, strategy)
    output = root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}" / f"temperature_{strategy}.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite a frozen temperature artifact: {output}")
    # Gate before reading the calibration manifest labels or constructing reader/model.
    locked = require_real_access(root, config)
    adapter = open_frozen_inference_adapter(root, config, model, seed,
                                            root / "data" / "raw" / "impactmesh" / "train", device_name)
    try:
        rows, owners = _manifest_inputs(root, config, "calibration")
        states = ("100",) if strategy == "original" else STATES if strategy == "multistate" else ()
        if not states:
            raise ValueError("strategy must be original or multistate")
        by_process: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            process, _ = owners[row["sample_id"]]
            by_process[process].append(row)
        # Retain only one filtered float32 process×state cell at a time during
        # construction; never hold tile predictions, owner arrays, or float64
        # copies alongside the complete five-state cache.
        cell_cache: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        cell_pixel_counts = {}
        for process in sorted(by_process):
            for state in states:
                logits_parts, label_parts = [], []
                for row in sorted(by_process[process], key=lambda value: value["sample_id"]):
                    logits, labels = adapter.predict_tile(row, state, purpose="calibration")
                    owner = owners[row["sample_id"]][1]
                    chosen = owner & (labels != 255)
                    if chosen.any():
                        # Store only fit support, float32 logits and int8 labels.
                        logits_parts.append(np.moveaxis(logits, 0, -1)[chosen].astype(np.float32, copy=True))
                        label_parts.append(labels[chosen].astype(np.uint8, copy=True))
                    del logits, labels, owner, chosen
                if not logits_parts:
                    raise ValueError(f"empty owned calibration process×state cell {(process, state)}")
                cell_logits = np.concatenate(logits_parts, axis=0)
                cell_labels = np.concatenate(label_parts, axis=0)
                if not np.isfinite(cell_logits).all():
                    raise ValueError(f"nonfinite calibration logits in {(process, state)}")
                cell_cache.append((process, state, cell_logits, cell_labels))
                cell_pixel_counts[f"{process}|{state}"] = int(len(cell_labels))
                del logits_parts, label_parts, cell_logits, cell_labels
        if len(cell_cache) != len(by_process) * len(states):
            raise RuntimeError("calibration cache is missing a frozen process×state cell")
        record = fit_global_temperature_streaming(cell_cache, model=model, seed=seed,
                                                  strategy=strategy,
                                                  pixel_counts=cell_pixel_counts, hashes=locked)
        del cell_cache
        record.update({"checkpoint_sha256": adapter.model_sha256,
                       "statistics_sha256": sha256(adapter.run_dir / "fit_only_statistics.json"),
                       "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
                       "role": "calibration", "retrospective_limit": True})
        write_json(output, record)
        record["output_path"] = str(output)
        return record
    finally:
        adapter.close()


def fit_global_temperature_streaming(cells: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
                                     *, model: str, seed: int, strategy: str,
                                     pixel_counts: Mapping[str, int],
                                     hashes: Mapping[str, str]) -> dict:
    """Fit on filtered float32 cells, casting at most one cell to float64 per step."""
    expected_states = ("100",) if strategy == "original" else STATES if strategy == "multistate" else ()
    if not expected_states:
        raise ValueError("strategy must be original or multistate")
    if not cells:
        raise ValueError("no calibration cells")
    keys = [(p, s) for p, s, _, _ in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate process×state calibration cell")
    processes = tuple(sorted({p for p, _, _, _ in cells}))
    cell_map = {(p, s): (z, y) for p, s, z, y in cells}
    expected = {(p, s) for p in processes for s in expected_states}
    if set(cell_map) != expected:
        raise ValueError("incomplete frozen calibration process×state grid")
    for key, (logits, labels) in cell_map.items():
        if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != 3:
            raise ValueError(f"calibration cache must be float32 N×3: {key}")
        if labels.dtype != np.uint8 or labels.ndim != 1 or labels.shape[0] != logits.shape[0] or not len(labels):
            raise ValueError(f"invalid/empty calibration labels: {key}")
        if not np.all(np.isin(labels, (0, 1, 2))) or not np.isfinite(logits).all():
            raise ValueError(f"invalid labels or logits: {key}")

    def objective(log_t: float) -> float:
        temperature = float(np.exp(log_t))
        losses = []
        # Sequential temporary float64 cast: bounded additional memory is one
        # process×state cell, independent of the full five-state cache size.
        for process in processes:
            for state in expected_states:
                logits32, labels8 = cell_map[(process, state)]
                scores = logits32.astype(np.float64, copy=True)
                scores /= temperature
                labels = labels8.astype(np.int64, copy=False)
                loss = logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels]
                losses.append(float(np.mean(loss, dtype=np.float64)))
                del scores, loss
        value = float(np.mean(losses))
        if not np.isfinite(value):
            raise ValueError("nonfinite global temperature objective")
        return value

    fit = minimize_scalar(objective, bounds=(-3., 3.), method="bounded",
                          options={"xatol": 0.001, "maxiter": 100})
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x):
        raise RuntimeError("frozen global temperature optimizer failed")
    log_t = float(fit.x)
    return {"model": model, "seed": int(seed), "calibration_strategy": strategy,
            "temperature": float(np.exp(log_t)), "log_temperature": log_t,
            "objective_nll": float(fit.fun), "boundary_hit": min(abs(log_t + 3), abs(log_t - 3)) <= .001,
            "optimizer_success": True, "evaluations": int(fit.nfev),
            "process_ids": list(processes), "state_ids": list(expected_states),
            "calibration_process_state_pixel_counts": dict(pixel_counts),
            "objective": "equal mean over process×state cell pixel-NLL means",
            "cache_dtype": "float32 logits + uint8 labels; owner/Ignore filtered during tile generation",
            "memory_policy": "stream tiles; retain only filtered cells; one float64 cell temporary per objective cell",
            "calibration_hashes": dict(hashes)}


def evaluate_frozen_internal(root: Path, config: dict, model: str, seed: int,
                             strategy: str, *, device_name: str = "xpu") -> dict:
    """One fixed internal retrospective evaluation; does not accept holdout as primary."""
    _validate_model_strategy(config, model, seed, strategy)
    expected_output = evaluation_output_path(
        root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}",
        "internal_retro", strategy)
    if expected_output.exists():
        raise RuntimeError(f"refusing to overwrite a one-time evaluation artifact: {expected_output}")
    locked = require_real_access(root, config)
    adapter = open_frozen_inference_adapter(root, config, model, seed,
                                            root / "data" / "raw" / "impactmesh" / "train", device_name)
    try:
        temperature_path = adapter.run_dir / f"temperature_{strategy}.json"
        if not temperature_path.is_file():
            raise RuntimeError("frozen calibration temperature file is missing")
        temp_record = json.loads(temperature_path.read_text(encoding="utf-8"))
        if (temp_record.get("model") != model or temp_record.get("seed") != seed
                or temp_record.get("calibration_strategy") != strategy
                or temp_record.get("checkpoint_sha256") != adapter.model_sha256
                or not np.isfinite(temp_record.get("temperature", np.nan))
                or temp_record["temperature"] <= 0):
            raise RuntimeError("temperature record does not match this frozen model/seed")
        rows, owners = _manifest_inputs(root, config, "internal_retro")
        by_process: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            process, _ = owners[row["sample_id"]]
            by_process[process].append(row)
        per_process = {}
        temperature = float(temp_record["temperature"])
        for process in sorted(by_process):
            ordered = sorted(by_process[process], key=lambda value: value["sample_id"])
            for state in (*STATES, *IRREGULAR_STATES):
                logit_parts, label_parts, owner_parts = [], [], []
                for row in ordered:
                    logits, labels = adapter.predict_tile(row, state, purpose="evaluation")
                    owner = owners[row["sample_id"]][1]
                    logit_parts.append(np.moveaxis(logits, 0, -1).reshape(-1, 3).astype(np.float32, copy=True))
                    label_parts.append(labels.reshape(-1).astype(np.uint8, copy=True))
                    owner_parts.append(owner.reshape(-1).copy())
                    del logits, labels, owner
                z = np.concatenate(logit_parts, axis=0)
                y = np.concatenate(label_parts)
                own = np.concatenate(owner_parts)
                del logit_parts, label_parts, owner_parts
                for tag, temp in (("raw", 1.), ("calibrated", temperature)):
                    scored = score_process_state(z, y, own, temperature=temp)
                    per_process[f"{model}|{seed}|{strategy}|{process}|{state}|{tag}"] = scored
                del z, y, own
        summaries = _summaries_for_single_evaluation(per_process, model, seed, strategy)
        result = {"role": "internal_retro",
                  "interpretation": "retrospective descriptive; not blind or confirmatory",
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "temperature_argmax_invariant": True,
                  "undefined_metric_policy": "null; no flood positives implies undefined AUPRC",
                  "per_process": per_process, "summary": summaries}
        result.update({"model": model, "seed": seed, "calibration_strategy": strategy,
                       "checkpoint_sha256": adapter.model_sha256,
                       "temperature_sha256": sha256(temperature_path),
                       "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
                       "locked_input_hashes": locked})
        output = evaluation_output_path(adapter.run_dir, "internal_retro", strategy)
        write_json(output, result)
        result["output_path"] = str(output)
        return result
    finally:
        adapter.close()


def evaluate_frozen_holdout(root: Path, config: dict, model: str, seed: int,
                            strategy: str, *, device_name: str = "xpu") -> dict:
    """Separate retrospective holdout summaries for the three metadata-fixed strata."""
    _validate_model_strategy(config, model, seed, strategy)
    expected_output = evaluation_output_path(
        root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}",
        "holdout_retro", strategy)
    if expected_output.exists():
        raise RuntimeError(f"refusing to overwrite a one-time evaluation artifact: {expected_output}")
    locked = require_real_access(root, config)
    auxiliary = config.get("auxiliary_metadata", {})
    for name in ("holdout_eligibility", "holdout_s1_anchor_audit_csv", "holdout_s1_anchor_audit_json"):
        if name not in auxiliary:
            raise RuntimeError(f"frozen config lacks required auxiliary metadata lock: {name}")
        item = auxiliary[name]
        _require_frozen_extra_asset(root, item["path"], item["sha256"])
    elig_rel = auxiliary["holdout_eligibility"]["path"]
    anchor_rel = auxiliary["holdout_s1_anchor_audit_csv"]["path"]
    # The test reader is constructed only after the full dual freeze + config
    # manifest/grid hashes + auxiliary metadata hashes have passed.
    adapter = open_frozen_inference_adapter(root, config, model, seed,
                                            root / "data" / "raw" / "impactmesh" / "test", device_name)
    try:
        temperature_path = adapter.run_dir / f"temperature_{strategy}.json"
        if not temperature_path.is_file():
            raise RuntimeError("frozen calibration temperature file is missing")
        temp_record = json.loads(temperature_path.read_text(encoding="utf-8"))
        if (temp_record.get("model") != model or temp_record.get("seed") != seed
                or temp_record.get("calibration_strategy") != strategy
                or temp_record.get("checkpoint_sha256") != adapter.model_sha256
                or not np.isfinite(temp_record.get("temperature", np.nan))
                or temp_record["temperature"] <= 0):
            raise RuntimeError("temperature record does not match this frozen model/seed")
        rows, owners = _manifest_inputs(root, config, "holdout_retro")
        with (root / elig_rel).open(newline="", encoding="utf-8") as f:
            eligibility_rows = list(csv.DictReader(f))
        with (root / anchor_rel).open(newline="", encoding="utf-8") as f:
            anchor_rows = list(csv.DictReader(f))
        strata = classify_holdout_strata(rows, eligibility_rows, anchor_rows)
        per_process = {}
        modes = {"paired_eligible": ((*STATES, *IRREGULAR_STATES), "paired"),
                 "low_visible": (("observed",), "low_visible"),
                 "no_qualifying_s2": (("observed_absent",), "absent")}
        for group_name, group_rows in strata.items():
            by_process: dict[str, list[dict]] = defaultdict(list)
            for row in group_rows:
                process, _ = owners[row["sample_id"]]
                by_process[process].append(row)
            states, optical_mode = modes[group_name]
            for process in sorted(by_process):
                ordered = sorted(by_process[process], key=lambda value: value["sample_id"])
                for state in states:
                    z_parts, y_parts, owner_parts = [], [], []
                    for row in ordered:
                        # Natural strata use their measured mask/absence once;
                        # they are never equated with a synthetic 0% state.
                        model_state = state if optical_mode == "paired" else "100"
                        logits, labels = adapter.predict_tile(
                            row, model_state, purpose="evaluation", optical_mode=optical_mode)
                        owner = owners[row["sample_id"]][1]
                        z_parts.append(np.moveaxis(logits, 0, -1).reshape(-1, 3).astype(np.float32, copy=True))
                        y_parts.append(labels.reshape(-1).astype(np.uint8, copy=True))
                        owner_parts.append(owner.reshape(-1).copy())
                        del logits, labels, owner
                    z = np.concatenate(z_parts, axis=0)
                    y = np.concatenate(y_parts)
                    owner = np.concatenate(owner_parts)
                    del z_parts, y_parts, owner_parts
                    for tag, temperature in (("raw", 1.), ("calibrated", float(temp_record["temperature"]))):
                        score = score_process_state(z, y, owner, temperature=temperature)
                        key = f"{model}|{seed}|{strategy}|{group_name}|{process}|{state}|{tag}"
                        per_process[key] = score
                    del z, y, owner
        # Strata remain separate and cannot be combined with the internal primary estimand.
        summaries = _summaries_for_holdout(per_process, model, seed, strategy)
        result = {"model": model, "seed": seed, "calibration_strategy": strategy,
                  "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
                  "role": "holdout_retro", "interpretation": "retrospective descriptive only",
                  "strata_counts": {name: len(group) for name, group in strata.items()},
                  "groups_not_pooled": True,
                  "natural_absence_not_comparable_to_synthetic_zero": True,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "temperature_argmax_invariant": True,
                  "per_process": per_process, "summary": summaries,
                  "checkpoint_sha256": adapter.model_sha256,
                  "temperature_sha256": sha256(temperature_path),
                  "locked_input_hashes": locked}
        output = evaluation_output_path(adapter.run_dir, "holdout_retro", strategy)
        write_json(output, result)
        result["output_path"] = str(output)
        return result
    finally:
        adapter.close()


def _summaries_for_holdout(per_process: Mapping[str, dict], model: str,
                           seed: int, strategy: str) -> dict:
    summaries = {}
    for group in ("paired_eligible", "low_visible", "no_qualifying_s2"):
        states = (*STATES, *IRREGULAR_STATES) if group == "paired_eligible" else (
            ("observed",) if group == "low_visible" else ("observed_absent",))
        for state in states:
            for tag in ("raw", "calibrated"):
                for metric in METRICS:
                    candidates = {}
                    for key, score in per_process.items():
                        parts = key.split("|")
                        if (parts[0] == model and int(parts[1]) == seed and parts[2] == strategy
                                and parts[3] == group and parts[5] == state and parts[6] == tag):
                            candidates[parts[4]] = _metric_value(score, metric)
                    support = sorted(p for p, value in candidates.items() if value is not None)
                    if support:
                        boot = grouped_process_bootstrap(
                            np.asarray([candidates[p] for p in support], dtype=np.float64)[:, None], support)
                        summary = {"n_processes": len(support), "process_ids": support,
                                   "mean": float(boot["mean"][0]),
                                   "ci95": [float(boot["ci95"][0, 0]), float(boot["ci95"][1, 0])]}
                    else:
                        summary = {"n_processes": 0, "process_ids": [], "mean": None, "ci95": None}
                    summaries[f"{group}|{state}|{tag}|{metric}"] = summary
    return summaries


# Explicit public spelling used by the coordinator for this separate role.
evaluate_holdout_frozen = evaluate_frozen_holdout


def evaluation_output_path(run_dir: Path, role: str, strategy: str) -> Path:
    if role not in {"internal_retro", "holdout_retro"} or strategy not in {"original", "multistate"}:
        raise ValueError("invalid evaluation role or calibration strategy")
    return run_dir / f"evaluation_{role}_{strategy}.json"


def _validate_model_strategy(config: dict, model: str, seed: int, strategy: str) -> None:
    if model not in config.get("models", {}):
        raise ValueError(f"model is not in the frozen candidate config: {model}")
    if int(seed) not in config.get("seeds", []):
        raise ValueError(f"seed is not in the frozen candidate config: {seed}")
    allowed = config["models"][model].get("calibration", [])
    if strategy not in allowed:
        raise ValueError(f"calibration strategy {strategy} is not frozen for model {model}")


def summarize_primary_frozen_outputs(root: Path, config: dict) -> dict:
    """Read the twelve frozen B5 internal result files and write only the primary contrast."""
    locked = require_real_access(root, config)
    seeds = tuple(int(seed) for seed in config["seeds"])
    model_for_arm = {"A": ("B5_whole", "original"),
                     "B": ("B5_whole", "multistate"),
                     "C": ("B5_local", "original"),
                     "D": ("B5_local", "multistate")}
    loaded = []
    for seed in seeds:
        for model, strategy in (model_for_arm[a] for a in ("A", "B", "C", "D")):
            path = (root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}"
                    / f"evaluation_internal_retro_{strategy}.json")
            if not path.is_file():
                raise RuntimeError(f"required fixed internal evaluation output is missing: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            run_dir = root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}"
            run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            temperature_path = run_dir / f"temperature_{strategy}.json"
            if (result.get("role") != "internal_retro" or result.get("model") != model
                    or result.get("seed") != seed or result.get("calibration_strategy") != strategy
                    or result.get("config_sha256") != sha256(root / "configs" / "upgrade_20260929.json")
                    or result.get("checkpoint_sha256") != run_record.get("model_only_sha256")
                    or not temperature_path.is_file()
                    or result.get("temperature_sha256") != sha256(temperature_path)
                    or result.get("locked_input_hashes") != locked):
                raise RuntimeError(f"result identity/hash mismatch: {path}")
            loaded.append(result)
    merged = combine_evaluation_results(loaded)
    manifest_item = config["manifests"]["internal_retro"]
    with (root / manifest_item["path"]).open(newline="", encoding="utf-8") as f:
        expected_processes = sorted({row["conservative_unit_id"] for row in csv.DictReader(f)})
    arm_keys = {arm: model_for_arm[arm] for arm in model_for_arm}
    summary = primary_training_calibration_interaction(
        merged, arm_keys=arm_keys, seeds=seeds, expected_process_ids=expected_processes)
    summary.update({"role": "internal_retro", "interpretation": "retrospective descriptive only",
                    "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
                    "locked_input_hashes": locked,
                    "source_evaluations": len(loaded)})
    output = root / "outputs" / "upgrade_20260929" / "primary_interaction_internal_retro.json"
    write_json(output, summary)
    summary["output_path"] = str(output)
    return summary


def write_matrix_summaries(root: Path, config: dict) -> dict:
    """Audit all frozen model/seed/strategy JSONs; write primary + separate CSVs.

    This consumes only completed evaluation outputs and their run/calibration
    records. It never constructs a reader or opens a new raster label.
    """
    locked = require_real_access(root, config)
    seeds = tuple(int(seed) for seed in config["seeds"])
    loaded_internal, loaded_holdout = [], []
    input_hashes = {}
    cache_hash: dict[Path, str] = {}

    def file_hash(path: Path) -> str:
        if path not in cache_hash:
            if not path.is_file():
                raise RuntimeError(f"required frozen output missing: {path}")
            cache_hash[path] = sha256(path)
        return cache_hash[path]

    for model, spec in config["models"].items():
        for seed in seeds:
            run_dir = root / "outputs" / "upgrade_20260929" / f"{model}_seed_{seed}"
            run_path = run_dir / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if (run.get("status") != "FIT_COMPLETE" or run.get("model") != model
                    or run.get("seed") != seed):
                raise RuntimeError(f"incomplete or misidentified fit run: {run_path}")
            input_hashes[run_path.relative_to(root).as_posix()] = file_hash(run_path)
            model_path = run_dir / "model_only.pt"
            if file_hash(model_path) != run.get("model_only_sha256"):
                raise RuntimeError(f"model-only checkpoint hash mismatch: {model_path}")
            input_hashes[model_path.relative_to(root).as_posix()] = file_hash(model_path)
            for strategy in spec.get("calibration", []):
                temp_path = run_dir / f"temperature_{strategy}.json"
                temp_record = json.loads(temp_path.read_text(encoding="utf-8"))
                if (temp_record.get("model") != model or temp_record.get("seed") != seed
                        or temp_record.get("calibration_strategy") != strategy
                        or temp_record.get("checkpoint_sha256") != run.get("model_only_sha256")):
                    raise RuntimeError(f"calibration/run mismatch: {temp_path}")
                input_hashes[temp_path.relative_to(root).as_posix()] = file_hash(temp_path)
                for role, collection in (("internal_retro", loaded_internal),
                                         ("holdout_retro", loaded_holdout)):
                    eval_path = evaluation_output_path(run_dir, role, strategy)
                    result = json.loads(eval_path.read_text(encoding="utf-8"))
                    if (result.get("role") != role or result.get("model") != model
                            or result.get("seed") != seed
                            or result.get("calibration_strategy") != strategy
                            or result.get("checkpoint_sha256") != run.get("model_only_sha256")
                            or result.get("temperature_sha256") != file_hash(temp_path)
                            or result.get("locked_input_hashes") != locked):
                        raise RuntimeError(f"evaluation/run/calibration/hash mismatch: {eval_path}")
                    input_hashes[eval_path.relative_to(root).as_posix()] = file_hash(eval_path)
                    collection.append(result)

    if not loaded_internal or not loaded_holdout:
        raise RuntimeError("matrix summary requires internal and separate holdout evaluations")
    combined_internal = combine_evaluation_results(loaded_internal)
    internal_item = config["manifests"]["internal_retro"]
    if sha256(root / internal_item["path"]) != internal_item["sha256"]:
        raise RuntimeError("internal retrospective role manifest hash changed")
    with (root / internal_item["path"]).open(newline="", encoding="utf-8") as f:
        expected_processes = sorted({row["conservative_unit_id"] for row in csv.DictReader(f)})
    primary = primary_training_calibration_interaction(
        combined_internal,
        arm_keys={"A": ("B5_whole", "original"), "B": ("B5_whole", "multistate"),
                  "C": ("B5_local", "original"), "D": ("B5_local", "multistate")},
        seeds=seeds, expected_process_ids=expected_processes)

    internal_rows = _rows_from_evaluation_summaries(loaded_internal, holdout=False)
    holdout_rows = _rows_from_evaluation_summaries(loaded_holdout, holdout=True)
    out_dir = root / "outputs" / "upgrade_20260929" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    primary.update({"role": "internal_retro", "interpretation": "retrospective descriptive only",
                    "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
                    "locked_input_hashes": locked, "expected_process_ids": expected_processes})
    primary_path = out_dir / "primary_interaction_internal_retro.json"
    write_json(primary_path, primary)
    internal_csv = out_dir / "internal_metric_summary.csv"
    holdout_csv = out_dir / "holdout_stratified_summary.csv"
    b1_csv = out_dir / "b1_hard_metrics.csv"
    _write_summary_csv(internal_csv, internal_rows, holdout=False)
    _write_summary_csv(holdout_csv, holdout_rows, holdout=True)
    b1_records = []
    b1_input_hashes = {}
    b1_spec = config.get("fixed_baselines", {}).get("B1")
    if not b1_spec:
        raise RuntimeError("matrix summary requires the frozen B1 hard baseline specification")
    b1_thresholds = root / b1_spec["thresholds_path"]
    b1_threshold_record = json.loads(b1_thresholds.read_text(encoding="utf-8"))
    if (b1_threshold_record.get("status") != "FIT_ROLE_ONLY_COMPLETE"
            or b1_threshold_record.get("fit_manifest_sha256") != config["manifests"]["fit"]["sha256"]
            or b1_threshold_record.get("fit_script_sha256") != b1_spec.get("fit_script_sha256")
            or b1_threshold_record.get("config_sha256") != sha256(root / "configs" / "upgrade_20260929.json")):
        raise RuntimeError("B1 fit artifact does not match the frozen fit-only procedure")
    b1_threshold_hash = file_hash(b1_thresholds)
    b1_input_hashes[b1_thresholds.relative_to(root).as_posix()] = b1_threshold_hash
    for role in ("internal_retro", "holdout_retro"):
        b1_path = root / "outputs" / "upgrade_20260929" / "B1" / f"evaluation_{role}.json"
        b1 = json.loads(b1_path.read_text(encoding="utf-8"))
        expected_counts = ({"internal_retro": 1906} if role == "internal_retro" else
                           {"paired_eligible": 432, "low_visible": 49, "no_qualifying_s2": 11})
        if (b1.get("model") != "B1" or b1.get("role") != role
                or b1.get("interpretation") != "retrospective descriptive only"
                or b1.get("thresholds_sha256") != b1_threshold_hash
                or b1.get("fit_manifest_sha256") != config["manifests"]["fit"]["sha256"]
                or b1.get("config_sha256") != sha256(root / "configs" / "upgrade_20260929.json")
                or b1.get("strata_counts") != expected_counts):
            raise RuntimeError(f"B1 hard-baseline output failed frozen-role audit: {b1_path}")
        b1_input_hashes[b1_path.relative_to(root).as_posix()] = file_hash(b1_path)
        for stratum, payload in b1["results"].items():
            for process, metrics in payload["processes"].items():
                b1_records.append({"role": role, "stratum": stratum, "process_id": process,
                                   **{key: metrics.get(key) for key in
                                      ("tiles", "pixels", "nonwater_iou", "permanent_iou",
                                       "flood_iou", "macro_iou", "flood_precision",
                                       "flood_recall", "flood_f1", "pre_event_fallback_tiles")}})
            for metric, summary in payload["summary"].items():
                b1_records.append({"role": role, "stratum": stratum, "process_id": "PROCESS_MACRO",
                                   "metric": metric, "n_processes": summary["n_processes"],
                                   "mean": summary["mean"], "ci95_low": (summary["ci95"] or [None, None])[0],
                                   "ci95_high": (summary["ci95"] or [None, None])[1]})
    b1_columns = ["role", "stratum", "process_id", "metric", "n_processes", "tiles", "pixels",
                  "nonwater_iou", "permanent_iou", "flood_iou", "macro_iou", "flood_precision",
                  "flood_recall", "flood_f1", "pre_event_fallback_tiles", "mean", "ci95_low", "ci95_high"]
    with b1_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=b1_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(b1_records)
    audit = {"status": "COMPLETE", "interpretation": "retrospective descriptive only",
             "roles_separate": True, "primary_interaction": primary,
             "internal_summary_rows": len(internal_rows), "holdout_summary_rows": len(holdout_rows),
             "outputs": {path.relative_to(root).as_posix(): sha256(path)
                         for path in (primary_path, internal_csv, holdout_csv, b1_csv)},
             "input_file_hashes": dict(sorted(input_hashes.items())),
             "B1_hard_only_file_hashes": dict(sorted(b1_input_hashes.items())),
             "locked_data_hashes": locked,
             "evaluation_record_count": {"internal_retro": len(loaded_internal),
                                         "holdout_retro": len(loaded_holdout)}}
    audit_path = out_dir / "matrix_summary_audit.json"
    write_json(audit_path, audit)
    audit["audit_path"] = str(audit_path)
    return audit


def _rows_from_evaluation_summaries(results: Sequence[dict], *, holdout: bool) -> list[dict]:
    rows = []
    for result in results:
        model, seed, strategy = result["model"], int(result["seed"]), result["calibration_strategy"]
        row_map = {}
        metrics = ("binary_flood_brier", "flood_iou", "flood_f1", "flood_auprc",
                   "multiclass_nll", "multiclass_brier", "ece15")
        if holdout:
            for key, summary in result.get("summary", {}).items():
                group, state, tag, metric = key.split("|")
                if metric not in metrics:
                    continue
                row_key = (group, state, tag)
                row = row_map.setdefault(row_key, {"model": model, "seed": seed,
                    "calibration_strategy": strategy, "stratum": group,
                    "state": state, "score": tag})
                _insert_summary_metric(row, metric, summary)
        else:
            for key, summary in result.get("summary", {}).items():
                parts = key.split("|")
                if len(parts) != 6:
                    raise ValueError(f"unexpected internal summary key: {key}")
                _, _, _, state, tag, metric = parts
                if metric not in metrics:
                    continue
                row_key = (state, tag)
                row = row_map.setdefault(row_key, {"model": model, "seed": seed,
                    "calibration_strategy": strategy, "state": state, "score": tag})
                _insert_summary_metric(row, metric, summary)
        rows.extend(row_map[key] for key in sorted(row_map))
    return rows


def _insert_summary_metric(row: dict, metric: str, summary: dict) -> None:
    row[f"{metric}_n"] = summary.get("n_processes")
    row[f"{metric}_mean"] = summary.get("mean")
    ci = summary.get("ci95")
    row[f"{metric}_ci_low"] = None if ci is None else ci[0]
    row[f"{metric}_ci_high"] = None if ci is None else ci[1]


def _write_summary_csv(path: Path, rows: Sequence[dict], *, holdout: bool) -> None:
    metrics = ("binary_flood_brier", "flood_iou", "flood_f1", "flood_auprc",
               "multiclass_nll", "multiclass_brier", "ece15")
    prefix = ["model", "seed", "calibration_strategy"]
    prefix += ["stratum"] if holdout else []
    prefix += ["state", "score"]
    columns = prefix + [f"{metric}_{suffix}" for metric in metrics
                        for suffix in ("n", "mean", "ci_low", "ci_high")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summaries_for_single_evaluation(per_process: Mapping[str, dict], model: str,
                                    seed: int, strategy: str) -> dict:
    result = {}
    processes = sorted({key.split("|")[3] for key in per_process})
    for state in (*STATES, *IRREGULAR_STATES):
        for tag in ("raw", "calibrated"):
            for metric in METRICS:
                candidates = {}
                for process in processes:
                    key = f"{model}|{seed}|{strategy}|{process}|{state}|{tag}"
                    if key not in per_process:
                        continue
                    candidates[process] = _metric_value(per_process[key], metric)
                support = [process for process, value in candidates.items() if value is not None]
                if not support:
                    summary = {"n_processes": 0, "process_ids": [], "mean": None, "ci95": None}
                else:
                    values = np.asarray([candidates[p] for p in support], dtype=np.float64)[:, None]
                    boot = grouped_process_bootstrap(values, support)
                    summary = {"n_processes": len(support), "process_ids": support,
                               "mean": float(boot["mean"][0]),
                               "ci95": [float(boot["ci95"][0, 0]), float(boot["ci95"][1, 0])]}
                result[f"{model}|{seed}|{strategy}|{state}|{tag}|{metric}"] = summary
    return result


def _expected_file_hashes(config: dict) -> dict[str, str]:
    hashes = {}
    for section in ("manifests", "role_grids"):
        for name, item in config.get(section, {}).items():
            hashes[item["path"]] = item["sha256"]
            if section == "role_grids" and item.get("source") and item.get("source_sha256"):
                hashes[item["source"]] = item["source_sha256"]
    for item in config.get("auxiliary_metadata", {}).values():
        hashes[item["path"]] = item["sha256"]
    for item in config.get("fixed_baselines", {}).values():
        if item.get("fit_script_path") and item.get("fit_script_sha256"):
            hashes[item["fit_script_path"]] = item["fit_script_sha256"]
    for key in ("publisher_asset_manifest", "archive_audit"):
        item = config.get("data_source", {}).get(key)
        if item:
            hashes[item["path"]] = item["sha256"]
    return hashes


def require_real_access(root: Path, config: dict) -> dict:
    """Check dual freeze and every configured data-manifest hash before readers exist."""
    require_dual_freeze(root, config)
    freeze = json.loads((root / "G3_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    freeze_locks = {str(item.get("path", "")).replace("\\", "/"): item.get("sha256")
                    for item in freeze.get("locked_files", [])}
    observed = {}
    for relative, expected in _expected_file_hashes(config).items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"frozen input missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash mismatch: {relative}")
        if freeze_locks.get(relative.replace("\\", "/")) != expected:
            raise RuntimeError(f"input is not bound by the immutable freeze manifest: {relative}")
        observed[relative] = actual
    if len(observed) < 8:
        raise RuntimeError("freeze package does not lock all role manifests and grids")
    return observed


def run_real_after_freeze(root: Path, config: dict, reader_factory: Callable,
                          operation: Callable) -> object:
    """Gate first, then and only then construct a real-data reader."""
    locked_hashes = require_real_access(root, config)
    reader = reader_factory(locked_hashes)
    try:
        return operation(reader, locked_hashes)
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "evaluate", "summarize-primary"))
    parser.add_argument("model", nargs="?", choices=("B2", "B4", "P", "B5_whole", "B5_local", "SMAGNet"))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--strategy", choices=("original", "multistate"), default="original")
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--role", choices=("internal_retro", "holdout_retro"), default="internal_retro")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.command == "summarize-primary":
        result = summarize_primary_frozen_outputs(ROOT, config)
        print(json.dumps({"role": result["role"], "status": result["status"],
                          "process_count": result["n_processes"], "mean": result["mean"],
                          "ci95": result["ci95"], "output_path": result["output_path"]}, indent=2))
        return
    if args.model is None or args.seed is None:
        parser.error("model and seed are required for calibrate/evaluate")
    # Each operation's first action verifies dual freeze plus every configured
    # manifest/grid hash, before a project reader or model can be constructed.
    if args.command == "calibrate":
        result = calibrate_frozen_model(ROOT, config, args.model, args.seed,
                                        args.strategy, device_name=args.device)
        print(json.dumps({key: result[key] for key in
                          ("model", "seed", "calibration_strategy", "temperature",
                           "objective_nll", "output_path")}, indent=2))
    else:
        evaluator = (evaluate_frozen_internal if args.role == "internal_retro"
                     else evaluate_holdout_frozen)
        result = evaluator(ROOT, config, args.model, args.seed,
                           args.strategy, device_name=args.device)
        print(json.dumps({"model": result["model"], "seed": result["seed"],
                          "strategy": result["calibration_strategy"],
                          "role": result["role"], "output_path": result["output_path"]}, indent=2))


if __name__ == "__main__":
    main()
