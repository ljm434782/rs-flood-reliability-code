"""Run the frozen v6 training-source robustness supplement safely.

This wrapper reuses the audited v5 split, frame and fit-only statistics.  It
never reads v1 test/holdout outputs and refuses to overwrite an existing v6
run.  ``--freeze`` must be run before any training command.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flood_reliability.models import DualUNet
from scripts import train_impactmesh_exploratory_v5 as v5


CONFIG_PATH = ROOT / "configs" / "impactmesh_robustness_v6.json"
SOURCE_CONFIG_PATH = ROOT / "configs" / "impactmesh_exploratory_v5.json"
BASE = ROOT / "outputs" / "robustness_v6"
FRAME = BASE / "development_frame.json"
STATS = BASE / "fit_only_statistics.json"
FREEZE = BASE / "freeze_manifest.json"


def sha256(path: Path) -> str:
    return v5.sha256(path)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def freeze(config):
    if FREEZE.exists():
        existing = json.loads(FREEZE.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != sha256(CONFIG_PATH):
            raise RuntimeError("Existing v6 freeze manifest has a different config hash")
        print(f"Existing v6 freeze manifest verified: {FREEZE}", flush=True)
        return existing

    # Use the original v5 globals while validating the audited split metadata.
    old = (v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS)
    v5.CONFIG = SOURCE_CONFIG_PATH
    v5.BASE = ROOT / "outputs" / "exploratory_v5"
    v5.FRAME = v5.BASE / "development_frame.json"
    v5.STATS = v5.BASE / "fit_only_statistics.json"
    source_config = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    fit, source_frame, grids = v5.prepare(source_config)
    source_stats = json.loads(v5.STATS.read_text(encoding="utf-8"))
    v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS = old

    if len(fit) != config["fit_process_count"] or len(source_frame["paired"]) != config["development_process_count"]:
        raise RuntimeError("v6 source process counts do not match the frozen protocol")
    if len(source_frame["real_missing"]) != config["development_real_missing_process_count"]:
        raise RuntimeError("v6 source real-missing process count changed")

    BASE.mkdir(parents=True, exist_ok=True)
    frame = dict(source_frame)
    frame["seed"] = config["development_mask_seed"]
    frame["purpose"] = "v6 robustness development frame; exact v5 frame with frozen common mask seed"
    frame["source_v5_frame_sha256"] = sha256(ROOT / config["source_v5_frame"])
    FRAME.write_text(json.dumps(frame, indent=2) + "\n", encoding="utf-8")

    stats = dict(source_stats)
    stats.update({
        "config_sha256": sha256(CONFIG_PATH),
        "source_v5_stats_sha256": sha256(ROOT / config["source_v5_fit_statistics"]),
        "v5_split_sha256": sha256(ROOT / config["source_v5_split"]),
        "purpose": "v6 train-source reuse of frozen v5 fit-only normalization and class weights",
    })
    STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    code_files = [
        CONFIG_PATH, SOURCE_CONFIG_PATH, Path(__file__),
        ROOT / "scripts" / "train_impactmesh_exploratory_v5.py",
        ROOT / "scripts" / "train_impactmesh_exploratory_v2.py",
        ROOT / "scripts" / "train_impactmesh_formal.py",
        ROOT / "src" / "flood_reliability" / "models.py",
        ROOT / "src" / "flood_reliability" / "models_v5.py",
    ]
    manifest = {
        "purpose": config["purpose"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG_PATH),
        "source_v5_config_sha256": sha256(SOURCE_CONFIG_PATH),
        "source_v5_split_sha256": sha256(ROOT / config["source_v5_split"]),
        "source_v5_frame_sha256": sha256(ROOT / config["source_v5_frame"]),
        "source_v5_fit_statistics_sha256": sha256(ROOT / config["source_v5_fit_statistics"]),
        "source_split_json_sha256": sha256((ROOT / config["source_v5_split"]).with_suffix(".json")),
        "original_split_sha256": sha256(ROOT / "data/manifests/impactmesh_process_split_grouped.csv"),
        "v4_split_sha256": sha256(ROOT / "data/manifests/impactmesh_v4_training_dev_processes.csv"),
        "eligibility_sha256": sha256(ROOT / "data/manifests/impactmesh_all_train_frozen_eligibility.csv.gz"),
        "grid_sha256": sha256(ROOT / "data/manifests/impactmesh_train_mask_grid_index.csv"),
        "v6_frame_sha256": sha256(FRAME),
        "v6_fit_statistics_sha256": sha256(STATS),
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_files},
        "fit_process_count": len(fit),
        "development_process_count": len(source_frame["paired"]),
        "development_real_missing_process_count": len(source_frame["real_missing"]),
        "contains_test_or_holdout_data": False,
        "contains_model_scores": False,
    }
    FREEZE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created frozen v6 manifest: {FREEZE}", flush=True)
    return manifest


def run(candidate, seed, device_name, config):
    if not FREEZE.exists():
        raise RuntimeError("Run --freeze and record the v6 freeze manifest before training")
    if seed not in config["seeds"] or candidate not in config["candidates"]:
        raise RuntimeError("Unfrozen v6 seed or candidate")
    if device_name == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("XPU unavailable")
    base = BASE
    # Point the v5 training implementation at the v6 immutable assets.
    v5.CONFIG = CONFIG_PATH
    v5.BASE = base
    v5.FRAME = FRAME
    v5.STATS = STATS
    v5.SPLIT_V5 = ROOT / config["source_v5_split"]

    def model_for(name, cfg, device):
        spec = cfg["candidates"][name]
        model = DualUNet(2, 12, 3, cfg["base_width"], gated=spec["architecture"] == "dual_unet_gated")
        return model.to(device)

    v5.model_for = model_for
    # prepare() validates the source split, while the v6 frame/statistics are
    # already frozen and must be used by train().
    old_globals = (v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS, v5.SPLIT_V5)
    source_config = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    v5.CONFIG = SOURCE_CONFIG_PATH
    v5.BASE = ROOT / "outputs" / "exploratory_v5"
    v5.FRAME = v5.BASE / "development_frame.json"
    v5.STATS = v5.BASE / "fit_only_statistics.json"
    v5.SPLIT_V5 = ROOT / config["source_v5_split"]
    fit, _, grids = v5.prepare(source_config)
    v5.CONFIG, v5.BASE, v5.FRAME, v5.STATS, v5.SPLIT_V5 = old_globals
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    v5.train(candidate, seed, config, fit, frame, grids, torch.device(device_name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--candidate", choices=("B4_v6", "B5_v6", "P_v6"))
    parser.add_argument("--seed", type=int, choices=(20261011, 20261012))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    config = load_config()
    if args.freeze:
        freeze(config)
        if args.candidate is None:
            return
    if args.candidate is None or args.seed is None:
        parser.error("training requires --candidate and --seed; use --freeze first")
    run(args.candidate, args.seed, args.device, config)


if __name__ == "__main__":
    main()
