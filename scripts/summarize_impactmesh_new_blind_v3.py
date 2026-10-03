"""Summarize source-locked blind events using the predeclared paired score."""

import csv
import json
from pathlib import Path

import numpy as np

from scripts.train_impactmesh_exploratory_v3 import BASE, CONFIG
from scripts.train_impactmesh_formal import ROOT, sha256

STATES = ("full", "half", "zero")
MAN = ROOT / "data/manifests/impactmesh_new_blind_locked_tiles.csv"
OUT = BASE / "new_blind_summary.json"
TABLE = BASE / "new_blind_summary.csv"
SEED = 20260929
REPS = 10000


def event_score(record):
    if record["true_flood_pixels_owned"] <= 0:
        return None
    values = []
    for state in STATES:
        score = record["scores"][state]
        if score["flood_iou"] is None:
            raise RuntimeError("Flood-positive event has undefined Flood IoU")
        values.append(score["nll"] - score["flood_iou"])
    return float(np.mean(values))


def paired_bootstrap(left, right, events):
    differences = np.asarray([left[event] - right[event] for event in events], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(events), size=(REPS, len(events)))
    boot = differences[draws].mean(axis=1)
    return {"events": len(events), "mean_difference": float(differences.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "event_differences": {event: float(value) for event, value in zip(events, differences)}}


def main():
    if OUT.exists() or TABLE.exists():
        raise RuntimeError("Refusing to overwrite blind summary")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen_path = BASE / "candidate_selection_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    choice = frozen["choice"]
    data = {}
    for candidate in [*config["candidates"], "B4_v1", "B5_v1"]:
        path = BASE / candidate / "new_blind_paired.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload["candidate"] != candidate or payload["locked_manifest_sha256"] != sha256(MAN)
                or payload["selection_frozen_sha256"] != sha256(frozen_path)):
            raise RuntimeError("Blind outputs have mismatched source or selection hashes")
        data[candidate] = payload["results"]
    event_ids = sorted(data[choice])
    if any(sorted(data[candidate]) != event_ids for candidate in data):
        raise RuntimeError("Matched models have different blind events")
    positive = [event for event in event_ids if data[choice][event]["true_flood_pixels_owned"] > 0]
    if any([event for event in event_ids if data[candidate][event]["true_flood_pixels_owned"] > 0] != positive
           for candidate in data):
        raise RuntimeError("Blind label truth differs across models")
    event_scores = {candidate: {event: event_score(record)
                                for event, record in results.items() if event in positive}
                    for candidate, results in data.items()}
    summary = {}
    table = []
    for candidate, results in data.items():
        states = {}
        for state in STATES:
            metric = {key: float(np.mean([results[event]["scores"][state][key]
                                          for event in event_ids]))
                      for key in ("nll", "brier", "ece15")}
            for key in ("flood_iou", "flood_auprc"):
                values = [results[event]["scores"][state][key] for event in positive]
                metric[key] = float(np.mean(values)) if values else None
            states[state] = metric
            table.append({"candidate": candidate, "state": state, "events": len(event_ids),
                          "flood_positive_events": len(positive), **metric})
        summary[candidate] = {"states": states,
            "positive_event_composite_score": (float(np.mean(list(event_scores[candidate].values())))
                                               if positive else None),
            "event_composite_scores": event_scores[candidate]}
    differences = {}
    if positive:
        for candidate in ("P_sigmoid_v3", "P_residual_v3"):
            differences[candidate] = {control: paired_bootstrap(event_scores[candidate],
                event_scores[control], positive) for control in ("B4_v3", "B5_v3")}
    statement = (choice in differences and len(positive) >= 5 and
                 all(differences[choice][control]["mean_difference"] < 0 and
                     differences[choice][control]["ci95"][1] < 0
                     for control in ("B4_v3", "B5_v3")))
    time_aligned = [event for event in positive if event != "EMSR843"]
    time_aligned_scores = {candidate: (float(np.mean([event_scores[candidate][event]
                                                    for event in time_aligned])) if time_aligned else None)
                           for candidate in data}
    OUT.write_text(json.dumps({"purpose": "post-v1-test exploratory new-event blind summary",
        "source_lock_sha256": sha256(MAN), "selection_frozen_sha256": sha256(frozen_path),
        "all_locked_events": event_ids, "flood_positive_events": positive,
        "selection_choice_from_dev16": choice,
        "descriptive_selected_p_superiority_criterion_met": statement,
        "metric": "event_equal_mean_calibrated_nll_minus_flood_iou_over_full_half_zero",
        "bootstrap_repetitions": REPS, "bootstrap_seed": SEED,
        "results": summary, "p_vs_controls": differences,
        "source_time_sensitivity_excluding_EMSR843": {
            "flood_positive_events": time_aligned, "candidate_composite_scores": time_aligned_scores,
            "reason": "EMSR843 closest CEMS delineation source image is 4.43 h after event S1"},
        "no_new_blind_tuning": True}, indent=2) + "\n", encoding="utf-8")
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({"choice": choice, "events": event_ids,
                      "flood_positive_events": positive,
                      "scores": {c: summary[c]["positive_event_composite_score"] for c in summary},
                      "criterion_met": statement}, indent=2))


if __name__ == "__main__":
    main()
