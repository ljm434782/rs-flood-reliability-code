"""Generate the manuscript's exploratory v5 table from frozen selection."""

import hashlib
import json
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "exploratory_v5"
FROZEN = BASE / "candidate_selection_frozen.json"
DETAIL = BASE / "selected_development_detail.json"
CONFIG = ROOT / "configs" / "impactmesh_exploratory_v5.json"
OUT = ROOT / "paper" / "tables" / "v5_diagnostic.tex"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    detail = json.loads(DETAIL.read_text(encoding="utf-8"))
    if (frozen["config_sha256"] != sha256(CONFIG)
            or detail["selection_frozen_sha256"] != sha256(FROZEN)
            or detail["promising_internal_signal"] is not False
            or frozen["choice"] != "B5_v5"):
        raise RuntimeError("v5 source or stop-decision mismatch")
    rows = []
    by_candidate = {row["candidate"]: row for row in frozen["ranking"]}
    for candidate in ("B4_v5", "B5_v5", "F_aux_v5"):
        row = by_candidate[candidate]
        seeds = row["seeds"]
        score1 = seeds["20261001"]["score"]
        score2 = seeds["20261002"]["score"]
        epoch1 = seeds["20261001"]["epoch"]
        epoch2 = seeds["20261002"]["epoch"]
        params = seeds["20261001"]["parameter_count"]
        if params != seeds["20261002"]["parameter_count"]:
            raise RuntimeError("Seed parameter counts differ")
        name = candidate.replace("_", r"\_")
        rows.append(
            f"{name} & {params:,} & "
            f"{score1:.6f} ({epoch1}) & {score2:.6f} ({epoch2}) & "
            f"{row['mean_score']:.6f} " + r"\\"
        )
    content = "\n".join([
        "% Generated from outputs/exploratory_v5/candidate_selection_frozen.json.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Post-test exploratory v5 development scores (lower is better). The two fixed seeds use the same original paired training pool and budget within each seed. Parentheses give the selected epoch of 12; these are training-derived development results, not independent test scores.}",
        r"\label{tab:v5}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Model & Parameters & Seed 1 & Seed 2 & Mean " + r"\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    OUT.write_text(content, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
