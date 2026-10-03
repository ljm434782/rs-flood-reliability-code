"""Draw a diagnostic figure from the audited STURM candidate-mask table."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "manifests" / "sturm_candidate_mask_audit.csv"
UNMATCHED = ROOT / "data" / "manifests" / "sturm_unmatched_activation_audit.json"
DEST = ROOT / "figures"


def main() -> None:
    data = pd.read_csv(SOURCE)
    stages = [
        ("Same-name candidates", data),
        ("Any shared valid pixels", data[data.common_valid_pixels > 0]),
        ("At least 50% shared valid area", data[data.common_valid_fraction_of_s1 >= 0.5]),
        ("Also same CEMS map ID", data[(data.common_valid_fraction_of_s1 >= 0.5) & data.same_floodmap_id]),
        ("Also S2 acquired by S1 time", data[(data.common_valid_fraction_of_s1 >= 0.5) & data.same_floodmap_id & (data.delta_s2_minus_s1_days <= 0)]),
    ]
    counts = [len(frame) for _, frame in stages]
    processes = [frame.ems_code.nunique() for _, frame in stages]
    unmatched = json.loads(UNMATCHED.read_text(encoding="utf-8"))
    if unmatched["checked_masks"] != 2470 or any(item["positive_area_spatial_pairs"] for item in unmatched["activations"]):
        raise RuntimeError("Unmatched-activation footprint audit changed")
    both_modalities = processes[0] + len(unmatched["activations"])
    potential_spatial = both_modalities - len(unmatched["activations"])
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    fig.subplots_adjust(left=0.35, right=0.98, top=0.76, bottom=0.20)
    y = list(range(len(stages)))
    ax.barh(y, counts, color=["#8997a5"] * 3 + ["#507ea7", "#1965a0"], height=0.64)
    ax.set_yticks(y, [name for name, _ in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, 1250)
    ax.set_xlabel("Candidate tile pairs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#d9e0e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for row, (count, activation_count) in enumerate(zip(counts, processes)):
        ax.text(count + 10, row, f"{count} pairs / {activation_count} activations", va="center", fontsize=9)
    fig.suptitle("STURM-Flood pairing feasibility audit", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.03, 0.89, f"Activation inventory: {both_modalities} contain both modalities; {len(unmatched['activations'])} unmatched groups have zero spatial overlap.", fontsize=10, color="#263238")
    fig.text(0.03, 0.84, f"At most {potential_spatial} activations may contain a spatial pair; this does not establish qualified study processes.", fontsize=10, color="#155e91", fontweight="bold")
    fig.text(0.03, 0.06, "Bars: same-name tiles only. Filters are diagnostic; map ID and overlap do not establish label truth or a final sample.", fontsize=8, color="#4c5560")
    DEST.mkdir(exist_ok=True)
    fig.savefig(DEST / "sturm_pairing_audit.png", dpi=220)
    fig.savefig(DEST / "sturm_pairing_audit.pdf")
    plt.close(fig)
    print(f"Saved {DEST / 'sturm_pairing_audit.png'} and PDF")


if __name__ == "__main__":
    main()
