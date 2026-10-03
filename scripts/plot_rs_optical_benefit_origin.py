"""Draw the retrospective B4/B2 Figure 3 from frozen summary rows.

The figure shows paired process-equal mean gains and their already-computed
descriptive process-bootstrap intervals. It does not rerun inference or
calculate a new uncertainty interval.
"""
from __future__ import annotations

import math
from pathlib import Path


STATES = ("100", "75", "50", "25", "0")
ROLES = (("internal_retro", "(a) Internal retrospective", 18),
         ("holdout_retro", "(b) Holdout retrospective", 9))
SERIES = (
    ("flood_iou_gain", "Flood IoU", "#205E85", "o", "-", -0.075),
    ("binary_flood_brier_gain", "Flood Brier", "#B4572B", "s", "--", 0.075),
)


def _series(rows: list[dict], role: str, expected_n: int, metric: str):
    selected = [row for row in rows
                if row["role"] == role and row["comparator"] == "B4"]
    cells = {str(row["state"]): row for row in selected}
    if len(selected) != len(STATES) or set(cells) != set(STATES):
        raise ValueError(f"Figure 3 has incomplete B4 states for {role}")
    means, lows, highs = [], [], []
    for state in STATES:
        row = cells[state]
        if int(row["process_n_joint"]) != expected_n:
            raise ValueError(f"Figure 3 has wrong process support for {role}/{state}")
        mean = float(row[f"{metric}_mean"])
        low = float(row[f"{metric}_ci95_low"])
        high = float(row[f"{metric}_ci95_high"])
        if not all(map(math.isfinite, (mean, low, high))) or not low <= mean <= high:
            raise ValueError(f"Figure 3 has invalid interval for {role}/{state}/{metric}")
        means.append(mean)
        lows.append(mean - low)
        highs.append(high - mean)
    return means, (lows, highs)


def draw_origin_benefit(summary_rows: list[dict], pdf: Path,
                        png: Path | None = None) -> None:
    """Render a restrained, vector-first two-panel scientific figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pdf.parent.mkdir(parents=True, exist_ok=True)
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.titlesize": 12, "axes.labelsize": 11.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8, "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0),
                                 sharex=True, sharey=True)
        for ax, (role, title, expected_n) in zip(axes, ROLES):
            ax.axhline(0, color="#333333", linewidth=0.95, zorder=1)
            for metric, _, color, marker, linestyle, offset in SERIES:
                means, errors = _series(summary_rows, role, expected_n, metric)
                ax.errorbar([i + offset for i in range(len(STATES))], means,
                            yerr=errors, color=color, marker=marker,
                            linestyle=linestyle, linewidth=1.8, markersize=6.5,
                            markeredgecolor="white", markeredgewidth=0.65,
                            elinewidth=1.0, capsize=2.8, capthick=1.0,
                            zorder=3)
            ax.set_title(f"{title}  (n = {expected_n})", loc="left", pad=8)
            ax.set_xlim(-0.32, 4.32)
            ax.set_ylim(-0.115, 0.095)
            ax.set_xticks(range(len(STATES)), STATES)
            ax.set_yticks((-0.10, -0.05, 0, 0.05))
            ax.tick_params(axis="both", direction="out", length=3.5,
                           width=0.8, top=False, right=False)
            for spine in ax.spines.values():
                spine.set_color("#222222")
            ax.grid(False)
        fig.supylabel("B4 benefit vs B2", x=0.03, fontsize=11.5)
        fig.supxlabel("S2 retained (%)", x=0.55, y=0.035,
                     fontsize=11.5)
        handles = [Line2D([], [], color=color, marker=marker,
                          linestyle=linestyle, linewidth=1.8, markersize=6.5,
                          label=label)
                   for _, label, color, marker, linestyle, _ in SERIES]
        fig.legend(handles=handles, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.55, 1.0), frameon=False,
                   handlelength=2.7, columnspacing=2.0, fontsize=11.2)
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.14,
                            top=0.86, hspace=0.34)
        fig.savefig(pdf)
        if png is not None:
            fig.savefig(png, dpi=240)
        plt.close(fig)
