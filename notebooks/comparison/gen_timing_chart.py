"""
Generate stacked horizontal bar chart for training & inference timings.
Replaces the timing table in the poster.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

OUT = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")

# Phase colors
C_MAP  = "#888888"   # MAP Train
C_FIT  = "#4C72B0"   # Fit / VI Train
C_TUNE = "#DD8452"   # LA Tune
C_INF  = "#55A868"   # Inference

# ── Data (from poster table, numbers in seconds) ───────────────────────────
# Two Moons (5-seed means)
tm = {
    "variants": ["Flipout/moped", "BBB/moped", "Kron/last\ngridsearch"],
    "map":  [67.4,  67.4,  67.4],
    "fit":  [28.4,  29.4,   0.5],
    "tune": [ 0.0,   0.0,   2.8],
    "inf":  [0.046, 0.028, 0.001],
}

# Sinusoid (single seed)
sin = {
    "variants": ["Flipout/scratch", "BBB/scratch", "Kron/last\nmarglik"],
    "map":  [ 0.0,   0.0,  5.0],
    "fit":  [535.0, 565.3,  0.7],
    "tune": [ 0.0,   0.0,  0.04],
    "inf":  [0.019, 0.008, 0.095],
}


def draw_panel(ax, data, title):
    variants = data["variants"]
    n = len(variants)
    y = np.arange(n)
    h = 0.45

    for i in range(n):
        left = 0.0
        m, f, t, inf = data["map"][i], data["fit"][i], data["tune"][i], data["inf"][i]

        if m > 0:
            ax.barh(y[i], m, height=h, left=left, color=C_MAP, zorder=2)
            left += m
        ax.barh(y[i], f, height=h, left=left, color=C_FIT, zorder=2)
        left += f
        if t > 0:
            ax.barh(y[i], t, height=h, left=left, color=C_TUNE, zorder=2)
            left += t
        # Inference is too small to see — annotate as text
        total = left + inf
        ax.text(total + ax.get_xlim()[1] * 0.01, y[i],
                f"+{inf*1000:.0f} ms inf", va="center", fontsize=7, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=8.5)
    ax.set_xlabel("Time (s)", fontsize=8.5)
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))

draw_panel(ax1, tm,  "Two Moons")
draw_panel(ax2, sin, "Sinusoid")

# Fix x-limits so the inference annotations don't clip
for ax, data in [(ax1, tm), (ax2, sin)]:
    totals = [data["map"][i] + data["fit"][i] + data["tune"][i] + data["inf"][i]
              for i in range(len(data["variants"]))]
    ax.set_xlim(0, max(totals) * 1.22)

legend_patches = [
    mpatches.Patch(color=C_MAP,  label="MAP Train"),
    mpatches.Patch(color=C_FIT,  label="Fit / VI Train"),
    mpatches.Patch(color=C_TUNE, label="LA Tune"),
    mpatches.Patch(facecolor="white", edgecolor="#555", label="Inference (ms, annotated)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.13), fontsize=8, frameon=False)

fig.tight_layout(rect=[0, 0.02, 1, 1])
out_path = f"{OUT}/timing_chart.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
