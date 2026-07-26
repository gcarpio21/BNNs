"""
Generates the Two Moons inference-time bar charts used in the report's
results section: 5seed_two_moons_bayesian_inf_time_bar.png and
5seed_two_moons_laplace_inf_time_bar.png, from the 5-seed metrics CSVs.

Inference time is the per-variant "Inf_Time (s)" column (mean and std
already aggregated across 5 seeds).

Both charts share the same y-axis limits so that bar heights are directly,
visually comparable between bayesian-torch and laplace-torch, not just
readable off two independently auto-scaled axes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent

BT_CSV = ROOT / "notebooks" / "bayesian" / "results" / "metrics" / "two_moons_bayesian_5seed_metrics.csv"
LA_CSV = ROOT / "notebooks" / "laplace"  / "results" / "metrics" / "two_moons_laplace_5seed_metrics.csv"
FIGURES_DIR = ROOT / "notebooks" / "figures"

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})


def plot_inf_time_bar(df, title, fname, ylim):
    labels = [v.replace(" / ", "\n") for v in df["Variant"]]
    x = np.arange(len(labels))
    means = df["Inf_Time (s)_mean"].to_numpy()
    stds  = df["Inf_Time (s)_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.7, edgecolor="k")
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Inference Time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


bt_df = pd.read_csv(BT_CSV)
la_df = pd.read_csv(LA_CSV)

all_hi = np.concatenate([bt_df["Inf_Time (s)_mean"] + bt_df["Inf_Time (s)_std"],
                          la_df["Inf_Time (s)_mean"] + la_df["Inf_Time (s)_std"]])
ylim = (0, all_hi.max() * 1.15)

plot_inf_time_bar(bt_df, "Two Moons Bayesian — Inference Time (mean ± std, 5 seeds)",
                   "5seed_two_moons_bayesian_inf_time_bar.png", ylim)
plot_inf_time_bar(la_df, "Two Moons Laplace — Inference Time (mean ± std, 5 seeds)",
                   "5seed_two_moons_laplace_inf_time_bar.png", ylim)
print("Done.")
