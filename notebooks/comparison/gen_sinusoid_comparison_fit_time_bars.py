"""
Generates the Sinusoid comparison-dataset (big/second set) total-training-time
bar charts used in the report's results section:
5seed_sinusoid_comparison_bayesian_total_time_bar.png and
5seed_sinusoid_comparison_laplace_total_time_bar.png, from the comparison
notebook's 5-seed metrics CSV.

Total time is the fair, comparable wall-clock cost per variant: for
bayesian-torch, fit time plus the shared MAP fit for MOPED variants (fit
time alone for scratch); for laplace-torch, the shared MAP fit plus fit
plus tune time (every variant requires the MAP fit). No error bars are
shown, since the underlying per-seed values for this combined quantity
are not available (only the aggregated per-component means are).

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

CSV = ROOT / "notebooks" / "comparison" / "results" / "metrics" / "sinusoid_comparison_5seed_metrics.csv"
FIGURES_DIR = ROOT / "notebooks" / "figures"
MAP_TIME = 13.668

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

ABBREV = {"gridsearch": "gs", "all weights": "aw", "last layer": "ll", "marglik": "ml",
          "adam loop": "al"}


def abbreviate_title(v):
    for full, ab in ABBREV.items():
        v = v.replace(full, ab)
    return v.replace(" / ", "/")


def total_time(df, is_la):
    if is_la:
        return (MAP_TIME + df["Fit_Time (s)_mean"] + df["Tune_Time (s)_mean"]).to_numpy()
    return (df["Fit_Time (s)_mean"] + df["Variant"].apply(lambda v: MAP_TIME if "moped" in v else 0.0)).to_numpy()


def plot_total_time_bar(df, means, title, fname, ylim):
    labels = [abbreviate_title(v) for v in df["Variant"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, color="steelblue", alpha=0.7, edgecolor="k")
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha="right")
    ax.set_ylabel("Total Time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


df = pd.read_csv(CSV)
bt_df = df[df["Method"] == "BT"]
la_df = df[df["Method"] == "Laplace"]
bt_means = total_time(bt_df, is_la=False)
la_means = total_time(la_df, is_la=True)

all_means = np.concatenate([bt_means, la_means])
ylim = (0, all_means.max() * 1.15)

plot_total_time_bar(bt_df, bt_means, "Sinusoid Comparison Bayesian — Total Time (mean, 5 seeds)",
                     "5seed_sinusoid_comparison_bayesian_total_time_bar.png", ylim)
plot_total_time_bar(la_df, la_means, "Sinusoid Comparison Laplace — Total Time (mean, 5 seeds)",
                     "5seed_sinusoid_comparison_laplace_total_time_bar.png", ylim)
print("Done.")
