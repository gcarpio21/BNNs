"""
Generates the Sinusoid comparison-dataset (big/second set) calibration-error
bar charts used in the report's results section:
5seed_sinusoid_comparison_bayesian_calibration_error_bar.png and
5seed_sinusoid_comparison_laplace_calibration_error_bar.png, from the
comparison notebook's 5-seed metrics CSV.
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

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

ABBREV = {"gridsearch": "gs", "all weights": "aw", "last layer": "ll", "marglik": "ml",
          "adam loop": "al"}


def abbreviate_title(v):
    for full, ab in ABBREV.items():
        v = v.replace(full, ab)
    return v.replace(" / ", "/")


def plot_calibration_bar(df, title, fname):
    labels = [abbreviate_title(v) for v in df["Variant"]]
    means = df["Calibration_Error_mean"].to_numpy()
    stds  = df["Calibration_Error_std"].to_numpy()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.7, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha="right")
    ax.set_ylabel("Calibration Error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


df = pd.read_csv(CSV)
bt_df = df[df["Method"] == "BT"]
la_df = df[df["Method"] == "Laplace"]

plot_calibration_bar(bt_df, "Sinusoid Comparison Bayesian — Calibration Error (mean ± std, 5 seeds)",
                      "5seed_sinusoid_comparison_bayesian_calibration_error_bar.png")
plot_calibration_bar(la_df, "Sinusoid Comparison Laplace — Calibration Error (mean ± std, 5 seeds)",
                      "5seed_sinusoid_comparison_laplace_calibration_error_bar.png")
print("Done.")
