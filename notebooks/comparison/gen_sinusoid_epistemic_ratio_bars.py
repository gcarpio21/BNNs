"""
Generates the Sinusoid OOD-vs-ID epistemic ratio bar charts used in the report's
results section (big/comparison dataset: 5seed_sinusoid_bayesian_epistemic_ratio_bar.png
and 5seed_sinusoid_laplace_epistemic_ratio_bar.png) and Appendix (small dataset:
5seed_sinusoid_small_bayesian_epistemic_ratio_bar.png and
5seed_sinusoid_small_laplace_epistemic_ratio_bar.png), from each dataset's
ID-vs-OOD metrics CSV.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent

OOD_CSV = ROOT / "notebooks" / "comparison" / "results" / "metrics" / "sinusoid_comparison_5seed_ood_metrics.csv"
SMALL_BT_OOD_CSV = ROOT / "notebooks" / "bayesian" / "results" / "metrics" / "sinusoid_bayesian_5seed_ood_metrics.csv"
SMALL_LA_OOD_CSV = ROOT / "notebooks" / "laplace"  / "results" / "metrics" / "sinusoid_laplace_5seed_ood_metrics.csv"
FIGURES_DIR = ROOT / "notebooks" / "figures"

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

ABBREV = {"gridsearch": "gs", "all weights": "aw", "last layer": "ll", "marglik": "ml",
          "adam loop": "al"}


def abbreviate_title(v):
    for full, ab in ABBREV.items():
        v = v.replace(full, ab)
    return v.replace(" / ", "/")


def parse_mean_std(s):
    m = re.match(r"\s*([\d.eE+-]+)\s*\xb1\s*([\d.eE+-]+)\s*", s)
    return float(m.group(1)), float(m.group(2))


def plot_epistemic_ratio_bar(df, title, fname):
    labels = [abbreviate_title(v) for v in df["Variant"]]
    means_stds = [parse_mean_std(v) for v in df["Epi Ratio"]]
    means = np.array([m for m, s in means_stds])
    stds = np.array([s for m, s in means_stds])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.7, edgecolor="k")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="Ratio = 1 (no OOD growth)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha="right")
    ax.set_ylabel("Epistemic Ratio (OOD / ID)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


df = pd.read_csv(OOD_CSV)
bt_df = df[df["Method"] == "BT"]
la_df = df[df["Method"] == "Laplace"]

plot_epistemic_ratio_bar(bt_df, "Sinusoid Comparison — Epistemic Ratio, bayesian-torch (5 seeds)",
                          "5seed_sinusoid_bayesian_epistemic_ratio_bar.png")
plot_epistemic_ratio_bar(la_df, "Sinusoid Comparison — Epistemic Ratio, laplace-torch (5 seeds)",
                          "5seed_sinusoid_laplace_epistemic_ratio_bar.png")

small_bt_df = pd.read_csv(SMALL_BT_OOD_CSV)
small_la_df = pd.read_csv(SMALL_LA_OOD_CSV)

plot_epistemic_ratio_bar(small_bt_df, "Sinusoid Small — Epistemic Ratio, bayesian-torch (5 seeds)",
                          "5seed_sinusoid_small_bayesian_epistemic_ratio_bar.png")
plot_epistemic_ratio_bar(small_la_df, "Sinusoid Small — Epistemic Ratio, laplace-torch (5 seeds)",
                          "5seed_sinusoid_small_laplace_epistemic_ratio_bar.png")
print("Done.")
