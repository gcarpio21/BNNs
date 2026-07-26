"""
Regenerate the dataset display figures for the poster:
  two_moons_dataset.png    — two moons scatter (no title)
  povertymap_samples.png   — 4 images 2x2
  sinusoid_dataset.png     — train + OOD test points + ID test points (no title)

Run from repo root or notebooks/comparison/.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

OUT = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")

# ── 0. Two Moons: scatter plot, no title ──────────────────────────────────────
print("Generating two_moons_dataset.png ...")
from shared import load_two_moons
tm_data = load_two_moons(seed=42, noise=0.3, batch_train=32, batch_eval=64, device=torch.device("cpu"))
X_all, y_all = tm_data["X"], tm_data["y"]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(X_all[y_all == 0, 0], X_all[y_all == 0, 1], c="blue", alpha=0.5, s=15, label="Class 0")
ax.scatter(X_all[y_all == 1, 0], X_all[y_all == 1, 1], c="red",  alpha=0.5, s=15, label="Class 1")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(f"{OUT}/two_moons_dataset.png", dpi=150, bbox_inches="tight")
print("  saved two_moons_dataset.png")

# ── 1. PovertyMap: 4 images from diverse countries ────────────────────────────
print("Generating povertymap_samples.png (4 images) ...")

from wilds import get_dataset
WILDS_ROOT = str(ROOT / "data" / "wilds")

def to_rgb(x):
    """Per-channel percentile stretch of bands 3,2,1 (Red,Green,Blue 0-indexed)."""
    rgb = x[[3, 2, 1]].permute(1, 2, 0).numpy().copy()
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb

# Pick 2 samples per country (high + low wealth), Angola first then Benin
dataset = get_dataset('poverty', download=False, root_dir=WILDS_ROOT, fold='A')
train_sub = dataset.get_subset('train', transform=lambda x: x.float())

by_country = {}  # country -> list of (x, wealth)
for idx in range(len(train_sub)):
    x, y, meta = train_sub[idx]
    ci = int(meta[0].item())
    country = dataset.metadata_map['country'][ci].title()
    by_country.setdefault(country, []).append((x, float(y), country))

samples = []
for country_samples in by_country.values():
    sorted_cs = sorted(country_samples, key=lambda t: t[1])
    samples.append(sorted_cs[0])                      # lowest wealth
    samples.append(sorted_cs[len(sorted_cs)//2])      # median wealth
    if len(samples) >= 4:
        break

fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.5))
for ax, (x, wealth, country) in zip(axes.flat, samples):
    ax.imshow(to_rgb(x))
    ax.set_title(f"{country}\nwealth={wealth:.2f}", fontsize=8)
    ax.axis("off")

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/povertymap_samples.png", bbox_inches="tight")
print(f"  saved povertymap_samples.png")


# ── 2. Sinusoid: add ID + OOD test scatter points ────────────────────────────
print("Generating sinusoid_dataset.png (with OOD test points) ...")

from shared import load_sinusoid

SEED = 711
TRAIN_LO, TRAIN_HI = 0.0, 8.0

data = load_sinusoid(n_data=600, sigma_noise=0.3, batch_size=64, seed=SEED,
                     device=torch.device("cpu"), n_val=200, n_test_id=200, n_test_ood=200)
X_train_np    = data["X_train"].flatten().numpy()
y_train_np    = data["y_train"].flatten().numpy()
X_test_id_np  = data["X_test_id"].flatten().numpy()
y_test_id_np  = data["y_test_id"].flatten().numpy()
X_test_ood_np = data["X_test_ood"].flatten().numpy()
y_test_ood_np = data["y_test_ood"].flatten().numpy()
x_grid = np.linspace(-5, 13, 500)

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.axvspan(-5, TRAIN_LO, color="gray", alpha=0.10, label="OOD region")
ax.axvspan(TRAIN_HI, 13, color="gray", alpha=0.10)
ax.plot(x_grid, np.sin(x_grid), "k--", lw=1, alpha=0.6, label="True sin(x)")
ax.scatter(X_train_np,    y_train_np,    c="tab:blue",  s=8,  alpha=0.3, label="Train (600)")
ax.scatter(X_test_id_np,  y_test_id_np,  c="tab:green", marker="^", s=20, alpha=0.6, label="ID test (200)")
ax.scatter(X_test_ood_np, y_test_ood_np, c="tab:red",   marker="x", s=25, alpha=0.6, label="OOD test (200)")
ax.set_xlim(-5, 13); ax.set_ylim(-3, 3)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
fig.savefig(f"{OUT}/sinusoid_dataset.png", dpi=150, bbox_inches="tight")
print(f"  saved sinusoid_dataset.png")

print("\nDone.")
