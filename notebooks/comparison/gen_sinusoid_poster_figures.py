"""
Generates sinusoidal poster figures from 5-seed averaged results:
  merged_sin_cal_mae.png          — Cal MAE bar chart (mean ± std, 5 seeds)
  reliability_sin_all_variants.png — Reliability curves (mean ± 1σ band, 5 seeds)

5 variants: Flipout scratch, BBB scratch, Kron last marglik,
            Kron all marglik, Full all gridsearch.

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
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from scipy.stats import norm as _norm

from shared import load_checkpoint, seed_everything, load_sinusoid

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS       = [711, 42, 123, 456, 789]
sigma_noise = 0.3
CKPT        = str(ROOT / "results" / "checkpoints" / "sinusoid_comparison_5seed")
OUT         = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")
CONF        = np.linspace(0.05, 0.95, 20)

print(f"Device: {device}")

# Test data on device (seed 711 — fixed split)
data711   = load_sinusoid(n_data=600, sigma_noise=sigma_noise, batch_size=64, seed=711,
                          device=device, n_val=200, n_test_id=200, n_test_ood=200)
X_test_id = data711["X_test_id"].to(device)
y_id      = data711["y_test_id"].cpu().flatten().numpy()

train_loader_dev = data711["train_loader"]

# --- helpers ------------------------------------------------------------------

def get_net(seed):
    seed_everything(seed)
    return nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(device)

def bnn_predict(net, X, n=100):
    net.eval()
    with torch.no_grad():
        preds = torch.stack([net(X) for _ in range(n)])
    return preds.mean(0).squeeze().cpu().numpy(), preds.std(0).squeeze().cpu().numpy()

def la_predict(la, X):
    f_mu, f_var = la(X)
    return f_mu.squeeze().detach().cpu().numpy(), f_var.squeeze().sqrt().detach().cpu().numpy()

def cal_mae_and_curve(y, mu, total_std):
    z   = np.abs((y - mu) / np.clip(total_std, 1e-12, None))
    obs = np.array([float(np.mean(z <= _norm.ppf((1 + c) / 2))) for c in CONF])
    return float(np.mean(np.abs(obs - CONF))), obs

def load_bnn_one_seed(seed, fname, model_type):
    net = get_net(seed)
    dnn_to_bnn(net, {"prior_mu": 0.0, "prior_sigma": 1.0,
                     "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
                     "type": model_type, "moped_enable": False, "moped_delta": 0.5})
    net.to(device)
    ck = load_checkpoint(f"{CKPT}/{fname}", map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    mu_id, epi_id = bnn_predict(net, X_test_id)
    total_id = np.sqrt(epi_id**2 + sigma_noise**2)
    return cal_mae_and_curve(y_id, mu_id, total_id)

def load_la_one_seed(seed, fname, subset, hessian):
    net = get_net(seed)
    map_ck = load_checkpoint(f"{CKPT}/sinusoid_map_seed{seed}.pt", map_location=device)
    net.load_state_dict(map_ck["model_state_dict"])
    la = Laplace(net, "regression", subset_of_weights=subset, hessian_structure=hessian)
    ck = load_checkpoint(f"{CKPT}/{fname}", map_location=device)
    la.load_state_dict(ck["la_state_dict"])
    sigma = float(la.sigma_noise.item())
    mu_id, epi_id = la_predict(la, X_test_id)
    total_id = np.sqrt(epi_id**2 + sigma**2)
    return cal_mae_and_curve(y_id, mu_id, total_id)

def aggregate(loader_fn):
    maes, curves = [], []
    for s in SEEDS:
        mae, curve = loader_fn(s)
        maes.append(mae); curves.append(curve)
    curves = np.stack(curves)
    return np.mean(maes), np.std(maes), curves.mean(0), curves.std(0)

# --- load all 5 variants (5 seeds each) ---------------------------------------
print("Loading variants across 5 seeds...")

variants = {}

print("  Flipout / scratch")
variants["Flipout / scratch"] = aggregate(
    lambda s: load_bnn_one_seed(s, f"sinusoid_flipout_scratch_seed{s}.pt", "Flipout"))

print("  BBB / scratch")
variants["BBB / scratch"] = aggregate(
    lambda s: load_bnn_one_seed(s, f"sinusoid_bbb_scratch_seed{s}.pt", "Reparameterization"))

print("  Kron / last / marglik")
variants["Kron / last / marglik"] = aggregate(
    lambda s: load_la_one_seed(s, f"sinusoid_kron_last_layer_marglik_seed{s}.pt", "last_layer", "kron"))

print("  Kron / all / marglik")
variants["Kron / all / marglik"] = aggregate(
    lambda s: load_la_one_seed(s, f"sinusoid_la_kron_all_marglik_seed{s}.pt", "all", "kron"))

print("  Full / all / gridsearch")
variants["Full / all / gridsearch"] = aggregate(
    lambda s: load_la_one_seed(s, f"sinusoid_la_full_all_gridsearch_seed{s}.pt", "all", "full"))

# ---- figure 1: merged_sin_cal_mae — bar chart with error bars ----------------
print("\nGenerating merged_sin_cal_mae.png ...")

labels = list(variants.keys())
means  = [variants[k][0] for k in labels]
stds   = [variants[k][1] for k in labels]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

fig, ax = plt.subplots(figsize=(5.5, 3))
bars = ax.bar(range(len(labels)), means, color=colors, edgecolor="white", linewidth=0.5)
ax.errorbar(range(len(labels)), means, yerr=stds, fmt="none", color="black",
            capsize=4, linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
ax.set_ylabel("Calibration MAE")
ax.set_title("Sinusoidal — Cal MAE (5 seeds)")
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.3)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.0005,
            f"{m:.4f}", ha="center", va="bottom", fontsize=7.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/merged_sin_cal_mae.png")
print("  saved merged_sin_cal_mae.png")

# ---- figure 2: reliability_sin_all_variants — mean ± 1σ bands ---------------
print("Generating reliability_sin_all_variants.png ...")

rel_entries = [
    ("Flipout / scratch",       colors[0], "-"),
    ("BBB / scratch",           colors[1], "--"),
    ("Kron / last / marglik",   colors[2], "-"),
    ("Kron / all / marglik",    colors[3], "--"),
    ("Full / all / gridsearch", colors[4], ":"),
]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect", zorder=5)
for label, color, ls in rel_entries:
    _, _, mean_cov, std_cov = variants[label]
    ax.plot(CONF, mean_cov, ls=ls, color=color, lw=1.5, marker="o", ms=4, label=label)
    ax.fill_between(CONF, mean_cov - std_cov, mean_cov + std_cov,
                    color=color, alpha=0.15)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Expected confidence")
ax.set_ylabel("Observed coverage")
ax.set_title("Sinusoidal — Calibration Reliability (5 seeds)")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/reliability_sin_all_variants.png")
print("  saved reliability_sin_all_variants.png")

print("\nDone. Figures written to", OUT)
