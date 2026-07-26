"""
Generates 3 Two Moons poster figures:
  cmp_fb_tm_ece.png              Flipout vs BBB ECE bar (best per variant, from 5-seed CSV)
  struct_tm_ece.png              Laplace structure ECE bar (from 5-seed CSV)
  reliability_tm_all_variants.png  merged reliability curves (mean ± 1σ, 5 seeds, checkpoints)
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from laplace import Laplace
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from shared import TinyMLP, load_checkpoint, seed_everything, load_two_moons

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

SEEDS     = [42, 711, 123, 456, 789]
CKPT_BNN  = str(ROOT / "results" / "checkpoints" / "two_moons_bayesian_5seed")
CKPT_LAP  = str(ROOT / "results" / "checkpoints" / "two_moons_laplace_5seed")
BNN_CSV   = str(ROOT / "notebooks" / "bayesian" / "results" / "metrics" / "two_moons_bayesian_5seed_metrics.csv")
LA_CSV    = str(ROOT / "notebooks" / "laplace"  / "results" / "metrics" / "two_moons_laplace_5seed_metrics.csv")
OUT       = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")
N_BINS    = 15
N_SAMPLES = 50
bins      = np.linspace(0, 1, N_BINS + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── helpers ────────────────────────────────────────────────────────────────────
def reliability_fracs(probs, y):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    accs, confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append((lo + hi) / 2)
        else:
            accs.append(float((pred[mask] == y[mask]).mean()))
            confs.append(float(conf[mask].mean()))
    return np.array(accs), np.array(confs)

def bnn_predict(X_np, model):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_np), 512):
            X_t = torch.tensor(X_np[i:i+512], dtype=torch.float32).to(device)
            sims = np.stack([torch.softmax(model(X_t), dim=1).cpu().numpy()
                             for _ in range(N_SAMPLES)])
            parts.append(sims.mean(axis=0))
    return np.concatenate(parts, axis=0)

def load_bnn(fname, model_type):
    net = TinyMLP(hidden=64)
    dnn_to_bnn(net, {"prior_mu": 0.0, "prior_sigma": 1.0,
                     "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
                     "type": model_type, "moped_enable": False, "moped_delta": 0.5})
    ck = load_checkpoint(f"{CKPT_BNN}/{fname}", map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    return net.to(device).eval()

def load_la(fname, subset, hessian, seed, train_loader=None):
    base = TinyMLP(hidden=64)
    map_ck = load_checkpoint(f"{CKPT_LAP}/two_moons_map_seed{seed}.pt", map_location=device)
    base.load_state_dict(map_ck["model_state_dict"])
    base.to(device)
    la = Laplace(base, "classification", subset_of_weights=subset, hessian_structure=hessian)
    ck = load_checkpoint(f"{CKPT_LAP}/{fname}", map_location=device)
    la.load_state_dict(ck["la_state_dict"])
    return la

def la_predict(la_v, X_np):
    results = []
    with torch.no_grad():
        for i in range(0, len(X_np), 512):
            X_t = torch.tensor(X_np[i:i+512], dtype=torch.float32).to(device)
            p = la_v(X_t, pred_type="glm", link_approx="probit")
            results.append(p.detach().cpu().numpy())
    return np.concatenate(results, axis=0)

def best_ece(df, prefix):
    """Return (label, mean, std) of the variant with lowest ECE_mean matching prefix."""
    rows = df[df['Variant'].str.startswith(prefix)]
    idx = rows['ECE_mean'].idxmin()
    return (str(rows.loc[idx, 'Variant']),
            float(rows.loc[idx, 'ECE_mean']),
            float(rows.loc[idx, 'ECE_std']))

# ── Figure 1: merged_tm_ece.png — all 5 variants from both CSVs ───────────────
print("Generating merged_tm_ece.png (from CSVs) ...")
bnn_df = pd.read_csv(BNN_CSV)
la_df  = pd.read_csv(LA_CSV)
# 5 variants: 2 BNN (moped only) + 3 Laplace
all_items = [
    best_ece(bnn_df, "Flipout / moped"),
    best_ece(bnn_df, "BBB / moped"),
    best_ece(la_df,  "Kron / last layer / gridsearch"),
    best_ece(la_df,  "Kron / all weights / gridsearch"),
    best_ece(la_df,  "Full / all weights / gridsearch"),
]
all_names  = [x[0] for x in all_items]
all_means  = [x[1] for x in all_items]
all_stds   = [x[2] for x in all_items]
# colors match reliability diagram below
bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

fig, ax = plt.subplots(figsize=(5.5, 3))
bars = ax.bar(range(len(all_names)), all_means, yerr=all_stds, capsize=5,
              color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=22, ha="right", fontsize=8)
ax.set_ylabel("ECE")
ax.set_title("Two Moons — ECE (5 seeds)")
ax.set_ylim(0, max(all_means[i] + all_stds[i] for i in range(len(all_means))) * 1.35)
for bar, m, s in zip(bars, all_means, all_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.0002, f"{m:.4f}",
            ha="center", va="bottom", fontsize=7.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/merged_tm_ece.png")
print("  saved merged_tm_ece.png")

# ── Figure 2: reliability — load checkpoints for 5 seeds ──────────────────────
# 5 variants: Flipout/moped ELBO, BBB/moped ELBO, all 3 Laplace gridsearch
print("\nCollecting reliability curves (5 seeds, from checkpoints)...")

bnn_rel_specs = [
    ("Flipout / moped", "two_moons_flipout_moped_elbo_seed{s}.pt", "Flipout"),
    ("BBB / moped",     "two_moons_bbb_moped_elbo_seed{s}.pt",     "Reparameterization"),
]
la_rel_specs = [
    ("Kron / last", "two_moons_gridsearch_kron_last_layer_seed{s}.pt", "last_layer", "kron"),
    ("Kron / all",  "two_moons_gridsearch_kron_all_weights_seed{s}.pt", "all",       "kron"),
    ("Full / all",  "two_moons_gridsearch_full_all_weights_seed{s}.pt", "all",       "full"),
]

bnn_rel = {n: {"accs": [], "confs": []} for n, _, _ in bnn_rel_specs}  # noqa: E501
la_rel  = {n: {"accs": [], "confs": []} for n, _, _, _ in la_rel_specs}

for seed in SEEDS:
    print(f"  Seed {seed}...")
    data   = load_two_moons(seed=seed, noise=0.3, batch_train=32, batch_eval=64, device=device)
    X_test = data["X_test"]
    y_test = data["y_test"]

    for name, tmpl, mtype in bnn_rel_specs:
        net   = load_bnn(tmpl.format(s=seed), mtype)
        probs = bnn_predict(X_test, net)
        a, c  = reliability_fracs(probs, y_test)
        bnn_rel[name]["accs"].append(a)
        bnn_rel[name]["confs"].append(c)

    for name, tmpl, subset, hessian in la_rel_specs:
        la_v  = load_la(tmpl.format(s=seed), subset, hessian, seed,
                        train_loader=data["train_loader"])
        probs = la_predict(la_v, X_test)
        a, c  = reliability_fracs(probs, y_test)
        la_rel[name]["accs"].append(a)
        la_rel[name]["confs"].append(c)

print("Generating reliability_tm_all_variants.png ...")
rel_styles = [
    ("Flipout / moped", bnn_rel, "tab:blue",   "-"),
    ("BBB / moped",     bnn_rel, "tab:orange", "--"),
    ("Kron / last",     la_rel,  "tab:green",  "-"),
    ("Kron / all",      la_rel,  "tab:purple", "--"),
    ("Full / all",      la_rel,  "tab:red",    ":"),
]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect", zorder=5)
for name, rel_dict, color, ls in rel_styles:
    all_accs  = np.array(rel_dict[name]["accs"])
    all_confs = np.array(rel_dict[name]["confs"])
    mean_obs  = np.nanmean(all_accs,  axis=0)
    std_obs   = np.nanstd(all_accs,   axis=0)
    mean_conf = np.nanmean(all_confs, axis=0)
    valid = ~np.isnan(mean_obs)
    ax.plot(mean_conf[valid], mean_obs[valid], ls=ls, color=color, lw=1.5,
            marker="o", ms=4, label=name)
    ax.fill_between(mean_conf[valid],
                    (mean_obs - std_obs)[valid],
                    (mean_obs + std_obs)[valid],
                    alpha=0.15, color=color)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Mean Confidence")
ax.set_ylabel("Accuracy")
ax.set_title("Two Moons — Calibration Reliability (5 seeds)")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/reliability_tm_all_variants.png")
print("  saved reliability_tm_all_variants.png")

print("\nDone.")
