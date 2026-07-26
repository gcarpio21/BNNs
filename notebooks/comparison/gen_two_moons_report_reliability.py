"""
Generates the combined Two Moons reliability diagram used in the report's
results section: 5seed_two_moons_combined_reliability.png, overlaying the
4 best-performing variants (2 bayesian-torch, 2 laplace-torch) on one plot.
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
from laplace import Laplace
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from shared import TinyMLP, load_checkpoint, load_two_moons

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

SEEDS     = [42, 711, 123, 456, 789]
CKPT_BNN  = str(ROOT / "results" / "checkpoints" / "two_moons_bayesian_5seed")
CKPT_LAP  = str(ROOT / "results" / "checkpoints" / "two_moons_laplace_5seed")
CKPT_MAP  = str(ROOT / "results" / "checkpoints" / "two_moons_map_5seed")
FIGURES_DIR = str(ROOT / "notebooks" / "figures")
N_BINS    = 15
N_SAMPLES = 50
bins      = np.linspace(0, 1, N_BINS + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


def load_la(fname, subset, hessian, seed):
    base = TinyMLP(hidden=64)
    map_ck = load_checkpoint(f"{CKPT_MAP}/two_moons_map_seed{seed}.pt", map_location=device)
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


bnn_rel_specs = [
    ("Flipout / scratch / ELBO", "two_moons_flipout_scratch_elbo_seed{s}.pt", "Flipout"),
    ("BBB / MOPED / ELBO",       "two_moons_bbb_moped_elbo_seed{s}.pt",       "Reparameterization"),
]
la_rel_specs = [
    ("Full / all / gridsearch",  "two_moons_gridsearch_full_all_weights_seed{s}.pt",  "all",        "full"),
    ("Kron / LL / gridsearch",   "two_moons_gridsearch_kron_last_layer_seed{s}.pt",   "last_layer", "kron"),
]

bnn_rel = {n: {"accs": [], "confs": []} for n, _, _ in bnn_rel_specs}
la_rel  = {n: {"accs": [], "confs": []} for n, _, _, _ in la_rel_specs}

print("Collecting reliability curves (5 seeds, from checkpoints)...")
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
        la_v  = load_la(tmpl.format(s=seed), subset, hessian, seed)
        probs = la_predict(la_v, X_test)
        a, c  = reliability_fracs(probs, y_test)
        la_rel[name]["accs"].append(a)
        la_rel[name]["confs"].append(c)

print("Generating 5seed_two_moons_combined_reliability.png ...")
rel_styles = [
    ("Flipout / scratch / ELBO", bnn_rel, "tab:blue",   "-"),
    ("BBB / MOPED / ELBO",       bnn_rel, "tab:orange", "--"),
    ("Full / all / gridsearch",  la_rel,  "tab:green",  "-"),
    ("Kron / LL / gridsearch",   la_rel,  "tab:red",    "--"),
]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", zorder=5)
for name, rel_dict, color, ls in rel_styles:
    all_accs  = np.array(rel_dict[name]["accs"])
    all_confs = np.array(rel_dict[name]["confs"])
    mean_obs  = np.nanmean(all_accs,  axis=0)
    std_obs   = np.nanstd(all_accs,   axis=0)
    mean_conf = np.nanmean(all_confs, axis=0)
    valid = ~np.isnan(mean_obs)
    ax.plot(mean_conf[valid], mean_obs[valid], ls=ls, color=color, lw=1.8,
            marker="o", ms=5, label=name)
    ax.fill_between(mean_conf[valid],
                    (mean_obs - std_obs)[valid],
                    (mean_obs + std_obs)[valid],
                    alpha=0.15, color=color)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Mean Confidence")
ax.set_ylabel("Accuracy")
ax.set_title(f"Two Moons — Combined Reliability Diagram ({len(SEEDS)} seeds)")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(True, alpha=0.3)
fig.savefig(f"{FIGURES_DIR}/5seed_two_moons_combined_reliability.png")
print("  saved 5seed_two_moons_combined_reliability.png")
print("\nDone.")
