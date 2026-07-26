"""
Generates the combined Sinusoid calibration diagram used in the report's
results section: 5seed_sinusoid_combined_calibration.png, overlaying the
best-calibrated bayesian-torch variants (Flipout/scratch, BBB/scratch) and
the best-calibrated laplace-torch variant of each of the Full/all and
Kron/last-layer families (both gridsearch-tuned).
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
from shared import (get_sinusoid_mlp, load_checkpoint, load_sinusoid,
                    build_sinusoid_bt, build_laplace_variant, bt_predict,
                    la_predict, optimize_noise_std_bt, combine_predictive_std,
                    regression_calibration_curve)

plt.rcParams.update({"font.family": "serif", "font.size": 12,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})

SEEDS       = [711, 42, 123, 456, 789]
SIGMA_NOISE = 0.3
CKPT_BT     = str(ROOT / "results" / "checkpoints" / "sinusoid_bayesian_5seed")
CKPT_LA     = str(ROOT / "results" / "checkpoints" / "sinusoid_laplace_5seed")
CKPT_MAP    = str(ROOT / "results" / "checkpoints" / "sinusoid_map_5seed")
FIGURES_DIR = str(ROOT / "notebooks" / "figures")
N_LEVELS    = 20
levels      = np.linspace(0.0, 1.0, N_LEVELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

bt_specs = [
    ("Flipout / scratch", "Flipout", "sinusoid_flipout_scratch_seed{s}.pt"),
    ("BBB / scratch",     "Reparameterization", "sinusoid_bbb_scratch_seed{s}.pt"),
]
la_specs = [
    ("Full / all / gridsearch",
     {"hessian": "full", "subset": "all", "tune": "gridsearch"},
     "sinusoid_laplace_full_all_gridsearch_seed{s}.pt"),
    ("Kron / last layer / gridsearch",
     {"hessian": "kron", "subset": "last_layer", "tune": "gridsearch"},
     "sinusoid_laplace_kron_last_layer_gridsearch_seed{s}.pt"),
]

curves = {name: [] for name, _, _ in bt_specs}
for name, _, _ in la_specs:
    curves[name] = []

print("Collecting calibration curves (5 seeds, from checkpoints)...")
for seed in SEEDS:
    print(f"  Seed {seed}...")
    # bayesian-torch side runs entirely on CPU, matching the notebook's own pipeline.
    d_cpu = load_sinusoid(n_data=150, sigma_noise=SIGMA_NOISE, batch_size=150, seed=seed,
                          device=torch.device("cpu"))
    X_test_id_cpu = d_cpu["X_test_id"]
    y_test_id_np = d_cpu["y_test_id"].cpu().flatten().numpy()
    val_loader = d_cpu["val_loader"]

    for name, model_type, tmpl in bt_specs:
        net = build_sinusoid_bt(model_type, False, seed=seed)
        ck = load_checkpoint(f"{CKPT_BT}/{tmpl.format(s=seed)}", map_location="cpu")
        net.load_state_dict(ck["model_state_dict"])
        sigma_bt = optimize_noise_std_bt(net, val_loader)
        mu_id, epi_id = bt_predict(net, X_test_id_cpu, n_samples=100, seed=seed)
        total_id = combine_predictive_std(epi_id, sigma_bt)
        _, observed = regression_calibration_curve(y_test_id_np, mu_id, total_id, n_levels=N_LEVELS)
        curves[name].append(observed)

    # laplace-torch side runs on device (cuda if available), matching that notebook.
    d = load_sinusoid(n_data=150, sigma_noise=SIGMA_NOISE, batch_size=150, seed=seed, device=device)
    X_test_id = d["X_test_id"]

    map_model = get_sinusoid_mlp(seed, device=device)
    map_ck = load_checkpoint(f"{CKPT_MAP}/sinusoid_map_seed{seed}.pt", map_location=device)
    map_model.load_state_dict(map_ck["model_state_dict"])
    map_model.eval()

    for name, spec, tmpl in la_specs:
        la = build_laplace_variant(get_sinusoid_mlp(seed, device=device), map_model, spec,
                                    "regression", subset_key="subset", hessian_key="hessian")
        ck = load_checkpoint(f"{CKPT_LA}/{tmpl.format(s=seed)}", map_location=device)
        la.load_state_dict(ck["la_state_dict"])
        sigma = float(la.sigma_noise.item())
        mu_id, epi_id = la_predict(la, X_test_id)
        total_id = combine_predictive_std(epi_id, sigma)
        _, observed = regression_calibration_curve(y_test_id_np, mu_id, total_id, n_levels=N_LEVELS)
        curves[name].append(observed)

print("Generating 5seed_sinusoid_combined_calibration.png ...")
styles = [
    ("Flipout / scratch",             "tab:blue",   "-"),
    ("BBB / scratch",                 "tab:orange", "--"),
    ("Full / all / gridsearch",       "tab:green",  "-"),
    ("Kron / last layer / gridsearch", "tab:red",   "--"),
]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", zorder=5)
for name, color, ls in styles:
    obs = np.stack(curves[name])
    mean_obs = obs.mean(axis=0)
    std_obs  = obs.std(axis=0)
    ax.plot(levels, mean_obs, ls=ls, color=color, lw=1.8, marker="o", ms=5, label=name)
    ax.fill_between(levels, mean_obs - std_obs, mean_obs + std_obs, alpha=0.15, color=color)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Nominal Coverage")
ax.set_ylabel("Observed Coverage")
ax.set_title(f"Sinusoid — Combined Calibration Diagram ({len(SEEDS)} seeds)")
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(True, alpha=0.3)
fig.savefig(f"{FIGURES_DIR}/5seed_sinusoid_combined_calibration.png")
print("  saved 5seed_sinusoid_combined_calibration.png")
print("\nDone.")
