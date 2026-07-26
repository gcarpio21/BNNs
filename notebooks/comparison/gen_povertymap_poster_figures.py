"""
Generates 3 PovertyMap poster figures:
  cmp_lbnn_pov_rmse.png       MAP vs Laplace vs BNN — RMSE bar
  cmp_lbnn_pov_cal_mae.png    MAP vs Laplace vs BNN — Calibration MAE bar
  reliability_povertymap.png  Laplace vs BNN calibration curves (mean ± 1σ, 5 folds)

Bar charts read from povertymap_full_metrics.csv (pre-computed 5-fold).
Reliability diagram loads checkpoints for all 5 folds.

Run from repo root or notebooks/comparison/.
"""
import sys, warnings, os, time
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from laplace import Laplace
from scipy.stats import norm as _norm
from wilds import get_dataset

from shared import load_checkpoint, seed_everything, regression_metrics

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "savefig.dpi": 180, "savefig.bbox": "tight"})

SEED = 42
WILDS_ROOT = str(ROOT / "data" / "wilds")
CKPT = str(ROOT / "notebooks" / "comparison" / "results" / "checkpoints" / "bnn_comparison_povertymap")
METRICS_CSV = str(ROOT / "notebooks" / "comparison" / "results" / "metrics" / "povertymap_full_metrics.csv")
OUT = str(ROOT / "submission" / "CPS_Poster_Template" / "figures")
FOLDS = ['A', 'B', 'C', 'D', 'E']
BATCH_SIZE = 256
N_MC_BAYES = 50
CONF_LEVELS = np.linspace(0.05, 0.95, 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── model ─────────────────────────────────────────────────────────────────────
class PovertyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


def create_model(seed=SEED):
    torch.manual_seed(seed)
    return PovertyResNet()


# ── data ──────────────────────────────────────────────────────────────────────
def poverty_transform(img):
    return img.float()


class DropMeta:
    """Wraps a WILDS DataLoader (X, y, meta) -> (X, y) for Laplace."""
    def __init__(self, loader):
        self.loader = loader
    @property
    def dataset(self):
        return self.loader.dataset
    def __iter__(self):
        for X, y, _ in self.loader:
            yield X, y
    def __len__(self):
        return len(self.loader)


def get_loaders(dataset):
    loaders = {}
    for split in ['train', 'id_val', 'id_test', 'test']:
        sd = dataset.get_subset(split, transform=poverty_transform, frac=1.0)
        loaders[split] = DataLoader(sd, batch_size=BATCH_SIZE, shuffle=(split == 'train'),
                                    num_workers=4, pin_memory=True)
    return loaders


def predict_bnn(model, loader, n_mc=N_MC_BAYES):
    model.eval()
    all_mu, all_std, all_y = [], [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            samples = torch.stack([model(X).squeeze(-1) for _ in range(n_mc)])
            mu = samples.mean(0)
            std = samples.std(0).clamp(min=1e-6)
            all_mu.append(mu.cpu())
            all_std.append(std.cpu())
            all_y.append(y.squeeze())
    return torch.cat(all_mu).numpy(), torch.cat(all_std).numpy(), torch.cat(all_y).numpy()


def predict_laplace(la, loader):
    la.model.eval()
    all_mu, all_std, all_y = [], [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            f_mu, f_var = la(X, pred_type="glm")
            sigma = la.sigma_noise.item() if la.sigma_noise is not None else 0.0
            pred_std = (f_var.squeeze() + sigma ** 2).sqrt().clamp(min=1e-6)
            all_mu.append(f_mu.squeeze().cpu())
            all_std.append(pred_std.cpu())
            all_y.append(y.squeeze())
    return torch.cat(all_mu).numpy(), torch.cat(all_std).numpy(), torch.cat(all_y).numpy()


# ── bar charts from CSV ──────────────────────────────────────────────────────
print("Reading metrics CSV for bar charts...")
df = pd.read_csv(METRICS_CSV)

def _val(df, split, method, col):
    row = df[(df['Split'] == split) & (df['Method'] == method)]
    return row[f'{col}_mean'].values[0], row[f'{col}_std'].values[0]

methods_csv     = ['MAP', 'Laplace', 'Bayesian-Torch']
methods_display = ['MAP', 'Full / last layer', 'Flipout / scratch']
bar_colors = ['#d62728', '#1f77b4', '#2ca02c']

# Figure 1: RMSE
print("Generating cmp_lbnn_pov_rmse.png ...")
rmse_vals = [_val(df, 'ID', m, 'RMSE') for m in methods_csv]
means_rmse = [v[0] for v in rmse_vals]
stds_rmse  = [v[1] for v in rmse_vals]

fig, ax = plt.subplots(figsize=(4.0, 3))
bars = ax.bar(range(len(methods_display)), means_rmse, yerr=stds_rmse,
              capsize=5, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(methods_display)))
ax.set_xticklabels(methods_display, fontsize=9)
ax.set_ylabel("RMSE")
ax.set_title("PovertyMap — RMSE")
ax.set_ylim(0, max(means_rmse + stds_rmse) * 1.35)
for bar, m, s in zip(bars, means_rmse, stds_rmse):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cmp_lbnn_pov_rmse.png")
print("  saved cmp_lbnn_pov_rmse.png")

# Figure 2: Calibration MAE — grouped ID + OOD (Laplace vs VI)
print("Generating cmp_lbnn_pov_cal_mae.png ...")
# csv_name -> display name; colors match reliability diagram
method_map  = [('Laplace', 'Full / last layer'), ('Bayesian-Torch', 'Flipout / scratch')]
tick_labels = [disp for _, disp in method_map]
colors_id   = ['#d62728', '#1f77b4']
colors_ood  = ['#d62728', '#1f77b4']   # same color, hatching distinguishes OOD

cal_id  = [_val(df, 'ID',  csv, 'Calibration_MAE') for csv, _ in method_map]
cal_ood = [_val(df, 'OOD', csv, 'Calibration_MAE') for csv, _ in method_map]
means_id,  stds_id  = [v[0] for v in cal_id],  [v[1] for v in cal_id]
means_ood, stds_ood = [v[0] for v in cal_ood], [v[1] for v in cal_ood]

x = np.arange(len(method_map))
w = 0.35
fig, ax = plt.subplots(figsize=(4.5, 3))
# ID bars: solid fill; OOD bars: same color + hatching (mirrors dashed=OOD in reliability)
entries = [
    (means_id[0],  stds_id[0],  x[0]-w/2, colors_id[0],  None,    'Full / last layer (ID)'),
    (means_ood[0], stds_ood[0], x[0]+w/2, colors_ood[0], '///',   'Full / last layer (OOD)'),
    (means_id[1],  stds_id[1],  x[1]-w/2, colors_id[1],  None,    'Flipout / scratch (ID)'),
    (means_ood[1], stds_ood[1], x[1]+w/2, colors_ood[1], '///',   'Flipout / scratch (OOD)'),
]
for m, s, xi, c, hatch, lbl in entries:
    ax.bar(xi, m, w, yerr=s, capsize=4, color=c, edgecolor='white',
           linewidth=0.5, hatch=hatch, alpha=0.6 if hatch else 1.0, label=lbl)
    ax.text(xi, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylabel("Calibration MAE")
ax.set_title("PovertyMap — Cal MAE (ID vs OOD)")
ylim = max(m + s for m, s, *_ in entries) * 1.45
ax.set_ylim(0, ylim)
ax.legend(fontsize=7, ncol=2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cmp_lbnn_pov_cal_mae.png")
print("  saved cmp_lbnn_pov_cal_mae.png")


# ── reliability diagram from checkpoints ─────────────────────────────────────
print("\nComputing reliability curves from checkpoints (5 folds)...")
rel_data = {
    'laplace_id':        {'mu': [], 'std': [], 'y': []},
    'laplace_ood':       {'mu': [], 'std': [], 'y': []},
    'bayesian_torch_id': {'mu': [], 'std': [], 'y': []},
    'bayesian_torch_ood':{'mu': [], 'std': [], 'y': []},
}

for fold in FOLDS:
    t0 = time.time()
    print(f"\n  Fold {fold}...")

    fold_dataset = get_dataset('poverty', download=False, root_dir=WILDS_ROOT, fold=fold)
    loaders = get_loaders(fold_dataset)

    map_ckpt = f"{CKPT}/map_fold_{fold}_seed{SEED}.pt"
    lap_ckpt = f"{CKPT}/laplace_fold_{fold}_seed{SEED}_hyper.pt"
    bnn_ckpt = f"{CKPT}/bnn_fold_{fold}_seed{SEED}.pt"

    # MAP
    map_model = create_model()
    map_ck = load_checkpoint(map_ckpt)
    map_model.load_state_dict(map_ck['model_state_dict'])
    map_model = map_model.to(device)
    map_model.eval()

    # Estimate sigma_noise from val residuals
    val_resid = []
    with torch.no_grad():
        for X, y, _ in loaders['id_val']:
            pred = map_model(X.to(device)).squeeze().cpu()
            val_resid.append((pred - y.squeeze()) ** 2)
    map_sigma_noise = float(torch.cat(val_resid).mean().sqrt())

    # ── Laplace ──
    seed_everything(SEED)
    la_model = create_model()
    la_model.load_state_dict(map_model.state_dict())
    la_model = la_model.to(device)
    la_model.eval()  # must precede Laplace()/load_state_dict(): the internal
    # _find_last_layer() forward pass runs in whatever mode the model is currently in,
    # and in train() mode it silently corrupts BatchNorm running stats
    la = Laplace(la_model, likelihood="regression",
                 subset_of_weights="last_layer", hessian_structure="full")
    hyper = load_checkpoint(lap_ckpt, map_location=device)
    if 'la_state_dict' in hyper:
        la.load_state_dict(hyper['la_state_dict'])
    else:
        la.fit(DropMeta(loaders['train']))
        la.prior_precision = hyper['prior_precision'].to(device)
        la.sigma_noise = hyper['sigma_noise'].to(device)

    for split_key, loader in [('laplace_id', loaders['id_test']), ('laplace_ood', loaders['test'])]:
        mu, std, y = predict_laplace(la, loader)
        rel_data[split_key]['mu'].append(mu)
        rel_data[split_key]['std'].append(std)
        rel_data[split_key]['y'].append(y)

    # ── BNN ──
    seed_everything(SEED)
    bnn_model = create_model()
    bnn_params = {
        "prior_mu": 0.0, "prior_sigma": (1.0 / 5e-4) ** 0.5,
        "posterior_mu_init": 0.0, "posterior_rho_init": -3.0,
        "type": "Flipout", "moped_enable": False,
    }
    dnn_to_bnn(bnn_model, bnn_params)
    bnn_ck = load_checkpoint(bnn_ckpt)
    bnn_model.load_state_dict(bnn_ck['model_state_dict'])
    bnn_model = bnn_model.to(device)

    for split_key, loader in [('bayesian_torch_id', loaders['id_test']), ('bayesian_torch_ood', loaders['test'])]:
        mu, std, y = predict_bnn(bnn_model, loader)
        rel_data[split_key]['mu'].append(mu)
        rel_data[split_key]['std'].append(std)
        rel_data[split_key]['y'].append(y)

    print(f"  Done in {time.time() - t0:.1f}s")

# Average per-fold curves (mean ± 1σ) and plot ID + OOD
print("\nGenerating reliability_povertymap.png ...")
rel_styles = [
    ('laplace_id',         'Full / last layer (ID)',       '#d62728', 'o', '-'),
    ('laplace_ood',        'Full / last layer (OOD)',      '#d62728', 'o', '--'),
    ('bayesian_torch_id',  'Flipout / scratch (ID)',       '#1f77b4', 's', '-'),
    ('bayesian_torch_ood', 'Flipout / scratch (OOD)',      '#1f77b4', 's', '--'),
]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect', zorder=5)

for key, label, color, marker, ls in rel_styles:
    curves = []
    for mu, std, y in zip(rel_data[key]['mu'], rel_data[key]['std'], rel_data[key]['y']):
        z_abs = np.abs((y - mu) / np.clip(std, 1e-12, None))
        curves.append([float(np.mean(z_abs <= _norm.ppf((1.0 + c) / 2.0))) for c in CONF_LEVELS])
    curves = np.array(curves)
    mean_obs = curves.mean(0)
    std_obs  = curves.std(0)
    cal_mae = float(np.mean(np.abs(mean_obs - CONF_LEVELS)))
    ax.plot(CONF_LEVELS, mean_obs, color=color, lw=2, marker=marker, ms=5, ls=ls,
            label=label)
    ax.fill_between(CONF_LEVELS, mean_obs - std_obs, mean_obs + std_obs, color=color, alpha=0.15)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Expected confidence')
ax.set_ylabel('Observed coverage')
ax.set_title('PovertyMap — Calibration Reliability (mean ± 1σ, 5 folds)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/reliability_povertymap.png")
print("  saved reliability_povertymap.png")

print("\nDone.")
