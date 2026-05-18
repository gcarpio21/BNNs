# BNN Classification on Two Moons using Bayesian-Torch

This notebook implements a Bayesian Neural Network (BNN) for the Two Moons classification task using `bayesian-torch`.
It mirrors the **exact same dataset setup** and **visualization code** as the Laplace approximation baseline to provide an equitable comparison.
Instead of a post-hoc methodology, this notebook employs **Stochastic Variational Inference (SVI)** using both **Flipout** and **Bayes-by-Backprop (BBB - Reparameterization)** layers.



```
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# bayesian-torch imports
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Format plots for LaTeX reports
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "savefig.dpi": 150,
    "savefig.bbox": 'tight',
})

os.makedirs('../figures', exist_ok=True)

```


```
# ─── Dataset Generation ───
X, y = make_moons(n_samples=10000, noise=0.3, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

def to_tensors(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

train_ds = to_tensors(X_train, y_train)
val_ds   = to_tensors(X_val,   y_val)
test_ds  = to_tensors(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

```


```
# ─── Visualization Helpers (Matching Laplace) ───

def plot_boundary(predict_fn, X_data, y_data, title, ax=None):
    """predict_fn: takes (N,2) numpy array, returns (N,) prob of class 1"""
    h = 0.05
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = predict_fn(grid).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    cf = ax.contourf(xx, yy, probs, levels=50, cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1.5)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='bwr', edgecolors='k', linewidths=0.4, s=30, zorder=3)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return cf, probs

def uncertainty_map(predict_fn, X_data, y_data, title, ax):
    """Visualise epistemic uncertainty as distance from 0.5 probability."""
    h = 0.05
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = predict_fn(grid).reshape(xx.shape)
    uncertainty = 1 - 2 * np.abs(probs - 0.5)  # 1 = max uncertainty, 0 = certain

    cf = ax.contourf(xx, yy, uncertainty, levels=50, cmap='YlOrRd', vmin=0, vmax=1)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='bwr', edgecolors='k', linewidths=0.4, s=30, zorder=3)
    ax.set_title(title)
    return cf, uncertainty

```


```
# ─── Model Architecture ───
class TinyMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)

def train_bnn(model_type="Flipout"):
    model = TinyMLP().to(device)
    
    bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": model_type,
        "moped_enable": False, 
    }
    
    dnn_to_bnn(model, bnn_prior_parameters)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    epochs = 20
    num_samples = len(train_ds)
    
    print(f"\n--- Training BNN with {model_type} ---")
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            nll_loss = criterion(logits, labels)
            kl_loss = get_kl_loss(model)
            batch_loss = nll_loss + (kl_loss / num_samples)
            batch_loss.backward()
            optimizer.step()
            
    print("Training Complete.")
    return model

model_flipout = train_bnn("Flipout")
model_bbb = train_bnn("Reparameterization")

```


```
# ─── Prediction Wrappers ───
def bnn_predict(X_np, model, n_samples=30):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(probs)
    return np.mean(probs_list, axis=0)

```


```
# ─── Figure: Decision Boundaries ───
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

cf0, _ = plot_boundary(lambda X: bnn_predict(X, model_flipout), X_test, y_test, 'SVI (Flipout)', axes[0])
cf1, _ = plot_boundary(lambda X: bnn_predict(X, model_bbb), X_test, y_test, 'SVI (BBB / Reparam)', axes[1])

for cf, ax in zip([cf0, cf1], axes):
    plt.colorbar(cf, ax=ax, label='P(class 1)')

plt.suptitle('Bayesian-Torch — Flipout vs BBB — decision boundaries', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_boundary.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── Figure: Uncertainty Heatmaps ───
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

cf0, _ = uncertainty_map(lambda X: bnn_predict(X, model_flipout), X_test, y_test, 'Uncertainty (Flipout)', axes[0])
cf1, _ = uncertainty_map(lambda X: bnn_predict(X, model_bbb), X_test, y_test, 'Uncertainty (BBB / Reparam)', axes[1])

for cf, ax in zip([cf0, cf1], axes):
    plt.colorbar(cf, ax=ax, label='Uncertainty')

plt.suptitle('Bayesian-Torch — Flipout vs BBB — uncertainty heatmaps', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()

```


```
# ─── Compute per-sample predictions ───
def get_confidence_and_uncertainty(model, n_samples=30):
    model.eval()
    probs_list = []
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        for _ in range(n_samples):
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    
    probs_stack = np.stack(probs_list)
    mean_probs = np.mean(probs_stack, axis=0) # shape: (len(X_test), 2)
    conf = mean_probs.max(axis=1) # Max probability
    unc = 1 - conf # Epistemic representation mapped from prob margin
    return conf, unc

flip_conf, flip_unc = get_confidence_and_uncertainty(model_flipout)
bbb_conf, bbb_unc = get_confidence_and_uncertainty(model_bbb)

# ─── Figure: Analysis Distributions ───
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
COLOR_FLIP = '#FF6347' # tomato
COLOR_BBB  = '#228B22' # green

# 1. Confidence distributions
axes[0].hist(flip_conf, bins=40, alpha=0.5, label='Flipout', color=COLOR_FLIP, density=True)
axes[0].hist(bbb_conf, bins=40, alpha=0.5, label='BBB/Reparam', color=COLOR_BBB, density=True)
axes[0].set_xlabel('Max probability')
axes[0].set_title('Confidence distributions')
axes[0].legend()

# 2. Scatter comparison (Laplace notebook style)
def make_scatter_comparison(ax, x_label, x_data, y_label, y_data, title, c_data=y_test):
    ax.scatter(x_data, y_data, alpha=0.4, s=15, c=c_data, cmap='bwr', edgecolors='k', linewidths=0.2)
    lims = [0, 1.01]
    ax.plot(lims, lims, 'k--', linewidth=1, label='no change')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

make_scatter_comparison(axes[1], 'Flipout confidence', flip_conf,
                        'BBB confidence', bbb_conf,
                        'Flipout vs BBB — per sample')

# 3. Uncertainty distributions
axes[2].hist(flip_unc, bins=40, alpha=0.5, label='Flipout', color=COLOR_FLIP, density=True)
axes[2].hist(bbb_unc, bins=40, alpha=0.5, label='BBB/Reparam', color=COLOR_BBB, density=True)
axes[2].set_xlabel('Uncertainty')
axes[2].set_title('Uncertainty distributions')
axes[2].legend()

plt.suptitle('Flipout vs BBB — Confidence and Uncertainty analysis', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

```


```
# ─── Comprehensive UQ Metrics Computation ───
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def expected_calibration_error(probs, labels, n_bins=15):
    """Compute ECE: |sum_k (n_k/N) * |acc_k - conf_k||"""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            acc_bin = accuracies[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += np.abs(acc_bin - conf_bin) * in_bin.sum() / len(labels)
    return ece

def evaluate_uq_metrics(model, model_name, n_samples=50):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    inf_start = time.time()
    logits_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_t)
            logits_list.append(logits.cpu().numpy())
    inf_time = time.time() - inf_start

    logits_stack = np.stack(logits_list)  # (n_samples, N, 2)
    probs_stack = np.stack([
        torch.softmax(torch.tensor(l), dim=1).numpy() for l in logits_list
    ])  # (n_samples, N, 2)

    # Per-sample mean & std of predictive distribution
    mean_probs = probs_stack.mean(axis=0)  # (N, 2)
    std_probs  = probs_stack.std(axis=0)   # (N, 2)
    mean_conf = mean_probs.max(axis=1)
    preds = mean_probs.argmax(axis=1)

    # ── Standard classification metrics ──
    accuracy = (preds == y_test).mean()
    nll = -np.log(mean_probs[np.arange(len(y_test)), y_test] + 1e-12).mean()
    y_true_onehot = np.zeros_like(mean_probs)
    y_true_onehot[np.arange(len(y_test)), y_test] = 1
    brier = np.mean(np.sum((mean_probs - y_true_onehot)**2, axis=1))
    ece = expected_calibration_error(mean_probs, y_test, n_bins=15)

    # ── ROC-AUC ──
    try:
        roc_auc = roc_auc_score(y_test, mean_probs[:, 1])
    except ValueError:
        roc_auc = float('nan')

    # ── Precision / Recall / F1 ──
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    # ── Predictive Std (uncertainty) ──
    pred_std = std_probs[np.arange(len(y_test)), preds].mean()
    pred_std_median = np.median(std_probs[np.arange(len(y_test)), preds])
    pred_std_std = std_probs[np.arange(len(y_test)), preds].std()

    # ── Uncertainty Decomposition (epistemic / aleatoric / total) ──
    # Total: H[E[p(y|x)]]  — entropy of the mean probability
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(p * np.log(p), axis=-1)
    total_unc = entropy(mean_probs).mean()

    # Aleatoric: E[H[p(y|x)]] — expected entropy across MC samples
    aleatoric_unc = entropy(probs_stack).mean()

    # Epistemic: total - aleatoric = I[y; theta] = mutual information
    epistemic_unc = total_unc - aleatoric_unc

    # ── Bayesian confidence interval width (95% CI) ──
    lower = np.percentile(probs_stack[:, np.arange(len(y_test)), preds], 5, axis=0)
    upper = np.percentile(probs_stack[:, np.arange(len(y_test)), preds], 95, axis=0)
    ci95_width = (upper - lower).mean()

    # ── Mean / Median confidence ──
    mean_confidence = mean_conf.mean()
    median_confidence = np.median(mean_conf)

    return {
        "Accuracy":              accuracy,
        "NLL":                   nll,
        "Brier Score":           brier,
        "ECE":                   ece,
        "ROC-AUC":               roc_auc,
        "Precision":             precision,
        "Recall":                recall,
        "F1 Score":              f1,
        "Mean Confidence":       mean_confidence,
        "Median Confidence":     median_confidence,
        "Predictive Std (mean)": pred_std,
        "Predictive Std (median)": pred_std_median,
        "Predictive Std (std)":   pred_std_std,
        "CI95 Width (mean)":     ci95_width,
        "Aleatoric Uncertainty": aleatoric_unc,
        "Epistemic Uncertainty": epistemic_unc,
        "Total Uncertainty":     total_unc,
        "Inference Time (s)":    inf_time,
    }

metrics_dict = {}
metrics_dict["Bayesian-Torch (Flipout)"] = evaluate_uq_metrics(model_flipout, "Flipout")
metrics_dict["Bayesian-Torch (BBB)"] = evaluate_uq_metrics(model_bbb, "BBB")

df_metrics = pd.DataFrame(metrics_dict).T
df_metrics.index.name = "Model Type"

# Format display
df_display = df_metrics.copy()
for col in df_display.columns:
    if pd.api.types.is_float_dtype(df_display[col]):
        df_display[col] = df_display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

print("=== Two Moons Bayesian-Torch — Comprehensive UQ Metrics ===")
display(df_display)

# ── CSV Export ──
csv_path = "twomoons_bayesian_torch_metrics.csv"
df_metrics.to_csv(csv_path)
print(f"\nSaved metrics to {csv_path}")

# ── LaTeX Table ──
print("\n=== LaTeX Table Code ===")
latex_str = df_display.style.format(precision=4).to_latex()
print(latex_str)

# ── Figure: Metrics table ──
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                     rowLabels=df_display.index, cellLoc='center', loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(8)
the_table.scale(1.2, 1.8)
plt.savefig('../figures/bayesian_torch_metrics_table.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/bayesian_torch_metrics_table.png')

```


```
# ─── Figure: Calibration Reliability Diagram ───
def reliability_diagram(probs, labels, ax, n_bins=15, label_prefix=""):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confs = []
    bin_accs = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_confs.append(confidences[in_bin].mean())
            bin_accs.append(accuracies[in_bin].mean())
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(bin_confs, bin_accs, 'o-', linewidth=2, markersize=6,
            label=f'{label_prefix}ECE={expected_calibration_error(probs, labels, n_bins):.4f}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Calibration Reliability Diagram')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# Compute probs for reliability diagram
X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
def get_mean_probs(model, n_samples=50):
    model.eval()
    nll_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_t)
            nll_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.mean(np.stack(nll_probs), axis=0)

mean_probs_flip = get_mean_probs(model_flipout)
mean_probs_bbb = get_mean_probs(model_bbb)

fig, ax = plt.subplots(figsize=(5.5, 5))
reliability_diagram(mean_probs_flip, y_test, ax, label_prefix='Flipout ')
reliability_diagram(mean_probs_bbb, y_test, ax, label_prefix='BBB ')
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_reliability.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/bayesian_torch_reliability.png')

```


```
# ─── Figure: Predictive Std Spatial Maps ───
def get_std_map(predict_fn_std, X_data, y_data, title, ax):
    h = 0.05
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    std_vals = predict_fn_std(grid).reshape(xx.shape)
    cf = ax.contourf(xx, yy, std_vals, levels=50, cmap='YlOrRd', alpha=0.8)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='bwr', edgecolors='k', linewidths=0.4, s=30, zorder=3)
    ax.set_title(title)
    return cf

def bnn_predict_std(X_np, model, n_samples=50):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(probs)
    return np.std(probs_list, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
cf0 = get_std_map(lambda X: bnn_predict_std(X, model_flipout), X_test, y_test, 'Predictive Std — Flipout', axes[0])
cf1 = get_std_map(lambda X: bnn_predict_std(X, model_bbb), X_test, y_test, 'Predictive Std — BBB / Reparam', axes[1])
for cf, ax in zip([cf0, cf1], axes):
    plt.colorbar(cf, ax=ax, label='Std(P(class=1))')
plt.suptitle('Bayesian-Torch — Predictive Std Spatial Maps', fontsize=13)
plt.tight_layout()
plt.savefig('../figures/bayesian_torch_std_map.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/bayesian_torch_std_map.png')

```


```
# ─── Summary Print ───
for name, metrics in metrics_dict.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:              {metrics['Accuracy']*100:.2f}%")
    print(f"  NLL:                   {metrics['NLL']:.4f}")
    print(f"  Brier Score:           {metrics['Brier Score']:.4f}")
    print(f"  ECE:                   {metrics['ECE']:.4f}")
    print(f"  ROC-AUC:               {metrics['ROC-AUC']:.4f}")
    print(f"  Precision / Recall / F1: {metrics['Precision']:.4f} / {metrics['Recall']:.4f} / {metrics['F1 Score']:.4f}")
    print(f"  Mean Confidence:       {metrics['Mean Confidence']:.4f}")
    print(f"  Predictive Std:        {metrics['Predictive Std (mean)']:.4f} (median: {metrics['Predictive Std (median)']:.4f}, std: {metrics['Predictive Std (std)']:.4f})")
    print(f"  95% CI Width:          {metrics['CI95 Width (mean)']:.4f}")
    print(f"  Aleatoric Uncertainty: {metrics['Aleatoric Uncertainty']:.4f}")
    print(f"  Epistemic Uncertainty: {metrics['Epistemic Uncertainty']:.4f}")
    print(f"  Total Uncertainty:     {metrics['Total Uncertainty']:.4f}")
    print(f"  Inference Time:        {metrics['Inference Time (s)']:.2f}s")
    print(f"{'='*60}")

```
