import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Intro
cells.append(nbf.v4.new_markdown_cell("""# BNN Classification on Two Moons using Bayesian-Torch

This notebook implements a Bayesian Neural Network (BNN) for the Two Moons classification task using `bayesian-torch`.
It mirrors the **exact same dataset setup** and **visualization code** as the Laplace approximation baseline to provide an equitable comparison.
Instead of a post-hoc methodology, this notebook employs **Stochastic Variational Inference (SVI)** using both **Flipout** and **Bayes-by-Backprop (BBB - Reparameterization)** layers.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
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
"""))

# Cell 3: Data
cells.append(nbf.v4.new_code_cell("""# ─── Dataset Generation ───
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
"""))

# Cell 4: Visualization helpers
cells.append(nbf.v4.new_code_cell("""# ─── Visualization Helpers (Matching Laplace) ───

def plot_boundary(predict_fn, X_data, y_data, title, ax=None):
    \"\"\"predict_fn: takes (N,2) numpy array, returns (N,) prob of class 1\"\"\"
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
    \"\"\"Visualise epistemic uncertainty as distance from 0.5 probability.\"\"\"
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
"""))

# Cell 5: Model
cells.append(nbf.v4.new_code_cell("""# ─── Model Architecture ───
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
    
    print(f"\\n--- Training BNN with {model_type} ---")
    
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
"""))

# Cell 6: Predict FN
cells.append(nbf.v4.new_code_cell("""# ─── Prediction Wrappers ───
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
"""))

# Cell 7: Plots
cells.append(nbf.v4.new_code_cell("""# ─── Figure: Decision Boundaries ───
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
"""))

nb['cells'] = cells
with open('laplace_torch/two_moons_bayesian.ipynb', 'w') as f:
    nbf.write(nb, f)

