import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Intro
cells.append(nbf.v4.new_markdown_cell("""# BNN Classification on Two Moons using Bayesian-Torch

This notebook implements a Bayesian Neural Network (BNN) for the Two Moons classification task using `bayesian-torch`.
It mirrors the **exact same dataset setup** and **architecture** as the Laplace approximation baseline to provide an equitable comparison.
Instead of a post-hoc methodology, this notebook employs **Stochastic Variational Inference (SVI)** with **Flipout layers** to actively learn the variational posterior across epochs.

Plots are automatically configured and saved for quick inclusion into LaTeX reports."""))

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
    "savefig.dpi": 300,
    "savefig.bbox": 'tight',
    "axes.grid": True,
    "grid.alpha": 0.3
})

os.makedirs('../figures', exist_ok=True)
"""))

# Cell 3: Data
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Generation 
We map exactly the `make_moons` generation from the Laplace script: 10,000 samples, 0.3 noise, split into train, validation, and test datasets."""))

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

print(f"Train size: {len(train_ds)}")
print(f"Val size:   {len(val_ds)}")
print(f"Test size:  {len(test_ds)}")

# Quick overview of data
plt.figure(figsize=(6, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.3, s=10)
plt.title("Two Moons Dataset (Train split)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('../figures/two_moons_dataset.pdf')
plt.show()
"""))

# Cell 4: Model
cells.append(nbf.v4.new_markdown_cell("""## 2. Model Definition & Conversion
We define `TinyMLP` identically. We then convert the network to a BNN using `dnn_to_bnn` from `bayesian-torch` (Flipout scheme)."""))

cells.append(nbf.v4.new_code_cell("""# ─── Model ───
class TinyMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),   # raw logits (2 classes)
        )
    def forward(self, x):
        return self.net(x)

model = TinyMLP().to(device)

# Conversion Parameters for bayesian-torch
bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Flipout",
    "moped_enable": False, 
}

# Convert the deterministic network to a Bayesian Flipout Network
dnn_to_bnn(model, bnn_prior_parameters)
print("Conversion Complete. Network Structure:")
print(model)
"""))

# Cell 5: Training Loop
cells.append(nbf.v4.new_markdown_cell("""## 3. Variational Training (ELBO)
Evidence Lower Bound (ELBO) training: Balancing classification NLL (CrossEntropy) with Kullback-Leibler (KL) divergence cost."""))

cells.append(nbf.v4.new_code_cell("""# ─── SVI ELBO Training ───
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 15
num_samples = len(train_ds)

print("Starting SVI Training...")
for epoch in range(epochs):
    model.train()
    total_loss, total_nll, total_kl = 0.0, 0.0, 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass (Stochastic)
        logits = model(inputs)
        
        # 1. Negative Log-Likelihood (CrossEntropy)
        nll_loss = criterion(logits, labels)
        
        # 2. Kullback-Leibler Divergence
        kl_loss = get_kl_loss(model)
        
        # 3. ELBO Loss computation (Mean NLL + Mean KL per batch item)
        batch_loss = nll_loss + (kl_loss / num_samples)
        
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_nll += nll_loss.item()
        total_kl += (kl_loss / num_samples).item()
        
    num_batches = len(train_loader)
    print(f"Epoch {epoch+1:02d} | Avg Loss: {total_loss/num_batches:.4f} | Avg NLL: {total_nll/num_batches:.4f} | Avg KL: {total_kl/num_batches:.4f}")
"""))

# Cell 6: Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 4. Inference & Predictive Distributions
We perform Monte Carlo multiple passes to obtain our predictive mean and standard deviation matrices across the spatial grid bounds."""))

cells.append(nbf.v4.new_code_cell("""# ─── MC Sampling on Grid ───
model.eval()

# Create a mesh grid reflecting standard boundaries
margin = 0.5
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

grid_inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

n_samples = 30 # Number of MC inference passes
preds = []

with torch.no_grad():
    for _ in range(n_samples):
        # Raw logits output
        logits = model(grid_inputs)
        # Softmax to get probability bounds for class 1
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds.append(probs.cpu().numpy())
        
preds = np.stack(preds) # Shape: (n_samples, 10000)

mean_preds_grid = preds.mean(axis=0).reshape(xx.shape)
std_preds_grid = preds.std(axis=0).reshape(xx.shape)
"""))

# Cell 7: Plotting
cells.append(nbf.v4.new_code_cell("""# ─── Plotting bounds ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Predictive Mean Bound
c1 = ax1.contourf(xx, yy, mean_preds_grid, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=0.8)
ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.5, s=15)
ax1.set_title("Mean Predictive Distribution (Bayesian-Torch)")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
cbar1 = fig.colorbar(c1, ax=ax1)
cbar1.set_label("P(class = 1)")

# Plot 2: Predictive Uncertainty Boundary
# Standard deviation peaks significantly where class boundaries overlap
c2 = ax2.contourf(xx, yy, std_preds_grid, levels=20, cmap='viridis', alpha=0.9)
ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.5, s=15)
ax2.set_title("Predictive Uncertainty (Std Deviation)")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
cbar2 = fig.colorbar(c2, ax=ax2)
cbar2.set_label("Standard Deviation")

plt.savefig('../figures/two_moons_svi_predictions.pdf')
plt.show()
"""))

nb['cells'] = cells
with open('laplace_torch/two_moons_bayesian.ipynb', 'w') as f:
    nbf.write(nb, f)

