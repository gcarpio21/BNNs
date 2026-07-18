from __future__ import annotations

import copy
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint, seed_everything


class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    criterion=nn.CrossEntropyLoss(),
    seed: int | None = 42,
    checkpoint_path: str | None = None,
    val_loader: DataLoader | None = None,
    patience: int | None = None,
):
    """Train a MAP model. Returns the trained model, per-epoch losses, and fit_time.

    If both val_loader and patience are given, tracks validation loss each epoch and
    stops early (keeping the best-val-loss state) once patience epochs pass without
    improvement. With either left as None, behaves exactly as before (fixed epochs,
    no early stopping) so existing callers are unaffected.
    """
    if checkpoint_exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint.get("train_losses", []), checkpoint.get("fit_time", 0.0)

    if seed is not None:
        seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = val_loader is not None and patience is not None
    best_val_loss, best_state, wait = float("inf"), None, 0

    train_losses = []
    start_time = time.time()
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        if early_stopping:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    fit_time = time.time() - start_time

    if checkpoint_path:
        save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "fit_time": fit_time,
                "seed": seed,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            checkpoint_path,
        )

    return model, train_losses, fit_time


def get_sinusoid_mlp(seed: int, device: torch.device | None = None) -> nn.Module:
    """1 -> 50 -> Tanh -> 50 -> 1 MLP, seeded, for sinusoid regression."""
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
    return model.to(device) if device is not None else model


def la_predict(la_model, X: torch.Tensor):
    """Laplace GLM predictive mean/std on a raw tensor batch."""
    f_mu, f_var = la_model(X)
    return f_mu.squeeze().detach().cpu().numpy(), f_var.squeeze().sqrt().detach().cpu().numpy()


def build_laplace_variant(
    fresh_model: nn.Module,
    base_model: nn.Module,
    spec: Dict,
    likelihood: str,
    subset_key: str = "subset_of_weights",
    hessian_key: str = "hessian_structure",
):
    """Wrap a fresh copy of base_model's weights in an unfitted Laplace object.

    fresh_model must be a newly constructed model instance, not base_model
    itself and not reused across variants: Laplace stores last-layer
    resolution state on the model object itself (model.last_layer), so
    reusing one instance across variants would make each later variant see
    the last layer as already resolved from an earlier variant's fit and
    raise "Last layer is already known". A fresh instance per variant
    avoids the collision at the source.

    Args:
        fresh_model: Newly constructed model instance to copy weights into.
        base_model: Trained model to copy weights from.
        spec: Variant config dict holding the subset/hessian keys.
        likelihood: "classification" or "regression", passed to Laplace().
        subset_key: Key in spec for subset_of_weights.
        hessian_key: Key in spec for hessian_structure.

    Returns:
        An unfitted Laplace object; call .fit(train_loader) before use.
    """
    from laplace import Laplace

    fresh_model.load_state_dict(base_model.state_dict())
    fresh_model.eval()
    return Laplace(
        fresh_model,
        likelihood,
        subset_of_weights=spec[subset_key],
        hessian_structure=spec[hessian_key],
    )


def bt_mc_forward(net: nn.Module, X: torch.Tensor, n_samples: int) -> torch.Tensor:
    """MC-sample a Bayesian-Torch model: tiles the batch for Flipout (independent
    per-row samples in one forward pass), loops for Reparameterization/BBB (one shared
    weight sample per forward pass, so tiling would just repeat it). Dispatches on
    net.model_type. Returns (n_samples, batch, ...).

    Tiling needs n_samples copies of the batch to fit in memory. For large models
    pass an explicit always-loop callable instead to functions that accept one
    (e.g. optimize_noise_std_bt's mc_forward_fn).
    """
    if getattr(net, "model_type", None) == "Flipout":
        X_tiled = X.repeat(n_samples, *([1] * (X.dim() - 1)))
        out = net(X_tiled)
        return out.reshape(n_samples, X.shape[0], -1)
    return torch.stack([net(X) for _ in range(n_samples)])


def bt_predict(net: nn.Module, X: torch.Tensor, n_samples: int):
    """MC mean/std regression prediction via bt_mc_forward."""
    net.eval()
    with torch.no_grad():
        preds = bt_mc_forward(net, X, n_samples)
    return preds.mean(0).squeeze().cpu().numpy(), preds.std(0).squeeze().cpu().numpy()


def optimize_noise_std_bt(
    net: nn.Module,
    val_loader: DataLoader,
    device: torch.device | None = None,
    lr: float = 1e-1,
    n_epochs: int = 10,
    n_mc: int = 20,
    mc_forward_fn=bt_mc_forward,
) -> float:
    """Fit a scalar aleatoric sigma by minimizing Gaussian NLL on val_loader (Adam),
    using the MC mean of mc_forward_fn as the fixed target mean each step.
    val_loader must yield plain (X, y) pairs.
    """
    net.eval()
    log_sigma_noise = nn.Parameter(
        torch.zeros(1, device=device) if device is not None else torch.zeros(1)
    )
    optimizer = torch.optim.Adam([log_sigma_noise], lr=lr)
    gaussian_nll_loss = nn.GaussianNLLLoss(full=True)
    for _ in range(n_epochs):
        for xb, yb in val_loader:
            if device is not None:
                xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = mc_forward_fn(net, xb, n_mc).mean(0)
            nll = gaussian_nll_loss(y_pred, yb, torch.ones_like(y_pred) * log_sigma_noise.exp() ** 2)
            nll.backward()
            optimizer.step()
    return log_sigma_noise.exp().item()


# Monkeypatch LLLaplace.load_state_dict to be tolerant of stored sample data.
try:
    from laplace.lllaplace import LLLaplace

    _orig_ll_load = LLLaplace.load_state_dict

    def _patched_ll_load(self, state_dict):
        sd = state_dict.copy() if isinstance(state_dict, dict) else state_dict
        # Only drop 'data' once the last layer is resolved: re-running
        # _find_last_layer() on an already-resolved model raises "already known".
        if isinstance(sd, dict) and 'data' in sd and self.model.last_layer is not None:
            sd['data'] = None
        try:
            return _orig_ll_load(self, sd)
        except ValueError as e:
            if 'Last layer is already known' in str(e):
                try:
                    self.data = None
                    return _orig_ll_load(self, sd)
                except Exception:
                    pass
            raise

    LLLaplace.load_state_dict = _patched_ll_load
except Exception:
    pass
