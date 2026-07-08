from __future__ import annotations

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
):
    """Train a MAP model. Returns the trained model, per-epoch losses, and fit_time."""
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


def eval_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs_list.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs_list).numpy()


def predict_probs_from_model_or_fn(
    model_or_fn,
    X_np: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return (N, C) numpy probabilities for either a callable predictor or a torch module."""
    try:
        out = model_or_fn(X_np)
        out = np.array(out)
        if out.ndim == 1:
            p1 = out
            return np.vstack([1 - p1, p1]).T
        return out
    except Exception:
        pass

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_or_fn
    model = model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu()
            results.append(probs)
    return torch.cat(results).numpy()


def safe_load_laplace_state(la_obj, state_dict: dict):
    """Load a Laplace object's state_dict, clearing stored `data` to avoid
    last-layer re-discovery errors in some laplace-torch versions."""
    state = state_dict.copy() if isinstance(state_dict, dict) else state_dict
    if isinstance(state, dict) and 'data' in state:
        state['data'] = None
    return la_obj.load_state_dict(state)


# Monkeypatch LLLaplace.load_state_dict to be tolerant of stored sample data.
try:
    from laplace.lllaplace import LLLaplace

    _orig_ll_load = LLLaplace.load_state_dict

    def _patched_ll_load(self, state_dict):
        sd = state_dict.copy() if isinstance(state_dict, dict) else state_dict
        # Only drop the stored data batch if the last layer is already resolved
        # (that's when re-running _find_last_layer raises "already known"). If it's
        # still unresolved, load_state_dict needs this batch to detect it — that's
        # the library's documented/tested mechanism (one-sample forward pass), not
        # something to bypass.
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
