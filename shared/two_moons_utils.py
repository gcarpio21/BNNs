"""Shared utilities for the project-wide Two Moons experiments.

This lives at the repository root so notebooks for Laplace and Bayesian models
can import the same dataset, model, and evaluation code.
"""

from __future__ import annotations

import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reduce nondeterminism from cuDNN where possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def checkpoint_exists(checkpoint_path: str | None) -> bool:
    return bool(checkpoint_path) and os.path.exists(checkpoint_path)


def load_checkpoint(checkpoint_path: str, map_location: torch.device | str | None = None):
    return torch.load(checkpoint_path, map_location=map_location)


def save_checkpoint(payload: dict, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_two_moons(
    n_samples: int = 10000,
    noise: float = 0.3,
    test_size: float = 0.2,
    val_size: float = 0.25,
    batch_train: int = 32,
    batch_eval: int = 64,
    seed: int = 42,
    device: torch.device | None = None,
) -> Dict[str, object]:
    """Generate a deterministic split and return arrays, datasets, and loaders."""
    seed_everything(seed)

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed
    )

    def to_dataset(X_part: np.ndarray, y_part: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(X_part, dtype=torch.float32),
            torch.tensor(y_part, dtype=torch.long),
        )

    train_ds = to_dataset(X_train, y_train)
    val_ds = to_dataset(X_val, y_val)
    test_ds = to_dataset(X_test, y_test)

    # Use a generator seeded from `seed` so DataLoader shuffle is reproducible
    g = torch.Generator()
    try:
        g.manual_seed(seed)
    except Exception:
        # Fallback if Generator not available on older torch
        g = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_train,
        shuffle=True,
        generator=(g if g is not None else None),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test_tensor": torch.tensor(X_test, dtype=torch.float32, device=device),
        "y_test_tensor": torch.tensor(y_test, dtype=torch.long, device=device),
        "scaler": scaler,
    }


def save_splits(path: str, data_dict: Dict[str, object]) -> None:
    """Save numeric splits and scaler to a .npz and scaler pickle in `path`."""
    os.makedirs(path, exist_ok=True)
    np.savez(
        os.path.join(path, "splits.npz"),
        X=data_dict["X"],
        y=data_dict["y"],
        X_train=data_dict["X_train"],
        X_val=data_dict["X_val"],
        X_test=data_dict["X_test"],
        y_train=data_dict["y_train"],
        y_val=data_dict["y_val"],
        y_test=data_dict["y_test"],
    )
    with open(os.path.join(path, "scaler.pkl"), "wb") as f:
        pickle.dump(data_dict["scaler"], f)


def load_splits(path: str, device: torch.device | None = None) -> Dict[str, object]:
    """Load previously saved splits from `path` (returns the same keys as load_two_moons)."""
    arrs = np.load(os.path.join(path, "splits.npz"))
    with open(os.path.join(path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dataset(X_part: np.ndarray, y_part: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(X_part, dtype=torch.float32),
            torch.tensor(y_part, dtype=torch.long),
        )

    X = arrs["X"]
    y = arrs["y"]
    X_train = arrs["X_train"]
    X_val = arrs["X_val"]
    X_test = arrs["X_test"]
    y_train = arrs["y_train"]
    y_val = arrs["y_val"]
    y_test = arrs["y_test"]

    train_ds = to_dataset(X_train, y_train)
    val_ds = to_dataset(X_val, y_val)
    test_ds = to_dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_test_tensor": torch.tensor(X_test, dtype=torch.float32, device=device),
        "y_test_tensor": torch.tensor(y_test, dtype=torch.long, device=device),
        "scaler": scaler,
    }


def load_sinusoid(n_data: int = 150, sigma_noise: float = 0.3, batch_size: int = 150, seed: int = 42, device: torch.device | None = None) -> Dict[str, object]:
    """Generate a simple sinusoid regression dataset and return loaders + arrays.

    Returns a dict with keys similar to load_two_moons where appropriate.
    """
    seed_everything(seed)

    X_train_np = (np.random.rand(n_data, 1) * 8).astype(np.float32)
    y_train_np = (np.sin(X_train_np) + np.random.randn(*X_train_np.shape) * sigma_noise).astype(np.float32)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_ds": train_ds,
        "train_loader": train_loader,
        "X_test": X_test,
        "X_test_tensor": X_test.to(device),
    }


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
    """Train a MAP model. If `seed` is provided, re-seed for deterministic init and training.

    Returns the trained model, per-epoch losses, and fit_time.
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


def eval_probs(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> np.ndarray:
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


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    acc = float(accuracy_score(labels, preds))

    max_conf = probs.max(axis=1)
    uncertainty = 1.0 - max_conf

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0

    eps = 1e-12
    return {
        "Accuracy": acc,
        "Mean_Uncertainty": float(uncertainty.mean()),
        "Std_Uncertainty": float(uncertainty.std()),
        "Brier_Score": float(((probs - one_hot) ** 2).mean()),
        "NLL": float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean()),
    }
