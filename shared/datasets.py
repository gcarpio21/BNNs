from __future__ import annotations

import os
import pickle
from typing import Dict

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .checkpoints import seed_everything


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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            torch.tensor(X_part, dtype=torch.float32).to(device),
            torch.tensor(y_part, dtype=torch.long).to(device),
        )

    train_ds = to_dataset(X_train, y_train)
    val_ds   = to_dataset(X_val,   y_val)
    test_ds  = to_dataset(X_test,  y_test)

    g = torch.Generator()
    try:
        g.manual_seed(seed)
    except Exception:
        g = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_train, shuffle=True,
        generator=(g if g is not None else None),
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val,  "y_test": y_test,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
        "X_test_tensor": torch.tensor(X_test, dtype=torch.float32, device=device),
        "y_test_tensor": torch.tensor(y_test, dtype=torch.long,    device=device),
        "scaler": scaler,
    }


def save_splits(path: str, data_dict: Dict[str, object]) -> None:
    """Save numeric splits and scaler to a .npz and scaler pickle in `path`."""
    os.makedirs(path, exist_ok=True)
    np.savez(
        os.path.join(path, "splits.npz"),
        X=data_dict["X"], y=data_dict["y"],
        X_train=data_dict["X_train"], X_val=data_dict["X_val"], X_test=data_dict["X_test"],
        y_train=data_dict["y_train"], y_val=data_dict["y_val"], y_test=data_dict["y_test"],
    )
    with open(os.path.join(path, "scaler.pkl"), "wb") as f:
        pickle.dump(data_dict["scaler"], f)


def load_splits(path: str, device: torch.device | None = None) -> Dict[str, object]:
    """Load previously saved splits from `path`."""
    arrs = np.load(os.path.join(path, "splits.npz"))
    with open(os.path.join(path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dataset(X_part, y_part):
        return TensorDataset(
            torch.tensor(X_part, dtype=torch.float32).to(device),
            torch.tensor(y_part, dtype=torch.long).to(device),
        )

    X_train, X_val, X_test = arrs["X_train"], arrs["X_val"], arrs["X_test"]
    y_train, y_val, y_test = arrs["y_train"], arrs["y_val"], arrs["y_test"]
    train_ds = to_dataset(X_train, y_train)
    val_ds   = to_dataset(X_val,   y_val)
    test_ds  = to_dataset(X_test,  y_test)

    return {
        "X": arrs["X"], "y": arrs["y"],
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "train_loader": DataLoader(train_ds, batch_size=32, shuffle=True),
        "val_loader":   DataLoader(val_ds,   batch_size=64, shuffle=False),
        "test_loader":  DataLoader(test_ds,  batch_size=64, shuffle=False),
        "X_test_tensor": torch.tensor(X_test, dtype=torch.float32, device=device),
        "y_test_tensor": torch.tensor(y_test, dtype=torch.long,    device=device),
        "scaler": scaler,
    }


def load_sinusoid(
    n_data: int = 150,
    sigma_noise: float = 0.3,
    batch_size: int = 150,
    seed: int = 42,
    device: torch.device | None = None,
    n_val: int = 50,
    n_test_id: int = 250,
    n_test_ood: int = 250,
) -> Dict[str, object]:
    """Generate a sinusoid regression dataset with train/val/test-ID/test-OOD splits.

    Train/val/test-ID are sampled uniformly from [0, 8] (in-distribution).
    Test-OOD covers [-5, 0] (n_test_ood//2 pts) and [8, 13] (n_test_ood//2 pts).
    All splits are labeled (y = sin(x) + noise).
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_split(X_np: np.ndarray) -> tuple:
        y_np = (np.sin(X_np) + np.random.randn(*X_np.shape) * sigma_noise).astype(np.float32)
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_np, dtype=torch.float32).to(device)
        return X_t, y_t

    # Train
    X_train_np = (np.random.rand(n_data, 1) * 8).astype(np.float32)
    X_train, y_train = make_split(X_train_np)
    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Val (in-distribution)
    X_val_np = (np.random.rand(n_val, 1) * 8).astype(np.float32)
    X_val, y_val = make_split(X_val_np)
    val_ds     = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=n_val, shuffle=False)

    # Test ID: evenly spaced in [0, 8]
    X_test_id_np = np.linspace(0, 8, n_test_id).reshape(-1, 1).astype(np.float32)
    X_test_id, y_test_id = make_split(X_test_id_np)
    test_id_ds     = TensorDataset(X_test_id, y_test_id)
    test_id_loader = DataLoader(test_id_ds, batch_size=n_test_id, shuffle=False)

    # Test OOD: n_test_ood//2 pts in [-5, 0] + n_test_ood//2 pts in [8, 13]
    half = n_test_ood // 2
    X_ood_np = np.concatenate([
        np.linspace(-5, 0, half).reshape(-1, 1),
        np.linspace(8, 13, n_test_ood - half).reshape(-1, 1),
    ], axis=0).astype(np.float32)
    X_test_ood, y_test_ood = make_split(X_ood_np)
    test_ood_ds     = TensorDataset(X_test_ood, y_test_ood)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=n_test_ood, shuffle=False)

    return {
        "X_train": X_train, "y_train": y_train,
        "train_ds": train_ds, "train_loader": train_loader,
        "X_val": X_val, "y_val": y_val,
        "val_ds": val_ds, "val_loader": val_loader,
        "X_test_id": X_test_id, "y_test_id": y_test_id,
        "test_id_ds": test_id_ds, "test_id_loader": test_id_loader,
        "X_test_ood": X_test_ood, "y_test_ood": y_test_ood,
        "test_ood_ds": test_ood_ds, "test_ood_loader": test_ood_loader,
        "X_test_id_tensor": X_test_id.to(device),
        "X_test_ood_tensor": X_test_ood.to(device),
        "sigma_noise": sigma_noise,
    }
