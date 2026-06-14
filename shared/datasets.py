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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            torch.tensor(X_part, dtype=torch.float32),
            torch.tensor(y_part, dtype=torch.long),
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
) -> Dict[str, object]:
    """Generate a simple sinusoid regression dataset and return loaders + arrays."""
    seed_everything(seed)

    X_train_np = (np.random.rand(n_data, 1) * 8).astype(np.float32)
    y_train_np = (np.sin(X_train_np) + np.random.randn(*X_train_np.shape) * sigma_noise).astype(np.float32)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    X_test       = torch.linspace(-5, 13, 500).unsqueeze(-1)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "X_train": X_train, "y_train": y_train,
        "train_ds": train_ds, "train_loader": train_loader,
        "X_test": X_test, "X_test_tensor": X_test.to(device),
    }
