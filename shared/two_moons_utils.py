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
from sklearn.metrics import log_loss
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


def load_checkpoint(
    checkpoint_path: str,
    map_location: torch.device | str | None = None,
    weights_only: bool = False,
):
    """Load checkpoints with compatibility across PyTorch versions.

    PyTorch 2.6 changed torch.load default `weights_only=True`, which breaks
    object checkpoints (e.g., Laplace objects) unless explicitly disabled.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        # Older torch versions do not support the `weights_only` argument.
        try:
            return torch.load(checkpoint_path, map_location=map_location)
        except Exception as e:
            # Bubble up for the outer handler below which will handle corrupted files.
            last_exc = e
    except Exception as e:
        last_exc = e

    # If we reach here, torch.load raised an exception (corrupt or unreadable file).
    # Handle common corruption/unpickling errors gracefully so notebooks can continue
    # and decide to recompute the checkpoint instead of crashing.
    try:
        msg = str(last_exc)
    except Exception:
        msg = repr(last_exc)

    # If the file appears corrupted, move it aside and return None so callers
    # can fallback to recomputing the checkpoint.
    corrupt_indicators = [
        'PytorchStreamReader failed locating file data.pkl',
        'miniz error',
        'file not found',
        'UnpicklingError',
        'pickle.UnpicklingError',
    ]
    if any(ind.lower() in msg.lower() for ind in corrupt_indicators):
        try:
            corrupt_path = checkpoint_path + '.corrupt'
            os.replace(checkpoint_path, corrupt_path)
        except Exception:
            # If we can't move the file, ignore — we'll still return None.
            pass
        print(f"Warning: checkpoint appears corrupted. Renamed to: {corrupt_path}")
        return None

    # For other errors (permission, version mismatch, etc.), re-raise to surface
    # unexpected conditions to the user.
    raise last_exc


def save_checkpoint(payload: dict, checkpoint_path: str) -> None:
    # Always sanitize incoming payloads to avoid torch.save trying to pickle
    # non-serialisable internals (e.g., local closures/hooks inside Laplace wrappers).
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _sanitize(obj_dict: dict) -> dict:
        safe = {}
        for k, v in (obj_dict or {}).items():
            # If object exposes a state_dict, store its state dict instead of the object
            if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                try:
                    safe_key = k if k.endswith("_state_dict") else f"{k}_state_dict"
                    safe[safe_key] = v.state_dict()
                    continue
                except Exception:
                    # Skip if state_dict extraction fails
                    continue

            # Common simple types that are safe to pickle
            try:
                import pickle as _pickle

                _pickle.dumps(v)
                safe[k] = v
            except Exception:
                # Skip anything not picklable
                continue
        return safe

    safe_payload = _sanitize(payload)

    # Write atomically: save to a temp file then rename
    tmp_path = checkpoint_path + ".tmp"
    try:
        torch.save(safe_payload, tmp_path)
        try:
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            # Best-effort cleanup/rename
            os.remove(tmp_path) if os.path.exists(tmp_path) else None
            raise
    except Exception as e:
        # As a last resort, attempt to further prune the payload and retry
        try:
            pruned = {}
            for k, v in safe_payload.items():
                # only keep tensors and small scalars/containers
                if isinstance(v, (torch.Tensor, int, float, str, dict, list, tuple, bool, type(None))):
                    pruned[k] = v
            torch.save(pruned, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            # Give up and raise original exception to surface to user
            raise e


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


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Return average predictive entropy for a (N, C) probs array."""
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(ent.mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute (scalar) ECE using max-prob binning.

    probs: (N, C) predicted probabilities
    labels: (N,) integer labels
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        acc_bin = (preds[mask] == labels[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(conf_bin - acc_bin)
    return float(ece)


def calibration_curve_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Return calibration-bin statistics for reliability-style plots."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    centers = []
    accuracies = []
    mean_confidences = []
    expected_uncertainties = []
    actual_errors = []
    counts = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue

        acc_bin = float((preds[mask] == labels[mask]).mean())
        conf_bin = float(confidences[mask].mean())
        count = int(mask.sum())

        centers.append(float((low + high) / 2.0))
        accuracies.append(acc_bin)
        mean_confidences.append(conf_bin)
        expected_uncertainties.append(float(1.0 - conf_bin))
        actual_errors.append(float(1.0 - acc_bin))
        counts.append(count)

    return {
        "bin_center": np.asarray(centers, dtype=float),
        "accuracy": np.asarray(accuracies, dtype=float),
        "mean_confidence": np.asarray(mean_confidences, dtype=float),
        "expected_uncertainty": np.asarray(expected_uncertainties, dtype=float),
        "actual_error": np.asarray(actual_errors, dtype=float),
        "count": np.asarray(counts, dtype=int),
    }


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Average one-vs-rest ECE across classes."""
    if probs.ndim != 2 or probs.shape[1] < 2:
        return float("nan")

    values = []
    for class_idx in range(probs.shape[1]):
        binary_probs = np.column_stack([1.0 - probs[:, class_idx], probs[:, class_idx]])
        binary_labels = (labels == class_idx).astype(int)
        values.append(expected_calibration_error(binary_probs, binary_labels, n_bins=n_bins))
    return float(np.mean(values))


def uncertainty_calibration_summary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Summarize calibration and expected-vs-actual uncertainty behavior."""
    curve = calibration_curve_data(probs, labels, n_bins=n_bins)
    confidences = probs.max(axis=1)
    errors = (probs.argmax(axis=1) != labels).astype(float)

    expected_uncertainty = 1.0 - confidences
    actual_uncertainty = errors

    if len(curve["count"]) > 0:
        weights = curve["count"] / curve["count"].sum()
        uncertainty_gap = np.abs(curve["expected_uncertainty"] - curve["actual_error"])
        uncertainty_gap_mae = float(np.average(uncertainty_gap, weights=weights))
        uncertainty_gap_mse = float(np.average(uncertainty_gap ** 2, weights=weights))

        if len(curve["expected_uncertainty"]) > 1 and np.std(curve["expected_uncertainty"]) > 0 and np.std(curve["actual_error"]) > 0:
            uncertainty_corr = float(np.corrcoef(curve["expected_uncertainty"], curve["actual_error"])[0, 1])
        else:
            uncertainty_corr = float("nan")

        max_gap = float(np.max(uncertainty_gap))
    else:
        uncertainty_gap_mae = float("nan")
        uncertainty_gap_mse = float("nan")
        uncertainty_corr = float("nan")
        max_gap = float("nan")

    return {
        "ECE": expected_calibration_error(probs, labels, n_bins=n_bins),
        "Classwise_ECE": classwise_ece(probs, labels, n_bins=n_bins),
        "Brier_Score": brier_score(probs, labels),
        "Mean_Confidence": float(confidences.mean()),
        "1 - Confidence": float(expected_uncertainty.mean()),
        "Mean_Entropy": predictive_entropy(probs),
        "ExpectedVsActual_Uncertainty_MAE": uncertainty_gap_mae,
        "ExpectedVsActual_Uncertainty_MSE": uncertainty_gap_mse,
        "ExpectedVsActual_Uncertainty_Corr": uncertainty_corr,
        "Max_Binned_Uncertainty_Gap": max_gap,
        "Mean_Actual_Error": float(actual_uncertainty.mean()),
    }


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean Brier score for multiclass probs (averaged over classes via one-hot MSE)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).mean())


def standard_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Canonical metric set computed identically for every two-moons model.

    Both the Laplace and Bayesian notebooks call this so every metric has a
    single definition. `probs` is (N, C) predicted probabilities; `labels` is
    (N,) integer labels.

    Uncertainty is defined ONE way: `1 - Confidence = mean(1 - max_prob)`.
    Predictive entropy is reported separately as `Mean_Entropy`. All
    calibration/Brier/ECE values come from `uncertainty_calibration_summary`
    so they cannot drift between notebooks.
    """
    eps = 1e-12
    preds = probs.argmax(axis=1)
    accuracy = float((preds == labels).mean())
    nll = float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())
    summary = uncertainty_calibration_summary(probs, labels, n_bins=n_bins)
    return {"Accuracy": accuracy, "NLL": nll, **summary}


def regression_metrics(y_true, mean, total_std, n_levels: int = 11) -> dict:
    """Scored regression metrics for the sinusoid notebooks (single source).

    Needs ground-truth targets. `mean`/`total_std` are the predictive Gaussian's
    mean and std (total = sqrt(epistemic^2 + aleatoric^2)). Mirrors the two-moons
    metric philosophy: a point-error term (RMSE/MAE), a proper-scoring term (NLL),
    and an actual-vs-expected calibration block.

    Calibration uses the predictive CDF: for a well-calibrated model the CDF value
    p_i = Phi((y_i - mean_i)/std_i) is Uniform(0,1). For each expected level q we
    compare it against the *observed* fraction with p_i <= q; the gap between
    expected (q) and actual (observed) coverage is the regression analog of the
    expected-vs-actual uncertainty block on two moons.
    """
    import math
    y = np.asarray(y_true, dtype=float).ravel()
    mu = np.asarray(mean, dtype=float).ravel()
    sd = np.clip(np.asarray(total_std, dtype=float).ravel(), 1e-12, None)
    resid = y - mu

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    # Gaussian negative log-likelihood (lower = better; punishes over- and under-confidence)
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2) + resid ** 2 / (2.0 * sd ** 2)))

    z = resid / sd
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))   # predictive CDF at each true y
    qs = np.linspace(0.0, 1.0, n_levels)
    observed = np.array([float(np.mean(cdf <= q)) for q in qs])
    gap = np.abs(observed - qs)
    cal_corr = float(np.corrcoef(qs, observed)[0, 1]) if observed.std() > 0 else float("nan")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "NLL": nll,
        "Calibration_MAE": float(gap.mean()),
        "Calibration_MSE": float((gap ** 2).mean()),
        "Calibration_Corr": cal_corr,
        "Max_Calibration_Gap": float(gap.max()),
        "Coverage_68": float(np.mean(np.abs(z) <= 1.0)),
        "Coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
    }


def regression_uncertainty_stats(epistemic_std, aleatoric_std) -> dict:
    """Canonical uncertainty summary for the sinusoid regression notebooks.

    Computed identically by both the Laplace and Bayesian notebooks so the
    numbers cannot drift. `epistemic_std` is a per-point (N,) array (function /
    posterior std); `aleatoric_std` is a scalar or (N,) observation-noise std.
    Total predictive std = sqrt(epistemic^2 + aleatoric^2).
    """
    epi = np.asarray(epistemic_std, dtype=float).ravel()
    ale = np.asarray(aleatoric_std, dtype=float)
    ale = np.full_like(epi, float(ale)) if ale.ndim == 0 else ale.ravel()
    total = np.sqrt(epi ** 2 + ale ** 2)
    return {
        "Mean_Epistemic_Std": float(epi.mean()),
        "Mean_Aleatoric_Std": float(ale.mean()),
        "Mean_Total_Std": float(total.mean()),
    }


def predict_probs_from_model_or_fn(model_or_fn, X_np: np.ndarray, batch_size: int = 256, device: torch.device | None = None) -> np.ndarray:
    """Return (N, C) numpy probabilities for either a callable predictor or a torch module.

    If `model_or_fn` is a callable that accepts numpy arrays and returns either a (N,) class-1 prob or (N, C) probs, it will be used directly. If it is a torch.nn.Module, it will be applied in batches.
    """
    # Try calling as a numpy-aware predictor
    try:
        out = model_or_fn(X_np)
        out = np.array(out)
        if out.ndim == 1:
            p1 = out
            return np.vstack([1 - p1, p1]).T
        return out
    except Exception:
        pass

    # Otherwise, assume it's a torch module
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


# --- Compatibility helpers for laplace-torch state_dict loading ---
def safe_load_laplace_state(la_obj, state_dict: dict):
    """Load a laplace object's state_dict while avoiding feature-extractor
    discovery from stored sample `data` which can trigger "Last layer is
    already known." ValueError in some laplace-torch versions.

    This makes a shallow copy of the dict and clears the `data` entry
    before delegating to the laplace object's `load_state_dict`.
    """
    state = state_dict.copy() if isinstance(state_dict, dict) else state_dict
    if isinstance(state, dict) and 'data' in state:
        state['data'] = None
    return la_obj.load_state_dict(state)


# Monkeypatch LLLaplace.load_state_dict to be tolerant when possible.
try:
    from laplace.lllaplace import LLLaplace

    _orig_ll_load = LLLaplace.load_state_dict

    def _patched_ll_load(self, state_dict):
        # Null out stored sample `data` to avoid triggering feature-extractor
        # discovery that can raise when last_layer is already set.
        sd = state_dict.copy() if isinstance(state_dict, dict) else state_dict
        if isinstance(sd, dict) and 'data' in sd:
            sd['data'] = None
        try:
            return _orig_ll_load(self, sd)
        except ValueError as e:
            msg = str(e)
            if 'Last layer is already known' in msg:
                # Try clearing instance data and retry once.
                try:
                    self.data = None
                    return _orig_ll_load(self, sd)
                except Exception:
                    pass
            # Re-raise if we can't handle it here.
            raise

    LLLaplace.load_state_dict = _patched_ll_load
except Exception:
    # laplace not available or monkeypatch failed; ignore and fall back to
    # caller-side fixes in notebooks.
    pass
