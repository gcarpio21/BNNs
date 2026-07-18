from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm


# ---------------------------------------------------------------------------
# Classification metrics (two_moons and any binary/multiclass task)
# ---------------------------------------------------------------------------

def entropy_per_example(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy along the last axis, no reduction (one value per row)."""
    return -(probs * np.log(probs + eps)).sum(axis=-1)


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Average predictive entropy for a (N, C) probs array."""
    return float(entropy_per_example(probs, eps).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean Brier score for multiclass probs (averaged over classes via one-hot MSE)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).mean())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Scalar ECE using max-prob confidence binning."""
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
        acc_bin  = (preds[mask] == labels[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(conf_bin - acc_bin)
    return float(ece)


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Average one-vs-rest ECE across classes."""
    if probs.ndim != 2 or probs.shape[1] < 2:
        return float("nan")
    values = []
    for class_idx in range(probs.shape[1]):
        binary_probs  = np.column_stack([1.0 - probs[:, class_idx], probs[:, class_idx]])
        binary_labels = (labels == class_idx).astype(int)
        values.append(expected_calibration_error(binary_probs, binary_labels, n_bins=n_bins))
    return float(np.mean(values))


def calibration_curve_data(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """Return calibration-bin statistics for reliability-style plots."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins  = np.linspace(0.0, 1.0, n_bins + 1)

    centers, accuracies, mean_confidences = [], [], []
    expected_uncertainties, actual_errors, counts = [], [], []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        acc_bin  = float((preds[mask] == labels[mask]).mean())
        conf_bin = float(confidences[mask].mean())
        count    = int(mask.sum())

        centers.append(float((low + high) / 2.0))
        accuracies.append(acc_bin)
        mean_confidences.append(conf_bin)
        expected_uncertainties.append(float(1.0 - conf_bin))
        actual_errors.append(float(1.0 - acc_bin))
        counts.append(count)

    return {
        "bin_center":            np.asarray(centers,               dtype=float),
        "accuracy":              np.asarray(accuracies,            dtype=float),
        "mean_confidence":       np.asarray(mean_confidences,      dtype=float),
        "expected_uncertainty":  np.asarray(expected_uncertainties, dtype=float),
        "actual_error":          np.asarray(actual_errors,         dtype=float),
        "count":                 np.asarray(counts,                dtype=int),
    }


def uncertainty_calibration_summary(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict:
    """Calibration summary matching Laplace Redux's reported classification metrics
    (ECE, Brier score, confidence; see Fig. 6/10 in Daxberger et al. 2021)."""
    confidences = probs.max(axis=1)
    return {
        "ECE":              expected_calibration_error(probs, labels, n_bins=n_bins),
        "Brier_Score":      brier_score(probs, labels),
        "Mean_Confidence":  float(confidences.mean()),
    }


def standard_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Canonical classification metric set (two_moons, and any multiclass task).

    Single definition used by both Laplace and Bayesian notebooks so numbers
    cannot drift between them.
    """
    eps   = 1e-12
    preds = probs.argmax(axis=1)
    accuracy = float((preds == labels).mean())
    nll      = float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())
    summary  = uncertainty_calibration_summary(probs, labels, n_bins=n_bins)
    return {"Accuracy": accuracy, "NLL": nll, **summary}


# ---------------------------------------------------------------------------
# Regression metrics (sinusoid, PovertyMap, and any Gaussian-output task)
# ---------------------------------------------------------------------------

def regression_metrics(y_true, mean, total_std, n_levels: int = 20) -> dict:
    """Regression metrics matching Laplace Redux's reported metrics: RMSE (their MSE,
    rooted), NLL, and two-sided calibration MAE (see Fig. 6/10 in Daxberger et al. 2021).

    Calibration uses two-sided prediction intervals: for each confidence level c_j,
    the observed coverage is the fraction of test points where |z_i| <= Phi^{-1}((1+c_j)/2).
    The calibration error is the mean absolute gap between expected and observed coverage.
    """
    y   = np.asarray(y_true,     dtype=float).ravel()
    mu  = np.asarray(mean,       dtype=float).ravel()
    sd  = np.clip(np.asarray(total_std, dtype=float).ravel(), 1e-12, None)
    resid = y - mu

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    nll  = float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2) + resid ** 2 / (2.0 * sd ** 2)))

    z_abs       = np.abs(resid / sd)
    conf_levels = np.linspace(0.05, 0.95, n_levels)
    observed    = np.array([
        float(np.mean(z_abs <= _norm.ppf((1.0 + c) / 2.0))) for c in conf_levels
    ])
    gap = np.abs(observed - conf_levels)

    return {
        "RMSE":            rmse,
        "NLL":             nll,
        "Calibration_MAE": float(gap.mean()),
    }


def combine_predictive_std(epistemic_std, aleatoric_std):
    """Total predictive std = sqrt(epistemic^2 + aleatoric^2). Works on numpy
    arrays, python floats, or torch tensors, with no numpy coercion.
    """
    return (epistemic_std ** 2 + aleatoric_std ** 2) ** 0.5
