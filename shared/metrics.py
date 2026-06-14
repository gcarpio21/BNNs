from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm


# ---------------------------------------------------------------------------
# Classification metrics (two_moons and any binary/multiclass task)
# ---------------------------------------------------------------------------

def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    """Average predictive entropy for a (N, C) probs array."""
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(ent.mean())


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
    """Summarize calibration and expected-vs-actual uncertainty behaviour."""
    curve       = calibration_curve_data(probs, labels, n_bins=n_bins)
    confidences = probs.max(axis=1)
    errors      = (probs.argmax(axis=1) != labels).astype(float)

    expected_uncertainty = 1.0 - confidences
    actual_uncertainty   = errors

    if len(curve["count"]) > 0:
        weights        = curve["count"] / curve["count"].sum()
        uncertainty_gap = np.abs(curve["expected_uncertainty"] - curve["actual_error"])
        uncertainty_gap_mae = float(np.average(uncertainty_gap, weights=weights))
        uncertainty_gap_mse = float(np.average(uncertainty_gap ** 2, weights=weights))

        if (len(curve["expected_uncertainty"]) > 1
                and np.std(curve["expected_uncertainty"]) > 0
                and np.std(curve["actual_error"]) > 0):
            uncertainty_corr = float(
                np.corrcoef(curve["expected_uncertainty"], curve["actual_error"])[0, 1]
            )
        else:
            uncertainty_corr = float("nan")

        max_gap = float(np.max(uncertainty_gap))
    else:
        uncertainty_gap_mae = uncertainty_gap_mse = uncertainty_corr = max_gap = float("nan")

    return {
        "ECE":                               expected_calibration_error(probs, labels, n_bins=n_bins),
        "Classwise_ECE":                     classwise_ece(probs, labels, n_bins=n_bins),
        "Brier_Score":                       brier_score(probs, labels),
        "Mean_Confidence":                   float(confidences.mean()),
        "1 - Confidence":                    float(expected_uncertainty.mean()),
        "Mean_Entropy":                      predictive_entropy(probs),
        "ExpectedVsActual_Uncertainty_MAE":  uncertainty_gap_mae,
        "ExpectedVsActual_Uncertainty_MSE":  uncertainty_gap_mse,
        "ExpectedVsActual_Uncertainty_Corr": uncertainty_corr,
        "Max_Binned_Uncertainty_Gap":        max_gap,
        "Mean_Actual_Error":                 float(actual_uncertainty.mean()),
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
    """Regression metrics following Kuleshov et al. (2018) and Daxberger et al. (2021).

    Calibration uses two-sided prediction intervals: for each confidence level c_j,
    the observed coverage is the fraction of test points where |z_i| <= Phi^{-1}((1+c_j)/2).
    The calibration error is the mean absolute gap between expected and observed coverage.
    This matches the regression calibration error used in Laplace Redux (reference [71]).
    """
    y   = np.asarray(y_true,     dtype=float).ravel()
    mu  = np.asarray(mean,       dtype=float).ravel()
    sd  = np.clip(np.asarray(total_std, dtype=float).ravel(), 1e-12, None)
    resid = y - mu

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae  = float(np.mean(np.abs(resid)))
    nll  = float(np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2) + resid ** 2 / (2.0 * sd ** 2)))

    z_abs       = np.abs(resid / sd)
    conf_levels = np.linspace(0.05, 0.95, n_levels)
    observed    = np.array([
        float(np.mean(z_abs <= _norm.ppf((1.0 + c) / 2.0))) for c in conf_levels
    ])
    gap      = np.abs(observed - conf_levels)
    cal_corr = float(np.corrcoef(conf_levels, observed)[0, 1]) if observed.std() > 0 else float("nan")

    return {
        "RMSE":               rmse,
        "MAE":                mae,
        "NLL":                nll,
        "Calibration_MAE":    float(gap.mean()),
        "Calibration_MSE":    float((gap ** 2).mean()),
        "Calibration_Corr":   cal_corr,
        "Max_Calibration_Gap": float(gap.max()),
        "Coverage_68":        float(np.mean(z_abs <= 1.0)),
        "Coverage_95":        float(np.mean(z_abs <= 1.959963984540054)),
    }


def regression_uncertainty_stats(epistemic_std, aleatoric_std) -> dict:
    """Canonical uncertainty decomposition for regression notebooks.

    Total predictive std = sqrt(epistemic^2 + aleatoric^2).
    """
    epi = np.asarray(epistemic_std, dtype=float).ravel()
    ale = np.asarray(aleatoric_std, dtype=float)
    ale = np.full_like(epi, float(ale)) if ale.ndim == 0 else ale.ravel()
    total = np.sqrt(epi ** 2 + ale ** 2)
    return {
        "Mean_Epistemic_Std": float(epi.mean()),
        "Mean_Aleatoric_Std": float(ale.mean()),
        "Mean_Total_Std":     float(total.mean()),
    }
