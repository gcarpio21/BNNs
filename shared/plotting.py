from __future__ import annotations

import os

import numpy as np

from .metrics import combine_predictive_std


def plot_reliability_diagrams(
    variant_specs, all_seed_probs, all_seed_y_test, seeds, figures_dir,
    filename_prefix, n_bins: int = 15,
):
    """Per-variant reliability diagram (confidence vs accuracy), mean +/- std across
    seeds. Saves one figure per variant as
    {figures_dir}/{filename_prefix}_{variant_name}_reliability.png. Method-agnostic,
    used identically for Laplace and Bayesian-Torch variants on two_moons.
    """
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, n_bins + 1)
    for spec in variant_specs:
        vname = spec["variant_name"]
        title = spec["plot_title"]

        all_accs, all_confs = [], []
        for seed in seeds:
            probs = all_seed_probs.get(seed, {}).get(vname)
            if probs is None:
                continue
            y_true = all_seed_y_test[seed]
            conf = probs.max(axis=1)
            pred = probs.argmax(axis=1)
            accs_bin, confs_bin = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (conf > lo) & (conf <= hi)
                if mask.sum() == 0:
                    accs_bin.append(np.nan)
                    confs_bin.append((lo + hi) / 2)
                else:
                    accs_bin.append(float((pred[mask] == y_true[mask]).mean()))
                    confs_bin.append(float(conf[mask].mean()))
            all_accs.append(accs_bin)
            all_confs.append(confs_bin)

        all_accs  = np.array(all_accs)
        all_confs = np.array(all_confs)
        mean_obs  = np.nanmean(all_accs, axis=0)
        std_obs   = np.nanstd(all_accs, axis=0)
        mean_conf = np.nanmean(all_confs, axis=0)
        valid = ~np.isnan(mean_obs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_conf[valid], mean_obs[valid], "o-", label=f"Model (mean, {len(seeds)} seeds)")
        ax.fill_between(mean_conf[valid],
                        (mean_obs - std_obs)[valid],
                        (mean_obs + std_obs)[valid],
                        alpha=0.2, label=f"±1σ across {len(seeds)} seeds")
        ax.set_xlabel("Mean Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} — Reliability Diagram ({len(seeds)} seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fname = f"{filename_prefix}_{vname}_reliability.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {fname}")
    print(f"\nDone: {len(variant_specs)} reliability figures")


def plot_boundary(predict_fn, X_data, y_data, title, ax=None):
    """2D decision-boundary contour + scatter, for any binary classifier's predict_fn
    (probability of class 1 given an (N, 2) array). Method-agnostic, used identically
    for both Laplace and Bayesian-Torch variants on two_moons.
    """
    import matplotlib.pyplot as plt

    h = 0.05
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    probs = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu_r", alpha=0.8, vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap="bwr",
               edgecolors="k", linewidths=0.4, s=30, zorder=3)
    ax.set_title(title)


def plot_regression_prediction_bands(
    variant_specs, all_seed_preds, seeds, x_grid, train_lo, train_hi,
    figures_dir, filename_prefix, true_fn=np.sin, xlim=(-5, 13), ylim=(-3, 3),
):
    """Per-variant prediction figure: mean prediction plus total, epistemic, and
    seed-variability bands, averaged across seeds. Method-agnostic, used identically
    for Laplace and Bayesian-Torch variants on the sinusoid task.

    Args:
        variant_specs: List of variant spec dicts with "variant_name"/"plot_title".
        all_seed_preds: {seed: {variant_name: (mean, epistemic_std, aleatoric_sigma)}}.
        seeds: Seeds to average over.
        x_grid: X values the predictions were evaluated on.
        train_lo: Lower bound of the training range, for OOD shading.
        train_hi: Upper bound of the training range, for OOD shading.
        figures_dir: Directory to save figures into.
        filename_prefix: Prefix for saved filenames.
        true_fn: Ground-truth function to overlay, evaluated on x_grid.
        xlim: X-axis limits.
        ylim: Y-axis limits.
    """
    import matplotlib.pyplot as plt

    for spec in variant_specs:
        vname = spec["variant_name"]
        title = spec["plot_title"]

        all_mus = np.stack([all_seed_preds[s][vname][0] for s in seeds])
        all_epis = np.stack([all_seed_preds[s][vname][1] for s in seeds])
        all_sigs = [all_seed_preds[s][vname][2] for s in seeds]

        mean_mu = all_mus.mean(axis=0)
        std_mu = all_mus.std(axis=0)
        mean_epi = all_epis.mean(axis=0)
        mean_sig = float(np.mean(all_sigs))
        mean_total = combine_predictive_std(mean_epi, mean_sig)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.axvspan(xlim[0], train_lo, color="gray", alpha=0.08)
        ax.axvspan(train_hi, xlim[1], color="gray", alpha=0.08, label="OOD (extrapolation)")
        ax.plot(x_grid, true_fn(x_grid), "k--", lw=1, alpha=0.6, label="True sin(x)")
        ax.plot(x_grid, mean_mu, color="tab:orange", lw=2, label=f"Mean prediction ({len(seeds)} seeds)")
        ax.fill_between(x_grid, mean_mu - 2 * mean_total, mean_mu + 2 * mean_total,
                        color="tab:orange", alpha=0.20, label="±2σ total (avg)")
        ax.fill_between(x_grid, mean_mu - 2 * mean_epi, mean_mu + 2 * mean_epi,
                        color="tab:purple", alpha=0.30, label="±2σ epistemic (avg)")
        ax.fill_between(x_grid, mean_mu - std_mu, mean_mu + std_mu,
                        color="tab:green", alpha=0.25, label="±1σ seed variability")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title} — {len(seeds)}-Seed Mean Predictive Uncertainty")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fname = f"{filename_prefix}_{vname}_prediction.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {fname}")
    print(f"\nDone: {len(variant_specs)} prediction figures")


def plot_regression_ood_epistemic(
    variant_specs, all_seed_preds, seeds, x_grid, train_lo, train_hi,
    figures_dir, filename_prefix, xlim=(-5, 13),
):
    """Per-variant epistemic-uncertainty-vs-x figure, mean plus seed-variability band.
    Method-agnostic, used identically for Laplace and Bayesian-Torch variants on the
    sinusoid task.

    Args:
        variant_specs: List of variant spec dicts with "variant_name"/"plot_title".
        all_seed_preds: {seed: {variant_name: (mean, epistemic_std, aleatoric_sigma)}}.
        seeds: Seeds to average over.
        x_grid: X values the predictions were evaluated on.
        train_lo: Lower bound of the training range, for OOD shading.
        train_hi: Upper bound of the training range, for OOD shading.
        figures_dir: Directory to save figures into.
        filename_prefix: Prefix for saved filenames.
        xlim: X-axis limits.
    """
    import matplotlib.pyplot as plt

    for spec in variant_specs:
        vname = spec["variant_name"]
        title = spec["plot_title"]

        all_epis = np.stack([all_seed_preds[s][vname][1] for s in seeds])
        mean_epi = all_epis.mean(axis=0)
        std_epi = all_epis.std(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        ax.axvspan(xlim[0], train_lo, color="gray", alpha=0.10, label="OOD (extrapolation)")
        ax.axvspan(train_hi, xlim[1], color="gray", alpha=0.10)
        ax.axvline(train_lo, color="k", ls=":", lw=1)
        ax.axvline(train_hi, color="k", ls=":", lw=1)
        ax.plot(x_grid, mean_epi, color="tab:purple", lw=2, label="Mean epistemic std")
        ax.fill_between(x_grid, mean_epi - std_epi, mean_epi + std_epi,
                        color="tab:purple", alpha=0.20, label="±1σ seed variability")
        ax.set_xlim(*xlim)
        ax.set_xlabel("x")
        ax.set_ylabel("Epistemic std")
        ax.set_title(f"{title} — {len(seeds)}-Seed Epistemic Uncertainty vs x")
        ax.legend(loc="upper center", fontsize=8)
        ax.grid(True, alpha=0.3)
        fname = f"{filename_prefix}_{vname}_ood_epistemic.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {fname}")
    print(f"\nDone: {len(variant_specs)} OOD epistemic figures")


def plot_regression_calibration(
    variant_specs, seed_calibration, levels, seeds, figures_dir, filename_prefix,
):
    """Per-variant one-sided calibration diagram, as used by Laplace Redux and adapted
    from Kuleshov et al. 2018, mean plus std band across seeds. Method-agnostic, used
    identically for Laplace and Bayesian-Torch variants on the sinusoid task.

    Args:
        variant_specs: List of variant spec dicts with "variant_name"/"plot_title".
        seed_calibration: {seed: {variant_name: observed_p_hat_array}}.
        levels: Nominal Coverages the p_hat arrays were evaluated at.
        seeds: Seeds to average over.
        figures_dir: Directory to save figures into.
        filename_prefix: Prefix for saved filenames.
    """
    import matplotlib.pyplot as plt

    for spec in variant_specs:
        vname = spec["variant_name"]
        title = spec["plot_title"]
        avail = [seed_calibration[s][vname] for s in seeds if vname in seed_calibration[s]]
        if not avail:
            print(f"  Skipping {vname} (no data)")
            continue
        obs_matrix = np.stack(avail)
        mean_obs = obs_matrix.mean(axis=0)
        std_obs = obs_matrix.std(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(levels, mean_obs, "o-", label="Model mean")
        ax.fill_between(levels, mean_obs - std_obs, mean_obs + std_obs,
                        alpha=0.2, label=f"±1σ across {len(seeds)} seeds")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(f"{title} — {len(seeds)}-Seed Calibration Diagram")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f"{filename_prefix}_{vname}_calibration.png"
        fig.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {fname}")
    print(f"\nDone: {len(variant_specs)} calibration figures")
