from __future__ import annotations

import os

import numpy as np


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
