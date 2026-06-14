# BNN Project — Session Handoff

_Generated: 2026-06-14_

---

## Project Overview

This is a BNN (Bayesian Neural Network) comparison project evaluating **Laplace approximation** and **Bayesian-Torch** methods across two toy datasets (two-moons classification, sinusoidal regression) and one real dataset (PovertyMap).

**Repo root:** `/u/halle/carg/home_at/Documents/BNNs`

**Shared module** (`shared/`): fully refactored. Exports `TinyMLP`, `train_map`, `seed_everything`, `checkpoint_exists`, `load_checkpoint`, `save_checkpoint`, `load_two_moons`, `load_sinusoid`, `standard_metrics`, `regression_metrics`, `regression_uncertainty_stats`, etc.

---

## Notebook Inventory

### Single-seed notebooks (run and producing checkpoints)

| Notebook | Dataset | Method | Variants | SEED | Checkpoint dir |
|---|---|---|---|---|---|
| `notebooks/bayesian/two_moons_bayesian.ipynb` | Two Moons | Bayesian-Torch | 8 (Flipout/BBB × scratch/moped × elbo/avuc) | 42 | `results/checkpoints/` (relative to notebook dir) |
| `notebooks/laplace/two_moons_laplace.ipynb` | Two Moons | Laplace | 10 (diag/kron/full × all; full/kron × last_layer; × marglik/gridsearch) | 42 | `results/checkpoints/` (relative) |
| `notebooks/bayesian/sinusoidal_bayesian.ipynb` | Sinusoid | Bayesian-Torch | 4 (Flipout/BBB × scratch/moped, ELBO only — no AvUC for regression) | 711 | `results/checkpoints/sinusoid_bayesian/` (relative) |
| `notebooks/laplace/sinusoidal_laplace.ipynb` | Sinusoid | Laplace | 10 (diag/kron/full × all; full/kron × last_layer; × marglik/gridsearch) | 711 | ROOT `results/checkpoints/` (absolute via ROOT resolution) |

### 5-seed notebooks (created this session, NOT YET RUN)

| Notebook | Dataset | Method | Variants × Seeds | SEEDS | Checkpoint dir |
|---|---|---|---|---|---|
| `notebooks/bayesian/sinusoidal_bayesian_5seed.ipynb` | Sinusoid | Bayesian-Torch | 4 × 5 = 20 fits | [711,42,123,456,789] | ROOT `results/checkpoints/sinusoid_bayesian_5seed/` |
| `notebooks/laplace/sinusoidal_laplace_5seed.ipynb` | Sinusoid | Laplace | 12 × 5 = 60 fits | [711,42,123,456,789] | ROOT `results/checkpoints/sinusoid_laplace_5seed/` |
| `notebooks/bayesian/two_moons_bayesian_5seed.ipynb` | Two Moons | Bayesian-Torch | 8 × 5 = 40 fits | [42,711,123,456,789] | ROOT `results/checkpoints/two_moons_bayesian_5seed/` |
| `notebooks/laplace/two_moons_laplace_5seed.ipynb` | Two Moons | Laplace | 10 × 5 = 50 fits | [42,711,123,456,789] | ROOT `results/checkpoints/two_moons_laplace_5seed/` |

> **sinusoidal_laplace_5seed has 12 variants** (vs 10 in single-seed): it adds `kron_all_adam_loop` and `kron_last_layer_adam_loop`. Checkpoints for these two exist in ROOT `results/checkpoints/` from a prior run (`sinusoid_laplace_kron_all_adam_loop_seed711.pt`, etc.).

### Comparison notebook

| Notebook | Status |
|---|---|
| `notebooks/comparison/bnn_comparison_povertymap.ipynb` | ✅ Good — 5 folds, Laplace vs Bayesian-Torch, correct calibration format |

### Stale / legacy notebooks (pending decision/deletion)

| Notebook | Status |
|---|---|
| `notebooks/laplace/two_moons_laplace_copy.ipynb` | Stale copy of `two_moons_laplace.ipynb` — should be deleted |
| `notebooks/bayesian/bnn_bayesian_torch1.ipynb` | Old 18-cell legacy notebook — should be deleted or archived |

---

## Plot Formatting Convention

**All reliability/calibration diagrams must follow this style** (matches `two_moons_bayesian.ipynb`):

```python
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")  # diagonal
ax.plot(conf_levels, observed, "o-", label=f"Model (...)")    # model curve
ax.legend(); ax.grid(True, alpha=0.3)
# NO fill_between shading for over/underconfident regions
# NO ax.text annotations
```

- **Classification** axes: "Mean Confidence" vs "Accuracy"
- **Regression** axes: "Expected confidence" vs "Observed coverage" (Kuleshov calibration: z-score coverage)

**5-seed notebooks** may add `ax.fill_between(conf_levels, mean_obs - std_obs, mean_obs + std_obs, alpha=0.2, label="±1σ across N seeds")` for seed variability — this is acceptable.

**Current formatting status:**

| Notebook | `k--` diagonal | `o-` model | Bad shading | ax.text |
|---|---|---|---|---|
| `two_moons_bayesian.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `two_moons_laplace.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `sinusoidal_bayesian.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `sinusoidal_laplace.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `sinusoidal_bayesian_5seed.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `sinusoidal_laplace_5seed.ipynb` | ✅ | ✅ | ✅ none | ✅ none |
| `two_moons_bayesian_5seed.ipynb` | ⚠️ **MISSING** | ⚠️ **MISSING** | — | — |
| `two_moons_laplace_5seed.ipynb` | ⚠️ **MISSING** | ⚠️ **MISSING** | — | — |
| `bnn_comparison_povertymap.ipynb` | ✅ | ✅ | ✅ none | ✅ none |

---

## What Was Done This Session

1. **Renamed** `notebooks/laplace/bnn_laplace_visualization_files/` → `sinusoidal_laplace_files/` (all 22 PNG files renamed with matching prefix change)
2. **Renamed** `notebooks/notes/bnn_laplace_visualization_notebook.md` → `sinusoidal_laplace_notebook.md`
3. **Restored** `notebooks/bayesian/two_moons_bayesian.ipynb` from git commit `aa96c53`
4. **Fixed legacy import** in `two_moons_bayesian.ipynb`: `from shared.two_moons_utils import standard_metrics` → `from shared import standard_metrics`
5. **Created 4 five-seed notebooks** from scratch (see table above)
6. **Updated calibration formatting** in `sinusoidal_laplace.ipynb` and `sinusoidal_bayesian.ipynb` to match the two_moons style
7. **Verified** `bnn_comparison_povertymap.ipynb` already has correct formatting — no changes needed

---

## Open Tasks (for next session)

### HIGH PRIORITY

#### 1. Run all 5-seed notebooks
None have been executed. Run them in Jupyter in order:
```
notebooks/bayesian/sinusoidal_bayesian_5seed.ipynb
notebooks/laplace/sinusoidal_laplace_5seed.ipynb
notebooks/bayesian/two_moons_bayesian_5seed.ipynb
notebooks/laplace/two_moons_laplace_5seed.ipynb
```
Each should save checkpoints to their respective `results/checkpoints/<name>/` dirs and figures to `notebooks/figures/` (shared). CSVs go to `results/metrics/<name>_metrics.csv`.

#### 2. Add reliability diagrams to two_moons 5-seed notebooks
`two_moons_bayesian_5seed.ipynb` and `two_moons_laplace_5seed.ipynb` have **no calibration/reliability cells** at all. The single-seed versions both have reliability diagrams (classification: confidence vs accuracy). These cells need to be added to the 5-seed versions with the standard `k--` + `o-` format, aggregated across seeds (mean ± std band).

For two_moons (classification), the reliability cell should use ECE-style confidence-accuracy binning, not the regression Kuleshov coverage. Reference the reliability cell in `two_moons_bayesian.ipynb` cell 6 for the correct approach.

#### 3. Decide on sinusoidal_laplace.ipynb adam_loop variants
The single-seed `sinusoidal_laplace.ipynb` has 10 variants; the 5-seed has 12 (adds `kron_all_adam_loop` and `kron_last_layer_adam_loop`). Checkpoints for the adam_loop variants already exist for seed 711. Decide: add the two adam_loop variants to the single-seed notebook for consistency, or leave as-is.

### MEDIUM PRIORITY

#### 4. Delete stale notebooks
```python
# Delete:
notebooks/laplace/two_moons_laplace_copy.ipynb
notebooks/bayesian/bnn_bayesian_torch1.ipynb
```
Confirm with the user before deleting `bnn_bayesian_torch1.ipynb` — it may contain reference code.

#### 5. Clean up checkpoint directory inconsistency
Single-seed notebooks have inconsistent checkpoint root paths:
- `two_moons_*.ipynb` save to `notebooks/<type>/results/checkpoints/` (relative to notebook's CWD)
- `sinusoidal_laplace.ipynb` saves to ROOT `results/checkpoints/` (absolute via ROOT resolution)
- `sinusoidal_bayesian.ipynb` saves to `notebooks/bayesian/results/checkpoints/sinusoid_bayesian/` (relative)

Consider standardizing all single-seed notebooks to save to ROOT `results/checkpoints/<dataset>_<method>_1seed/` — same pattern as the 5-seed notebooks.

#### 6. Stale Laplace/ directory
There is an untracked directory `Laplace/` at the repo root containing what appears to be the laplace-torch library source. Confirm whether this is intentional (local dev install) or a leftover — if the latter, add to `.gitignore` or delete.

---

## Checkpoint State Summary

```
results/checkpoints/          (ROOT — currently holds single-seed sinusoid laplace)
├── sinusoid_map_seed711.pt
├── sinusoid_laplace_diag_all_marglik_seed711.pt
├── sinusoid_laplace_diag_all_gridsearch_seed711.pt
├── sinusoid_laplace_full_all_marglik_seed711.pt
├── sinusoid_laplace_full_all_gridsearch_seed711.pt
├── sinusoid_laplace_full_last_layer_marglik_seed711.pt
├── sinusoid_laplace_full_last_layer_gridsearch_seed711.pt
├── sinusoid_laplace_kron_all_marglik_seed711.pt
├── sinusoid_laplace_kron_all_gridsearch_seed711.pt
├── sinusoid_laplace_kron_all_adam_loop_seed711.pt
├── sinusoid_laplace_kron_last_layer_marglik_seed711.pt
├── sinusoid_laplace_kron_last_layer_gridsearch_seed711.pt
├── sinusoid_laplace_kron_last_layer_adam_loop_seed711.pt
├── two_moons_map_seed42.pt
└── legacy_backup/

notebooks/laplace/results/checkpoints/    (two_moons laplace + old sinusoid laplace)
├── two_moons_gridsearch_diag_all_weights_seed42.pt  (×10 variants)
└── ...

# 5-seed dirs do NOT exist yet (notebooks haven't been run):
results/checkpoints/sinusoid_laplace_5seed/    (MISSING — notebook not run)
results/checkpoints/sinusoid_bayesian_5seed/   (MISSING)
results/checkpoints/two_moons_laplace_5seed/   (MISSING)
results/checkpoints/two_moons_bayesian_5seed/  (MISSING)
```

---

## Key Technical Notes

- **ROOT resolution**: All notebooks walk up from `Path.cwd()` until they find a directory containing `shared/`. This means ROOT = repo root regardless of where the notebook is. **5-seed notebooks use `ROOT /` for checkpoint paths (absolute). Single-seed notebooks are inconsistent — some use ROOT, some use relative `os.path.join(...)` from the notebook's CWD.**

- **x_grid for sinusoid**: `torch.linspace(-5, 13, 500)` — deterministic, seed-independent. Set once before the seed loop in 5-seed notebooks.

- **two_moons data is seed-dependent**: `load_two_moons(seed=seed, ...)` generates different train/val/test splits per seed. This is intentional — variance across seeds captures real generalization uncertainty.

- **sinusoidal_laplace uses `fit_or_load_variant` with `la` object**: pickled Laplace object stored directly in checkpoint. If loading fails, falls back to re-fitting and saving `la_state_dict`.

- **AvULoss**: Only used in two_moons classification BNN variants. Not applicable to sinusoidal regression.

- **Figure prefix convention**: All 5-seed figures use `5seed_<dataset>_<vname>_<type>.png` to avoid collisions with single-seed figures in the shared `notebooks/figures/` directory.
