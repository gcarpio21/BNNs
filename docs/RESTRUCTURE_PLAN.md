# Restructure: Minimal `praktikum/` Directory

## Context

The current project (`/u/halle/carg/home_at/Documents/BNNs`) is 21GB and has accumulated a lot of cruft over many sessions: duplicate figure/checkpoint directories (`figures/` vs `figures_copy/`, `checkpoints/` vs `checkpoints_copy/`), a full separate git clone of the `laplace-torch` source (`Laplace/`, 50MB, confirmed unused — the conda env has a normal non-editable pip install of `laplace-torch`), three draft report `.tex` files at root, ~15 session-handoff/investigation markdown files, 14 literature PDFs (251MB), a 13GB WILDS dataset copy, and 11 total notebooks when only 6 are actually needed. The user wants a clean sibling directory, `praktikum/`, containing *only* what's needed to run the six notebooks whose filenames contain `5seed`, plus `bnn_comparison_povertymap.ipynb` — nothing else. This supports a focused deliverable without the accumulated research/exploration debris.

## Scope (confirmed with user)

**Notebooks to keep (6, exact filename match on `*5seed*` + povertymap):**
1. `notebooks/bayesian/two_moons_bayesian_5seed.ipynb`
2. `notebooks/bayesian/sinusoidal_bayesian_5seed.ipynb`
3. `notebooks/laplace/two_moons_laplace_5seed.ipynb`
4. `notebooks/laplace/sinusoidal_laplace_5seed.ipynb`
5. `notebooks/comparison/bnn_comparison_sinusoid_5seed.ipynb`
6. `notebooks/comparison/bnn_comparison_povertymap.ipynb`

**WILDS dataset**: not copied (13GB). `download=False` → `download=True` in the copied PovertyMap notebook so it fetches automatically on first run if missing.

**Checkpoints**: directory skeleton only, no files copied — same treatment as `figures/` and `data/wilds/` below. The submission should be code-only; nothing precomputed ships with it, but the folder structure the notebooks already expect (via `checkpoint_exists`/relative-path construction) is preserved so a fresh run has nowhere to error out on a missing parent directory.

**Target location**: `/u/halle/carg/home_at/Documents/praktikum` (sibling directory).

## What gets copied

### `shared/` — keep whole package, drop `helper/`
Verified via direct import trace of all 6 notebooks (`grep` of every `from shared import (...)` block): the union of names needed is `checkpoint_exists, load_checkpoint, save_checkpoint, seed_everything, regression_metrics, regression_uncertainty_stats, load_sinusoid, TinyMLP, train_map, load_two_moons, standard_metrics` — effectively all of `checkpoints.py`, `datasets.py`, `models.py`, and `metrics.py` (whose functions are transitively used via `standard_metrics`/`uncertainty_calibration_summary`). Package is tiny so no benefit to hand-trimming individual functions — copy these 6 files as-is:
- `shared/__init__.py`, `shared/checkpoints.py`, `shared/datasets.py`, `shared/models.py`, `shared/metrics.py`, `shared/plotting.py`

**`plotting.py` is a hard requirement, not optional**: `shared/__init__.py` unconditionally does `from .plotting import (...)` (5 plotting functions), so *any* `from shared import (...)` — which all 6 notebooks do — executes this at import time. Without `plotting.py` present, every notebook's first cell raises `ModuleNotFoundError: No module named 'shared.plotting'`, regardless of whether that notebook calls a plotting function itself. (This file didn't exist, or wasn't yet wired into `__init__.py`, when this plan was first drafted — re-verified directly against the current source before adding it here.) `plotting.py` itself only needs `numpy` and `shared.metrics` (already copied) plus a lazily-imported `matplotlib`, so it's a clean addition.

Confirmed `shared/__init__.py` never imports `.helper` (read directly), and no target notebook references `shared.helper` — safe to drop `shared/helper/` entirely.

**Not copied**: `Laplace/` (verified via `pip show laplace-torch` in the `bnn` conda env — normal site-packages install, `Location:` points to `site-packages/laplace/`, no editable-install marker; the local `Laplace/` git clone at project root is unused at runtime).

### Notebooks — copied as-is, one edit
Copy the 6 files listed above into the equivalent subdirectory structure (`notebooks/bayesian/`, `notebooks/laplace/`, `notebooks/comparison/`). Each notebook's `ROOT` discovery (`while ROOT != ROOT.parent and not (ROOT / "shared").exists(): ROOT = ROOT.parent`) is portable as long as the notebook sits at `praktikum/notebooks/<subdir>/<file>.ipynb` with `praktikum/shared/` present — no path logic needs editing.

**One required edit**: in the copied `bnn_comparison_povertymap.ipynb`, replace every occurrence of `download=False` with `download=True` in `get_dataset(dataset='poverty', download=False, ...)` calls. Re-verified directly against the current notebook (parsed as JSON, not grepped as raw text, since raw-text grep on notebook JSON is unreliable): **5** occurrences now, in cells `c005`, `c017`, `nllgridsearch01`, `tauschedule01`, `batchsizebnn01`. The count and cell list in an earlier version of this plan (8 occurrences, including cells `reduxlafit01`, `lasaveloadcheck01`, `gridsearchnosigma01`) is stale — those three cells have since been deleted from the notebook entirely, not just edited. Doing this as a blanket find-replace across the whole notebook (rather than editing specific cell IDs by name) avoids depending on the exact cell list staying current. This is the only content change; everything else copies verbatim.

### Checkpoints — directory skeleton only, no contents copied
Same names as the original `results/checkpoints/` subdirectories, but created empty — no `.pt` files copied over:
- `praktikum/results/checkpoints/two_moons_bayesian_5seed/`
- `praktikum/results/checkpoints/two_moons_laplace_5seed/`
- `praktikum/results/checkpoints/sinusoid_bayesian_5seed/`
- `praktikum/results/checkpoints/sinusoid_laplace_5seed/`
- `praktikum/results/checkpoints/sinusoid_comparison_5seed/`
- `praktikum/results/checkpoints/two_moons_map_5seed/`
- `praktikum/results/checkpoints/sinusoid_map_5seed/`

The last two (`two_moons_map_5seed`, `sinusoid_map_5seed`) were missing from every earlier version of this plan — found by checking which checkpoint paths the 6 notebooks actually reference, not by trusting the plan's own prior list. They hold the shared MAP-fit checkpoint reused across a dataset's bayesian/laplace notebook pair (the "shared MAP fit time" the report writes up); 5 of the 6 notebooks reference one or the other (only `bnn_comparison_povertymap.ipynb` doesn't, since PovertyMap's MAP is a downloaded pretrained checkpoint, not locally fit).

PovertyMap's nested checkpoint dir, also created empty: `praktikum/notebooks/comparison/results/checkpoints/bnn_comparison_povertymap/`.

Since nothing is copied, the earlier concern about the PovertyMap checkpoint directory being actively written to by an in-progress run no longer applies — there's no file content to race against, just an empty directory to create.

**Not created/copied at all**: `notebooks/bayesian/results/checkpoints/{sinusoid_bayesian,two_moons_bayesian}/` (1-seed variants, out of scope), `notebooks/laplace/results/checkpoints/` (empty), any `metrics/`, `*_copy/` directories, `results/checkpoints/legacy_backup/`, `*_1seed/` checkpoint dirs (out of scope) — these don't even get an empty-directory placeholder, unlike the eight locations above (seven under `results/checkpoints/`, one nested under `notebooks/comparison/`).

## Target directory structure

```
praktikum/
├── shared/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── datasets.py
│   ├── models.py
│   ├── metrics.py
│   └── plotting.py
├── notebooks/
│   ├── bayesian/
│   │   ├── two_moons_bayesian_5seed.ipynb
│   │   └── sinusoidal_bayesian_5seed.ipynb
│   ├── laplace/
│   │   ├── two_moons_laplace_5seed.ipynb
│   │   └── sinusoidal_laplace_5seed.ipynb
│   ├── comparison/
│   │   ├── bnn_comparison_sinusoid_5seed.ipynb
│   │   ├── bnn_comparison_povertymap.ipynb   (download=True edit applied)
│   │   └── results/
│   │       └── checkpoints/
│   │           └── bnn_comparison_povertymap/   (empty)
│   └── figures/            (empty dir; notebooks write here via relative "../figures")
├── results/
│   └── checkpoints/
│       ├── two_moons_bayesian_5seed/       (empty)
│       ├── two_moons_laplace_5seed/        (empty)
│       ├── sinusoid_bayesian_5seed/        (empty)
│       ├── sinusoid_laplace_5seed/         (empty)
│       ├── sinusoid_comparison_5seed/      (empty)
│       ├── two_moons_map_5seed/            (empty)
│       └── sinusoid_map_5seed/             (empty)
└── data/
    └── wilds/               (empty; WILDS auto-downloads poverty_v1.1 here on first run)
```

## Execution steps

1. Create directory skeleton at `/u/halle/carg/home_at/Documents/praktikum` (all subdirs above).
2. Copy the 6 `shared/*.py` files (including `plotting.py`).
3. Copy the 6 notebooks into their respective subdirs.
4. Edit the copied `bnn_comparison_povertymap.ipynb`: replace every `download=False` → `download=True` (5 occurrences as of the last check against the actual notebook — reconfirm at execution time rather than trusting this number, since it's already drifted once).
5. Create the eight empty checkpoint directories (five dataset/method ones plus the two shared MAP ones, seven total under `results/checkpoints/`, plus one more nested under `notebooks/comparison/results/checkpoints/`) — no files copied into any of them.
6. Create empty `notebooks/figures/` and `data/wilds/` placeholder directories so relative paths resolve without error on first run.

## Verification

- `find praktikum -name "*.ipynb"` → exactly 6 files, matching the list above.
- `python3 -c "import sys; sys.path.insert(0,'praktikum'); import shared; print(shared.__all__ if hasattr(shared,'__all__') else dir(shared))"` from within `praktikum/` → imports cleanly, no `ModuleNotFoundError` for `shared.helper` or `shared.plotting`.
- Open each of the 6 notebooks in `praktikum/` and run the first cell (imports + `ROOT` discovery) — confirm `ROOT` resolves to `praktikum/` and no `FileNotFoundError` on `shared`.
- For `bnn_comparison_povertymap.ipynb`: confirm `grep -c "download=False"` on the copied file returns `0` (raw-text grep is fine for verification, just not for locating cells to edit — see note above on why).
- `find praktikum -type d -name checkpoints -o -type d -path "*checkpoints/*"` → confirm all eight checkpoint directories exist and are empty (no `.pt` files).
- Spot-check one non-PovertyMap notebook (e.g. `sinusoidal_bayesian_5seed.ipynb`) runs its checkpoint-loading cell and reports "trained" (not "loaded") for every variant, confirming it correctly finds no cached checkpoint at the new path and fits fresh instead.
