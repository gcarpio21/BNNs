# Session Handoff — BNNs Project

## 1. Project Structure

```
BNNs/
├── shared/
│   ├── __init__.py
│   ├── models.py          # TinyMLP, train_map (shared), eval_probs
│   ├── datasets.py        # load_two_moons, load_sinusoid
│   └── ...
├── notebooks/
│   ├── bayesian/
│   │   ├── two_moons_bayesian.ipynb       ← main focus this session
│   │   ├── two_moons_bayesian_5seed.ipynb
│   │   ├── sinusoidal_bayesian.ipynb
│   │   └── sinusoidal_bayesian_5seed.ipynb
│   ├── laplace/
│   │   ├── two_moons_laplace.ipynb
│   │   ├── two_moons_laplace_5seed.ipynb
│   │   ├── sinusoidal_laplace.ipynb
│   │   └── sinusoidal_laplace_5seed.ipynb
│   ├── comparison/
│   │   ├── bnn_comparison_povertymap.ipynb
│   │   └── bnn_comparison_sinusoid.ipynb
│   └── notes/
├── Laplace/               # local dev install of laplace-torch (intentional, do not gitignore)
└── submission/template_report/template.tex
```

---

## 2. State of `two_moons_bayesian.ipynb`

### Cells added this session (at the end of the notebook)

| Cell ID | Type | Content |
|---|---|---|
| `97ddc4a4` | markdown | Header: "## Error Analysis Maps" with description of A and C |
| `43adeaca` | code | **Cell A**: Misclassification map — confidence heatmap background + test points sized by confidence, correct=limegreen circle, incorrect=red X |
| `6a8cc12d` | code | **Cell C**: Overconfidence scatter (current state, see below) |

### Cell C — current state (cell `6a8cc12d`)

```python
# C: Overconfidence scatter — per test point: predicted_confidence − actual_correctness
# Omits points whose 95% MC credible interval falls entirely on the correct side of 0.5

for spec in variant_specs:
    ...
    mc = bnn_predict_samples(X_test, net, n_samples=50)   # re-runs MC
    p1_lo = np.percentile(mc[:, :, 1], 2.5, axis=0)
    p1_hi = np.percentile(mc[:, :, 1], 97.5, axis=0)
    within_95 = (
        ((y_test_np == 1) & (p1_lo > 0.5)) |
        ((y_test_np == 0) & (p1_hi < 0.5))
    )
    show = ~within_95
    conf     = probs.max(axis=1)
    correct  = (probs.argmax(axis=1) == y_test_np).astype(float)  # 1 if correct, 0 if not
    overconf = conf - correct   # always negative for correct, always positive for incorrect
    ...scatter plot of X_test[show], colored by overconf[show]...
```

### Known issue with Cell C

The `correct` variable is binary (0 or 1), so:
- **Correct predictions**: `overconf = conf − 1` → always negative → always blue
- **Incorrect predictions**: `overconf = conf − 0 = conf` → always positive → always red

This means the plot is essentially a **confidence-weighted error map**, not a true overconfidence map in the calibration sense. The user asked "what is correctness?" but did not explicitly request a change. Possible next steps:
- Keep as-is (it's still informative: shade of blue = how uncertain a correct pred is, shade of red = how confident a wrong pred is)
- Or redefine to use local accuracy (k-NN) but only at test point locations (scatter, no grid) — avoids the grid extrapolation problem the user objected to
- k-NN with k=100 was discussed and rejected because the user questioned whether grid extrapolation is meaningful

### Previous Cell C iterations (all rejected/superseded)
1. `contourf` + k=5 k-NN + fixed `vabs=0.15` → too much white, too little gradation
2. Power transform (`gamma=0.5`, `gamma=0.3`) → user said no real change visible
3. Custom colormap (immediate jump to medium color at zero) → rejected before running
4. `imshow` with `RdBu_r` (accepted as approach)
5. `imshow` + k=100 k-NN → user rejected: questioned whether grid extrapolation is meaningful
6. Scatter (current) — no k-NN, per-point overconfidence, 95% CI filtering

### Figures style (all notebooks, LaTeX-ready)
- No axis labels (x1, x2 removed in a version "on disk" the user referenced)
- No verbose subtitle/explanation in title — just variant name
- All figures saved to `FIGURES_DIR` for LaTeX import

---

## 3. MAP / Deterministic Model — Findings and Open Question

### What exists

| Notebook | Model | Function | True MAP? |
|---|---|---|---|
| `two_moons_laplace`, `sinusoidal_laplace` (+ 5-seed) | TinyMLP(hidden=64) | `shared/train_map` | **Approximately yes** — Adam + `weight_decay=1e-4`, no early stopping, converges to penalized loss minimum |
| `two_moons_bayesian`, `sinusoidal_bayesian` (+ 5-seed) | TinyMLP(hidden=64) | `shared/train_map` | Same weights, used only as **MOPED warm-start**, not as Bayesian expansion point |
| `bnn_comparison_povertymap` | ResNet18 | Local `train_map` (in notebook) | **No** — has early stopping on val loss; the returned model is not at a loss minimum, so it is not a MAP point |

### Why the label matters / doesn't
- For Laplace: MAP is the expansion point for the Gaussian posterior approximation — the Hessian is computed there. Theoretically requires a true MAP point. Two moons/sinusoidal Laplace uses it correctly.
- For MOPED/BNN: the MAP weights are just a good initializer for posterior means. Calling it "MAP" here is technically correct but misleading — it's functioning as a pretrained network.
- For povertymap: early stopping breaks the MAP property entirely.

### Open decision
The user raised the question of renaming "MAP" to "**baseline NN**" or "**deterministic NN**" across all notebooks and the report. This was not resolved — no changes were made. It is a valid suggestion especially for povertymap and the BNN warm-start use cases.

---

## 4. bayesian-torch Prior — Confirmed Finding

From reading the installed source at:
`/u/halle/carg/home_at/anaconda3/envs/bnn/lib/python3.12/site-packages/bayesian_torch/layers/`

- `prior_weight_mu` and `prior_weight_sigma` are registered via `register_buffer` — **not** `nn.Parameter`
- They are filled with `prior_mean=0.0` and `prior_variance=1.0` at init and **never updated** during training
- Only `mu_weight` and `rho_weight` are `nn.Parameter` (the variational posterior)
- **The prior is completely fixed at N(0,1) in bayesian-torch — there is no prior optimization**

This contrasts with Laplace, where `prior_precision` is tuned via `marglik`, `gridsearch`, or `adam_loop`.

MOPED (`moped_enable=True`, `moped_delta=0.5`) only affects posterior initialization:
- `posterior_mu` ← MAP weights
- `posterior_sigma` ← `|MAP weight| × delta`
It does NOT change the prior buffers.

---

## 5. Sparsity Priors — Clarification

A question arose about why sparsity priors are not used in the Laplace implementations. The textbook (PML Advanced Topics ch.17) was cited:
> "For some applications, it is useful to use sparsity promoting priors, such as the Laplace, which encourage most of the weights to be zero."

**Key distinction** (I initially conflated them):
- **Laplace prior**: the double-exponential distribution p(w) ∝ exp(-λ|w|) — a sparsity-promoting prior
- **Laplace approximation**: the inference method that fits a Gaussian to the posterior at the MAP

They share the name "Laplace" but are unrelated concepts. The textbook statement is correct — sparsity priors CAN be used with BNNs. Combining them with the Laplace **approximation** specifically is harder (non-differentiability at zero, Hessian issues) but possible with modifications (prune then apply LA). `laplace-torch` only exposes Gaussian priors via `prior_precision`.

---

## 6. Dataset and Notebook Coverage

Three datasets:
1. **Two moons** — classification, 2D, synthetic
2. **Sinusoidal** — regression, 1D, synthetic
3. **Povertymap** — regression, multi-dim, real (WILDS benchmark, ResNet18)

Povertymap has **no standalone Laplace or BNN notebook** — everything is in `bnn_comparison_povertymap.ipynb`, which covers both Laplace and BNN in one comparison notebook.

---

## 7. `load_sinusoid` — State (from prior session)

In `shared/datasets.py`, `load_sinusoid` was updated to proper labeled train/val/test-ID/test-OOD splits with single `seed_everything(seed)` call (sequential RNG). This caused gridsearch variants in the sinusoidal notebooks to select very high prior precision (~1555) because the new sequential val set is more structured — this is a known finding, not a bug.

---

## 8. Pending / Unresolved Items

1. **Cell C overconfidence definition**: Decide whether `overconf = conf − correct` (binary, current) is the right metric, or switch to something else (e.g. k-NN local accuracy at test point locations only, not grid).

2. **MAP → rename decision**: Whether to rename the deterministic baseline to "baseline NN" / "pretrained NN" across all notebooks and the report. No changes made yet. Povertymap is the clearest case where "MAP" is wrong.

3. **Axis labels removed "on disk"**: The user referenced a version on disk with axis labels removed from all figures. Verify that the notebook on disk matches (cell A at `43adeaca` may still have axis labels — check).

4. **Cell A misclassification map**: Not deeply reviewed this session. Assumed working from prior session. No changes made to it this session.

5. **Figures for LaTeX**: All figures in `notebooks/figures/` should be clean for LaTeX import — no axis labels, minimal titles. Verify Cell A and C both comply when re-run.

---

## 9. Key File Paths

| File | Purpose |
|---|---|
| `shared/models.py` | TinyMLP definition, `train_map` (shared) |
| `shared/datasets.py` | `load_two_moons`, `load_sinusoid` |
| `notebooks/bayesian/two_moons_bayesian.ipynb` | Main notebook edited this session |
| `notebooks/comparison/bnn_comparison_povertymap.ipynb` | Povertymap — local `train_map`, ResNet18 |
| `notebooks/figures/` | Output figures for LaTeX |
| `submission/template_report/template.tex` | Report — user was editing during session |
| `/u/halle/carg/home_at/anaconda3/envs/bnn/` | Conda env with bayesian-torch installed |
| `Laplace/` | Local dev install of laplace-torch (intentional, do not delete) |
