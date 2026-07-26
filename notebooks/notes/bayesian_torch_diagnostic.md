# Bayesian-Torch Diagnostic — KL Scaling & Prior Precision

## 1. What was wrong in PovertyMap originally

`train_bayesian_torch` in `bnn_comparison_povertymap.ipynb` had:

```python
loss = mse + 0.1 * kl               # flat constant, no normalization at all
"prior_sigma": (1.0 / 5e-4) ** 0.5  # hardcoded, borrowed from Laplace Redux's MNIST/CIFAR baseline
```

plus a checkpoint-loading bug (`dnn_to_bnn()` missing before `load_state_dict()`, causing a
key-mismatch crash) and an epoch count (25) far short of what any of the borrowed hyperparameters
were tuned against (fixed to 1000, matching `flipout_scratch` elsewhere in the repo).

## 2. Where `0.1` and `5e-4` actually came from — traced to the source, not assumed

Both numbers were lifted from Laplace Redux's paper (Daxberger et al. 2021), Appendix C.2.1,
describing their "VB" baseline for the **MNIST/CIFAR-10** predictive-uncertainty comparison
(a different dataset/architecture than PovertyMap):

> "The prior precision is set to 5×10⁻⁴ to match the MAP-trained network, while the KL-term
> downscaling factor is set to 0.1, following [13]."

Reference [13] is Osawa et al., "Practical Deep Learning with Bayesian Principles" (NeurIPS 2019)
— the paper introducing **VOGN** (Variational Online Gauss-Newton), a *different inference
algorithm* than the Flipout-based mean-field VI actually used in Redux's "VB" baseline (and in
this repo). Fetching the actual paper (arXiv 1906.02506) showed the τ=0.1 is not meant as a fixed
constant at all — it's the *starting value of a warm-up schedule* that anneals τ from 0.1 up to
1.0 over training.

Laplace Redux's own reference [13] is also cited elsewhere in the *same paper* (Appendix C.2.1,
"Other baselines") to justify **excluding** VOGN as an active baseline in this same section: "A
recent VI method called Variational Online Gauss-Newton (VOGN) [13] also seems to underperform.
For example, Fig. 5 of Osawa et al. [13] shows that on OOD detection with CIFAR-10 vs. SVHN,
MC-dropout and VOGN only achieve AUROC values of 81.9 and 80.0" — i.e. they cite Osawa's own
reported numbers, they don't re-run VOGN themselves, in this section. Separately (Section 4.4,
continual learning, Figure 7), Redux *does* run VOGN as a live, plotted baseline ("VB(VOGN)" in
the figure legend, "closely followed by VOGN" in the text) — so the authors clearly know the
method firsthand; it's specifically in the MNIST/CIFAR predictive-uncertainty section that they
borrow just the τ=0.1 number from a different algorithm's paper into their own, unrelated Flipout
baseline, without re-deriving whether it transfers.

## 3. Checking Laplace Redux's *actual code*, not just the paper's prose

`baselines/bbb/train.py` (their real BBB/VB training script, on GitHub) does NOT match the
paper's simplified prose. The prose says "prior precision is set to 5×10⁻⁴"; the code computes:

```python
parser.add_argument('--var0', type=float, default=1,
                    help='Gaussian prior variance. If None, it will be computed to emulate weight decay')
...
if args.var0 is None:
    args.var0 = 1/(5e-4*len(train_loader.dataset))   # var0 is prior VARIANCE
```

i.e. `prior_precision = 1/var0 = weight_decay * N` — not `weight_decay` alone. The paper's prose
dropped the `*N` factor; the shipped code includes it. The loss itself:

```python
loss = F.cross_entropy(out.squeeze(), y) + args.tau/num_data*kl
```

confirms both the `tau` (fixed, not annealed, in the shipped code — `--tau` is a plain argparse
float with no scheduler object anywhere in the file) and the `/num_data` (= N) scaling.

## 4. Why `weight_decay * N` (not `weight_decay` alone) is required, algebraically

The equivalence "prior_precision = weight_decay" is real and standard — L2 regularization
(weight decay λ) during MAP training is mathematically the MAP estimate under a Gaussian prior
of precision λ (`-log p(θ) = (λ/2)‖θ‖² + const`). Laplace Redux states this explicitly in its
own derivation (main text, Section 2): *"the widely used weight regularizer r(θ) = (γ⁻²/2)‖θ‖²
(a.k.a. weight decay) corresponds to a centered Gaussian prior p(θ) = N(θ;0,γ²I)"*.

This transfers cleanly to the **Laplace approximation** (a Gaussian built directly around a MAP
point). For **mean-field variational inference** trained via minibatches, it needs a correction.
Gaussian KL divergence per weight:

`KL = log(σ_p/σ_q) + (σ_q² + μ_q²)/(2σ_p²) − 0.5`

With `precision = weight_decay·N` (so `σ_p² = 1/(weight_decay·N)`), substituting into `(τ/N)·KL`
and expanding:

```
(τ/N)·KL = [τ·weight_decay/2 · (σ_q² + μ_q²)]     <- N cancels; behaves like weight decay
         + (τ/N)·[−log(σ_q) − 0.5·log(weight_decay·N) − 0.5]   <- N does NOT cancel
```

The first term is the weight-decay-equivalent penalty on the posterior mean/variance — this is
*why* the `*N` scaling on the prior is needed. Without it, dividing the loss by N with
`precision=weight_decay` alone leaves an effectively-zero regularization strength
(`weight_decay/N`). The second term contains `−log(σ_q)`, the **entropy term** that resists
posterior collapse into overconfidence — it keeps an explicit `1/N` that never cancels, and
gets weaker as N grows. This is a known, real effect in mean-field VI on large datasets and
plausibly a real contributor to the original miscalibration (NLL 5-8x worse than MAP/Laplace,
alongside comparable RMSE — a pattern consistent with badly overconfident predicted variance
rather than bad point predictions).

The same derivation redone with `batch_size` (M) instead of N shows the cancellation holds
symmetrically — `prior_precision` must be scaled by *whatever* the loss divides KL by:

```
(τ/M)·KL = [τ·weight_decay/2 · (σ_q² + μ_q²)] + (τ/M)·[−log(σ_q) − 0.5·log(weight_decay·M) − 0.5]
```

Since `batch_size << N` generally, using batch_size scaling keeps the entropy-preserving second
term relatively *stronger* than N-based scaling would — plausibly working better against the
overconfidence failure mode independent of "it's the official library convention" alone.

## 5. Comparing against `bayesian_torch`'s own official conventions — a real, unresolved tension

Researching the actual `IntelLabs/bayesian-torch` GitHub repo (README, `dnn_to_bnn.py`, every
`examples/main_bayesian_*.py` script) shows the library's own shipped convention is *different*
from what Redux's separate custom script does:

- **KL scaling: `kl / batch_size`, always.** Every official example (CIFAR, ImageNet, MNIST,
  `dnn_to_bnn`-based or manually-constructed layers) does this identically:
  ```python
  cross_entropy_loss = criterion(output, target_var)
  scaled_kl = kl / args.batch_size
  loss = cross_entropy_loss + scaled_kl
  ```
  Two of the scripts even have a *commented-out* dataset-size alternative
  (`# scaled_kl = kl / len_trainset`), showing the authors tried N-based scaling and shipped
  batch_size scaling instead.
- **`prior_sigma = 1.0`, always, a flat literal.** README, the `dnn_to_bnn.py` module docstring,
  and every example script. No guidance anywhere ties it to dataset size or weight decay.
- **No tau/tempering of any kind.** The MOPED paper (Krishnan et al., AAAI 2020 — the paper
  behind this library) states plainly: *"we use... full-scale KL-divergence term in ELBO"*,
  contrasting this with (not rejecting, just distinguishing from) Sønderby et al.'s separate
  KL-annealing/warm-up technique for VAEs.
- **MOPED (informed-prior init from a pretrained model) is explicitly recommended for large
  models specifically because mean-field VI struggles to converge from random init at scale**
  — directly relevant to PovertyMap's ResNet-18, though not adopted here per explicit preference.

**Decision made this session:** implement `prior_precision = weight_decay * batch_size`,
`loss = NLL + get_kl_loss(model)/batch_size`, no tau, as an additional comparison variant in
PovertyMap — matching `bayesian_torch`'s own library convention exactly. The two existing
N-based variants (fixed-tau=0.1 and annealed-tau) stay in place for comparison.

## 6. Repo-wide inventory of every Bayesian-Torch usage

| File | KL scaling | prior_sigma | Notes |
|---|---|---|---|
| `bnn_comparison_povertymap.ipynb` (`train_bayesian_torch`) | `(tau/N)*kl`, tau=0.1 fixed | `weight_decay*N` derived | Redux-style; `WEIGHT_DECAY` now a single notebook-global constant, also used by `train_map`'s optimizer |
| `bnn_comparison_povertymap.ipynb` (`train_bayesian_torch_tau_schedule`) | `(tau/N)*kl`, tau annealed 0.1→1.0 | same | Osawa-style schedule |
| `bnn_comparison_povertymap.ipynb` (`train_bayesian_torch_batch_size`) | `(tau/batch_size)*kl`, tau=0.1 fixed | `weight_decay*batch_size` derived | New variant this session — isolates N vs. batch_size, holding tau fixed (not yet the "pure" official-convention no-tau version) |
| `gen_povertymap_poster_figures.py` | N/A (inference-only) | still hardcoded `(1/5e-4)**0.5` | **Stale** — not updated to match the notebook's fix. Low risk (only affects `dnn_to_bnn` layer *construction*, not the loaded weights) but should be reconciled |
| `bnn_comparison_sinusoid.ipynb` / `_5seed.ipynb` | `get_kl_loss(net)/num_samp` (= kl/N, implicit tau=1) | flat `1.0` | No weight_decay in MAP training here, so nothing to match; internally consistent but doesn't use the weight-decay-equivalence idea at all |
| `sinusoidal_bayesian.ipynb` / `_5seed.ipynb` | same `kl/num_samples` | flat `1.0` | Likely the original source the comparison notebooks were adapted from |
| `two_moons_bayesian.ipynb` / `_5seed.ipynb` | same `kl/num_samples` (+ optional AvUC term) | flat `1.0` | Epoch counts (100/300, moped/scratch) have no justification comment, unlike the regression notebooks' commented split (`400 if moped else 1000`) |
| `gen_sinusoid_poster_figures.py`, `gen_two_moons_poster_figures.py`, `measure_timing.py` | N/A (inference-only) | flat `1.0`, consistent with their training notebooks | Correct `dnn_to_bnn`-before-`load_state_dict` order |
| `report/code/bayesian_code.py` | `kl/N` | `1.0` | Report excerpt matching two-moons notebook, not independently executed |

## 7. Outstanding action items

1. ~~Add a third Bayesian-Torch comparison variant to `bnn_comparison_povertymap.ipynb`~~ — done
   this session (`train_bayesian_torch_batch_size`), though it keeps `tau=0.1` fixed rather than
   dropping it entirely; a "pure" no-tau official-convention variant is not yet a separate cell.
2. Reconcile `gen_povertymap_poster_figures.py`'s stale `prior_sigma=(1/5e-4)**0.5`.
3. Bring sinusoid/two-moons notebooks onto the `bayesian_torch`-official convention
   (`prior_precision = weight_decay * batch_size`, `kl/batch_size`, no tau) — requires first
   deciding whether to introduce `weight_decay` into their MAP training (currently absent), since
   without it there's nothing for the prior to match.
