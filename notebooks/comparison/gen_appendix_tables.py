"""
Generates full per-variant LaTeX tables (best value per column in bold) for
every experiment, to paste into the report's Appendix. Reads the same 5-seed
metrics CSVs used throughout the rest of the pipeline.

Usage: python gen_appendix_tables.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / "shared").exists():
    ROOT = ROOT.parent

TWO_MOONS_MAP_TIME = (9.416, 1.593)
SINUSOID_MAP_TIME  = (59.017, 9.843)
SINUSOID_BIG_MAP_TIME = (13.668, 0.676)


def fmt(mean, std, dp=3):
    return f"${mean:.{dp}f} \\pm {std:.{dp}f}$"


def bold_best(strs, means, mode):
    """mode: 'min' or 'max'. Bolds the best entry via \\bm{} *inside* the
    math delimiters -- \\textbf{$x$} does not actually bold math content,
    since \\textbf only affects the surrounding text font, not the nested
    math font used inside $...$.
    """
    idx = int(np.argmin(means)) if mode == "min" else int(np.argmax(means))
    out = list(strs)
    s = out[idx]
    assert s.startswith("$") and s.endswith("$"), s
    out[idx] = "$\\bm{" + s[1:-1] + "}$"
    return out


def parse_pm(s):
    m = re.match(r"\s*([\d.eE+-]+)\s*\xb1\s*([\d.eE+-]+)\s*", s)
    return float(m.group(1)), float(m.group(2))


def timing_row_fmt(vals):
    return bold_best([f"${v:.2f}$" for v in vals], vals, "min")


def inf_row_fmt_ms(vals_seconds):
    """Inference time gets its own formatter, in milliseconds: at 2 decimal
    places in seconds every variant rounds to 0.00-0.01, hiding real
    per-variant differences that are only visible at millisecond resolution.
    """
    ms = [v * 1000.0 for v in vals_seconds]
    return bold_best([f"${v:.2f}$" for v in ms], ms, "min")


def bt_total_time(df, map_t):
    """Total wall-clock training cost for bayesian-torch: fit time alone for
    scratch (no MAP needed), fit time plus the shared MAP step for MOPED
    (which depends on it as a warm start)."""
    return df["Fit_Time (s)_mean"] + df["Variant"].apply(lambda v: map_t if "moped" in v else 0.0)


def la_total_time(df, map_t):
    """Total wall-clock training cost for laplace-torch: the shared MAP step
    plus fit plus tune time. Every laplace-torch variant requires the MAP
    fit, so it is included for all of them."""
    return map_t + df["Fit_Time (s)_mean"] + df["Tune_Time (s)_mean"]


# ─── Two Moons: bayesian-torch ────────────────────────────────────────────────
def two_moons_bt():
    df = pd.read_csv(ROOT / "notebooks/bayesian/results/metrics/two_moons_bayesian_5seed_metrics.csv")
    map_t, _ = TWO_MOONS_MAP_TIME
    acc  = bold_best([fmt(r.Accuracy_mean, r.Accuracy_std, 4) for r in df.itertuples()], df.Accuracy_mean, "max")
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 4) for r in df.itertuples()], df.NLL_mean, "min")
    bs   = bold_best([fmt(r.Brier_Score_mean, r.Brier_Score_std, 4) for r in df.itertuples()], df.Brier_Score_mean, "min")
    ece  = bold_best([fmt(r.ECE_mean, r.ECE_std, 4) for r in df.itertuples()], df.ECE_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init, loss = v.split(" / ")
        rows.append(f"{typ} & {init} & {loss} & {acc[i]} & {nll[i]} & {bs[i]} & {ece[i]} \\\\")
    print("% ── Two Moons bayesian-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcccc}\n\\toprule")
    print("Type & Init & Loss & Acc & NLL & BS & ECE \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{bayesian-torch} on Two Moons: 5-seed metrics (mean $\\pm$ std).}")
    print("\\label{tab:app:tm_bt}\n\\end{table}\n")

    fit = timing_row_fmt(df["Fit_Time (s)_mean"])
    inf = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(bt_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init, loss = v.split(" / ")
        rows.append(f"{typ} & {init} & {loss} & {fit[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Two Moons bayesian-torch: timing ──")
    print("\\begin{table}[H]\n\\centering")
    print("\\begin{tabular}{lllccc}\n\\toprule")
    print("Type & Init & Loss & Fit (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}")
    print("\\caption{\\texttt{bayesian-torch} on Two Moons: 5-seed timing (mean $\\pm$ std). "
          f"Total time is fit time plus the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) for MOPED variants, fit time alone for scratch.}}")
    print("\\label{tab:app:tm_bt_timing}\n\\end{table}\n")


# ─── Two Moons: laplace-torch ──────────────────────────────────────────────────
def two_moons_la():
    df = pd.read_csv(ROOT / "notebooks/laplace/results/metrics/two_moons_laplace_5seed_metrics.csv")
    map_t, _ = TWO_MOONS_MAP_TIME
    acc_mean, acc_std = df.Accuracy_mean.iloc[0], df.Accuracy_std.iloc[0]
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 4) for r in df.itertuples()], df.NLL_mean, "min")
    bs   = bold_best([fmt(r.Brier_Score_mean, r.Brier_Score_std, 4) for r in df.itertuples()], df.Brier_Score_mean, "min")
    ece  = bold_best([fmt(r.ECE_mean, r.ECE_std, 4) for r in df.itertuples()], df.ECE_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune} & {nll[i]} & {bs[i]} & {ece[i]} \\\\")
    print("% ── Two Moons laplace-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllccc}\n\\toprule")
    print("Hessian & Scope & Tuning & NLL & BS & ECE \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Two Moons: 5-seed metrics (mean $\\pm$ std). "
          f"Accuracy is identical across all variants (${acc_mean:.4f} \\pm {acc_std:.4f}$), since it depends only on the shared MAP mean.}}")
    print("\\label{tab:app:tm_la}\n\\end{table}\n")

    fit  = timing_row_fmt(df["Fit_Time (s)_mean"])
    tune = timing_row_fmt(df["Tune_Time (s)_mean"])
    inf  = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(la_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune_m = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune_m} & {fit[i]} & {tune[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Two Moons laplace-torch: timing ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcccc}\n\\toprule")
    print("Hessian & Scope & Tuning & Fit (s) & Tune (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Two Moons: 5-seed timing (mean $\\pm$ std). "
          f"Total time is the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) plus fit plus tune time, since every variant requires the MAP step.}}")
    print("\\label{tab:app:tm_la_timing}\n\\end{table}\n")


# ─── Sinusoid (small dataset): bayesian-torch ─────────────────────────────────
def sinusoid_small_bt():
    df = pd.read_csv(ROOT / "notebooks/bayesian/results/metrics/sinusoid_bayesian_5seed_metrics.csv")
    map_t, _ = SINUSOID_MAP_TIME
    rmse = bold_best([fmt(r.RMSE_mean, r.RMSE_std, 3) for r in df.itertuples()], df.RMSE_mean, "min")
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 3) for r in df.itertuples()], df.NLL_mean, "min")
    cal  = bold_best([fmt(r.Calibration_Error_mean, r.Calibration_Error_std, 3) for r in df.itertuples()], df.Calibration_Error_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init = v.split(" / ")
        rows.append(f"{typ} & {init} & {rmse[i]} & {nll[i]} & {cal[i]} \\\\")
    print("% ── Sinusoid (small) bayesian-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Type & Init & RMSE & NLL & Cal. Error \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{bayesian-torch} on Sinusoid (small dataset): 5-seed metrics (mean $\\pm$ std).}")
    print("\\label{tab:app:sin_small_bt}\n\\end{table}\n")

    fit = timing_row_fmt(df["Fit_Time (s)_mean"])
    inf = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(bt_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init = v.split(" / ")
        rows.append(f"{typ} & {init} & {fit[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Sinusoid (small) bayesian-torch: timing ──")
    print("\\begin{table}[H]\n\\centering")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Type & Init & Fit (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}")
    print("\\caption{\\texttt{bayesian-torch} on Sinusoid (small dataset): 5-seed timing (mean $\\pm$ std). "
          f"Total time is fit time plus the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) for MOPED variants, fit time alone for scratch.}}")
    print("\\label{tab:app:sin_small_bt_timing}\n\\end{table}\n")


# ─── Sinusoid (small dataset): laplace-torch ──────────────────────────────────
def sinusoid_small_la():
    df = pd.read_csv(ROOT / "notebooks/laplace/results/metrics/sinusoid_laplace_5seed_metrics.csv")
    map_t, _ = SINUSOID_MAP_TIME
    rmse_mean, rmse_std = df.RMSE_mean.iloc[0], df.RMSE_std.iloc[0]
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 3) for r in df.itertuples()], df.NLL_mean, "min")
    cal  = bold_best([fmt(r.Calibration_Error_mean, r.Calibration_Error_std, 3) for r in df.itertuples()], df.Calibration_Error_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune} & {nll[i]} & {cal[i]} \\\\")
    print("% ── Sinusoid (small) laplace-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcc}\n\\toprule")
    print("Hessian & Scope & Tuning & NLL & Cal. Error \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Sinusoid (small dataset): 5-seed metrics (mean $\\pm$ std). "
          f"RMSE is identical across all variants (${rmse_mean:.3f} \\pm {rmse_std:.3f}$), since it depends only on the shared MAP mean.}}")
    print("\\label{tab:app:sin_small_la}\n\\end{table}\n")

    fit  = timing_row_fmt(df["Fit_Time (s)_mean"])
    tune = timing_row_fmt(df["Tune_Time (s)_mean"])
    inf  = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(la_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune_m = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune_m} & {fit[i]} & {tune[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Sinusoid (small) laplace-torch: timing ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcccc}\n\\toprule")
    print("Hessian & Scope & Tuning & Fit (s) & Tune (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Sinusoid (small dataset): 5-seed timing (mean $\\pm$ std). "
          f"Total time is the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) plus fit plus tune time, since every variant requires the MAP step.}}")
    print("\\label{tab:app:sin_small_la_timing}\n\\end{table}\n")


# ─── Sinusoid (big/comparison dataset): bayesian-torch ────────────────────────
def sinusoid_big_bt():
    df = pd.read_csv(ROOT / "notebooks/comparison/results/metrics/sinusoid_comparison_5seed_metrics.csv")
    df = df[df["Method"] == "BT"]
    map_t, _ = SINUSOID_BIG_MAP_TIME
    rmse = bold_best([fmt(r.RMSE_mean, r.RMSE_std, 3) for r in df.itertuples()], df.RMSE_mean, "min")
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 3) for r in df.itertuples()], df.NLL_mean, "min")
    cal  = bold_best([fmt(r.Calibration_Error_mean, r.Calibration_Error_std, 3) for r in df.itertuples()], df.Calibration_Error_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init = v.split(" / ")
        rows.append(f"{typ} & {init} & {rmse[i]} & {nll[i]} & {cal[i]} \\\\")
    print("% ── Sinusoid (big/comparison) bayesian-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Type & Init & RMSE & NLL & Cal. Error \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{bayesian-torch} on Sinusoid (comparison dataset), in-distribution: 5-seed metrics (mean $\\pm$ std).}")
    print("\\label{tab:app:sin_big_bt}\n\\end{table}\n")

    fit = timing_row_fmt(df["Fit_Time (s)_mean"])
    inf = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(bt_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        typ, init = v.split(" / ")
        rows.append(f"{typ} & {init} & {fit[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Sinusoid (big/comparison) bayesian-torch: timing ──")
    print("\\begin{table}[H]\n\\centering")
    print("\\begin{tabular}{llccc}\n\\toprule")
    print("Type & Init & Fit (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}")
    print("\\caption{\\texttt{bayesian-torch} on Sinusoid (comparison dataset): 5-seed timing (mean $\\pm$ std). "
          f"Total time is fit time plus the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) for MOPED variants, fit time alone for scratch.}}")
    print("\\label{tab:app:sin_big_bt_timing}\n\\end{table}\n")


# ─── Sinusoid (big/comparison dataset): laplace-torch ─────────────────────────
def sinusoid_big_la():
    df = pd.read_csv(ROOT / "notebooks/comparison/results/metrics/sinusoid_comparison_5seed_metrics.csv")
    df = df[df["Method"] == "Laplace"]
    map_t, _ = SINUSOID_BIG_MAP_TIME
    rmse_mean, rmse_std = df.RMSE_mean.iloc[0], df.RMSE_std.iloc[0]
    nll  = bold_best([fmt(r.NLL_mean, r.NLL_std, 3) for r in df.itertuples()], df.NLL_mean, "min")
    cal  = bold_best([fmt(r.Calibration_Error_mean, r.Calibration_Error_std, 3) for r in df.itertuples()], df.Calibration_Error_mean, "min")
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune} & {nll[i]} & {cal[i]} \\\\")
    print("% ── Sinusoid (big/comparison) laplace-torch: metrics ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcc}\n\\toprule")
    print("Hessian & Scope & Tuning & NLL & Cal. Error \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Sinusoid (comparison dataset), in-distribution: 5-seed metrics (mean $\\pm$ std). "
          f"RMSE is identical across all variants (${rmse_mean:.3f} \\pm {rmse_std:.3f}$), since it depends only on the shared MAP mean.}}")
    print("\\label{tab:app:sin_big_la}\n\\end{table}\n")

    fit  = timing_row_fmt(df["Fit_Time (s)_mean"])
    tune = timing_row_fmt(df["Tune_Time (s)_mean"])
    inf  = inf_row_fmt_ms(df["Inf_Time (s)_mean"])
    total = timing_row_fmt(la_total_time(df, map_t))
    rows = []
    for i, v in enumerate(df.Variant):
        hessian, scope, tune_m = v.split(" / ")
        rows.append(f"{hessian} & {scope} & {tune_m} & {fit[i]} & {tune[i]} & {inf[i]} & {total[i]} \\\\")
    print("% ── Sinusoid (big/comparison) laplace-torch: timing ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lllcccc}\n\\toprule")
    print("Hessian & Scope & Tuning & Fit (s) & Tune (s) & Inf (ms) & Total (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{laplace-torch} on Sinusoid (comparison dataset): 5-seed timing (mean $\\pm$ std). "
          f"Total time is the shared MAP step (${map_t:.1f}\\,\\text{{s}}$) plus fit plus tune time, since every variant requires the MAP step.}}")
    print("\\label{tab:app:sin_big_la_timing}\n\\end{table}\n")


# ─── Sinusoid (small dataset, ID vs OOD): both methods ────────────────────────
def sinusoid_small_ood():
    bt_df = pd.read_csv(ROOT / "notebooks/bayesian/results/metrics/sinusoid_bayesian_5seed_ood_metrics.csv")
    bt_df.insert(0, "Method", "BT")
    la_df = pd.read_csv(ROOT / "notebooks/laplace/results/metrics/sinusoid_laplace_5seed_ood_metrics.csv")
    la_df.insert(0, "Method", "Laplace")
    df = pd.concat([bt_df, la_df], ignore_index=True)
    cols = ["ID RMSE", "OOD RMSE", "ID NLL", "OOD NLL", "ID Epi", "OOD Epi", "Epi Ratio"]
    parsed = {c: [parse_pm(x) for x in df[c]] for c in cols}
    means = {c: [m for m, s in parsed[c]] for c in cols}
    fmted = {c: bold_best([fmt(m, s, 3) for m, s in parsed[c]], means[c], "min") for c in cols}
    rows = []
    for i, (v, method) in enumerate(zip(df.Variant, df.Method)):
        row = " & ".join(fmted[c][i] for c in cols)
        rows.append(f"{method} & {v} & {row} \\\\")
    print("% ── Sinusoid (small dataset) both methods, ID vs OOD ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccccccc}\n\\toprule")
    print("Method & Variant & " + " & ".join(cols) + " \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{Both methods on Sinusoid (small dataset), ID vs OOD: 5-seed metrics (mean $\\pm$ std).}")
    print("\\label{tab:app:sin_small_ood}\n\\end{table}\n")


# ─── Sinusoid (big/comparison dataset, ID vs OOD): both methods ──────────────
def sinusoid_big_ood():
    df = pd.read_csv(ROOT / "notebooks/comparison/results/metrics/sinusoid_comparison_5seed_ood_metrics.csv")
    cols = ["ID RMSE", "OOD RMSE", "ID NLL", "OOD NLL", "ID Epi", "OOD Epi", "Epi Ratio"]
    parsed = {c: [parse_pm(x) for x in df[c]] for c in cols}
    means = {c: [m for m, s in parsed[c]] for c in cols}
    fmted = {c: bold_best([fmt(m, s, 3) for m, s in parsed[c]], means[c], "min") for c in cols}
    rows = []
    for i, (v, method) in enumerate(zip(df.Variant, df.Method)):
        row = " & ".join(fmted[c][i] for c in cols)
        rows.append(f"{method} & {v} & {row} \\\\")
    print("% ── Sinusoid (big/comparison) both methods, ID vs OOD ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccccccc}\n\\toprule")
    print("Method & Variant & " + " & ".join(cols) + " \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{Both methods on Sinusoid (comparison dataset), ID vs OOD: 5-seed metrics (mean $\\pm$ std).}")
    print("\\label{tab:app:sin_big_ood}\n\\end{table}\n")


# ─── Sinusoid (big/comparison dataset): MOPED delta sweep ─────────────────────
def sinusoid_delta_sweep():
    df = pd.read_csv(ROOT / "notebooks/comparison/results/metrics/sinusoid_delta_sweep_5seed_metrics.csv")
    cols = ["ID RMSE", "ID NLL", "ID Cal Error", "OOD RMSE", "OOD NLL", "Epi Ratio"]
    parsed = {c: [parse_pm(x) for x in df[c]] for c in cols}
    means = {c: [m for m, s in parsed[c]] for c in cols}
    fmted = {c: bold_best([fmt(m, s, 3) for m, s in parsed[c]], means[c], "min") for c in cols}

    def fix_variant(v):
        return v.replace("δ=", "$\\delta=$")

    rows = []
    for i, v in enumerate(df.Variant):
        row = " & ".join(fmted[c][i] for c in cols)
        rows.append(f"{fix_variant(v)} & {row} \\\\")
    print("% ── Sinusoid (big/comparison) MOPED delta sweep ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lcccccc}\n\\toprule")
    print("Variant & " + " & ".join(cols) + " \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{\\texttt{bayesian-torch} MOPED $\\delta$ sweep on Sinusoid (comparison dataset): 5-seed metrics (mean $\\pm$ std). "
          "Scratch rows repeated for reference; the report's headline MOPED variants use $\\delta=0.5$.}")
    print("\\label{tab:app:sin_delta_sweep}\n\\end{table}\n")


# ─── PovertyMap: both methods, ID vs OOD ──────────────────────────────────────
def povertymap():
    df = pd.read_csv(ROOT / "notebooks/comparison/povertymap_bnn_comparison_metrics.csv")
    df = df.rename(columns={df.columns[0]: "Method"})
    df = df[df["Method"].str.lower() != "map"]
    cols_mean = ["ID_RMSE_mean", "OOD_RMSE_mean", "ID_NLL_mean", "OOD_NLL_mean", "ID_Cal_Error_mean", "OOD_Cal_Error_mean"]
    labels = ["ID RMSE", "OOD RMSE", "ID NLL", "OOD NLL", "ID Cal. Err.", "OOD Cal. Err."]
    fmted = {}
    for c in cols_mean:
        vals = df[c].to_numpy(dtype=float)
        strs = [f"${v:.4f}$" for v in vals]
        fmted[c] = bold_best(strs, vals, "min")
    time_cols = ["Mean_Train_Time (s)", "Mean_Inf_Time_ID (s)", "Mean_Inf_Time_OOD (s)"]
    time_fmt = {}
    for c in time_cols:
        vals = df[c].to_numpy(dtype=float)
        strs = [f"${v:.2f}$" for v in vals]
        time_fmt[c] = bold_best(strs, vals, "min")
    rows = []
    for i, method in enumerate(df["Method"]):
        row = " & ".join(fmted[c][i] for c in cols_mean) + " & " + " & ".join(time_fmt[c][i] for c in time_cols)
        rows.append(f"{method} & {row} \\\\")
    print("% ── PovertyMap both methods, ID vs OOD ──")
    print("\\begin{table}[H]\n\\centering\n\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{lccccccccc}\n\\toprule")
    print("Method & " + " & ".join(labels) + " & Train (s) & Inf ID (s) & Inf OOD (s) \\\\\n\\midrule")
    print("\n".join(rows))
    print("\\bottomrule\n\\end{tabular}}")
    print("\\caption{Both methods on PovertyMap, ID vs OOD: 5-fold metrics.}")
    print("\\label{tab:app:povertymap}\n\\end{table}\n")


if __name__ == "__main__":
    two_moons_bt()
    two_moons_la()
    sinusoid_small_bt()
    sinusoid_small_la()
    sinusoid_small_ood()
    sinusoid_big_bt()
    sinusoid_big_la()
    sinusoid_big_ood()
    sinusoid_delta_sweep()
    povertymap()
