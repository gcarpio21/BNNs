"""Generate LaTeX table rows from the 5-seed regression metrics CSVs, for pasting
into submission/template_report/template.tex. Keeps report tables in sync with
what the notebooks actually compute, instead of hand-typed numbers going stale.

Usage:
    python gen_report_tables.py bt notebooks/bayesian/results/metrics/sinusoid_bayesian_5seed_metrics.csv
    python gen_report_tables.py laplace notebooks/laplace/results/metrics/sinusoid_laplace_5seed_metrics.csv
"""
import sys
import pandas as pd


def fmt(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def gen_bt_table(csv_path):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        model_type, init = row["Variant"].split(" / ")
        init = init.replace("moped", "MOPED")
        print(f"{model_type} & {init:7s} & {fmt(row['RMSE_mean'], row['RMSE_std'])} & "
              f"{fmt(row['NLL_mean'], row['NLL_std'])} & "
              f"{fmt(row['Calibration_Error_mean'], row['Calibration_Error_std'])} \\\\")


def gen_laplace_table(csv_path):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        hessian, scope, tuning = row["Variant"].split(" / ")
        scope = scope.replace("last layer", "LL")
        print(f"{hessian} & {scope:4s} & {tuning:10s} & "
              f"{fmt(row['NLL_mean'], row['NLL_std'])} & "
              f"{fmt(row['Calibration_Error_mean'], row['Calibration_Error_std'])} \\\\")
    print(f"% RMSE = {fmt(df['RMSE_mean'].iloc[0], df['RMSE_std'].iloc[0])} for all variants")


if __name__ == "__main__":
    which, csv_path = sys.argv[1], sys.argv[2]
    if which == "bt":
        gen_bt_table(csv_path)
    elif which == "laplace":
        gen_laplace_table(csv_path)
    else:
        raise SystemExit(f"unknown table type: {which!r} (expected 'bt' or 'laplace')")
