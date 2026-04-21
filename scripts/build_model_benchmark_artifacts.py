"""
Build manuscript artifacts for model benchmarking:

Generates:
- Figure: R² distribution across models (box plot) -> exports/manuscript_exports/figure_models_r2_distribution.{png,pdf,svg}
- Table: Model metrics with 95% CIs (R², MAE, RMSE) -> exports/manuscript_exports/table_model_metrics_ci.csv and .md

Data source:
- Uses per-split results from the latest output/*_output/evaluation/cv_splits.xlsx
  (this workbook contains one sheet per model with columns: split, R2, MAE, RMSE)

Notes:
- CIs are computed via nonparametric bootstrap on the mean (percentile method).
"""
from __future__ import annotations

import os
from utils.paths import EVALUATION_DIR
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
analysis_root = str(os.environ.get("MLTRAINER_ANALYSIS_ROOT", "") or "").strip()
OUTDIR = os.path.join(analysis_root, "manuscript_exports") if analysis_root else os.path.join(ROOT, "exports", "manuscript_exports")
os.makedirs(OUTDIR, exist_ok=True)


def _find_latest_cv_splits() -> str | None:
    run_root = str(os.environ.get("MLTRAINER_RUN_ROOT", "")).strip()
    if run_root and os.path.exists(run_root):
        p = os.path.join(run_root, EVALUATION_DIR, "cv_splits.xlsx")
        if os.path.isfile(p):
            return p

    base = os.path.join(ROOT, "output")
    candidates = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            if name == "runs":
                runs_dir = os.path.join(base, "runs")
                for run_id in os.listdir(runs_dir):
                    p = os.path.join(runs_dir, run_id, EVALUATION_DIR, "cv_splits.xlsx")
                    if os.path.isfile(p):
                        candidates.append((os.path.getmtime(p), p))
            else:
                p = os.path.join(base, name, EVALUATION_DIR, "cv_splits.xlsx")
                if os.path.isfile(p):
                    candidates.append((os.path.getmtime(p), p))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # dynamic fallback
    import glob
    fallbacks = glob.glob(os.path.join(ROOT, "output", "*_output", EVALUATION_DIR, "cv_splits.xlsx"))
    if fallbacks:
        return sorted(fallbacks, key=os.path.getmtime, reverse=True)[0]
    return None


def _bootstrap_ci_mean(values: np.ndarray, alpha: float = 0.05, n_boot: int = 5000, random_state: int = 42) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap of the mean."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    mean_hat = float(np.mean(vals))
    if vals.size == 1:
        return mean_hat, mean_hat, mean_hat
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    samples = vals[idx]
    boot_means = samples.mean(axis=1)
    lo, hi = np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])
    return mean_hat, float(lo), float(hi)


def build_table_and_figure(cv_splits_path: str) -> Dict[str, str]:
    xls = pd.ExcelFile(cv_splits_path)
    models = xls.sheet_names

    # Collect per-split R2 for figure, and compute bootstrap CIs for table
    rows = []
    r2_long = []  # for figure
    for model in models:
        df = pd.read_excel(xls, sheet_name=model)
        # Normalize expected columns
        cols = {c.lower(): c for c in df.columns}
        r2_col = cols.get("r2")
        mae_col = cols.get("mae")
        rmse_col = cols.get("rmse")
        if r2_col is None or mae_col is None or rmse_col is None:
            # Skip sheets without required metrics
            continue
        r2 = pd.to_numeric(df[r2_col], errors='coerce').dropna().values
        mae = pd.to_numeric(df[mae_col], errors='coerce').dropna().values
        rmse = pd.to_numeric(df[rmse_col], errors='coerce').dropna().values

        # Store for figure
        for v in r2:
            r2_long.append({"model": model, "R2": float(v)})

        # Bootstrap CIs for means
        r2_mean, r2_lo, r2_hi = _bootstrap_ci_mean(r2)
        mae_mean, mae_lo, mae_hi = _bootstrap_ci_mean(mae)
        rmse_mean, rmse_lo, rmse_hi = _bootstrap_ci_mean(rmse)

        rows.append({
            "model": model,
            "n_splits": int(max(len(r2), len(mae), len(rmse))),
            "R2_mean": r2_mean, "R2_ci_low": r2_lo, "R2_ci_high": r2_hi,
            "MAE_mean": mae_mean, "MAE_ci_low": mae_lo, "MAE_ci_high": mae_hi,
            "RMSE_mean": rmse_mean, "RMSE_ci_low": rmse_lo, "RMSE_ci_high": rmse_hi,
        })

    # Create table
    tbl = pd.DataFrame(rows)
    # Sort by R2 descending for readability
    tbl = tbl.sort_values("R2_mean", ascending=False)

    csv_path = os.path.join(OUTDIR, "table_model_metrics_ci.csv")
    md_path = os.path.join(OUTDIR, "table_model_metrics_ci.md")
    tbl.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        # Markdown table header
        f.write("| Model | n | R² (mean [95% CI]) | MAE (mean [95% CI]) | RMSE (mean [95% CI]) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in tbl.iterrows():
            f.write(
                f"| {r['model']} | {int(r['n_splits'])} | "
                f"{r['R2_mean']:.3f} [{r['R2_ci_low']:.3f}, {r['R2_ci_high']:.3f}] | "
                f"{r['MAE_mean']:.3f} [{r['MAE_ci_low']:.3f}, {r['MAE_ci_high']:.3f}] | "
                f"{r['RMSE_mean']:.3f} [{r['RMSE_ci_low']:.3f}, {r['RMSE_ci_high']:.3f}] |\n"
            )

    # Create figure (R2 distribution across models)
    df_long = pd.DataFrame(r2_long)
    plt.figure(figsize=(10, max(5, 0.45 * df_long['model'].nunique())))
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    ax = sns.boxplot(data=df_long, x="R2", y="model", orient="h", showfliers=False, color="#99ccff")
    sns.stripplot(data=df_long, x="R2", y="model", orient="h", color="#1f77b4", size=3, alpha=0.5)
    ax.set_xlabel("R² (per-fold)")
    ax.set_ylabel("")
    ax.set_title("Distribution of R² across models")
    plt.tight_layout()
    fig_path_base = os.path.join(OUTDIR, "figure_models_r2_distribution")
    for ext in (".png", ".pdf", ".svg"):
        plt.savefig(fig_path_base + ext, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "table_csv": csv_path,
        "table_md": md_path,
        "figure_png": fig_path_base + ".png",
        "figure_pdf": fig_path_base + ".pdf",
        "figure_svg": fig_path_base + ".svg",
    }


def main():
    cv_path = _find_latest_cv_splits()
    if cv_path is None or not os.path.isfile(cv_path):
        print("[ERROR] Could not locate cv_splits.xlsx with per-model sheets.")
        return
    print("[INFO] Using cv_splits:", cv_path)
    out = build_table_and_figure(cv_path)
    for k, v in out.items():
        print(f"[OK] {k}: {v}")


if __name__ == "__main__":
    main()
