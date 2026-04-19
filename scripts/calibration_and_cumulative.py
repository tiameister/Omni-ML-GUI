"""
Run-scoped calibration summaries and cumulative feature-importance reports.

Preferred runtime (from GUI optional analysis runner):
- MLTRAINER_RUN_ROOT=<.../output/runs/<run_id>>
- MLTRAINER_ANALYSIS_ROOT=<.../output/runs/<run_id>/analysis>

If environment variables are missing, this script tries to locate the latest
run folder under output/runs.
"""

from __future__ import annotations

import glob
import json
import math
import os
from utils.paths import EVALUATION_DIR
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.logger import get_logger

LOGGER = get_logger(__name__)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _norm_token(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def resolve_run_root() -> str:
    env_root = str(os.environ.get("MLTRAINER_RUN_ROOT", "") or "").strip()
    if env_root and os.path.isdir(env_root):
        return env_root

    runs_root = os.path.join(REPO_ROOT, "output", "runs")
    if not os.path.isdir(runs_root):
        raise SystemExit("No run root available (missing MLTRAINER_RUN_ROOT and output/runs).")

    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(runs_root):
        path = os.path.join(runs_root, name)
        if os.path.isdir(path):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise SystemExit("No run folders found under output/runs.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_analysis_root(run_root: str) -> str:
    env_root = str(os.environ.get("MLTRAINER_ANALYSIS_ROOT", "") or "").strip()
    if env_root:
        return env_root
    return os.path.join(run_root, EVALUATION_DIR)


def linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Fit y_true = a + b * y_pred and return (a, b)."""
    X = np.vstack([np.ones_like(y_pred), y_pred]).T
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    return float(beta[0]), float(beta[1])


def r2_score_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def binned_errors(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred})
    df["bin"], edges = pd.qcut(df["yhat"], q=bins, retbins=True, labels=False, duplicates="drop")

    rows = []
    for b, g in df.groupby("bin"):
        if g.empty:
            continue
        err = g["y"].to_numpy() - g["yhat"].to_numpy()
        rows.append(
            {
                "bin": int(b),
                "n": int(len(g)),
                "y_mean": float(g["y"].mean()),
                "yhat_mean": float(g["yhat"].mean()),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(math.sqrt(np.mean(err ** 2))),
                "bin_left": float(edges[int(b)]),
                "bin_right": float(edges[int(b) + 1]),
            }
        )
    return pd.DataFrame(rows)


def _strip_training_prefix(name: str) -> str:
    out = str(name)
    prefix = "feature_engineering_"
    if out.startswith(prefix):
        out = out[len(prefix):]
    return out


def find_prediction_files(run_root: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(run_root, "models", "*", EVALUATION_DIR, "*_predictions_vs_actual.xlsx")
    by_model: Dict[str, Tuple[str, str]] = {}
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        suffix = "_predictions_vs_actual.xlsx"
        if not fname.endswith(suffix):
            continue
        model_name = _strip_training_prefix(fname[: -len(suffix)])
        model_key = _norm_token(model_name)
        existing = by_model.get(model_key)
        if existing is None or os.path.getmtime(path) > os.path.getmtime(existing[1]):
            by_model[model_key] = (model_name, path)
    return sorted(by_model.values(), key=lambda x: x[0].lower())


def load_metrics_df(run_root: str) -> pd.DataFrame | None:
    metrics_path = os.path.join(run_root, EVALUATION_DIR, "metrics.xlsx")
    if not os.path.exists(metrics_path):
        return None
    try:
        return pd.read_excel(metrics_path, sheet_name=0)
    except Exception:
        return None


def match_metrics_row(metrics_df: pd.DataFrame | None, model_name: str) -> pd.Series | None:
    if metrics_df is None or metrics_df.empty or "model" not in metrics_df.columns:
        return None
    model_key = _norm_token(model_name)
    if not model_key:
        return None

    exact = metrics_df.loc[metrics_df["model"].astype(str).map(lambda x: _norm_token(x) == model_key)]
    if not exact.empty:
        return exact.iloc[0]

    contains = metrics_df.loc[
        metrics_df["model"].astype(str).map(lambda x: model_key in _norm_token(x) or _norm_token(x) in model_key)
    ]
    if not contains.empty:
        return contains.iloc[0]
    return None


def calibration_workflow(run_root: str, analysis_root: str) -> None:
    out_tab_dir = os.path.join(analysis_root, "calibration")
    out_fig_dir = os.path.join(analysis_root, "figures")
    ensure_dir(out_tab_dir)
    ensure_dir(out_fig_dir)

    pred_files = find_prediction_files(run_root)
    if not pred_files:
        print("[WARN] No predictions_vs_actual files found for calibration.")
        return

    metrics_df = load_metrics_df(run_root)
    n = len(pred_files)
    ncols = 1 if n <= 1 else 2
    nrows = max(1, int(math.ceil(n / float(ncols))))
    fig_w = 7.4 * ncols
    fig_h = 5.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    cal_rows: List[dict] = []
    binned_all: List[pd.DataFrame] = []

    used_axes = 0
    for model_name, pred_path in pred_files:
        try:
            df = pd.read_excel(pred_path)
        except Exception as e:
            print(f"[WARN] Could not read predictions file for {model_name}: {e}")
            continue
        if not {"actual", "predicted"}.issubset(df.columns):
            print(f"[WARN] Missing required columns in {pred_path}")
            continue

        y = pd.to_numeric(df["actual"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(df["predicted"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(yhat)
        y = y[mask]
        yhat = yhat[mask]
        if len(y) < 5:
            print(f"[WARN] Not enough finite rows for calibration in {model_name}")
            continue

        intercept, slope = linear_calibration(y, yhat)
        r2_pred = r2_score_simple(y, yhat)

        row = {
            "model": model_name,
            "intercept": intercept,
            "slope": slope,
            "R2_pred": r2_pred,
        }
        metrics_row = match_metrics_row(metrics_df, model_name)
        if metrics_row is not None:
            r2_cv = metrics_row.get("R2_CV", np.nan)
            r2_train = metrics_row.get("R2_train", np.nan)
            row["R2_CV"] = float(r2_cv) if pd.notna(r2_cv) else np.nan
            row["R2_train"] = float(r2_train) if pd.notna(r2_train) else np.nan
            if pd.notna(row["R2_train"]) and pd.notna(row["R2_CV"]):
                row["delta_R2_train_minus_CV"] = float(row["R2_train"] - row["R2_CV"])
        cal_rows.append(row)

        bins_df = binned_errors(y, yhat, bins=10)
        if not bins_df.empty:
            bins_df.insert(0, "model", model_name)
            binned_all.append(bins_df)

        if used_axes < len(axes):
            ax = axes[used_axes]
            used_axes += 1
            ax.axis("on")
            ax.scatter(y, yhat, s=20, alpha=0.55, edgecolor="none", color="#4C78A8")

            combined = np.concatenate([y, yhat])
            try:
                lim_lo = float(np.nanpercentile(combined, 1.0))
                lim_hi = float(np.nanpercentile(combined, 99.0))
            except Exception:
                lim_lo = float(np.nanmin(combined))
                lim_hi = float(np.nanmax(combined))
            if not np.isfinite(lim_lo) or not np.isfinite(lim_hi) or lim_hi <= lim_lo:
                lim_lo = float(np.nanmin(combined))
                lim_hi = float(np.nanmax(combined))
            span = max(lim_hi - lim_lo, 1e-6)
            pad = 0.08 * span
            lim_lo -= pad
            lim_hi += pad

            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=1, label="1:1")
            if abs(slope) > 1e-8:
                x_line = np.linspace(lim_lo, lim_hi, 100)
                y_line = (x_line - intercept) / slope
                ax.plot(x_line, y_line, color="#2b8cbe", lw=1.5, label=f"calib: slope={slope:.3f}")

            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(f"Calibration - {model_name}", fontsize=11, pad=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
            ax.text(
                0.03,
                0.97,
                f"R²={r2_pred:.3f}\nSlope={slope:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
            )

    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(os.path.join(out_tab_dir, "calibration_summary.csv"), index=False)
    if binned_all:
        pd.concat(binned_all, ignore_index=True).to_csv(os.path.join(out_tab_dir, "binned_metrics.csv"), index=False)

    if used_axes > 0:
        fig.suptitle("Calibration Diagnostics", fontsize=14, y=0.99)
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.91, wspace=0.22, hspace=0.30)
        base = os.path.join(out_fig_dir, "calibration_models")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".svg", bbox_inches="tight")
        print(f"[OK] Saved calibration figure: {base}.png/.pdf/.svg")
    plt.close(fig)


def find_importance_files(run_root: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(run_root, "models", "*", "3_Manuscript_Figures", "*feature_importance.xlsx")
    files: List[Tuple[str, str]] = []
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        suffix = "_feature_importance.xlsx"
        if not fname.endswith(suffix):
            continue
        model_name = _strip_training_prefix(fname[: -len(suffix)])
        files.append((model_name, path))
    files.sort(key=lambda x: x[0].lower())
    return files


def load_importance_series(path: str) -> pd.Series:
    df = pd.read_excel(path, sheet_name=0)
    if df.empty or len(df.columns) < 2:
        return pd.Series(dtype=float)

    feature_col = None
    for col in df.columns:
        if str(col).strip().lower() == "feature":
            feature_col = col
            break
    if feature_col is None:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                feature_col = col
                break
    if feature_col is None:
        feature_col = df.columns[0]

    numeric_candidates = [
        col for col in df.columns if col != feature_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if numeric_candidates:
        importance_col = numeric_candidates[0]
    else:
        importance_col = None
        for col in df.columns:
            if col == feature_col:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                df[col] = coerced
                importance_col = col
                break
        if importance_col is None:
            return pd.Series(dtype=float)

    feat = df[feature_col].astype(str).str.strip()
    imp = pd.to_numeric(df[importance_col], errors="coerce")
    out = pd.Series(imp.values, index=feat).dropna()
    out = out[out.index != ""]
    if out.empty:
        return out
    return out.groupby(level=0).mean()


def resolve_best_model(run_root: str, metrics_df: pd.DataFrame | None) -> str:
    manifest_path = os.path.join(run_root, "run_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            best_model = str(manifest.get("best_model", "")).strip()
            if best_model:
                return best_model
        except Exception as e:
            LOGGER.exception("Failed reading run manifest: %s", manifest_path)
    if metrics_df is not None and not metrics_df.empty and "model" in metrics_df.columns:
        return str(metrics_df.iloc[0].get("model") or "").strip()
    return ""


def load_shap_means_for_model(run_root: str, model_name: str) -> Dict[str, float]:
    if not model_name:
        return {}
    key = _norm_token(model_name)
    pattern = os.path.join(run_root, "models", "*", "3_Manuscript_Figures", "*shap_values_top*.xlsx")
    candidates = []
    for path in glob.glob(pattern):
        if key and key not in _norm_token(os.path.basename(path)):
            continue
        candidates.append(path)
    if not candidates:
        return {}
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    try:
        sdf = pd.read_excel(candidates[0], sheet_name=0)
    except Exception:
        return {}

    means: Dict[str, float] = {}
    for col in sdf.columns:
        vals = pd.to_numeric(sdf[col], errors="coerce").dropna()
        if vals.empty:
            continue
        means[str(col)] = float(vals.mean())
    return means


def find_shap_mean(shap_means: Dict[str, float], feature: str) -> float | None:
    if feature in shap_means:
        return shap_means[feature]
    fkey = _norm_token(feature)
    for name, value in shap_means.items():
        if _norm_token(name) == fkey:
            return value
    return None


def cumulative_importance_workflow(run_root: str, analysis_root: str) -> None:
    cum_dir = os.path.join(analysis_root, "cumulative")
    fig_dir = os.path.join(analysis_root, "figures")
    ensure_dir(cum_dir)
    ensure_dir(fig_dir)

    imp_files = find_importance_files(run_root)
    if not imp_files:
        print("[WARN] No feature importance Excel files found for cumulative analysis.")
        return

    norm_series: Dict[str, pd.Series] = {}
    for model_name, path in imp_files:
        try:
            series = load_importance_series(path)
        except Exception as e:
            print(f"[WARN] Could not parse feature importance for {model_name}: {e}")
            continue
        if series.empty:
            continue
        series = series[series > 0]
        total = float(series.sum())
        if total <= 0:
            continue
        norm_series[model_name] = series / total

    if not norm_series:
        print("[WARN] No valid feature importance values found.")
        return

    combined = pd.concat(norm_series, axis=1).fillna(0.0)
    combined_mean = combined.mean(axis=1).sort_values(ascending=False)
    total_combined = float(combined_mean.sum())
    if total_combined <= 0:
        total_combined = 1.0

    df = pd.DataFrame(
        {
            "feature": combined_mean.index,
            "combined_mean": combined_mean.values,
        }
    )
    df["share"] = df["combined_mean"] / total_combined
    df["cum_share"] = df["share"].cumsum()
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(os.path.join(cum_dir, "cumulative_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["rank"], df["cum_share"], marker="o", lw=1.5)
    ax.set_xlabel("Top-N features")
    ax.set_ylabel("Cumulative normalized importance")
    ax.set_title("Cumulative importance across available models")
    ax.grid(True, alpha=0.3)
    try:
        n80 = int(np.searchsorted(df["cum_share"].values, 0.8) + 1)
        if n80 <= len(df):
            y80 = float(df.loc[df["rank"] == n80, "cum_share"].iloc[0])
            ax.axhline(0.8, color="red", ls="--", lw=1)
            ax.axvline(n80, color="red", ls=":", lw=1)
            ax.annotate(
                f"80% at N={n80}",
                xy=(n80, y80),
                xytext=(n80 + 1, min(0.95, y80 + 0.05)),
                arrowprops=dict(arrowstyle="->", color="red"),
            )
    except Exception as e:
        LOGGER.exception("Failed annotating cumulative importance plot")
    base = os.path.join(fig_dir, "cumulative_importance")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved cumulative figure: {base}.png/.pdf/.svg")

    metrics_df = load_metrics_df(run_root)
    best_model = resolve_best_model(run_root, metrics_df)
    shap_means = load_shap_means_for_model(run_root, best_model)

    top10 = df.head(10).copy()
    directions: List[str] = []
    notes: List[str] = []
    for feat in top10["feature"].astype(str):
        mean_val = find_shap_mean(shap_means, feat)
        if mean_val is None:
            directions.append("unknown")
            notes.append("not assessed")
            continue
        if abs(mean_val) < 1e-3:
            directions.append("mixed")
            notes.append("direction varies / near-zero mean SHAP")
        elif mean_val > 0:
            directions.append("+")
            notes.append("higher -> higher prediction")
        else:
            directions.append("-")
            notes.append("higher -> lower prediction")

    top10["direction"] = directions
    top10["effect_note"] = notes
    top10.to_csv(os.path.join(cum_dir, "top10_summary.csv"), index=False)
    print("[OK] Saved cumulative tables and top10 summary.")


def main() -> None:
    run_root = resolve_run_root()
    analysis_root = resolve_analysis_root(run_root)
    ensure_dir(analysis_root)

    print(f"[INFO] Run root: {run_root}")
    print(f"[INFO] Analysis root: {analysis_root}")

    calibration_workflow(run_root, analysis_root)
    cumulative_importance_workflow(run_root, analysis_root)


if __name__ == "__main__":
    main()