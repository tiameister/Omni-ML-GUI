import os
import json
import pandas as pd

from data.loader import read_csv_safely, detect_cols
from features.preprocess import build_preprocessor
from models.train import train_and_evaluate
from evaluation.metrics import save_model_metrics, dump_permutation, save_cv_splits
from evaluation.explain import explain_with_shap, generate_pdp
from evaluation.plots.curves import plot_learning_curve, plot_predictions_vs_actual
from config.columns import resolve_column_groups

from utils.paths import ensure_outdir

from config import RSTATE, PI_REPEATS, DO_SHAP, DATASET_PATH, get_output_folder
from config import CV_MODE, CV_FOLDS, CV_REPEATS, NESTED_OUTER_FOLDS, NESTED_INNER_FOLDS
from config import PI_ONLY_BEST_MODEL, PI_N_JOBS, PERM_IMPORTANCE_ENABLED, EVAL_PLOTS_ENABLED
from utils.logger import configure_logging, get_logger


configure_logging("ml_trainer.cli")
LOGGER = get_logger(__name__)


CSV_PATH = DATASET_PATH
# OUTDIR will be generated dynamically after training

# ---- 1. Load Data ----
df, _ = read_csv_safely(CSV_PATH)
target, bully, m_items, z_items, bully_subs, drop_cols = detect_cols(df)

# Determine input features
exclude = set([target] + m_items + z_items + bully_subs + drop_cols)
feature_cols = [c for c in df.columns if c not in exclude]
if bully not in feature_cols:
    feature_cols = [bully] + feature_cols

# Prepare target variable
y_raw = df[target]
y_num = pd.to_numeric(y_raw, errors="coerce")
if y_num.notna().mean() < 0.8:
    uniq = sorted(y_raw.dropna().unique(), key=lambda x: str(x))
    y = y_raw.map({v: i + 1 for i, v in enumerate(uniq)})
else:
    y = y_num

X = df[feature_cols].copy()
# Normalize binary Gender coding to 0/1 for clearer SHAP interpretation (Female=0, Male=1)
if 'Gender' in X.columns:
    g = pd.to_numeric(X['Gender'], errors='coerce')
    uniq = set(g.dropna().astype(int).unique().tolist())
    # If dataset uses 1/2, remap to 0/1
    if uniq.issubset({1, 2}) and len(uniq) > 0:
        X['Gender'] = g.map({1: 0, 2: 1})
    else:
        # keep existing 0/1 coding if already binary; otherwise leave as is
        X['Gender'] = g
# Resolve desired preprocessing groups by matching intended names to actual columns
num_cols, ordinal_cols, binary_cols, other_cat_cols = resolve_column_groups(list(X.columns))
cat_cols = list(other_cat_cols)

# Drop rows with missing data
mask = y.notna()
for col in num_cols:
    mask &= X[col].notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

LOGGER.info(
    "Target=%s | Features=%d (num=%d, ord=%d, bin=%d, cat=%d)",
    target,
    len(feature_cols),
    len(num_cols),
    len(ordinal_cols),
    len(binary_cols),
    len(cat_cols),
)

# ---- 2a. Quantify cleaning and imputation needs (for Methods reporting) ----
# Counts before/after cleaning
n_raw = int(df.shape[0])
n_clean = int(X.shape[0])
n_dropped = n_raw - n_clean

# Rows dropped specifically due to target/numeric missingness (approximation):
# We recompute on original df to avoid post-mask bias
try:
    _mask_target = df[target].notna()
    _mask_num = df[num_cols].notna().all(axis=1) if num_cols else pd.Series([True]*n_raw)
    n_drop_target_or_numeric = int((~(_mask_target & _mask_num)).sum())
except Exception:
    n_drop_target_or_numeric = n_dropped

# Among retained rows, how many cells will be imputed by the pipeline?
def _missing_cells_count(frame: pd.DataFrame, cols: list[str]) -> int:
    if not cols:
        return 0
    sub = frame[cols]
    try:
        return int(sub.isna().sum().sum())
    except Exception:
        return 0

imp_ord_cells = _missing_cells_count(X, ordinal_cols)
imp_bin_cells = _missing_cells_count(X, binary_cols)
imp_cat_cells = _missing_cells_count(X, cat_cols)

def _pct(part: int, whole: int) -> float:
    return (100.0 * part / whole) if whole else 0.0

# Denominators = number of available cells in each block among retained rows
ord_cells_total = int(len(ordinal_cols) * n_clean)
bin_cells_total = int(len(binary_cols) * n_clean)
cat_cells_total = int(len(cat_cols) * n_clean)

summary_cleaning = {
    "n_raw": n_raw,
    "n_clean": n_clean,
    "n_dropped_total": n_dropped,
    "n_dropped_target_or_numeric": n_drop_target_or_numeric,
    "imputed_cells": {
        "ordinal": {"n": imp_ord_cells, "pct": _pct(imp_ord_cells, ord_cells_total)},
        "binary": {"n": imp_bin_cells, "pct": _pct(imp_bin_cells, bin_cells_total)},
        "categorical_other": {"n": imp_cat_cells, "pct": _pct(imp_cat_cells, cat_cells_total)},
    },
    "denominators": {
        "ordinal_cells": ord_cells_total,
        "binary_cells": bin_cells_total,
        "categorical_other_cells": cat_cells_total,
    },
}

# Compose a publication-ready sentence and keep it to write once OUTDIR is known
cleaning_sentence = (
    f"After screening, we excluded {n_dropped} records with missing target or missing numeric predictors "
    f"({n_drop_target_or_numeric} by target/numeric criteria), yielding {n_clean} analyzable observations "
    f"out of {n_raw} total. Among retained rows, missing values were imputed within CV folds using the "
    f"most‑frequent level for ordinal (n={imp_ord_cells}, {_pct(imp_ord_cells, ord_cells_total):.1f}%) and binary "
    f"(n={imp_bin_cells}, {_pct(imp_bin_cells, bin_cells_total):.1f}%) variables and for other categorical predictors "
    f"(n={imp_cat_cells}, {_pct(imp_cat_cells, cat_cells_total):.1f}%). Numeric predictors were not imputed because "
    f"rows with numeric missingness were removed prior to modeling."
)

# ---- 2. Build Preprocessing Pipeline ----
preproc = build_preprocessor(num_cols, cat_cols, ordinal_cols=ordinal_cols, binary_cols=binary_cols)

# ---- 3. Train and Evaluate Models ----
metrics_df, fitted_models = train_and_evaluate(
    X, y, preproc,
    cv_mode=CV_MODE,
    cv_folds=CV_FOLDS
)
best_name = metrics_df.iloc[0]["model"]
LOGGER.info("Training complete. Best model: %s", best_name)

# ---- 4. Create and set output directory based on best model ----
out_folder = get_output_folder(best_name)  # e.g. output/linearregression
OUTDIR = ensure_outdir(out_folder)

# ---- 5. Save Metrics & Visuals ----
save_model_metrics(OUTDIR, metrics_df)
try:
    cv_scores_by_model = {name: model.get("cv_scores") for name, model in fitted_models.items()}
    save_cv_splits(OUTDIR, cv_scores_by_model)
except Exception as e:
    LOGGER.warning("Could not save CV split scores: %s", e)

if EVAL_PLOTS_ENABLED:
    # Generate learning curves and predictions vs actual for each model (saved under OUTDIR/evaluation)
    for name, model in fitted_models.items():
        try:
            plot_learning_curve(name, model["pipe"], X, y, OUTDIR)
        except Exception as e:
            LOGGER.warning("Learning curve failed for %s: %s", name, e)
        try:
            plot_predictions_vs_actual(name, model["pipe"], X, y, OUTDIR)
        except Exception as e:
            LOGGER.warning("Predictions vs Actual failed for %s: %s", name, e)

# Permutation Importances (optional for speed)
if PERM_IMPORTANCE_ENABLED:
    if PI_ONLY_BEST_MODEL:
        name = best_name
        model = fitted_models[name]
        dump_permutation(
            name,
            model["pipe"],
            model["holdout"][0],  # X used for fit (approximation for speed)
            model["holdout"][1],  # y used for fit (approximation for speed)
            num_cols,
            cat_cols,
            outdir=OUTDIR,
            n_repeats=PI_REPEATS,
            seed=RSTATE,
            n_jobs=PI_N_JOBS
        )
    else:
        for name, model in fitted_models.items():
            dump_permutation(
                name,
                model["pipe"],
                model["holdout"][0],
                model["holdout"][1],
                num_cols,
                cat_cols,
                outdir=OUTDIR,
                n_repeats=PI_REPEATS,
                seed=RSTATE,
                n_jobs=PI_N_JOBS
            )

# ---- 6. PDP for Best Model ----
imp_df_best = pd.read_csv(os.path.join(OUTDIR, f"feature_importance_{best_name}.csv"))
top_feats = list(imp_df_best["feature"])[:3]
generate_pdp(
    best_model_name=best_name,
    best_pipe=fitted_models[best_name]["pipe"],
    X=X,
    top_features=top_feats,
    outdir=OUTDIR
)

# ---- 7. SHAP (Optional) ----
if DO_SHAP:
    explain_with_shap(
        best_model_name=best_name,
        best_pipe=fitted_models[best_name]["pipe"],
        X=X,
        num_cols=num_cols,
        cat_cols=cat_cols,
        outdir=OUTDIR
    )

# ---- 8. Save Meta Info ----
meta = {
    "target": target,
    "bullying_total": bully,
    "excluded_m_items": m_items,
    "excluded_z_items": z_items + bully_subs,
    "dropped_geo_faaliyet_id": list(drop_cols),
    "used_features": feature_cols,
    "n_features": len(feature_cols),
    "best_model": best_name,
    "cv_config": {
        "mode": CV_MODE,
        "folds": CV_FOLDS,
        "repeats": CV_REPEATS,
        "nested_outer_folds": NESTED_OUTER_FOLDS,
        "nested_inner_folds": NESTED_INNER_FOLDS,
    }
}
with open(os.path.join(OUTDIR, "feature_selection_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# ---- 9. Persist data-cleaning summary for Methods ----
try:
    # JSON summary
    with open(os.path.join(OUTDIR, "data_cleaning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_cleaning, f, ensure_ascii=False, indent=2)
    # Human‑readable sentence
    with open(os.path.join(OUTDIR, "data_cleaning_summary.txt"), "w", encoding="utf-8") as f:
        f.write(cleaning_sentence + "\n")
    LOGGER.info("Data cleaning summary written to: %s", os.path.join(OUTDIR, "data_cleaning_summary.txt"))
    LOGGER.info(cleaning_sentence)
except Exception as _e:
    LOGGER.warning("Could not write data cleaning summary: %s", _e)
