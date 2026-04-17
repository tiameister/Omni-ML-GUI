import os
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from utils.helpers import save_bar
from utils.logger import get_logger
import numpy as np


LOGGER = get_logger(__name__)

def export_to_excel(df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1'):
    try:
        # Requires openpyxl or xlsxwriter
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
    except (ModuleNotFoundError, ImportError):
        # Skip Excel export if dependency missing
        LOGGER.warning("openpyxl not installed, skipping Excel export to %s", file_path)


def _safe_excel_sheet_name(name: str, fallback: str = "Sheet") -> str:
    forbidden = {":", "\\", "/", "?", "*", "[", "]"}
    cleaned = "".join("_" if ch in forbidden else ch for ch in str(name))
    cleaned = cleaned.strip().strip("'")
    cleaned = cleaned[:31]
    return cleaned or fallback


def _safe_file_token(name: str, fallback: str = "model") -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or fallback

def save_model_metrics(outdir: str, metrics_df: pd.DataFrame, filename_prefix: str = ""):
    """
    Saves model performance metrics and creates visual bar plots.
    """
    os.makedirs(outdir, exist_ok=True)
    eval_dir = os.path.join(outdir, '1_Overall_Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    metrics_out = metrics_df.copy(deep=True)
    if "R2_CV" in metrics_out.columns:
        metrics_out = metrics_out.sort_values("R2_CV", ascending=False).reset_index(drop=True)

    publication_view = metrics_out.copy(deep=True)
    if "R2_CV" in publication_view.columns:
        publication_view["rank_by_R2"] = publication_view["R2_CV"].rank(ascending=False, method="min").astype(int)
        try:
            best_r2 = float(publication_view["R2_CV"].max())
            publication_view["delta_R2_vs_best"] = publication_view["R2_CV"].astype(float) - best_r2
        except Exception:
            pass
    if "RMSE_CV" in publication_view.columns:
        try:
            best_rmse = float(publication_view["RMSE_CV"].min())
            publication_view["delta_RMSE_vs_best"] = publication_view["RMSE_CV"].astype(float) - best_rmse
        except Exception:
            pass

    preferred_cols = [
        "rank_by_R2",
        "model",
        "R2_CV",
        "RMSE_CV",
        "MAE_CV",
        "delta_R2_vs_best",
        "delta_RMSE_vs_best",
        "TrainingTime",
    ]
    publication_cols = [c for c in preferred_cols if c in publication_view.columns]
    publication_tail = [c for c in publication_view.columns if c not in publication_cols]
    publication_view = publication_view[publication_cols + publication_tail]

    # Save metrics workbook.
    base = f"{filename_prefix}metrics.xlsx" if filename_prefix else "metrics.xlsx"
    xlsx_path = os.path.join(eval_dir, base)
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            metrics_out.to_excel(writer, sheet_name="Metrics", index=False)
            publication_view.to_excel(writer, sheet_name="PublicationSummary", index=False)
    except (ModuleNotFoundError, ImportError):
        export_to_excel(metrics_out, xlsx_path)

    save_bar(
        os.path.join(eval_dir, f"{filename_prefix}metrics_R2_cv.png" if filename_prefix else "metrics_R2_cv.png"),
        list(metrics_out["model"]), list(metrics_out["R2_CV"]),
        "Models - R2 (CV)", "R2"
    )

    save_bar(
        os.path.join(eval_dir, f"{filename_prefix}metrics_RMSE_cv.png" if filename_prefix else "metrics_RMSE_cv.png"),
        list(metrics_out["model"]), list(metrics_out["RMSE_CV"]),
        "Models - RMSE (CV)", "RMSE"
    )

def save_cv_splits(outdir: str, cv_scores_by_model: dict):
    """Save per-split CV scores (R2/MAE/RMSE arrays) into an Excel workbook for comparison.
    Sheet per model with columns: split, R2, MAE, RMSE.
    """
    os.makedirs(outdir, exist_ok=True)
    eval_dir = os.path.join(outdir, '1_Overall_Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    xlsx_path = os.path.join(eval_dir, 'cv_splits.xlsx')
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            for model, scores in cv_scores_by_model.items():
                if not scores:
                    continue
                n = len(scores.get('R2', []))
                df = pd.DataFrame({
                    'split': np.arange(1, n+1),
                    'R2': scores.get('R2', []),
                    'MAE': scores.get('MAE', []),
                    'RMSE': scores.get('RMSE', []),
                })
                df.to_excel(writer, sheet_name=_safe_excel_sheet_name(model, fallback='model'), index=False)
    except (ModuleNotFoundError, ImportError):
        LOGGER.warning("openpyxl not installed; writing CV splits as per-model CSV files in %s", eval_dir)
        for model, scores in cv_scores_by_model.items():
            if not scores:
                continue
            n = len(scores.get('R2', []))
            df = pd.DataFrame({
                'split': np.arange(1, n+1),
                'R2': scores.get('R2', []),
                'MAE': scores.get('MAE', []),
                'RMSE': scores.get('RMSE', []),
            })
            token = _safe_file_token(model, fallback='model')
            df.to_csv(os.path.join(eval_dir, f'cv_splits_{token}.csv'), index=False)

def get_feature_names_from_pipe(pipe: Pipeline, num_cols: List[str] | None = None, cat_cols: List[str] | None = None) -> List[str]:
    """
    Extract final feature names after preprocessing (e.g., OHE expanded names).

    Robust to different OneHotEncoder step names ("onehot" vs "encoder") and
    will derive numeric/categorical column lists from the fitted ColumnTransformer
    when not provided.
    """
    ct = pipe.named_steps.get("prep")
    if ct is None:
        # No preprocessor; assume original columns are final names
        return list(getattr(pipe, "feature_names_in_", []))

    # Derive num/cat columns from ColumnTransformer if not provided
    derived_num: List[str] = []
    derived_cat: List[str] = []
    try:
        for name, trans, cols in getattr(ct, "transformers_", []):
            if name == "num":
                derived_num = list(cols) if cols is not None else []
            elif name == "cat":
                derived_cat = list(cols) if cols is not None else []
            elif name == "ord":
                # ordinal features treated like numeric in names (single column per feature)
                if cols is not None:
                    derived_num.extend(list(cols))
            elif name == "bin":
                # passthrough binary columns
                if cols is not None:
                    derived_num.extend(list(cols))
    except (AttributeError, TypeError) as exc:
        LOGGER.debug("Could not derive transformer columns from preprocessor: %s", exc)

    if not num_cols:
        num_cols = derived_num
    if not cat_cols:
        cat_cols = derived_cat

    names: List[str] = []
    if num_cols:
        names.extend(num_cols)
    if cat_cols:
        # Support either step name
        cat_pipeline = ct.named_transformers_.get("cat")
        if cat_pipeline is not None and hasattr(cat_pipeline, "named_steps"):
            ohe = cat_pipeline.named_steps.get("onehot") or cat_pipeline.named_steps.get("encoder")
            if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                try:
                    names.extend(list(ohe.get_feature_names_out(cat_cols)))
                except (TypeError, ValueError, AttributeError) as exc:
                    # Fallback to generic naming if encoder refuses cat_cols
                    LOGGER.debug("Falling back to generic encoded feature names due to: %s", exc)
                    names.extend([f"{c}_{i}" for c in cat_cols for i in range(1)])
            else:
                names.extend(cat_cols)
        else:
            names.extend(cat_cols)
    return names

def dump_permutation(name: str, pipe_fit: Pipeline, X_te, y_te,
                     num_cols: List[str], cat_cols: List[str],
                     outdir: str, n_repeats: int = 5, seed: int = 42,
                     n_jobs: int = -1) -> pd.DataFrame:
    """
    Saves permutation importances (CSV + Excel + TXT) and a .png bar plot.
    """
    pi = permutation_importance(pipe_fit, X_te, y_te, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)
    feat_names = get_feature_names_from_pipe(pipe_fit, num_cols, cat_cols)
    k = min(len(feat_names), len(pi.importances_mean))

    imp_df = pd.DataFrame({
        "feature": feat_names[:k],
        "perm_importance_mean": pi.importances_mean[:k],
        "perm_importance_std":  pi.importances_std[:k],
    }).sort_values(by="perm_importance_mean", ascending=False)

    # Save CSV (backward compatibility) and Excel (preferred)
    csv_path = os.path.join(outdir, f"feature_importance_{name}.csv")
    xlsx_path = os.path.join(outdir, f"feature_importance_{name}.xlsx")
    imp_df.to_csv(csv_path, index=False)
    export_to_excel(imp_df, xlsx_path, sheet_name="PermutationImportance")
    with open(os.path.join(outdir, f"feature_importance_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Permutation Importance - {name}\n")
        for _, r in imp_df.head(100).iterrows():
            f.write(f"{r['feature']}: {r['perm_importance_mean']:.6f} ± {r['perm_importance_std']:.6f}\n")

    # Save bar plot
    topN = imp_df.head(25)
    save_bar(
        os.path.join(outdir, f"feature_importance_{name}.png"),
        list(topN["feature"]), list(topN["perm_importance_mean"]),
        f"Top Feature Importances (Permutation) - {name}",
        "Mean importance (ΔR²)"
    )

    return imp_df
