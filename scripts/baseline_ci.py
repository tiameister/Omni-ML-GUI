"""
Baseline vs. Best Model with 95% Bootstrap CIs.
Outputs:
- output/baseline/baseline_metrics.xlsx
- output/baseline/baseline_vs_best.txt
"""
from __future__ import annotations

import os
from utils.paths import EVALUATION_DIR
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import config as cfg
    OUTPUT_ROOT = cfg.OUTPUT_DIR
    CV_FOLDS = int(cfg.CV_FOLDS)
except Exception:
    OUTPUT_ROOT = 'output'
    CV_FOLDS = 5

OUTDIR = str(os.environ.get("MLTRAINER_RUN_ROOT", "")).strip()
if OUTDIR and os.path.exists(OUTDIR):
    OUTDIR = os.path.join(OUTDIR, EVALUATION_DIR, "baseline")
else:
    OUTDIR = os.path.join(ROOT, OUTPUT_ROOT, 'baseline')
os.makedirs(OUTDIR, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = []
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(values[idx])))
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    def find_meta():
        run_root = os.environ.get("MLTRAINER_RUN_ROOT", "").strip()
        if run_root and os.path.isdir(run_root):
            manifest_path = os.path.join(run_root, 'run_manifest.json')
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    best_model = str(manifest.get('best_model') or '').strip()
                    if best_model:
                        return manifest
                except Exception as e:
                    LOGGER.exception("Failed reading run manifest: %s", manifest_path)

            for m_name in ('metrics.xlsx', 'feature_engineering_metrics.xlsx'):
                metrics_xlsx = os.path.join(run_root, EVALUATION_DIR, m_name)
                if os.path.exists(metrics_xlsx):
                    try:
                        mdf = pd.read_excel(metrics_xlsx, sheet_name=0)
                        if not mdf.empty and 'model' in mdf.columns:
                            best_model = str(mdf.iloc[0].get('model') or '').strip()
                            if best_model:
                                meta = {'best_model': best_model}
                                sel = os.path.join(run_root, '0_Feature_Selection', 'feature_selection_meta.json')
                                if os.path.exists(sel):
                                    with open(sel, 'r', encoding='utf-8') as f:
                                        sel_data = json.load(f)
                                        if 'features' in sel_data:
                                            meta['features'] = sel_data['features']
                                return meta
                    except Exception as e:
                        LOGGER.exception("Failed reading metrics workbook: %s", metrics_xlsx)

            sel = os.path.join(run_root, '0_Feature_Selection', 'feature_selection_meta.json')
            if os.path.exists(sel):
                try:
                    with open(sel, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    LOGGER.exception("Failed reading feature selection meta: %s", sel)

        metas = []
        out_root = os.path.join(ROOT, OUTPUT_ROOT)
        if os.path.isdir(out_root):
            for name in os.listdir(out_root):
                p = os.path.join(out_root, name)
                if os.path.isdir(p) and (name.endswith('_output') or ('_output_v' in name)):
                    mp = os.path.join(p, 'feature_selection_meta.json')
                    if os.path.exists(mp):
                        metas.append((os.path.getmtime(mp), mp))
        vc_root = os.path.join(ROOT, 'validation_compare')
        for sub in ['kfold', 'repeated', 'nested']:
            p = os.path.join(vc_root, sub)
            if os.path.isdir(p):
                for model_dir in os.listdir(p):
                    mp = os.path.join(p, model_dir, 'feature_selection_meta.json')
                    if os.path.exists(mp):
                        metas.append((os.path.getmtime(mp), mp))
        
        if metas:
            metas.sort(reverse=True)
            with open(metas[0][1], 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    from data.loader import read_csv_safely, detect_cols
    from config.columns import resolve_column_groups
    from features.preprocess import build_preprocessor
    from models.train import train_and_evaluate
    
    meta = find_meta()
    
    data_path = os.environ.get("DATASET_PATH", "").strip()
    if not data_path or not os.path.exists(data_path):
        data_path = os.path.join(ROOT, 'dataset', 'data_cleaned.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(ROOT, 'dataset', 'data.csv')
            
    df, _ = read_csv_safely(data_path)
    
    target = os.environ.get("TARGET_COL", "").strip()
    if not target:
        target, _, _, _, _, _ = detect_cols(df)

    if meta and 'features' in meta and isinstance(meta['features'], list):
        feature_cols = [c for c in meta['features'] if c in df.columns]
    elif os.environ.get("SELECTED_FEATURES"):
        import json
        try:
            sel_feats = json.loads(os.environ.get("SELECTED_FEATURES"))
            feature_cols = [c for c in sel_feats if c in df.columns]
        except Exception:
            feature_cols = []
    
    if not locals().get('feature_cols'):
        _, bully, m_items, z_items, bully_subs, drop_cols = detect_cols(df)
        exclude = set([target] + m_items + z_items + bully_subs + drop_cols)
        feature_cols = [c for c in df.columns if c not in exclude]
        if bully not in feature_cols:
            feature_cols = [bully] + feature_cols
            
    best_model_name = meta.get('best_model', 'BestModel') if meta else 'BestModel'
    
    y_raw = df[target]
    y_num = pd.to_numeric(y_raw, errors='coerce')
    if y_num.notna().mean() < 0.8:
        uniq = sorted(y_raw.dropna().unique(), key=lambda x: str(x))
        y = y_raw.map({v: i + 1 for i, v in enumerate(uniq)})
    else:
        y = y_num
    X = df[feature_cols].copy()
    num_cols, ordinal_cols, binary_cols, other_cat_cols = resolve_column_groups(list(X.columns))
    mask = y.notna()
    for c in num_cols:
        mask &= X[c].notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Baseline model: DummyRegressor (mean)
    dummy = DummyRegressor(strategy='mean')
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    yhat_dummy = cross_val_predict(dummy, X, y, cv=kf, n_jobs=None)

    # Load best model pipeline by refitting (simpler) using models/train and preprocessing
    from features.preprocess import build_preprocessor
    from models.train import train_and_evaluate
    preproc = build_preprocessor(num_cols, list(other_cat_cols), ordinal_cols=ordinal_cols, binary_cols=binary_cols)
    # Train only the best model for fair comparison
    metrics_df, fitted = train_and_evaluate(X, y, preproc, model_names=[best_model_name], cv_mode='kfold', cv_folds=CV_FOLDS)
    pipe = fitted[best_model_name]['pipe']
    yhat_best = cross_val_predict(pipe, X, y, cv=kf, n_jobs=None, method='predict')

    # Compute metrics
    def compute_all(y_true, y_pred):
        return {
            'R2': float(r2_score(y_true, y_pred)),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': rmse(y_true, y_pred),
        }
    m_dummy = compute_all(y, yhat_dummy)
    m_best = compute_all(y, yhat_best)

    # Bootstrap CIs on per-sample errors (for MAE/RMSE) and per-sample R2 proxy via residuals
    # Build per-sample contributions
    resid_dummy = (y - yhat_dummy).to_numpy()
    resid_best = (y - yhat_best).to_numpy()
    abs_dummy = np.abs(resid_dummy)
    abs_best = np.abs(resid_best)
    sq_dummy = resid_dummy ** 2
    sq_best = resid_best ** 2

    mae_ci_dummy = bootstrap_ci(abs_dummy)
    mae_ci_best = bootstrap_ci(abs_best)
    rmse_ci_dummy = tuple(float(np.sqrt(x)) for x in bootstrap_ci(sq_dummy))
    rmse_ci_best = tuple(float(np.sqrt(x)) for x in bootstrap_ci(sq_best))

    # For R2, bootstrap over indices and recompute R2
    def r2_boot_ci(y_true, y_pred, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y_true)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(r2_score(y_true[idx], y_pred[idx]))
        return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    r2_ci_dummy = r2_boot_ci(y.to_numpy(), yhat_dummy)
    r2_ci_best = r2_boot_ci(y.to_numpy(), yhat_best)

    rows = []
    rows.append({
        'model': 'DummyRegressor(mean)',
        'R2': m_dummy['R2'], 'R2_CI_low': r2_ci_dummy[0], 'R2_CI_high': r2_ci_dummy[1],
        'MAE': m_dummy['MAE'], 'MAE_CI_low': mae_ci_dummy[0], 'MAE_CI_high': mae_ci_dummy[1],
        'RMSE': m_dummy['RMSE'], 'RMSE_CI_low': rmse_ci_dummy[0], 'RMSE_CI_high': rmse_ci_dummy[1],
    })
    rows.append({
        'model': best_model_name,
        'R2': m_best['R2'], 'R2_CI_low': r2_ci_best[0], 'R2_CI_high': r2_ci_best[1],
        'MAE': m_best['MAE'], 'MAE_CI_low': mae_ci_best[0], 'MAE_CI_high': mae_ci_best[1],
        'RMSE': m_best['RMSE'], 'RMSE_CI_low': rmse_ci_best[0], 'RMSE_CI_high': rmse_ci_best[1],
    })
    out_xlsx = os.path.join(OUTDIR, 'baseline_metrics.xlsx')
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)

    with open(os.path.join(OUTDIR, 'baseline_vs_best.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Best model: {best_model_name}\n")
        f.write(json.dumps(rows, indent=2))
        f.write("\n")
    print('[OK] Wrote:', out_xlsx)


if __name__ == '__main__':
    main()
