"""
Diagnostics for best model: residual plots and heteroskedasticity test.
Outputs:
- output/diagnostics/residuals.png
- output/diagnostics/residual_distribution.png
- output/diagnostics/qq_plot.png
- output/diagnostics/heteroskedasticity.txt
"""
from __future__ import annotations

import os
from utils.paths import EVALUATION_DIR
import json
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = str(os.environ.get('MLTRAINER_RUN_ROOT', '') or '').strip()
ANALYSIS_ROOT = str(os.environ.get('MLTRAINER_ANALYSIS_ROOT', '') or '').strip()
try:
    import config as cfg
    OUTPUT_ROOT = cfg.OUTPUT_DIR
    CV_FOLDS = int(cfg.CV_FOLDS)
except Exception:
    OUTPUT_ROOT = 'output'
    CV_FOLDS = 5

if ANALYSIS_ROOT:
    OUTDIR = os.path.join(ANALYSIS_ROOT, '2_Model_Diagnostics')
elif RUN_ROOT and os.path.isdir(RUN_ROOT):
    OUTDIR = os.path.join(RUN_ROOT, EVALUATION_DIR, '2_Model_Diagnostics')
else:
    OUTDIR = os.path.join(ROOT, OUTPUT_ROOT, '2_Model_Diagnostics')
os.makedirs(OUTDIR, exist_ok=True)


def load_best_meta():
    if RUN_ROOT and os.path.isdir(RUN_ROOT):
        manifest_path = os.path.join(RUN_ROOT, 'run_manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                best_model = str(manifest.get('best_model') or '').strip()
                if best_model:
                    return manifest
            except Exception as e:
                LOGGER.exception("Failed reading run manifest: %s", manifest_path)

        # Check both potential metrics file names
        for m_name in ('metrics.xlsx', 'feature_engineering_metrics.xlsx'):
            metrics_xlsx = os.path.join(RUN_ROOT, EVALUATION_DIR, m_name)
            if os.path.exists(metrics_xlsx):
                try:
                    mdf = pd.read_excel(metrics_xlsx, sheet_name=0)
                    if not mdf.empty and 'model' in mdf.columns:
                        best_model = str(mdf.iloc[0].get('model') or '').strip()
                        if best_model:
                            meta = {'best_model': best_model}
                            # Also try loading feature list from selection if available
                            sel = os.path.join(RUN_ROOT, '0_Feature_Selection', 'feature_selection_meta.json')
                            if os.path.exists(sel):
                                with open(sel, 'r', encoding='utf-8') as f:
                                    sel_data = json.load(f)
                                    if 'features' in sel_data:
                                        meta['features'] = sel_data['features']
                            return meta
                except Exception as e:
                    LOGGER.exception("Failed reading metrics workbook: %s", metrics_xlsx)

        # Fallback to feature_selection_meta.json if manifest/metrics didn't work but we have RUN_ROOT
        sel = os.path.join(RUN_ROOT, '0_Feature_Selection', 'feature_selection_meta.json')
        if os.path.exists(sel):
            try:
                with open(sel, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                LOGGER.exception("Failed reading feature selection meta: %s", sel)

    # find latest feature_selection_meta.json under OUTPUT_ROOT or validation_compare
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
    if not metas:
        raise SystemExit('No feature_selection_meta.json found.')
    metas.sort(reverse=True)
    with open(metas[0][1], 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    meta = load_best_meta()
    best_model_name = meta.get('best_model', 'BestModel')

    from data.loader import read_csv_safely, detect_cols
    from config.columns import resolve_column_groups
    from features.preprocess import build_preprocessor
    from models.train import train_and_evaluate
    from evaluation.explain import plot_residuals, plot_residual_distribution, plot_qq
    
    # Respect DATASET_PATH if set, else fallback to standard lookup
    data_path = os.environ.get("DATASET_PATH", "").strip()
    if not data_path or not os.path.exists(data_path):
        data_path = os.path.join(ROOT, 'dataset', 'data_cleaned.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(ROOT, 'dataset', 'data.csv')
            
    df, _ = read_csv_safely(data_path)
    
    target = os.environ.get("TARGET_COL", "").strip()
    if not target:
        target, _, _, _, _, _ = detect_cols(df)

    if 'features' in meta and isinstance(meta['features'], list):
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

    y_raw = df[meta.get('target', target)]
    y_num = pd.to_numeric(y_raw, errors='coerce')
    y = y_num if y_num.notna().mean() >= 0.8 else y_raw.map({v: i + 1 for i, v in enumerate(sorted(y_raw.dropna().unique(), key=lambda x: str(x)))})
    X = df[feature_cols].copy()

    num_cols, ordinal_cols, binary_cols, other_cat_cols = resolve_column_groups(list(X.columns))
    mask = y.notna()
    for c in num_cols:
        mask &= X[c].notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    preproc = build_preprocessor(num_cols, list(other_cat_cols), ordinal_cols=ordinal_cols, binary_cols=binary_cols)
    metrics_df, fitted = train_and_evaluate(X, y, preproc, model_names=[best_model_name], cv_mode='kfold', cv_folds=CV_FOLDS)
    pipe = fitted[best_model_name]['pipe']

    # Fit full and get residuals
    pipe.fit(X, y)
    y_hat = pipe.predict(X)
    resid = y - y_hat

    # Plots
    plot_residuals(best_model_name, pipe, X, y, OUTDIR)
    plot_residual_distribution(best_model_name, pipe, X, y, OUTDIR)
    plot_qq(best_model_name, pipe, X, y, OUTDIR)

    # Heteroskedasticity (Breusch-Pagan)
    X_lin = sm.add_constant(pd.DataFrame(pipe.named_steps['prep'].fit_transform(X)))
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, X_lin)
    with open(os.path.join(OUTDIR, 'heteroskedasticity.txt'), 'w', encoding='utf-8') as f:
        f.write(f"LM={lm:.4f} (p={lm_pvalue:.4g}), F={fvalue:.4f} (p={f_pvalue:.4g})\n")
    print('[OK] Diagnostics written to', OUTDIR)


if __name__ == '__main__':
    main()
