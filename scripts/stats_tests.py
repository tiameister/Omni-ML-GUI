"""
Statistical tests across CV strategies or models using per-split scores.
Outputs:
- Corrected Resampled T-tests between top models and baselines
- FDR Adjusted p-values
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corrected_resampled_ttest(scores_a: np.ndarray, scores_b: np.ndarray, k: int, r: int = 1):
    """Nadeau & Bengio corrected resampled t-test.
    scores_*: per-split scores (same length); higher-is-better assumed.
    k: folds per repeat; r: repeats (1 for kfold; >1 for repeated kfold)
    Returns t_stat, p_value (two-sided, normal approx).
    """
    from math import sqrt
    from scipy.stats import t as student_t
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    n = diff.size
    mean_d = np.mean(diff)
    var_d = np.var(diff, ddof=1) if n > 1 else 0.0
    if r <= 1:
        c = 1.0 / k
    else:
        c = 1.0 / k + 1.0 / (k * (r - 1))
    se = sqrt((1.0 / n + c) * var_d)
    if se == 0:
        return 0.0, 1.0
    t_stat = mean_d / se
    df = n - 1
    p_val = 2 * (1 - student_t.cdf(abs(t_stat), df))
    return float(t_stat), float(p_val)

def _read_model_scores(run_root: str):
    """Read cv_splits.xlsx from the current run root."""
    path = os.path.join(run_root, "1_Overall_Evaluation", "cv_splits.xlsx")
    if not os.path.exists(path):
        # Fallback to CSVs
        csvs = glob.glob(os.path.join(run_root, "1_Overall_Evaluation", "cv_splits_*.csv"))
        data = {}
        for cp in csvs:
            model = os.path.basename(cp).replace("cv_splits_", "").replace(".csv", "")
            data[model] = pd.read_csv(cp)
        return data
    
    data = {}
    try:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            data[sheet_name] = df[['R2', 'RMSE', 'MAE']].dropna(how='all')
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    return data

def main():
    # If invoked by training runner, we receive the exact path to the output directory
    run_root = os.environ.get("MLTRAINER_RUN_ROOT")
    if run_root:
        out_dir = os.environ.get("MLTRAINER_ANALYSIS_ROOT", os.path.join(run_root, "analysis"))
        out_dir = os.path.join(out_dir, "stats_tests")
        os.makedirs(out_dir, exist_ok=True)
        # Read the cv_splits for models in the CURRENT run
        data = _read_model_scores(run_root)
        models = list(data.keys())
        if len(models) < 2:
            print("Not enough models to perform statistical comparison tests.")
            return

        from config import CV_FOLDS, CV_REPEATS, NESTED_OUTER_FOLDS, CV_MODE
        k = CV_FOLDS if CV_MODE != 'nested' else NESTED_OUTER_FOLDS
        r = CV_REPEATS if CV_MODE == 'repeated' else 1
        
        # Perform all pairwise combinations
        import itertools
        pairs = list(itertools.combinations(models, 2))
        rows_t = []
        metrics = ['R2', 'RMSE', 'MAE']
        
        for metric in metrics:
            for a, b in pairs:
                if metric not in data[a].columns or metric not in data[b].columns: continue
                s_a = pd.to_numeric(data[a][metric], errors='coerce').dropna().values
                s_b = pd.to_numeric(data[b][metric], errors='coerce').dropna().values
                
                # For RMSE and MAE, lower is better. We negate them so that positive t_stat means A is better!
                if metric in ['RMSE', 'MAE']:
                    s_a = -s_a
                    s_b = -s_b
                    
                n = min(len(s_a), len(s_b))
                if n < 2: continue
                s_a, s_b = s_a[:n], s_b[:n]
                t_stat, p_val = corrected_resampled_ttest(s_a, s_b, k, r)
                
                # Make A the "better" model directionally so reporting is easy
                if t_stat < 0:
                    a, b = b, a
                    t_stat = -t_stat
                    
                rows_t.append({
                    'metric': metric,
                    'Model_A_Better': a,
                    'Model_B_Worse': b,
                    't_stat': t_stat,
                    'p_value': p_val,
                    'n_splits': n
                })
                
        df_t = pd.DataFrame(rows_t)
        
        if not df_t.empty:
            df_t = df_t.sort_values(by=['metric', 'p_value'])
            csv_t = os.path.join(out_dir, 'model_pairwise_t_tests.csv')
            df_t.to_csv(csv_t, index=False)
            
            # FDR adjustment
            rows_fdr = []
            for metric, g in df_t.groupby('metric'):
                pvals = g['p_value'].values
                if len(pvals) == 0: continue
                rejected, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                for (idx, row), rj, pa in zip(g.iterrows(), rejected, p_adj):
                    rows_fdr.append({
                        'metric': metric,
                        'Model_A_Better': row['Model_A_Better'],
                        'Model_B_Worse': row['Model_B_Worse'],
                        'p_value': row['p_value'],
                        'p_value_fdr': float(pa),
                        'reject_H0_Alpha0.05': bool(rj)
                    })
            df_fdr = pd.DataFrame(rows_fdr)
            csv_fdr = os.path.join(out_dir, 'model_pairwise_fdr_adjusted.csv')
            df_fdr.to_csv(csv_fdr, index=False)
            print('[OK] Wrote:', csv_t)
            print('[OK] Wrote:', csv_fdr)

if __name__ == '__main__':
    main()
