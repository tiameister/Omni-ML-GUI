"""
Statistical tests across CV strategies/models using per-split scores.
Outputs:
- output/stats_tests/corrected_t_tests.csv
- output/stats_tests/fdr_adjusted_pvalues.csv
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'output', 'stats_tests')
os.makedirs(OUTDIR, exist_ok=True)

BASE_VC = os.path.join(ROOT, 'validation_compare')
STRATS = ['kfold', 'repeated', 'nested']
# Analyze the best model or first model across strategies instead of a fixed name


def _read_scores(strategy: str, model: str) -> pd.DataFrame:
    pattern = os.path.join(BASE_VC, strategy, f'{model}_output*', '1_Overall_Evaluation', 'cv_splits.xlsx')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['R2', 'RMSE', 'MAE'])
    path = files[-1]
    try:
        df = pd.read_excel(path, sheet_name=model)
    except Exception:
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, xls.sheet_names[0])
    return df[['R2', 'RMSE', 'MAE']].dropna(how='all')


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
    # Correction term c = 1/k + 1/(k*(r-1)) for repeated k-fold; for single kfold ~ 1/k
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


def _get_best_model_name(strategy):
    # Peek into the first available output directory
    pat = os.path.join(BASE, strategy, '*_output*', '1_Overall_Evaluation', 'cv_splits.xlsx')
    cands = glob.glob(pat)
    if cands:
        import re
        m = re.search(r"([A-Za-z0-9]+)_output", cands[0])
        if m: return m.group(1)
    return "RandomForest"

def main():
    rows_t = []
    metrics = ['R2', 'RMSE', 'MAE']

    MODEL = _get_best_model_name('kfold')
    
    # Pairwise tests between strategies for the best model
    data = {s: _read_scores(s, MODEL) for s in STRATS}
    # Determine k and r for each strategy from lengths (best-effort)
    # Assume k=10 for kfold; repeated ~ 10*r; nested: compare outer folds only (k=10)
    info = {
        'kfold': {'k': 10, 'r': 1},
        'repeated': {'k': 10, 'r': max(1, len(data['repeated']) // 10)},
        'nested': {'k': 10, 'r': 1},
    }

    pairs = [('kfold', 'repeated'), ('kfold', 'nested'), ('repeated', 'nested')]
    for metric in metrics:
        for a, b in pairs:
            if data[a].empty or data[b].empty:
                continue
            s_a = pd.to_numeric(data[a][metric], errors='coerce').dropna().values
            s_b = pd.to_numeric(data[b][metric], errors='coerce').dropna().values
            n = min(len(s_a), len(s_b))
            if n < 2:
                continue
            # Align lengths
            s_a = s_a[:n]
            s_b = s_b[:n]
            t_stat, p_val = corrected_resampled_ttest(s_a, s_b, info[a]['k'], info[a]['r'])
            rows_t.append({
                'metric': metric,
                'A': a,
                'B': b,
                't_stat': t_stat,
                'p_value': p_val,
                'n': n
            })

    df_t = pd.DataFrame(rows_t)
    csv_t = os.path.join(OUTDIR, 'corrected_t_tests.csv')
    df_t.to_csv(csv_t, index=False)

    # FDR adjustment across all tests per metric
    rows_fdr = []
    for metric, g in df_t.groupby('metric'):
        pvals = g['p_value'].values
        if len(pvals) == 0:
            continue
        rejected, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for (idx, row), rj, pa in zip(g.iterrows(), rejected, p_adj):
            rows_fdr.append({
                'metric': metric,
                'A': row['A'],
                'B': row['B'],
                'p_value': row['p_value'],
                'p_value_fdr': float(pa),
                'reject_H0': bool(rj)
            })
    df_fdr = pd.DataFrame(rows_fdr)
    csv_fdr = os.path.join(OUTDIR, 'fdr_adjusted_pvalues.csv')
    df_fdr.to_csv(csv_fdr, index=False)

    print('[OK] Wrote:', csv_t)
    print('[OK] Wrote:', csv_fdr)


if __name__ == '__main__':
    main()
