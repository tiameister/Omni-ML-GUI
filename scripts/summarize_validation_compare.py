"""
Summarize per-split CV metrics for RandomForest across kfold, repeated, nested.
Outputs:
- validation_compare/tables/cv_summary_stats.csv
- validation_compare/tables/cv_summary_stats.tex (LaTeX table)
"""
import os
from utils.paths import EVALUATION_DIR
import glob
import pandas as pd
import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'validation_compare')

STRATS = ['kfold', 'repeated', 'nested']


def _read_cv(strategy: str) -> pd.DataFrame:
    pattern = os.path.join(BASE, strategy, '*_output*', EVALUATION_DIR, 'cv_splits.xlsx')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['R2','RMSE','MAE'])
    path = files[-1]
    try:
        xls = pd.ExcelFile(path)
        df = pd.read_excel(xls, xls.sheet_names[0])
    except Exception:
        LOGGER.exception("Failed reading CV splits workbook: %s", path)
        return pd.DataFrame(columns=['R2','RMSE','MAE'])
    
    out = pd.DataFrame({
        'R2': pd.to_numeric(df.get('R2', pd.Series([], dtype=float)), errors='coerce'),
        'RMSE': pd.to_numeric(df.get('RMSE', pd.Series([], dtype=float)), errors='coerce'),
        'MAE': pd.to_numeric(df.get('MAE', pd.Series([], dtype=float)), errors='coerce'),
    }).dropna(how='all')
    out['strategy'] = strategy
    return out


def _ci95(x: pd.Series) -> tuple[float, float]:
    x = x.dropna().values
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return (mean - 1.96 * se, mean + 1.96 * se)


def main():
    frames = []
    for s in STRATS:
        frames.append(_read_cv(s))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        print('[WARN] No cv_splits.xlsx found under validation_compare. Run scripts/run_validation_compare.py first.')
        return
    os.makedirs(os.path.join(BASE, 'tables'), exist_ok=True)
    rows = []
    for strat, g in df.groupby('strategy'):
        for metric in ['R2','RMSE','MAE']:
            x = g[metric].dropna()
            if x.empty:
                continue
            ci_low, ci_high = _ci95(x)
            rows.append({
                'strategy': strat,
                'metric': metric,
                'n_splits': int(x.shape[0]),
                'mean': float(x.mean()),
                'std': float(x.std(ddof=1)) if x.shape[0] > 1 else float('nan'),
                'median': float(x.median()),
                'iqr': float(x.quantile(0.75) - x.quantile(0.25)),
                'ci95_low': ci_low,
                'ci95_high': ci_high,
            })
    summary = pd.DataFrame(rows)
    csv_path = os.path.join(BASE, 'tables', 'cv_summary_stats.csv')
    tex_path = os.path.join(BASE, 'tables', 'cv_summary_stats.tex')
    summary.sort_values(['metric','strategy']).to_csv(csv_path, index=False)
    try:
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(summary.pivot(index='strategy', columns='metric', values='mean').round(3).to_latex())
    except Exception as e:
        print('[WARN] Could not write LaTeX:', e)
    print('[OK] Wrote:', csv_path)
    print('[OK] Wrote:', tex_path)

if __name__ == '__main__':
    main()
