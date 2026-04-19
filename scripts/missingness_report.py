"""
Missingness report and optional Little's MCAR test.
Outputs under a dedicated folder for easy discovery:
- output/mcar/missingness_by_variable.csv
- output/mcar/missingness_summary.txt
Optionally:
- output/mcar/little_mcar_test.txt (if package available)
"""
import os
from utils.paths import EVALUATION_DIR
import pandas as pd
import numpy as np

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = str(os.environ.get('MLTRAINER_RUN_ROOT', '') or '').strip()
ANALYSIS_ROOT = str(os.environ.get('MLTRAINER_ANALYSIS_ROOT', '') or '').strip()
try:
    import config as cfg
    OUTPUT_ROOT = cfg.OUTPUT_DIR
except Exception:
    OUTPUT_ROOT = 'output'

if ANALYSIS_ROOT:
    BASE = os.path.join(ANALYSIS_ROOT, 'mcar')
elif RUN_ROOT and os.path.isdir(RUN_ROOT):
    BASE = os.path.join(RUN_ROOT, EVALUATION_DIR, 'mcar')
else:
    BASE = os.path.join(ROOT, OUTPUT_ROOT, 'mcar')
os.makedirs(BASE, exist_ok=True)

_env_ds = os.environ.get('DATASET_PATH')
if _env_ds and _env_ds.strip():
    DATASET_PATH = _env_ds
else:
    _clean = os.path.join(ROOT, 'dataset', 'data_cleaned.csv')
    _raw = os.path.join(ROOT, 'dataset', 'data.csv')
    DATASET_PATH = _clean if os.path.exists(_clean) else _raw
from data.loader import read_csv_safely


def little_mcar_test(df: pd.DataFrame) -> str:
    """Run Little's MCAR test.
    Preference order:
    1) pingouin.mcar if available
    2) built-in fallback (approximate Little 1988)
    Returns a formatted line with chi2, dof, p.
    """
    # Prepare a numeric-coded DataFrame preserving NaNs for categorical/object columns
    df_enc = df.copy()
    for col in df_enc.columns:
        s = df_enc[col]
        if not pd.api.types.is_numeric_dtype(s):
            try:
                cat = s.astype('category')
                codes = cat.cat.codes.replace(-1, np.nan)
                df_enc[col] = codes
            except Exception:
                LOGGER.exception("Failed encoding column for MCAR test: %s", col)

    # Try pingouin if it exposes mcar
    try:
        import pingouin as pg  # type: ignore
        if hasattr(pg, 'mcar'):
            res = pg.mcar(df_enc)
            chi2 = float(res.loc[0, 'chi2'])
            dof = int(res.loc[0, 'df'])
            pval = float(res.loc[0, 'p'])
            return f"Little's MCAR: chi2={chi2:.3f}, dof={dof}, p={pval:.4g}\n"
    except Exception:
        LOGGER.exception("Little's MCAR test failed (pingouin path)")

    # Fallback approximate implementation
    try:
        X = df_enc.apply(pd.to_numeric, errors='coerce')
        cols = X.columns.tolist()
        p = len(cols)
        cc = X.dropna()
        if cc.shape[0] < 3 or p == 0:
            raise ValueError('Insufficient complete cases for MCAR approximation')
        var = cc.var(axis=0)
        keep = var[var > 0].index.tolist()
        X = X[keep]
        cols = keep
        p = len(cols)
        cc = X.dropna()
        mu = cc.mean(axis=0).to_numpy()
        Sigma = np.cov(cc.to_numpy().T, ddof=1)
        if p == 1:
            Sigma = np.array([[Sigma]])
        T = 0.0
        df_out = 0
        mask_mat = ~X.isna()
        patterns = mask_mat.drop_duplicates()
        for _, pat in patterns.iterrows():
            o_idx = np.where(pat.values)[0]
            if len(o_idx) == 0:
                continue
            # select rows matching this exact missingness pattern
            rows_sel = (mask_mat.eq(pat)).all(axis=1)
            subset = X.loc[rows_sel]
            n_k = subset.shape[0]
            if n_k == 0:
                continue
            xbar_k = subset.iloc[:, o_idx].mean(axis=0).to_numpy()
            mu_o = mu[o_idx]
            Sigma_oo = Sigma[np.ix_(o_idx, o_idx)]
            inv_S = np.linalg.pinv(Sigma_oo)
            d = xbar_k - mu_o
            T += n_k * float(d.T @ inv_S @ d)
            df_out += len(o_idx)
        df_out = df_out - p if df_out > p else 1
        try:
            from scipy.stats import chi2  # type: ignore
            pval = float(1.0 - chi2.cdf(T, df_out))
        except Exception:
            pval = float('nan')
        return f"Little's MCAR (approx.): chi2={T:.3f}, dof={df_out}, p={pval:.4g}\n"
    except Exception as e:
        return f"Little's MCAR test failed: {e}\n"


def main():
    # Use robust loader to avoid encoding/format issues
    df, _ = read_csv_safely(DATASET_PATH)
    n = df.shape[0]

    miss_counts = df.isna().sum().rename('n_missing')
    miss_pct = (df.isna().mean() * 100).rename('pct_missing')
    by_var = pd.concat([miss_counts, miss_pct], axis=1).sort_values('pct_missing', ascending=False)
    out_csv = os.path.join(BASE, 'missingness_by_variable.csv')
    by_var.to_csv(out_csv)

    # Overall summary
    total_cells = int(df.size)
    total_missing = int(df.isna().sum().sum())
    overall_pct = 100.0 * total_missing / total_cells if total_cells else 0.0

    # Rows removed prior to modeling (approx.) — align with main.py logic
    # We assume numeric columns were required complete; compute how many rows have any numeric NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows_num_nan = int((~df[numeric_cols].notna().all(axis=1)).sum()) if numeric_cols else 0

    lines = []
    lines.append(f"Rows: {n}")
    lines.append(f"Total cells: {total_cells}")
    lines.append(f"Total missing cells: {total_missing} ({overall_pct:.1f}%)")
    lines.append(f"Rows with any numeric missingness: {rows_num_nan}")

    # Optional Little's MCAR test
    mcar_txt = little_mcar_test(df.copy())
    lines.append(mcar_txt.strip())

    with open(os.path.join(BASE, 'missingness_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    # Optional: variable-level missingness heatmap could be added if needed
    print('[OK] Wrote:', out_csv)
    print('[OK] Wrote:', os.path.join(BASE, 'missingness_summary.txt'))
    if "MCAR" in mcar_txt:
        with open(os.path.join(BASE, 'little_mcar_test.txt'), 'w', encoding='utf-8') as f:
            f.write(mcar_txt + "\n")
        print('[OK] Wrote:', os.path.join(BASE, 'little_mcar_test.txt'))


if __name__ == '__main__':
    main()
