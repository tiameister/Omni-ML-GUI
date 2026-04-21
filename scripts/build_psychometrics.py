"""Psychometric analysis for Happiness Score scale (m1–m9).

Outputs (created under supplements/):
 tables/
   psychometrics_item_stats.csv              # Descriptives + item-total r + loadings
   psychometrics_summary_table.csv           # Same + alpha row
   psychometrics_reliability_overview.csv    # Alpha, KMO, Bartlett, N items/obs
   psychometrics_factor_eigenvalues.csv      # Eigenvalues (if EFA or PCA)
   psychometrics_parallel_analysis_mean.csv  # Parallel analysis average eigenvalues (if feasible)
 figures/
   psychometrics_scree_plot.png
   psychometrics_parallel_analysis.png

Robust to missing factor_analyzer: falls back to PCA 1-component loadings.
Dataset auto-detected (semicolon or comma delimiter) from dataset/data_cleaned.csv then dataset/data.csv.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.paths import get_supplements_root

LOGGER = get_logger(__name__)

ITEM_COLS = [f"m{i}" for i in range(1,10)]  # m1..m9
DATA_CANDIDATES = [Path("dataset/data_cleaned.csv"), Path("dataset/data.csv")]
SUPP_ROOT = get_supplements_root()
OUT_TABLE = SUPP_ROOT / "tables"
OUT_FIG = SUPP_ROOT / "figures"

# Try to import factor_analyzer, else mark fallback
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    _FA_OK = True
except Exception:
    _FA_OK = False

def _read_dataset() -> pd.DataFrame:
    for p in DATA_CANDIDATES:
        if p.exists():
            # Try delimiter detection: attempt semicolon then comma
            for sep in [';','\t',',']:
                try:
                    df = pd.read_csv(p, sep=sep)
                    if set(ITEM_COLS) & set(df.columns):
                        return df
                except Exception:
                    LOGGER.exception("Dataset read failed (sep=%s) for %s", sep, p)
            # fallback to default
            return pd.read_csv(p)
    raise FileNotFoundError("No dataset found (data_cleaned.csv or data.csv)")

def cronbach_alpha(df: pd.DataFrame) -> float:
    df = df.dropna(how='any')
    k = df.shape[1]
    if k <= 1:
        return float('nan')
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float('nan')
    return float(k/(k-1) * (1 - item_vars.sum() / total_var))

def item_total_corr(df: pd.DataFrame) -> pd.Series:
    total = df.sum(axis=1)
    out = {}
    for c in df.columns:
        out[c] = df[c].corr(total - df[c])
    return pd.Series(out, name='item_total_r')

def floor_ceiling(series: pd.Series):
    mn, mx = series.min(), series.max()
    return (series == mn).mean(), (series == mx).mean()

def pca_one_component(df: pd.DataFrame) -> pd.Series:
    """Return first component loadings (standardized) as PCA fallback if factor_analyzer not available."""
    from sklearn.decomposition import PCA
    # standardize items (z-scores) to approximate correlation matrix PCA
    Z = (df - df.mean()) / df.std(ddof=1)
    Z = Z.replace([np.inf,-np.inf], np.nan).dropna()
    pca = PCA(n_components=1, random_state=42)
    pca.fit(Z.values)
    # Loadings approximated by component weights * sqrt(eigenvalue)
    eig = pca.explained_variance_[0]
    loadings = pca.components_[0] * np.sqrt(eig)
    return pd.Series(loadings, index=df.columns, name='loading_1f')

def pca_loadings(df: pd.DataFrame, n_components: int = 2):
    """Return PCA loadings for first n_components (standardized input)."""
    from sklearn.decomposition import PCA
    Z = (df - df.mean()) / df.std(ddof=1)
    Z = Z.replace([np.inf,-np.inf], np.nan).dropna()
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(Z.values)
    ev = pca.explained_variance_[:n_components]
    comps = pca.components_[:n_components]
    # scale by sqrt eigenvalue to get loadings
    load = comps.T * np.sqrt(ev)
    cols = [f'F{i+1}' for i in range(n_components)]
    return pd.DataFrame(load, index=df.columns, columns=cols)

def parallel_analysis(df: pd.DataFrame, B: int = 200, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    from sklearn.decomposition import PCA
    Z = (df - df.mean()) / df.std(ddof=1)
    Z = Z.replace([np.inf,-np.inf], np.nan).dropna()
    n, k = Z.shape
    rand_eigs = []
    for _ in range(B):
        # permute each column independently
        perm = np.column_stack([rng.permutation(Z.iloc[:,j].values) for j in range(k)])
        p = PCA().fit(perm)
        ev = p.explained_variance_[:k]
        rand_eigs.append(ev)
    rand_eigs = np.vstack(rand_eigs)
    return rand_eigs.mean(axis=0)

def main():
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df = _read_dataset()
    items = [c for c in ITEM_COLS if c in df.columns]
    if len(items) < 3:
        raise ValueError(f"Not enough item columns found in dataset: {items}")
    dfi = df[items].apply(pd.to_numeric, errors='coerce').dropna(how='any')
    n_obs = len(dfi)

    # Descriptives
    rows = []
    for c in items:
        s = dfi[c]
        f_pct, c_pct = floor_ceiling(s)
        rows.append({
            'item': c,
            'mean': s.mean(),
            'sd': s.std(ddof=1),
            'min': s.min(),
            'max': s.max(),
            'floor_pct': f_pct,
            'ceiling_pct': c_pct
        })
    desc = pd.DataFrame(rows)

    alpha = cronbach_alpha(dfi)
    itcorr = item_total_corr(dfi)
    desc = desc.merge(itcorr.reset_index().rename(columns={'index':'item'}), on='item', how='left')

    # Factor / Loadings
    kmo_val = bartlett_chi2 = bartlett_p = None
    loadings = None
    eigen_df = None
    if _FA_OK:
        try:
            # Full factor analyzer for eigenvalues
            fa_all = FactorAnalyzer(n_factors=len(items), rotation=None, method='ml')
            fa_all.fit(dfi)
            ev, _ = fa_all.get_eigenvalues()
            eigen_df = pd.DataFrame({'component': range(1, len(ev)+1), 'eigenvalue': ev})
            fa1 = FactorAnalyzer(n_factors=1, rotation=None, method='ml')
            fa1.fit(dfi)
            loadings = pd.Series(fa1.loadings_.flatten(), index=items, name='loading_1f')
            kmo_all, kmo_model = calculate_kmo(dfi)
            kmo_val = float(kmo_model)
            bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(dfi)
        except Exception:
            loadings = None
    if loadings is None:
        # PCA fallback
        loadings = pca_one_component(dfi)
        # Eigenvalues from PCA
        from sklearn.decomposition import PCA
        Z = (dfi - dfi.mean()) / dfi.std(ddof=1)
        Z = Z.replace([np.inf,-np.inf], np.nan).dropna()
        p = PCA().fit(Z.values)
        ev = p.explained_variance_[:len(items)]
        eigen_df = pd.DataFrame({'component': range(1, len(ev)+1), 'eigenvalue': ev})

    desc = desc.merge(loadings.reset_index().rename(columns={'index':'item'}), on='item', how='left')

    # Order by loading then item-total
    desc = desc.sort_values(['loading_1f','item_total_r'], ascending=[False, False])

    # Summary table with alpha row
    alpha_row = {c: np.nan for c in desc.columns}
    alpha_row['item'] = 'Cronbach_alpha'
    summary = pd.concat([desc, pd.DataFrame([alpha_row])], ignore_index=True)

    # Save core tables
    desc.to_csv(OUT_TABLE / 'psychometrics_item_stats.csv', index=False)
    summary.to_csv(OUT_TABLE / 'psychometrics_summary_table.csv', index=False)
    pd.DataFrame([{
        'alpha': alpha,
        'n_items': len(items),
        'n_obs': n_obs,
        'kmo': kmo_val,
        'bartlett_chi2': bartlett_chi2,
        'bartlett_p': bartlett_p
    }]).to_csv(OUT_TABLE / 'psychometrics_reliability_overview.csv', index=False)
    if eigen_df is not None:
        eigen_df.to_csv(OUT_TABLE / 'psychometrics_factor_eigenvalues.csv', index=False)

    # Scree plot
    if eigen_df is not None:
        plt.figure(figsize=(5,4))
        plt.plot(eigen_df['component'], eigen_df['eigenvalue'], marker='o')
        plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
        plt.title('Scree Plot')
        plt.xlabel('Component')
        plt.ylabel('Eigenvalue')
        plt.tight_layout()
        plt.savefig(OUT_FIG / 'psychometrics_scree_plot.png', dpi=300)
        plt.close()

    # Parallel analysis (always via PCA fallback method for speed)
    try:
        pa_mean = parallel_analysis(dfi)
        pa_df = pd.DataFrame({'component': range(1, len(pa_mean)+1), 'pa_mean_eigen': pa_mean})
        pa_df.to_csv(OUT_TABLE / 'psychometrics_parallel_analysis_mean.csv', index=False)
        if eigen_df is not None:
            plt.figure(figsize=(5,4))
            plt.plot(eigen_df['component'], eigen_df['eigenvalue'], marker='o', label='Observed')
            plt.plot(pa_df['component'], pa_df['pa_mean_eigen'], marker='x', label='Parallel Mean')
            plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
            plt.legend()
            plt.title('Parallel Analysis')
            plt.xlabel('Component')
            plt.ylabel('Eigenvalue')
            plt.tight_layout()
            plt.savefig(OUT_FIG / 'psychometrics_parallel_analysis.png', dpi=300)
            plt.close()
            # Determine number of components suggested by PA
            merged = eigen_df.merge(pa_df, on='component')
            n_factors_pa = int((merged['eigenvalue'] > merged['pa_mean_eigen']).sum())
            if n_factors_pa >= 2:
                # produce two-factor loadings (PCA fallback if FA not available)
                if _FA_OK:
                    try:
                        fa2 = FactorAnalyzer(n_factors=2, rotation='varimax', method='ml')
                        fa2.fit(dfi)
                        l2 = pd.DataFrame(fa2.loadings_, index=items, columns=['F1','F2'])
                    except Exception:
                        l2 = pca_loadings(dfi, 2)
                else:
                    l2 = pca_loadings(dfi, 2)
                l2.reset_index(names='item').to_csv(OUT_TABLE / 'psychometrics_two_factor_loadings.csv', index=False)
    except Exception:
        LOGGER.exception("Psychometrics pipeline failed")

    print(f"[OK] Items: {items}  N={n_obs}  Alpha={alpha:.3f}")
    if kmo_val is not None:
        print(f"[OK] KMO={kmo_val:.3f} Bartlett χ²={bartlett_chi2} p={bartlett_p}")
    print(f"[OK] Tables -> {OUT_TABLE}")
    print(f"[OK] Figures -> {OUT_FIG}")
    # If two-factor model added
    if os.path.exists(OUT_TABLE / 'psychometrics_two_factor_loadings.csv'):
        print("[OK] Two-factor loadings saved (parallel analysis indicated >=2 factors)")

if __name__ == '__main__':
    main()
