"""
XAI consistency: fuse feature rankings across multiple models using Borda count and bootstrap.
Outputs:
- output/xai_consistency/borda_ranking.csv
- output/xai_consistency/bootstrap_stability.csv
"""
from __future__ import annotations

import os
from utils.paths import EVALUATION_DIR
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run_root = os.environ.get("MLTRAINER_RUN_ROOT", "").strip()
if run_root and os.path.isdir(run_root):
    OUTDIR = os.path.join(run_root, EVALUATION_DIR, 'xai_consistency')
else:
    OUTDIR = os.path.join(ROOT, 'output', 'xai_consistency')
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ['RandomForest', 'HistGB', 'XGBoost']  # XGBoost included if importance file present


def load_importances(model: str, collapse_levels: bool = True) -> pd.DataFrame:
    # look for feature_importance_{model}.csv under output/ or MLTRAINER_RUN_ROOT
    base = os.path.join(ROOT, 'output')
    cand = []
    run_root = os.environ.get("MLTRAINER_RUN_ROOT", "").strip()
    
    search_dirs = [base]
    if run_root and os.path.isdir(run_root):
        search_dirs.append(run_root)

    for search_dir in search_dirs:
        for root_dir, dirs, files in os.walk(search_dir):
            fname = f'feature_importance_{model}.csv'
            if fname in files:
                p = os.path.join(root_dir, fname)
                cand.append((os.path.getmtime(p), p))

    if not cand:
        return pd.DataFrame(columns=['feature','perm_importance_mean'])
    cand.sort(reverse=True)
    path = cand[0][1]
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=['feature','perm_importance_mean','perm_importance_std'])
    # Ensure expected columns even if alternative naming occurs
    if 'perm_importance_mean' not in df.columns:
        # try to infer numeric columns
        num_cols = [c for c in df.columns if c != 'feature']
        if num_cols:
            df = df.rename(columns={num_cols[0]: 'perm_importance_mean'})
    if 'perm_importance_std' not in df.columns:
        df['perm_importance_std'] = 0.0
    df = df[['feature','perm_importance_mean','perm_importance_std']]
    if collapse_levels and not df.empty:
        # Collapse one-hot like patterns feature_variant (e.g., Reading Books (Frequency)_5)
        base_names = []
        for f in df['feature'].astype(str):
            # Split at last underscore if trailing token is short or numeric
            if '_' in f:
                head, tail = f.rsplit('_', 1)
                if tail.replace('.0','').isdigit() or len(tail) <= 3:
                    base_names.append(head)
                else:
                    base_names.append(f)
            else:
                base_names.append(f)
        df['_base'] = base_names
        grouped = df.groupby('_base', as_index=False).agg({
            'perm_importance_mean': 'sum',  # sum contributions of levels
            'perm_importance_std': 'mean',  # average std (approximate)
        }).rename(columns={'_base':'feature'})
        df = grouped.sort_values('perm_importance_mean', ascending=False)
    return df


def borda_rank(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # unify features
    feats = set()
    for df in dfs:
        feats.update(df['feature'].dropna().astype(str).tolist())
    feats = sorted(feats)
    # assign ranks within each model (higher importance -> better rank)
    ranks = {f: 0 for f in feats}
    for df in dfs:
        df = df.dropna(subset=['feature']).copy()
        df['rank'] = df['perm_importance_mean'].rank(ascending=False, method='min')
        rmap = df.set_index('feature')['rank'].to_dict()
        for f in feats:
            ranks[f] += rmap.get(f, len(feats))  # missing -> worst rank
    # lower total score is better; convert to final rank
    arr = [(f, s) for f, s in ranks.items()]
    arr.sort(key=lambda x: x[1])
    out = pd.DataFrame(arr, columns=['feature','borda_score'])
    out['borda_rank'] = out['borda_score'].rank(method='min')
    return out


def bootstrap_stability(
    dfs: list[pd.DataFrame],
    B: int = 1000,
    seed: int = 42,
    noise_scale: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stability via perturbation of importances using their empirical std and added scale.

    Enhancements:
      - Records full Borda rank distribution (mean, sd, quantiles) instead of only Top-10 hits.
      - Adds probability for Top-3 / Top-5 / Top-10.
      - Introduces a *noise_scale* factor to avoid degenerate 0/1 inclusion when raw std=0.

    Returns
    -------
    (stab_topk, rank_metrics)
      stab_topk: legacy style with top10 inclusion frequency.
      rank_metrics: extended metrics with rank distribution statistics.
    """
    rng = np.random.default_rng(seed)
    feats = sorted(set().union(*[set(df['feature'].dropna().astype(str)) for df in dfs]))
    # Storage for rank samples
    rank_samples = {f: [] for f in feats}
    top10_counts = {f: 0 for f in feats}
    top5_counts = {f: 0 for f in feats}
    top3_counts = {f: 0 for f in feats}

    # Pre-index each df by feature for rapid lookup
    lookup = []
    for df in dfs:
        d = df.set_index('feature') if not df.empty else pd.DataFrame(columns=['perm_importance_mean','perm_importance_std']).set_index(pd.Index([]))
        lookup.append(d)

    for _ in range(B):
        sample_dfs = []
        for d in lookup:
            if d.empty:
                continue
            means = d['perm_importance_mean']
            stds = d.get('perm_importance_std', pd.Series(0.0, index=d.index))
            # Expand zero or tiny std with global dispersion * noise_scale to introduce plausible rank jitter
            global_disp = means.std() if means.std() > 0 else (means.max() - means.min()) / 6.0 if means.max() > means.min() else 1e-6
            adj_stds = stds.replace(0, global_disp * noise_scale).fillna(global_disp * noise_scale)
            # Ensure strictly positive
            adj_stds = adj_stds.clip(lower=global_disp * noise_scale * 0.05)
            pert = means + rng.normal(0, adj_stds)
            sample_df = pd.DataFrame({
                'feature': means.index,
                'perm_importance_mean': pert.values,
                'perm_importance_std': adj_stds.values,
            })
            sample_dfs.append(sample_df)
        if not sample_dfs:
            continue
        br = borda_rank(sample_dfs)
        # Borda rank: lower is better
        for _, row in br.iterrows():
            rank_samples[row.feature].append(row.borda_rank)
        # Top-k memberships
        br_sorted = br.sort_values('borda_rank')
        top3 = set(br_sorted.head(3)['feature'])
        top5 = set(br_sorted.head(5)['feature'])
        top10 = set(br_sorted.head(min(10, len(br_sorted)))['feature'])
        for f in top3: top3_counts[f] += 1
        for f in top5: top5_counts[f] += 1
        for f in top10: top10_counts[f] += 1

    # Build legacy style
    stab = pd.DataFrame({'feature': feats, 'top10_freq': [top10_counts[f] for f in feats]})
    stab['top10_pct'] = stab['top10_freq'] / float(B)

    # Extended metrics
    stats_rows = []
    for f in feats:
        rs = rank_samples[f]
        if rs:
            arr = np.array(rs, dtype=float)
            stats_rows.append({
                'feature': f,
                'samples': len(arr),
                'mean_rank': arr.mean(),
                'median_rank': np.median(arr),
                'rank_std': arr.std(ddof=1) if arr.size > 1 else 0.0,
                'q05': np.quantile(arr, 0.05),
                'q25': np.quantile(arr, 0.25),
                'q75': np.quantile(arr, 0.75),
                'q95': np.quantile(arr, 0.95),
                'p_top3': top3_counts[f] / float(B),
                'p_top5': top5_counts[f] / float(B),
                'p_top10': top10_counts[f] / float(B),
            })
        else:
            stats_rows.append({
                'feature': f,
                'samples': 0,
                'mean_rank': np.nan,
                'median_rank': np.nan,
                'rank_std': np.nan,
                'q05': np.nan,
                'q25': np.nan,
                'q75': np.nan,
                'q95': np.nan,
                'p_top3': 0.0,
                'p_top5': 0.0,
                'p_top10': 0.0,
            })
    rank_metrics = pd.DataFrame(stats_rows)
    rank_metrics = rank_metrics.sort_values('mean_rank')
    return stab.sort_values('top10_pct', ascending=False), rank_metrics


def main():
    base = os.path.join(ROOT, 'output')
    run_root = os.environ.get("MLTRAINER_RUN_ROOT", "").strip()
    
    search_dirs = [base]
    if run_root and os.path.isdir(run_root):
        search_dirs.append(run_root)

    # Automatically resolve models by finding their importance files
    found_models = set()
    for search_dir in search_dirs:
        for root_dir, dirs, files in os.walk(search_dir):
            for fname in files:
                if fname.startswith('feature_importance_') and fname.endswith('.csv'):
                    model_name = fname[len('feature_importance_'):-4]
                    if model_name:
                        found_models.add(model_name)

    models_to_check = list(found_models)
    if not models_to_check:
        models_to_check = ['RandomForest', 'HistGB', 'XGBoost', 'LinearRegression', 'Ridge', 'LGBM', 'TabNet']

    dfs = []
    for m in models_to_check:
        df = load_importances(m, collapse_levels=True)
        if not df.empty:
            dfs.append(df[['feature','perm_importance_mean','perm_importance_std']].copy())
    if not dfs:
        print('[WARN] No importance files found under output/*_output/.')
        return
    br = borda_rank([d[['feature','perm_importance_mean']].copy() for d in dfs])
    br.to_csv(os.path.join(OUTDIR, 'borda_ranking.csv'), index=False)
    stab, rank_metrics = bootstrap_stability(dfs)
    stab.to_csv(os.path.join(OUTDIR, 'bootstrap_stability.csv'), index=False)
    rank_metrics.to_csv(os.path.join(OUTDIR, 'bootstrap_rank_metrics.csv'), index=False)
    print('[OK] Wrote:', os.path.join(OUTDIR, 'borda_ranking.csv'))
    print('[OK] Wrote:', os.path.join(OUTDIR, 'bootstrap_stability.csv'))
    print('[OK] Wrote:', os.path.join(OUTDIR, 'bootstrap_rank_metrics.csv'))


if __name__ == '__main__':
    main()
