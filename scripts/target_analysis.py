"""
Target variable descriptive analysis for publication.

Outputs:
- output/target/target_summary.json : numeric stats
- output/target/target_summary.txt  : Q1-ready sentence
- output/target/target_hist_kde.png/.pdf/.svg : histogram + KDE plot
- output/target/target_qq.png/.pdf/.svg       : Q-Q plot
- output/target/target_hist_qq.png/.pdf/.svg  : 2-panel composite (hist+Q-Q)
- output/target/target_box.png/.pdf/.svg      : boxplot

Logic mirrors main.py for target detection and cleaning so numbers match the modeling dataset.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.loader import detect_cols, read_csv_safely
from config import OUTPUT_DIR
from config.columns import resolve_column_groups


ROOT = os.path.dirname(os.path.dirname(__file__))


def _ensure_out(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_all(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.svg', bbox_inches='tight')


def describe_target(csv_path: str, out_dir: str) -> Tuple[dict, str]:
    df, _ = read_csv_safely(csv_path)
    target, bully, m_items, z_items, bully_subs, drop_cols = detect_cols(df)

    # Feature set and target preparation mirrors main.py.
    exclude = set([target] + m_items + z_items + bully_subs + drop_cols)
    feature_cols = [c for c in df.columns if c not in exclude]
    if bully not in feature_cols:
        feature_cols = [bully] + feature_cols

    y_raw = df[target]
    y_num = pd.to_numeric(y_raw, errors='coerce')
    if y_num.notna().mean() >= 0.8:
        y = y_num
        target_mode = 'numeric'
    else:
        uniq = sorted(y_raw.dropna().unique(), key=lambda x: str(x))
        y = y_raw.map({v: i + 1 for i, v in enumerate(uniq)})
        target_mode = 'ordinalized'

    # Drop rows with missing numeric predictors and target, mirroring training.
    X = df[feature_cols].copy()
    num_cols, ordinal_cols, binary_cols, other_cat_cols = resolve_column_groups(list(X.columns))
    _ = (ordinal_cols, binary_cols, other_cat_cols)
    mask = y.notna()
    for c in num_cols:
        mask &= X[c].notna()
    y = y.loc[mask].astype(float)

    n = int(y.shape[0])
    mean = float(y.mean()) if n else float('nan')
    std = float(y.std(ddof=1)) if n > 1 else float('nan')
    vmin = float(y.min()) if n else float('nan')
    vmax = float(y.max()) if n else float('nan')
    skew = float(pd.Series(y).skew()) if n else float('nan')
    kurt = float(pd.Series(y).kurt()) if n else float('nan')

    try:
        from scipy.stats import shapiro
        if n >= 3 and n <= 5000:
            w_stat, p_shapiro = shapiro(y.dropna().values)
        else:
            w_stat, p_shapiro = (np.nan, np.nan)
    except Exception:
        w_stat, p_shapiro = (np.nan, np.nan)

    transform_note = "No transformation applied (distribution reasonably symmetric)."
    if np.isfinite(skew) and abs(skew) > 0.8:
        transform_note = "Highly skewed; consider log/Box-Cox if substantive justification exists."
    elif np.isfinite(skew) and abs(skew) > 0.4:
        transform_note = "Mild-to-moderate skewness; transformation may be considered, but raw scale retained here."

    stats = {
        'n': n,
        'mean': mean,
        'sd': std,
        'min': vmin,
        'max': vmax,
        'skewness': skew,
        'kurtosis': kurt,
        'shapiro_W': w_stat,
        'shapiro_p': float(p_shapiro) if p_shapiro is not np.nan else np.nan,
        'target_mode': target_mode,
    }

    _ensure_out(out_dir)
    sns.set_theme(context='paper', style='whitegrid')

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    sns.histplot(y, kde=True, bins='auto', color='#1565C0', edgecolor='white', alpha=0.9, ax=ax)
    ax.set_xlabel(target)
    ax.set_title('Target distribution (histogram + KDE)')
    fig.tight_layout()
    _save_all(fig, os.path.join(out_dir, 'target_hist_kde.png'))
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    sns.boxplot(x=y, color='#42A5F5', ax=ax)
    ax.set_xlabel(target)
    ax.set_title('Target distribution (boxplot)')
    fig.tight_layout()
    _save_all(fig, os.path.join(out_dir, 'target_box.png'))
    plt.close(fig)

    from scipy import stats as spstats
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    (osm, osr), (slope, intercept, r_val) = spstats.probplot(y.dropna().values, dist='norm')
    _ = r_val
    ax.scatter(osm, osr, s=12, alpha=0.7, color='#424242', edgecolors='none')
    x_line = np.linspace(min(osm), max(osm), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', lw=1)
    ax.set_title('Q-Q plot vs Normal')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    fig.tight_layout()
    _save_all(fig, os.path.join(out_dir, 'target_qq.png'))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(y, kde=True, bins='auto', color='#1565C0', edgecolor='white', alpha=0.9, ax=axes[0])
    axes[0].set_xlabel(target)
    axes[0].set_title('Histogram + KDE')
    axes[1].scatter(osm, osr, s=12, alpha=0.7, color='#424242', edgecolors='none')
    axes[1].plot(x_line, y_line, 'r--', lw=1)
    axes[1].set_title('Q-Q plot vs Normal')
    axes[1].set_xlabel('Theoretical quantiles')
    axes[1].set_ylabel('Sample quantiles')
    fig.tight_layout()
    _save_all(fig, os.path.join(out_dir, 'target_hist_qq.png'))
    plt.close(fig)

    tail = []
    if np.isfinite(stats['skewness']):
        tail.append(f"skewness = {stats['skewness']:.2f}")
    if np.isfinite(stats['kurtosis']):
        tail.append(f"kurtosis = {stats['kurtosis']:.2f}")
    shape = ('; '.join(tail)) if tail else ''
    sent = (
        f"The target variable was analyzed after applying the same row-level screening used for modeling (i.e., "
        f"dropping rows with missing target or missing numeric predictors), yielding n = {n}. The mean was {mean:.2f} "
        f"(SD = {std:.2f}, range {vmin:.2f}-{vmax:.2f}){(' with ' + shape) if shape else ''}. "
        f"{transform_note}"
    )

    with open(os.path.join(out_dir, 'target_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'target_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(sent + '\n')

    return stats, sent


if __name__ == '__main__':
    csv_clean = os.path.join(ROOT, 'dataset', 'data_cleaned.csv')
    csv_path = csv_clean if os.path.exists(csv_clean) else os.path.join(ROOT, 'dataset', 'data.csv')
    out_dir = os.path.join(ROOT, OUTPUT_DIR, 'target')
    _stats, _sent = describe_target(csv_path, out_dir)
    print(_sent)
