"""Create Q1-level feature stability visualization combining rank distribution and inclusion probabilities.

Path policy: canonical folders only.

Data sources (produced by xai_consistency.py):
 - output/xai_consistency/bootstrap_rank_metrics.csv

Plot concept:
 Horizontal segments per feature representing interquartile range (Q25–Q75) of Borda rank (lower is better).
 Marker (•) for median rank; whiskers to Q05/Q95.
 Color encode Top-10 inclusion probability (p_top10) with perceptually uniform colormap.
 Annotate right side with p_top10 (and p_top3 if >0) for quick stability interpretation.
 Features ordered by mean_rank ascending.

Outputs:
 - output/xai_consistency/rank_stability_distribution.png
 - supplements/figures/S5_rank_stability_distribution.png (primary figure replacing previous bars)
"""
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.paths import EVALUATION_DIR, get_supplements_root

ROOT = Path(__file__).resolve().parents[1]
_analysis_root = str(os.environ.get("MLTRAINER_ANALYSIS_ROOT", "") or "").strip()
_run_root = str(os.environ.get("MLTRAINER_RUN_ROOT", "") or "").strip()
if _analysis_root:
    XAI_DIR = Path(_analysis_root) / "xai_consistency"
elif _run_root:
    XAI_DIR = Path(_run_root) / EVALUATION_DIR / "xai_consistency"
else:
    XAI_DIR = ROOT / "output" / "xai_consistency"

def main():
    supp_fig = get_supplements_root() / "figures"
    metrics_path = XAI_DIR / 'bootstrap_rank_metrics.csv'
    if not metrics_path.exists():
        raise FileNotFoundError('bootstrap_rank_metrics.csv not found. Run xai_consistency.py first.')
    df = pd.read_csv(metrics_path)
    # Order by mean_rank (lower is better)
    df = df.sort_values('mean_rank')

    # Normalize p_top10 for colormap
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    n = len(df)
    fig_height = max(5, 0.5 * n)
    plt.figure(figsize=(8, fig_height))
    ax = plt.gca()

    y_positions = np.arange(n)
    # We'll plot ranks on x-axis; shift so best rank at left.
    # Provide some horizontal padding.
    max_rank = df['q95'].max() + 0.5
    min_rank = df['q05'].min() - 0.5

    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        # Whisker (q05-q95)
        ax.plot([row.q05, row.q95], [y, y], color='#999999', linewidth=1.0, solid_capstyle='round')
        # IQR segment
        ax.plot([row.q25, row.q75], [y, y], color='#333333', linewidth=4.0, solid_capstyle='round')
        # Median marker
        ax.scatter(row.median_rank, y, s=55, color=cmap(norm(row.p_top10)), edgecolor='black', linewidth=0.6, zorder=3)
        # Annotation of inclusion probabilities to right
        annot = f"p10={row.p_top10:.2f}"
        if row.p_top3 > 0:
            annot = f"p3={row.p_top3:.2f} · " + annot
        ax.text(max_rank + 0.05, y, annot, va='center', ha='left', fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['feature'], fontsize=9)
    ax.set_xlabel('Borda Rank (Lower = More Important)')
    ax.set_xlim(min_rank, max_rank + 1.8)
    ax.invert_yaxis()  # Top = best
    ax.set_title('Bootstrap Feature Rank Distribution & Inclusion Probabilities')

    # Colorbar for p_top10
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Top-10 Inclusion Probability (p_top10)')

    # Gridlines to help interpret rank scale
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', color='#DDDDDD', linewidth=0.6, alpha=0.7)
    ax.yaxis.grid(False)

    plt.tight_layout()
    out = XAI_DIR / 'rank_stability_distribution.png'
    supp_fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.savefig(supp_fig / 'S5_rank_stability_distribution.png', dpi=300)
    plt.close()
    print('[OK] Wrote distribution figure:', out)

if __name__ == '__main__':
    main()
