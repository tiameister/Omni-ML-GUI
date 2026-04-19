"""Generate ranked bar chart of bootstrap Top-10 inclusion probability.

Replaces previous heatmap visualization with a cleaner, reviewer-friendly horizontal bar chart.

Rationale:
 - Directly communicates stability via bar lengths (no color scale cognition cost).
 - Annotated bars (probability and raw frequency) improve interpretability.
 - Highlights domain-critical features with zero inclusion (kept at bottom, lightly shaded).

Outputs:
 - output/xai_consistency/rank_stability_bars.png
 - supplements/figures/S5_rank_stability_bars.png (overwrites prior heatmap asset)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
XAI_DIR = ROOT / 'output' / 'xai_consistency'
SUPP_FIG = ROOT / 'supplements' / 'figures'

def main():
    boot_path = XAI_DIR / 'bootstrap_stability.csv'
    borda_path = XAI_DIR / 'borda_ranking.csv'
    if not boot_path.exists():
        print('[ERROR] bootstrap_stability.csv not found.')
        return
    boot = pd.read_csv(boot_path)
    if 'feature' not in boot.columns or 'top10_pct' not in boot.columns:
        print('[ERROR] Expected columns feature, top10_pct in bootstrap_stability.csv')
        return
    # Optional: merge borda rank to break ties and sort more informatively
    if borda_path.exists():
        borda = pd.read_csv(borda_path)
        boot = boot.merge(borda[['feature','borda_rank']], on='feature', how='left')
    else:
        boot['borda_rank'] = np.nan

    # Sort primary: descending probability; secondary: ascending Borda rank (if present)
    boot = boot.sort_values(['top10_pct','borda_rank'], ascending=[False, True])

    # Keep order; separate zero-prob features to append at bottom preserving original relative order among zeros.
    non_zero = boot[boot.top10_pct > 0]
    zeros = boot[boot.top10_pct == 0]
    boot_ordered = pd.concat([non_zero, zeros], axis=0)

    # Aesthetic parameters
    n = len(boot_ordered)
    height = max(4.5, 0.45 * n)
    plt.figure(figsize=(7, height))
    y_pos = np.arange(n)[::-1]  # top = highest prob
    probs = boot_ordered['top10_pct'].clip(0,1).values
    feats = boot_ordered['feature'].values
    freqs = boot_ordered['top10_freq'].values

    colors = ['#2c7fb8' if p>0 else '#cccccc' for p in probs]
    plt.barh(y_pos, probs, color=colors, edgecolor='#333333', linewidth=0.6)

    # Annotate: probability (%.2f) and raw freq in parentheses
    for y, p, f in zip(y_pos, probs, freqs):
        txt = f"{p:.2f} ({f})"
        if p >= 0.05:
            plt.text(p - 0.01, y, txt, va='center', ha='right', fontsize=9, color='white', fontweight='bold')
        else:
            plt.text(p + 0.01, y, txt, va='center', ha='left', fontsize=9, color='#444444')

    plt.yticks(y_pos, feats, fontsize=9)
    plt.xlabel('Top-10 Inclusion Probability (Bootstrap)', fontsize=10)
    plt.title('Bootstrap Feature Stability (Top-10 Inclusion)', fontsize=12, weight='bold')
    plt.xlim(0, 1.0)
    # Reference lines
    for ref in [0.25, 0.50, 0.75]:
        plt.axvline(ref, color='#bbbbbb', linestyle='--', linewidth=0.7)
    plt.text(0.50, -0.5, '0.50', ha='center', va='bottom', color='#666666', fontsize=8)
    plt.tight_layout()

    out1 = XAI_DIR / 'rank_stability_bars.png'
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()

    # Copy to supplements with new naming
    SUPP_FIG.mkdir(parents=True, exist_ok=True)
    import shutil
    suppl_path = SUPP_FIG / 'S5_rank_stability_bars.png'
    shutil.copy2(out1, suppl_path)
    print('[OK] Stability bar chart saved to:', out1)
    print('[OK] Supplement bar chart saved to:', suppl_path)

if __name__ == '__main__':
    main()
