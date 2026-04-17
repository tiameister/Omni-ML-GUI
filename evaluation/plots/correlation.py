import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from config import SAVE_PDF
except Exception:
    SAVE_PDF = False


def _save_fig_formats(fig_path_base: str):
    try:
        plt.savefig(fig_path_base + '.png', dpi=300, bbox_inches="tight")
    except Exception as e:
        with open(fig_path_base + '_save_warning.txt', 'w', encoding='utf-8') as f:
            f.write(f'PNG save failed: {e}')
    if SAVE_PDF:
        try:
            plt.savefig(fig_path_base + '.pdf', bbox_inches="tight")
        except Exception as e:
            with open(fig_path_base + '_save_warning.txt', 'a', encoding='utf-8') as f:
                f.write(f'\nPDF save failed: {e}')

def plot_correlation_matrix(model_name: str, X, y, outdir: str):
    import pandas as pd
    out_diag = os.path.join(outdir, '2_Model_Diagnostics', model_name)
    os.makedirs(out_diag, exist_ok=True)
    try:
        # Combine features with target and compute numeric-only correlations
        df = X.copy()
        target_col = 'target'
        if target_col in df.columns:
            base = 'target'
            suffix = 2
            while f"{base}_{suffix}" in df.columns:
                suffix += 1
            target_col = f"{base}_{suffix}"
        df[target_col] = y
        corr = df.corr(numeric_only=True)

        # Guard against empty correlation matrices
        if corr.empty or corr.shape[0] == 0:
            with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
                f.write(f'\n### {model_name} Correlation Warning\n```text\nNo numeric columns available to compute correlation.\n```\n')
            return

        # Save matrix to Excel for inspection (best effort)
        try:
            corr.to_excel(os.path.join(out_diag, f"{model_name}_correlation_matrix.xlsx"))
        except Exception as e:
            with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
                f.write(f'\n### {model_name} Correlation Excel Warning\n```text\n{e}\n```\n')

        # Figure size scales with variable count
        n = corr.shape[0]
        fig_width = max(8, n * 0.6)
        fig_height = max(6, n * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            annot_kws={'size': max(6, 12 - int(n*0.2))},
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Correlation', 'shrink': 0.75},
            ax=ax,
            square=True
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=max(8, 14 - int(n*0.2)))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=max(8, 14 - int(n*0.2)))
        ax.set_title(f'Correlation Matrix - {model_name}', fontsize=16)
        fig.tight_layout(pad=1.5)
        # Save as PNG (300 DPI) and PDF (vector)
        _save_fig_formats(os.path.join(out_diag, f'{model_name}_correlation_matrix'))
        plt.close(fig)
    except Exception as e:
        # Ensure failures are visible in the output folder
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Correlation Matrix Exception\n```text\n{e}\n```\n")

