"""
Residual analysis and diagnostic plotting utilities for MLTrainer.
Includes functions for saving residual plots and distributions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.paths import DIAGNOSTICS_DIR

try:
    from config import SAVE_PDF
except Exception:
    SAVE_PDF = False


def _save_fig_formats(fig_path_base: str):
    try:
        plt.savefig(fig_path_base + '.png', dpi=160, bbox_inches="tight")
    except Exception as e:
        with open(fig_path_base + '_save_warning.txt', 'w', encoding='utf-8') as f:
            f.write(f'PNG save failed: {e}')
    if SAVE_PDF:
        try:
            plt.savefig(fig_path_base + '.pdf', bbox_inches="tight")
        except Exception as e:
            with open(fig_path_base + '_save_warning.txt', 'a', encoding='utf-8') as f:
                f.write(f'\nPDF save failed: {e}')


def plot_residuals(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str,
    preds=None,
    cv=5):
    out_diag = os.path.join(outdir, DIAGNOSTICS_DIR, model_name)
    os.makedirs(out_diag, exist_ok=True)
    try:
        if preds is None:
            from sklearn.model_selection import cross_val_predict

            preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        resid = y - preds
        
        # Calculate true statistics before sampling plot rendering
        mu, sigma = resid.mean(), resid.std()
        
        # ML pipeline optimization: Downsample massive scatter/KDE layers to prevent memory explosion & viewer crashes
        if len(resid) > 5000:
            idx = np.random.RandomState(42).choice(len(resid), size=5000, replace=False)
            plot_preds = np.asarray(preds)[idx]
            plot_resid = np.asarray(resid)[idx]
        else:
            plot_preds = preds
            plot_resid = resid
            
        fig, ax = plt.subplots(figsize=(8, 6))
        # rasterized=True prevents SVG/PDF engines from freezing on millions of paths
        ax.scatter(plot_preds, plot_resid, s=30, alpha=0.5, c='tab:blue', edgecolor='none', rasterized=True)
        sns.kdeplot(x=plot_preds, y=plot_resid, levels=5, color='black', linewidths=1, alpha=0.5, ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        text = f'Mean={mu:.2f}\nStd={sigma:.2f}'
        ax.text(0.95, 0.95, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax.set_xlabel(f'Predicted {getattr(y, "name", "") or ""}'.strip(), fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'Residuals - {model_name}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        _save_fig_formats(os.path.join(out_diag, f'{model_name}_residuals'))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Residuals Exception\n```text\n{e}\n```\n")

def plot_residual_distribution(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str,
    preds=None,
    cv=5):
    out_diag = os.path.join(outdir, DIAGNOSTICS_DIR, model_name)
    os.makedirs(out_diag, exist_ok=True)
    try:
        if preds is None:
            from sklearn.model_selection import cross_val_predict

            preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        resid = y - preds
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(resid, bins=30, stat='density', color='skyblue', edgecolor='black', alpha=0.6, ax=ax)
        sns.kdeplot(resid, color='darkblue', linewidth=2, ax=ax)
        sns.rugplot(resid, color='black', alpha=0.5, height=0.05, ax=ax)
        mu, sigma = resid.mean(), resid.std()
        text = f'Mean={mu:.2f}\nStd={sigma:.2f}'
        ax.text(0.95, 0.95, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax.set_xlabel('Residual', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Residual Distribution - {model_name}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        _save_fig_formats(os.path.join(out_diag, f'{model_name}_residual_distribution'))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Residual Distribution Exception\n```text\n{e}\n```\n")

def plot_qq(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str,
    preds=None,
    cv=5):
    out_diag = os.path.join(outdir, DIAGNOSTICS_DIR, model_name)
    os.makedirs(out_diag, exist_ok=True)
    try:
        if preds is None:
            from sklearn.model_selection import cross_val_predict

            preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        resid = y - preds
        (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist='norm')
        r2 = r**2
        fig, ax = plt.subplots(figsize=(8, 6))
        pct = np.percentile(osr, [2.5, 97.5])
        mask_tail = (osr < pct[0]) | (osr > pct[1])
        ax.scatter(osm[~mask_tail], osr[~mask_tail], alpha=0.5, color='tab:blue', label='Data')
        ax.scatter(osm[mask_tail], osr[mask_tail], alpha=0.8, color='red', label='Tails')
        x_line = np.array([osm.min(), osm.max()])
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax.set_ylabel('Ordered Residuals', fontsize=12)
        ax.set_title(f'Q-Q Plot - {model_name}', fontsize=14)
        ax.text(0.05, 0.95, f'$R^2$={r2:.2f}', transform=ax.transAxes,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        _save_fig_formats(os.path.join(out_diag, f'{model_name}_qq_plot'))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Q-Q Plot Exception\n```text\n{e}\n```\n")

