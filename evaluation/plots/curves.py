import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve


try:
    from config import SAVE_PDF
except Exception:
    SAVE_PDF = False


def _save_fig_formats_local(fig_path_base: str):
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


def _apply_plot_header(fig, title: str) -> float:
    fig.suptitle(title, fontsize=14, y=0.975)
    return 0.88


def plot_learning_curve(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str,
    cv=5):
    out_eval = os.path.join(outdir, '3_Manuscript_Figures', model_name)
    os.makedirs(out_eval, exist_ok=True)
    try:
        train_sizes, train_scores, cv_scores = learning_curve(
            pipe, X, y, cv=cv, scoring='r2', n_jobs=-1,
            train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        cv_mean = cv_scores.mean(axis=1)
        cv_std = cv_scores.std(axis=1)
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        ax.plot(train_sizes, train_mean, 'o-', color='tab:blue', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='tab:blue')
        ax.plot(train_sizes, cv_mean, 's--', color='tab:orange', label='Cross-validation score')
        ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, color='tab:orange')
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        top_margin = _apply_plot_header(fig, f'Learning Curve - {model_name}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)

        y_stack = np.concatenate([train_mean, cv_mean])
        y_min = float(np.nanmin(y_stack))
        y_max = float(np.nanmax(y_stack))
        y_span = max(y_max - y_min, 1e-6)
        y_pad = 0.12 * y_span
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        x_min = float(np.nanmin(train_sizes))
        x_max = float(np.nanmax(train_sizes))
        x_span = max(x_max - x_min, 1.0)
        x_last = float(train_sizes[-1])
        x_pos = x_last + 0.02 * x_span
        x_ha = 'left'
        if x_pos >= x_max:
            x_pos = x_last - 0.02 * x_span
            x_ha = 'right'

        ax.text(
            x_pos,
            float(train_mean[-1]) + 0.02 * y_span,
            f'{train_mean[-1]:.2f}',
            color='tab:blue',
            va='bottom',
            ha=x_ha,
            fontsize=10)
        ax.text(
            x_pos,
            float(cv_mean[-1]) - 0.02 * y_span,
            f'{cv_mean[-1]:.2f}',
            color='tab:orange',
            va='top',
            ha=x_ha,
            fontsize=10)

        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=top_margin)
        _save_fig_formats_local(os.path.join(out_eval, f'{model_name}_learning_curve'))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Learning Curve Exception\n```text\n{e}\n```\n")

def plot_predictions_vs_actual(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str,
    preds=None,
    cv=5):
    out_eval = os.path.join(outdir, '2_Model_Diagnostics', model_name)
    os.makedirs(out_eval, exist_ok=True)
    try:
        if preds is None:
            from sklearn.model_selection import cross_val_predict

            preds = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Prediction Failed\n```text\nPrediction failed: {e}\n```\n")
        return

    # Ensure index alignment does not corrupt exports.
    y_arr = np.asarray(y)
    preds_arr = np.asarray(preds)
    df_pa = pd.DataFrame({'actual': y_arr, 'predicted': preds_arr})
    try:
        df_pa.to_excel(os.path.join(out_eval, f"{model_name}_predictions_vs_actual.xlsx"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Predictions Excel Warning\n```text\n{e}\n```\n")
    try:
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_arr, preds_arr)
        try:
            from sklearn.metrics import root_mean_squared_error
            rmse_val = root_mean_squared_error(y_arr, preds_arr)
        except ImportError:
            rmse_val = mean_squared_error(y_arr, preds_arr, squared=False)
        fig, ax = plt.subplots(figsize=(7.8, 7.2))
        ax.scatter(y_arr, preds_arr, alpha=0.5, c='tab:green', edgecolor='none')
        lims = [min(float(np.min(y_arr)), float(np.min(preds_arr))), max(float(np.max(y_arr)), float(np.max(preds_arr)))]
        ax.plot(lims, lims, 'r--', linewidth=1)
        text = f'$R^2$={r2:.2f}\nRMSE={rmse_val:.2f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax.set_xlabel(f'{getattr(y, "name", "Actual") or "Actual"}', fontsize=12)
        ax.set_ylabel(f'Predicted {getattr(y, "name", "") or ""}'.strip(), fontsize=12)
        top_margin = _apply_plot_header(fig, f'Predictions vs Actual - {model_name}')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=top_margin)
        _save_fig_formats_local(os.path.join(out_eval, f'{model_name}_predictions_vs_actual'))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(outdir, "Run_Log_and_Warnings.md"), 'a', encoding='utf-8') as f:
            f.write(f"\n### {model_name} Predictions vs Actual Exception\n```text\n{e}\n```\n")

