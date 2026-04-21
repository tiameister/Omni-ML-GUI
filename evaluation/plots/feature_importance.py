import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# Force a font with wide Unicode coverage and proper minus sign handling
rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
})


try:
    from config import SAVE_PDF
except Exception:
    SAVE_PDF = False

# Normalize curly quotes/backticks to ASCII to avoid missing glyphs across OSes
try:
    from utils.text import normalize_quotes_ascii as _qascii
except Exception:
    def _qascii(s):
        return str(s)


def _strip_pipeline_prefix(name: str) -> str:
    txt = str(name)
    for prefix in ("num__", "cat__", "ord__", "bin__"):
        if txt.startswith(prefix):
            return txt[len(prefix):]
    return txt


def _resolve_transformed_feature_names(pipe, X, n_expected: int) -> list[str]:
    try:
        from evaluation.metrics import get_feature_names_from_pipe

        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns.tolist() if c not in num_cols]
        names = list(get_feature_names_from_pipe(pipe, num_cols, cat_cols))
        if len(names) == n_expected:
            return names
    except Exception:
        pass
    return [f"F{i}" for i in range(n_expected)]


def _raw_feature_from_transformed(transformed_name: str, raw_columns_sorted: list[str]) -> str:
    name = _strip_pipeline_prefix(transformed_name)
    if name in raw_columns_sorted:
        return name
    for raw in raw_columns_sorted:
        if name.startswith(f"{raw}_"):
            return raw
    return name


def plot_feature_importance_heatmap(
    model_name: str,
    pipe,
    X,
    y,
    outdir: str):
    out_expl = os.path.join(outdir, '3_Manuscript_Figures', model_name)
    os.makedirs(out_expl, exist_ok=True)
    try:
        model = pipe.named_steps.get('model', None)
        if model is None:
            return
        imp = None
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = model.coef_
        else:
            # Try permutation importance as a fallback
            try:
                from sklearn.inspection import permutation_importance
                # Keep this serial to avoid thread-config propagation warnings on some sklearn/joblib combos.
                res = permutation_importance(pipe, X, y, n_repeats=5, random_state=42, n_jobs=1)
                imp = res.importances_mean
            except Exception:
                imp = None
        if imp is None:
            # No feature importance available; write a notice and exit
            with open(os.path.join(out_expl, f"{model_name}_feature_importance_warning.txt"), 'w', encoding='utf-8') as f:
                f.write('No feature importance available for this model and permutation importance failed.')
            return

        imp_arr = np.asarray(imp, dtype=float)
        if imp_arr.ndim > 1:
            imp_arr = np.mean(np.abs(imp_arr), axis=0)
        else:
            imp_arr = np.abs(imp_arr)

        transformed_names = _resolve_transformed_feature_names(pipe, X, n_expected=len(imp_arr))
        raw_columns = [str(c) for c in X.columns.tolist()] if hasattr(X, 'columns') else []
        raw_columns_sorted = sorted(raw_columns, key=len, reverse=True)

        raw_importance: dict[str, float] = {}
        for idx, raw_val in enumerate(imp_arr):
            transformed = transformed_names[idx] if idx < len(transformed_names) else f"F{idx}"
            raw_name = _raw_feature_from_transformed(transformed, raw_columns_sorted)
            raw_importance[raw_name] = float(raw_importance.get(raw_name, 0.0) + float(abs(raw_val)))

        imp_series = pd.Series(raw_importance).sort_values(ascending=True)
        imp_series.index = [_qascii(str(n)) for n in imp_series.index]

        export_desc = imp_series.sort_values(ascending=False)
        export_total = float(export_desc.sum()) if len(export_desc) > 0 else 0.0
        export_df = pd.DataFrame(
            {
                "feature": export_desc.index,
                "importance": export_desc.values,
            }
        )
        export_df.insert(0, "rank", np.arange(1, len(export_df) + 1, dtype=int))
        if export_total > 0:
            export_df["importance_pct"] = (export_df["importance"] / export_total) * 100.0
            export_df["cumulative_pct"] = export_df["importance_pct"].cumsum()
        else:
            export_df["importance_pct"] = 0.0
            export_df["cumulative_pct"] = 0.0

        try:
            export_df.to_excel(os.path.join(out_expl, f"{model_name}_feature_importance.xlsx"), index=False)
        except Exception as e:
            with open(os.path.join(out_expl, f"{model_name}_feature_importance_save_warning.txt"), 'w', encoding='utf-8') as f:
                f.write(str(e))
        n_features = len(imp_series)
        max_label_len = max((len(str(lbl)) for lbl in imp_series.index), default=1)
        fig_h = max(3.2, min(14.0, 0.45 * n_features + 1.8))
        fig_w = max(6.5, min(11.5, 6.0 + 0.03 * max_label_len))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        colors = plt.cm.viridis(imp_series.values / (imp_series.max() if imp_series.max() != 0 else 1))
        y_pos = np.arange(len(imp_series), dtype=float)
        y_labels = [_qascii(str(t)) for t in imp_series.index]
        ax.barh(y_pos, imp_series.values, color=colors)
        # Use fixed tick positions to avoid category inference and locator warnings.
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

        max_val = float(np.nanmax(imp_series.values)) if len(imp_series) > 0 else 0.0
        x_pad = max(1e-6, max_val * 0.18)
        ax.set_xlim(0.0, max(1e-6, max_val + x_pad))
        x_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-6)
        text_offset = 0.015 * x_span
        for idx, (feat, val) in enumerate(imp_series.items()):
            ax.text(float(val) + text_offset, idx, f'{val:.3f}', va='center', ha='left', fontsize=8)
        ax.set_xlabel('Importance', fontsize=12)

        fig.suptitle(f'Feature Importances - {model_name}', fontsize=14, y=0.975)
        top_margin = 0.88

        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        left_margin = min(0.52, max(0.22, 0.16 + 0.006 * max_label_len))
        fig.subplots_adjust(left=left_margin, right=0.96, bottom=0.12, top=top_margin)
        try:
            fig.savefig(
                os.path.join(out_expl, f"{model_name}_feature_importance.png"),
                bbox_inches="tight",
                dpi=300,
            )
        except Exception as e:
            with open(os.path.join(out_expl, f"{model_name}_feature_importance_png_warning.txt"), 'w', encoding='utf-8') as f:
                f.write(str(e))
        if SAVE_PDF:
            try:
                fig.savefig(
                    os.path.join(out_expl, f"{model_name}_feature_importance.pdf"),
                    bbox_inches="tight",
                )
            except Exception as e:
                with open(os.path.join(out_expl, f"{model_name}_feature_importance_pdf_warning.txt"), 'w', encoding='utf-8') as f:
                    f.write(str(e))
        plt.close(fig)
    except Exception as e:
        with open(os.path.join(out_expl, f"{model_name}_feature_importance_exception.txt"), 'w', encoding='utf-8') as f:
            f.write(str(e))

