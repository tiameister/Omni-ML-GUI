import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from config import FEATURE_NAME_MAP
except Exception:
    FEATURE_NAME_MAP = {}
from utils.plotting_helpers import CATEGORY_LABELS as _CAT_LABELS, display_name as _disp, _norm as _norm  # type: ignore

try:
    from config import SAVE_PDF  # Optional: whether to also save PDF plots
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

def generate_regression_stats(model_name: str, best_pipe, X, y, outdir: str):
    # Direct outputs into evaluation subfolder for modularized structure
    outdir_eval = os.path.join(outdir, '1_Overall_Evaluation')
    os.makedirs(outdir_eval, exist_ok=True)
    try:
        try:
            import statsmodels.api as sm
        except ImportError as e:
            with open(os.path.join(outdir_eval, f"{model_name}_regression_stats_warning.txt"), 'w', encoding='utf-8') as f:
                f.write(f"statsmodels not available: {e}")
            return None

        # Transform features with preprocessor; ensure dense for statsmodels
        X_proc = best_pipe.named_steps['prep'].transform(X)
        if hasattr(X_proc, 'toarray'):
            X_proc = X_proc.toarray()
        from evaluation.metrics import get_feature_names_from_pipe
        # Derive feature names from fitted preprocessor, including OHE-expanded names
        feat_names = get_feature_names_from_pipe(best_pipe)
        # Extract original raw categorical and numeric columns for label mapping
        raw_num_cols, raw_cat_cols = [], []
        try:
            ct = best_pipe.named_steps.get('prep')
            for name, trans, cols in getattr(ct, 'transformers_', []):
                if name == 'num':
                    raw_num_cols = list(cols) if cols is not None else []
                elif name == 'cat':
                    raw_cat_cols = list(cols) if cols is not None else []
        except Exception:
            pass
        # Align names length with transformed width; fallback to generic names if mismatch
        n_out = X_proc.shape[1]
        if not feat_names or len(feat_names) != n_out:
            feat_names = [f'X{i}' for i in range(n_out)]
        X_sm = pd.DataFrame(X_proc, columns=feat_names)
        X_sm = sm.add_constant(X_sm)
        y_sm = y.values
        model = sm.OLS(y_sm, X_sm).fit()

        # Write textual regression summary
        try:
            summary_path = os.path.join(outdir_eval, f'{model_name}_regression_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(model.summary().as_text())
        except Exception as e:
            with open(os.path.join(outdir_eval, f"{model_name}_regression_stats_warning.txt"), 'a', encoding='utf-8') as f:
                f.write(f"\nSummary write failed: {e}")

        # Build coefficient stats dataframe
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        df_stats = pd.DataFrame({
            'coef': params,
            'std_err': bse,
            't_value': tvalues,
            'p_value': pvalues
        })
        ci = model.conf_int()
        df_stats['ci_lower'] = ci[0]
        df_stats['ci_upper'] = ci[1]

        # Compute standardized coefficients (Beta) and their 95% CI via scaling
        # beta_j = b_j * (sd_xj / sd_y)
        try:
            y_std = float(pd.Series(y_sm).std(ddof=1))
            X_std = X_sm.std(ddof=1)
            beta_vals = []
            beta_ci_l = []
            beta_ci_u = []
            for term in df_stats.index:
                if term == 'const' or term not in X_std:
                    beta_vals.append(np.nan)
                    beta_ci_l.append(np.nan)
                    beta_ci_u.append(np.nan)
                else:
                    sx = float(X_std[term])
                    if y_std > 0 and sx > 0:
                        scale = (sx / y_std)
                        b = float(df_stats.loc[term, 'coef'])
                        lo = float(df_stats.loc[term, 'ci_lower'])
                        up = float(df_stats.loc[term, 'ci_upper'])
                        beta_vals.append(b * scale)
                        beta_ci_l.append(lo * scale)
                        beta_ci_u.append(up * scale)
                    else:
                        beta_vals.append(np.nan)
                        beta_ci_l.append(np.nan)
                        beta_ci_u.append(np.nan)
            df_stats['beta'] = beta_vals
            df_stats['beta_ci_lower'] = beta_ci_l
            df_stats['beta_ci_upper'] = beta_ci_u
        except Exception:
            df_stats['beta'] = np.nan
            df_stats['beta_ci_lower'] = np.nan
            df_stats['beta_ci_upper'] = np.nan

        # Pretty label mapping function used in Excel and plots
        def _pretty(name: str) -> str:
            try:
                if name in FEATURE_NAME_MAP:
                    return str(FEATURE_NAME_MAP[name])
                n = str(name)
                # strip pipeline prefixes
                if n.startswith('num__') or n.startswith('cat__'):
                    n = n.split('__', 1)[1]
                # Map OneHot features: base_level
                # Find a base that matches an original categorical column
                for base in raw_cat_cols:
                    pref = f"{base}_"
                    if n.startswith(pref):
                        level = n[len(pref):]
                        # Try numeric level mapping first
                        label = None
                        try:
                            lvl_int = int(level)
                        except Exception:
                            lvl_int = None
                        key = _norm(base)
                        mapping = _CAT_LABELS.get(key)
                        if mapping is not None:
                            if lvl_int is not None and lvl_int in mapping:
                                label = mapping[lvl_int]
                            elif level in mapping.values():
                                label = level
                        if label is None:
                            # fallback: use literal level text
                            label = level
                        return f"{_disp(base)}: {label}"
                # Numeric feature or unmatched term
                return _disp(n)
            except Exception:
                return str(name)

        # Excel output (robust): include raw and a sorted view
        try:
            # Q1 journal-style table: B, Std. Error, Beta, t, p with pretty term names

            df_q1 = df_stats[['coef', 'std_err', 'beta', 't_value', 'p_value']].copy()
            df_q1.rename(columns={
                'coef': 'B',
                'std_err': 'Std. Error',
                'beta': 'Beta',
                't_value': 't',
                'p_value': 'p'
            }, inplace=True)
            # Move/rename terms and prettify
            df_q1 = df_q1.reset_index(names=['term'])
            df_q1['term'] = df_q1['term'].apply(lambda s: 'Intercept' if s == 'const' else _pretty(s))

            # Also prepare a raw sheet with CI for completeness
            df_raw = df_stats.reset_index(names=['term'])

            xlsx_path = os.path.join(outdir_eval, f'{model_name}_regression_stats.xlsx')
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df_q1.to_excel(writer, index=False, sheet_name='Q1_Table')
                # Sorted by absolute Beta (exclude intercept rows that may be NaN)
                df_sorted_beta = df_q1[~df_q1['Beta'].isna()].copy()
                df_sorted_beta = df_sorted_beta.reindex(df_sorted_beta['Beta'].abs().sort_values(ascending=False).index)
                df_sorted_beta.to_excel(writer, index=False, sheet_name='SortedByAbsBeta')
                df_raw.to_excel(writer, index=False, sheet_name='Raw_With_CI')
        except Exception as e:
            with open(os.path.join(outdir_eval, f"{model_name}_regression_stats_warning.txt"), 'a', encoding='utf-8') as f:
                f.write(f"\nExcel save failed: {e}")

        # Build plotting frame: exclude intercept for readability, sort by |beta|
        df_plot = df_stats.copy()
        intercept_value = None
        if 'const' in df_plot.index:
            intercept_value = float(df_plot.loc['const', 'coef'])
            df_plot = df_plot.drop(index='const')
        if not df_plot.empty:
            # Prefer standardized beta for ordering when available
            if df_plot['beta'].notna().any():
                order = df_plot['beta'].abs().sort_values(ascending=True).index
            else:
                order = df_plot['coef'].abs().sort_values(ascending=True).index
            df_plot = df_plot.loc[order]

        # Horizontal bar chart with CI lines and exact value annotations
        n = len(df_plot)
        fig_h = max(3.4, min(12.0, 0.42 * n + 2.2))
        fig_w = 6.8 if n <= 2 else 9.8
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        if n > 0:
            y = np.arange(n)
            labels = [ _pretty(idx) for idx in df_plot.index ]
            # Use standardized Beta for bars; fall back to raw coef
            vals = df_plot['beta'].values if df_plot['beta'].notna().any() else df_plot['coef'].values
            ci_l = (df_plot['beta_ci_lower'].values if df_plot['beta'].notna().any() else df_plot['ci_lower'].values)
            ci_u = (df_plot['beta_ci_upper'].values if df_plot['beta'].notna().any() else df_plot['ci_upper'].values)
            pvals = df_plot['p_value'].values
            colors = ['#2E86C1' if p < 0.05 else '#B0BEC5' for p in pvals]
            ax.barh(y, vals, color=colors, edgecolor='#263238', alpha=0.9)
            # 95% CI lines
            for i, (yl, lo, up) in enumerate(zip(y, ci_l, ci_u)):
                ax.hlines(yl, lo, up, colors='#455A64', linewidth=2)
                ax.plot([lo, up], [yl, yl], 'o', color='#455A64', markersize=3)
            # Zero line
            ax.axvline(0, color='#9E9E9E', linewidth=1)
            # Labels and ticks
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            xlabel = 'Standardized Coefficient (Beta)' if df_plot['beta'].notna().any() else 'Coefficient (β)'
            ax.set_xlabel(xlabel)
            title = f'Coefficients - {model_name}'
            if intercept_value is not None:
                title += f'  (Intercept = {intercept_value:.3f})'
            ax.set_title(title)
            ax.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.5)

            try:
                finite_l = np.asarray(ci_l, dtype=float)
                finite_u = np.asarray(ci_u, dtype=float)
                finite_v = np.asarray(vals, dtype=float)
                lo = np.nanmin(np.concatenate([finite_l, finite_v]))
                hi = np.nanmax(np.concatenate([finite_u, finite_v]))
                span = max(float(hi - lo), 1e-6)
                pad = 0.14 * span
                ax.set_xlim(float(lo - pad), float(hi + pad))
            except Exception:
                pass

            # Annotate exact values at bar ends
            xlim = ax.get_xlim()
            span = max(xlim[1] - xlim[0], 1e-6)
            for yi, v in zip(y, vals):
                if np.isnan(v):
                    continue
                off = span * 0.015
                if v >= 0:
                    ax.text(v + off, yi, f'{v:.3f}', va='center', ha='left', fontsize=9)
                else:
                    ax.text(v - off, yi, f'{v:.3f}', va='center', ha='right', fontsize=9)

            max_label_len = max((len(str(lbl)) for lbl in labels), default=1)
            left_margin = min(0.50, max(0.20, 0.14 + 0.006 * max_label_len))
            fig.subplots_adjust(left=left_margin, right=0.96, bottom=0.13, top=0.90)
        else:
            ax.text(0.5, 0.5, 'No coefficients to display', transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
        _save_fig_formats(os.path.join(outdir_eval, f'{model_name}_beta_coefficients'))
        plt.close(fig)

        # F-statistic text
        try:
            with open(os.path.join(outdir_eval, f'{model_name}_f_statistic.txt'), 'w', encoding='utf-8') as f:
                f.write(f'F-statistic: {model.fvalue}\n')
        except Exception as e:
            with open(os.path.join(outdir_eval, f"{model_name}_regression_stats_warning.txt"), 'a', encoding='utf-8') as f:
                f.write(f"\nF-statistic write failed: {e}")

        return df_stats
    except Exception as e:
        with open(os.path.join(outdir_eval, f"{model_name}_regression_stats_warning.txt"), 'a', encoding='utf-8') as f:
            f.write(str(e))
        return None

