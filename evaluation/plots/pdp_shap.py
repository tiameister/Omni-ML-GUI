import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
})
import pandas as pd
import seaborn as sns
from typing import Optional, List, Tuple
from config import SHAP_TOP_N, SHAP_VAR_THRESH, FEATURE_NAME_MAP, SHAP_ALWAYS_INCLUDE
try:
    from config import SHAP_BEESWARM_TRIM_PCT, SHAP_BEESWARM_MIN_CAP
except Exception:
    SHAP_BEESWARM_TRIM_PCT = 1.0
    SHAP_BEESWARM_MIN_CAP = None
from sklearn.inspection import PartialDependenceDisplay
from utils.plotting_helpers import (
    map_labels,
    clip_outliers,
    plot_shap_dependence as plot_shap_dependence_helper,
    top_raw_features_by_shap)

# Optional: SHAP
try:
    import shap
    SHAP_OK = True
except ImportError:
    shap = None
    SHAP_OK = False

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

from string import capwords as _capwords
# Centralized quote/backtick normalization
try:
    from utils.text import normalize_quotes_ascii as _qascii
except Exception:
    def _qascii(s: str) -> str:
        return str(s)

def _pretty_label(name: str) -> str:
    # Remove pipeline prefixes and beautify
    try:
        if name in FEATURE_NAME_MAP:
            nm = _qascii(FEATURE_NAME_MAP[name])
            out = _capwords(nm)
            # preserve common acronyms after capitalization
            out = re.sub(r"\bTv\b", "TV", out)
            out = re.sub(r"\bAi\b", "AI", out)
            out = re.sub(r"\bGdp\b", "GDP", out)
            return out
        n = _qascii(name)
        # strip common prefixes
        for pref in ('num__', 'cat__'):
            if n.startswith(pref):
                n = n[len(pref):]
        n = n.replace('__', '_')
        # turn base_level into "base: level" when it looks categorical
        if '_' in n:
            base, level = n.split('_', 1)
            if level and not level.isnumeric():
                n = f"{base}: {level}"
        # final formatting (word-based capitalize, preserve apostrophes in words)
        n = n.replace('_', ' ')
        out = _capwords(n)
        out = re.sub(r"\bTv\b", "TV", out)
        out = re.sub(r"\bAi\b", "AI", out)
        out = re.sub(r"\bGdp\b", "GDP", out)
        return out
    except Exception:
        return name


def _dynamic_left_margin(labels: list[str], *, base: float = 0.24, max_left: float = 0.52) -> float:
    longest = max((len(str(lbl)) for lbl in labels), default=1)
    return min(max_left, max(base, 0.17 + 0.006 * float(longest)))


def _apply_plot_header(fig, title: str) -> float:
    fig.suptitle(title, fontsize=14, y=0.976)
    return 0.89


def _normalize_rule_token(value) -> str:
    txt = str(value if value is not None else "").strip().lower()
    txt = txt.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
    txt = txt.replace("ş", "s").replace("ö", "o").replace("ç", "c")
    return txt


def _resolve_feature_value_label_map(
    feature_name: str,
    feature_value_labels: dict[str, dict[str, str]] | None) -> dict[str, str]:
    if not isinstance(feature_value_labels, dict) or not feature_value_labels:
        return {}

    feature_norm = _normalize_rule_token(feature_name)
    for raw_key, raw_map in feature_value_labels.items():
        if not isinstance(raw_map, dict):
            continue
        key_txt = str(raw_key).strip()
        if key_txt == feature_name or _normalize_rule_token(key_txt) == feature_norm:
            out: dict[str, str] = {}
            for src, dst in raw_map.items():
                src_norm = _normalize_rule_token(src)
                dst_txt = str(dst).strip()
                if src_norm and dst_txt and src_norm not in out:
                    out[src_norm] = dst_txt
            return out
    return {}


def _raise_if_cancelled(cancel_cb=None):
    if callable(cancel_cb):
        try:
            if cancel_cb():
                raise RuntimeError("Cancelled by user")
        except RuntimeError:
            raise
        except Exception:
            # Ignore callback failures to avoid breaking plotting.
            return

def generate_pdp(best_model_name: str, best_pipe, X, top_features: list, outdir: str):
    out_expl = os.path.join(outdir, '3_Manuscript_Figures', best_model_name)
    os.makedirs(out_expl, exist_ok=True)
    try:
        X = X.copy().astype(float)
        for feat in top_features:
            plt.figure(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(best_pipe, X, [feat], grid_resolution=40, n_jobs=-1)
            plt.title(f"PDP - {best_model_name} - {feat}")
            plt.tight_layout()
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", feat)
            plt.savefig(
                os.path.join(out_expl, f"{best_model_name}_PDP_{safe}.png"),
                bbox_inches="tight",
                dpi=160,
            )
            plt.close()
    except Exception as e:
        with open(os.path.join(out_expl, "pdp_warning.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

def _compute_shap(best_pipe, X, num_cols, cat_cols, seed: int = 42, cancel_cb=None):
    _raise_if_cancelled(cancel_cb)
    from evaluation.metrics import get_feature_names_from_pipe
    feat_names = get_feature_names_from_pipe(best_pipe, num_cols, cat_cols)
    X_proc = best_pipe.named_steps["prep"].transform(X)
    _raise_if_cancelled(cancel_cb)
    model_obj = best_pipe.named_steps["model"]
    n_sample = min(500, X_proc.shape[0])
    idx = np.random.RandomState(seed).choice(X_proc.shape[0], size=n_sample, replace=False)
    Xs = X_proc[idx]
    
    # ML Pipeline Optimization: Fast TreeExplainer for ensemble models (O(1) background), 
    # and downsampled background for Exact/Kernel explainers to prevent GUI freeze.
    model_name = model_obj.__class__.__name__
    is_tree = any(t in model_name for t in ("RandomForest", "GradientBoosting", "XGB", "HistGradientBoosting", "Tree"))
    
    explainer = None
    if is_tree:
        try:
            explainer = shap.TreeExplainer(model_obj)
            shap_output = explainer(Xs)
        except Exception:
            explainer = None
            
    if explainer is None:
        # Prevent massive background datasets from crashing ExactExplainer
        bg_size = min(100, X_proc.shape[0])
        bg = X_proc[np.random.RandomState(seed).choice(X_proc.shape[0], size=bg_size, replace=False)]
        explainer = shap.Explainer(model_obj, bg)
        shap_output = explainer(Xs)
    _raise_if_cancelled(cancel_cb)
    if hasattr(shap_output, 'values'):
        shap_values = shap_output.values
    else:
        shap_values = shap_output
    # Unpack multi-output shap_values if returned as list or tuple
    if isinstance(shap_values, (list, tuple)):
        shap_values = shap_values[0]
    return feat_names, Xs, shap_values, idx


def generate_shap_summary(
    best_model_name: str,
    best_pipe,
    X,
    num_cols,
    cat_cols,
    outdir: str,
    seed: int = 42,
    top_n: int | None = SHAP_TOP_N,
    var_thresh: float | None = SHAP_VAR_THRESH,
    cancel_cb=None):
    out_expl = os.path.join(outdir, '3_Manuscript_Figures', best_model_name)
    os.makedirs(out_expl, exist_ok=True)
    if not SHAP_OK:
        with open(os.path.join(out_expl, f"{best_model_name}_shap_warning.txt"), "w", encoding="utf-8") as f:
            f.write("SHAP library not installed.")
        return
    try:
        _raise_if_cancelled(cancel_cb)
        feat_names, Xs, shap_values, _ = _compute_shap(best_pipe, X, num_cols, cat_cols, seed, cancel_cb=cancel_cb)
        _raise_if_cancelled(cancel_cb)
        # Rank by importance (mean |SHAP|) and drop near-zero variance columns
        imp = np.mean(np.abs(shap_values), axis=0)
        var = np.var(Xs, axis=0)
        idx_sorted = np.argsort(imp)[::-1]
        if var_thresh is None:
            idx_filtered = list(idx_sorted)
        else:
            idx_filtered = [i for i in idx_sorted if var[i] >= var_thresh]
        # Determine how many to keep; when top_n <= 0 or None -> keep all
        if top_n is None:
            k = len(idx_filtered)
        else:
            try:
                tn = int(top_n)
            except Exception:
                tn = len(idx_filtered)
            if tn <= 0:
                k = len(idx_filtered)
            else:
                k = min(len(idx_filtered), tn)
        top_idx = idx_filtered[:k]
        # Build pretty names for all, and for selected
        feat_pretty_all = [_pretty_label(n) for n in feat_names]
        sel_names = [feat_names[i] for i in top_idx]
        sel_pretty = [feat_pretty_all[i] for i in top_idx]
        # If no features selected, skip summary plots
        if len(top_idx) == 0:
            return

        # Save values and summary (selected only)
        try:
            shap_df = pd.DataFrame(shap_values[:, top_idx], columns=sel_names)
            shap_df.to_excel(os.path.join(out_expl, f"{best_model_name}_shap_values_top{len(top_idx)}.xlsx"), index=False)
            summary_df = pd.DataFrame({
                'feature': sel_names,
                'pretty': sel_pretty,
                'mean_abs_shap': imp[top_idx]
            }).sort_values('mean_abs_shap', ascending=False)
            summary_df.to_excel(os.path.join(out_expl, f"{best_model_name}_shap_summary_top{len(top_idx)}.xlsx"), index=False)
        except Exception:
            pass
        _raise_if_cancelled(cancel_cb)

    # Beeswarm (selected only) with dynamic height and margins
        nfeat = len(top_idx)
        max_label_len = max((len(str(lbl)) for lbl in sel_pretty), default=1)
        height = min(18.0, max(4.6, 0.42 * nfeat + 1.9))
        width = min(15.0, max(9.8, 8.6 + 0.045 * max_label_len))
        plt.figure(figsize=(width, height))
        # Dynamic lower outlier threshold for beeswarm: drop extreme bottom values
        try:
            vals = shap_values[:, top_idx]
            Xs_vals = Xs[:, top_idx]
            trim_pct = float(SHAP_BEESWARM_TRIM_PCT) if SHAP_BEESWARM_TRIM_PCT is not None else None
            if trim_pct is not None and 0.0 < trim_pct < 50.0:
                flat_vals = np.ravel(vals)
                lower_thresh = float(np.nanpercentile(flat_vals, trim_pct))
                mask = np.all(vals >= lower_thresh, axis=1)
                # guard: if mask prunes almost all rows, relax trimming
                if mask.sum() < max(10, int(0.3 * len(mask))):
                    mask = np.ones(len(mask), dtype=bool)
                vals = vals[mask]
                Xs_vals = Xs_vals[mask]
            # optional min-cap: if provided, clip values to this lower bound for display stability
            if SHAP_BEESWARM_MIN_CAP is not None:
                try:
                    lb = float(SHAP_BEESWARM_MIN_CAP)
                    vals = np.maximum(vals, lb)
                except Exception:
                    pass
        except Exception:
            vals = shap_values[:, top_idx]
            Xs_vals = Xs[:, top_idx]
        # Normalize quotes in feature names to avoid rendering issues
        try:
            from utils.text import normalize_quotes_ascii as _qascii
            sel_pretty_plot = [_qascii(n) for n in sel_pretty]
        except Exception:
            sel_pretty_plot = sel_pretty
        shap.summary_plot(vals, Xs_vals, feature_names=sel_pretty_plot, show=False)
        fig_bee = plt.gcf()
        top_margin = _apply_plot_header(fig_bee, f"SHAP Summary - {best_model_name}")
        left_margin = _dynamic_left_margin(sel_pretty_plot, base=0.24, max_left=0.52)
        right_margin = 0.90 if len(fig_bee.axes) > 1 else 0.96
        bottom_margin = 0.14 if nfeat <= 3 else 0.10
        try:
            fig_bee.subplots_adjust(
                left=left_margin,
                right=right_margin,
                bottom=bottom_margin,
                top=top_margin)
            if len(fig_bee.axes) > 1:
                cbar_ax = fig_bee.axes[-1]
                cbar_pos = cbar_ax.get_position()
                cb_left = min(0.95, right_margin + 0.02)
                cbar_ax.set_position([cb_left, cbar_pos.y0 + 0.02, 0.018, max(0.2, cbar_pos.height - 0.04)])
                cbar_ax.tick_params(labelsize=9, pad=2)
                cbar_ax.set_ylabel("Feature value", fontsize=9, labelpad=8)
        except Exception:
            pass
        _save_fig_formats(os.path.join(out_expl, f"{best_model_name}_shap_summary_beeswarm"))
        plt.close()
        _raise_if_cancelled(cancel_cb)
        # Bar (selected only)
        bar_h = min(14.0, max(3.0, 0.5 * nfeat + 1.4))
        bar_w = min(12.5, max(6.4, 5.6 + 0.04 * max_label_len))
        plt.figure(figsize=(bar_w, bar_h))
        try:
            from utils.text import normalize_quotes_ascii as _qascii
            sel_pretty_bar = [_qascii(n) for n in sel_pretty]
        except Exception:
            sel_pretty_bar = sel_pretty
        shap.summary_plot(shap_values[:, top_idx], Xs[:, top_idx], feature_names=sel_pretty_bar, plot_type="bar", show=False)
        fig_bar = plt.gcf()
        top_margin_bar = _apply_plot_header(fig_bar, f"SHAP Summary (Bar) - {best_model_name}")
        left_margin_bar = _dynamic_left_margin(sel_pretty_bar, base=0.24, max_left=0.52)
        try:
            fig_bar.subplots_adjust(left=left_margin_bar, right=0.96, bottom=0.13, top=top_margin_bar)
            if fig_bar.axes:
                ax_bar = fig_bar.axes[0]
                widths = [float(p.get_width()) for p in list(getattr(ax_bar, "patches", []) or [])]
                xmax = max(widths) if widths else 0.0
                if np.isfinite(xmax):
                    ax_bar.set_xlim(0.0, max(1e-6, xmax * 1.2))
                ax_bar.margins(y=0.12)
        except Exception:
            pass
        _save_fig_formats(os.path.join(out_expl, f"{best_model_name}_shap_summary_bar"))
        plt.close()
        _raise_if_cancelled(cancel_cb)
    except Exception as e:
        if "cancelled by user" in str(e).strip().lower():
            raise
        with open(os.path.join(out_expl, f"{best_model_name}_shap_summary_warning.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))


def generate_shap_dependence(
    best_model_name: str,
    best_pipe,
    X,
    num_cols,
    cat_cols,
    outdir: str,
    top_n: int | None = SHAP_TOP_N,
    seed: int = 42,
    var_thresh: float | None = SHAP_VAR_THRESH,
    always_include: Optional[List[str]] = None,
    feature_value_labels: dict[str, dict[str, str]] | None = None,
    cancel_cb=None,
    *,
    y_limit: Optional[Tuple[float, float]] = None,
    clip_limit: Optional[float] = None):
    out_expl = os.path.join(outdir, '3_Manuscript_Figures', best_model_name)
    os.makedirs(out_expl, exist_ok=True)
    if not SHAP_OK:
        with open(os.path.join(out_expl, f"{best_model_name}_shap_warning.txt"), "w", encoding="utf-8") as f:
            f.write("SHAP library not installed.")
        return
    try:
        _raise_if_cancelled(cancel_cb)
        feat_names, Xs, shap_values, idx = _compute_shap(best_pipe, X, num_cols, cat_cols, seed, cancel_cb=cancel_cb)
        _raise_if_cancelled(cancel_cb)
        # Determine top raw features by aggregated SHAP importance
        top_raw = top_raw_features_by_shap(
            shap_values=shap_values,
            feat_names=feat_names,
            X_raw=X,
            num_cols=num_cols,
            cat_cols=cat_cols,
            top_n=top_n,
            var_thresh=var_thresh)
        # Ensure always-include raw features
        include_list = always_include if always_include is not None else SHAP_ALWAYS_INCLUDE
        for r in include_list or []:
            if r in top_raw:
                continue
            # include if present in original columns
            if r in X.columns:
                top_raw.append(r)

        # Base random sample for continuous features
        X_sample = X.iloc[idx].reset_index(drop=True)
        # Plot each selected raw feature
        for raw_feat in top_raw:
            _raise_if_cancelled(cancel_cb)
            # For discrete/ordinal features, compute SHAP on full data to capture all levels
            if raw_feat in cat_cols or raw_feat in num_cols and X[raw_feat].dropna().nunique() <= 20 and X[raw_feat].dtype.kind in 'iu':
                # full sample
                feat_names2, Xs2, shap2, idx2 = _compute_shap(best_pipe, X, num_cols, cat_cols, seed, cancel_cb=cancel_cb)
                X_plot = X.iloc[idx2].reset_index(drop=True)
                shap_plot = shap2
            else:
                # continuous: use random subset
                X_plot = X_sample
                shap_plot = shap_values
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw_feat))
            out_base = os.path.join(out_expl, f"{best_model_name}_shap_dependence_{safe}")
            try:
                value_label_map = _resolve_feature_value_label_map(raw_feat, feature_value_labels)
                fig, ax = plot_shap_dependence_helper(
                    feature=raw_feat,
                    shap_values=shap_plot,
                    X_raw=X_plot,
                    feat_names=feat_names,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    clip_limit=clip_limit,
                    y_limit=y_limit,
                    value_label_map=value_label_map,
                    out_path=out_base)
                plt.close(fig)
                _raise_if_cancelled(cancel_cb)
            except Exception as e:
                if "cancelled by user" in str(e).strip().lower():
                    raise
                # Log and skip failures
                warn_file = os.path.join(out_expl, f"{best_model_name}_shap_dependence_warnings.txt")
                with open(warn_file, "a", encoding="utf-8") as f:
                    f.write(f"Error plotting {raw_feat}: {e}\n")
    except Exception as e:
        if "cancelled by user" in str(e).strip().lower():
            raise
        with open(os.path.join(out_expl, f"{best_model_name}_shap_dependence_warning.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))


def explain_with_shap(best_model_name: str, best_pipe, X, num_cols, cat_cols, outdir: str, seed: int = 42, top_n: int = SHAP_TOP_N, var_thresh: float = SHAP_VAR_THRESH, always_include: Optional[List[str]] = None):
    """Backward-compatible aggregator: generates both SHAP summary and dependence (top 3)."""
    generate_shap_summary(best_model_name, best_pipe, X, num_cols, cat_cols, outdir, seed, top_n=top_n, var_thresh=var_thresh)
    generate_shap_dependence(best_model_name, best_pipe, X, num_cols, cat_cols, outdir, top_n=top_n, seed=seed, var_thresh=var_thresh, always_include=always_include)

