import logging
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')

LOGGER = logging.getLogger(__name__)
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
from config import SHAP_DEPENDENCE_MODE
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
    # Remove pipeline prefixes and beautify without aggressively destroying existing capitalization
    try:
        if name in FEATURE_NAME_MAP:
            nm = _qascii(FEATURE_NAME_MAP[name])
            # Just return the mapped name as-is
            return nm
        n = _qascii(name)
        # strip common prefixes
        for pref in ('num__', 'cat__'):
            if n.startswith(pref):
                n = n[len(pref):]
        n = n.replace('__', '_')
        if '_' in n:
            base, level = n.split('_', 1)
            if level and not level.isnumeric() and len(level) > 0 and len(base) > 0 and not n.islower():
                # Note: if it's purely a user feature mapping like "televizyon_sure", we don't want a colon!
                # Actually, categorical variables often look like "Gender_Female", so if it has an uppercase letter, it might be OHE.
                # Just fallback to space replacement unless it really looks like a category mapping
                pass

        n = n.replace('_', ' ')
        n = n.strip()
        # If it is entirely lowercase and length > 0, capitalize just the first letter instead of capwords,
        # otherwise leave user's custom capitalizations intact! (For example, Acronyms, 'of', etc.)
        if n and n.islower():
            n = n[0].upper() + n[1:]
            
        # Preserve specific common acronyms purely safely
        out = re.sub(r"\btv\b", "TV", n)
        out = re.sub(r"\bai\b", "AI", out)
        out = re.sub(r"\bgdp\b", "GDP", out)
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
                dpi=300,
            )
            plt.close()
    except Exception as e:
        with open(os.path.join(out_expl, "pdp_warning.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

def _compute_shap(
    best_pipe,
    X,
    num_cols,
    cat_cols,
    seed: int = 42,
    cancel_cb=None,
    *,
    dependence_mode: str | None = None,
):
    _raise_if_cancelled(cancel_cb)
    from evaluation.metrics import get_feature_names_from_pipe
    feat_names = get_feature_names_from_pipe(best_pipe, num_cols, cat_cols)
    X_proc = best_pipe.named_steps["prep"].transform(X)
    is_df = hasattr(X_proc, "iloc")
    if is_df:
        feat_names = list(X_proc.columns)
    _raise_if_cancelled(cancel_cb)
    model_obj = best_pipe.named_steps["model"]
    n_sample = min(500, X_proc.shape[0])
    idx = np.random.RandomState(seed).choice(X_proc.shape[0], size=n_sample, replace=False)
    Xs = X_proc.iloc[idx] if is_df else X_proc[idx]

    dep_mode = str(dependence_mode or SHAP_DEPENDENCE_MODE or "interventional").strip().lower()
    
    # ML pipeline optimization: fast TreeExplainer for tree models when appropriate,
    # and downsampled background for maskers/explainers to keep GUI responsive.
    model_name = model_obj.__class__.__name__
    is_tree = any(t in model_name for t in ("RandomForest", "GradientBoosting", "XGB", "HistGradientBoosting", "Tree"))

    # Background sample (used for interventional SHAP and for correlation-aware masking)
    bg_size = min(200, X_proc.shape[0])
    bg_idx = np.random.RandomState(seed).choice(X_proc.shape[0], size=bg_size, replace=False)
    bg = X_proc.iloc[bg_idx] if is_df else X_proc[bg_idx]

    explainer = None
    shap_output = None

    # Correlation-aware SHAP: partition masker groups correlated features (Owen values)
    if dep_mode in {"partition", "correlation", "correlated", "grouped"}:
        try:
            try:
                masker = shap.maskers.Partition(bg, clustering="correlation")
            except TypeError:
                masker = shap.maskers.Partition(bg)
            explainer = shap.Explainer(model_obj, masker)
            shap_output = explainer(Xs)
        except Exception:
            explainer = None
            shap_output = None

    # Interventional / legacy tree SHAP for tree models (fast and common in applied work)
    if shap_output is None and is_tree and dep_mode in {"interventional", "independent", "tree_path_dependent"}:
        try:
            if dep_mode == "tree_path_dependent":
                explainer = shap.TreeExplainer(model_obj)
            else:
                try:
                    explainer = shap.TreeExplainer(model_obj, data=bg, feature_perturbation="interventional")
                except TypeError:
                    explainer = shap.TreeExplainer(model_obj, bg)
            shap_output = explainer(Xs)
        except Exception:
            explainer = None
            shap_output = None

    # Generic fallback: shap.Explainer with Independent masker (interventional, assumes feature independence)
    if shap_output is None:
        try:
            try:
                masker = shap.maskers.Independent(bg)
                explainer = shap.Explainer(model_obj, masker)
            except Exception:
                explainer = shap.Explainer(model_obj, bg)
            shap_output = explainer(Xs)
        except Exception:
            # Last resort: try TreeExplainer without background
            if is_tree:
                explainer = shap.TreeExplainer(model_obj)
                shap_output = explainer(Xs)
            else:
                raise
    _raise_if_cancelled(cancel_cb)
    if hasattr(shap_output, 'values'):
        shap_values = shap_output.values
    else:
        shap_values = shap_output
    # Unpack multi-output shap_values if returned as list or tuple
    if isinstance(shap_values, (list, tuple)):
        shap_values = shap_values[0]
        
    # ** OHE Feature Grouping for Academic Interpretability **
    grouped_shap = []
    grouped_Xs = []
    grouped_feat_names = []
    
    if is_df and len(feat_names) == shap_values.shape[1]:
        # Identify groups (OneHotEncoders usually output cat__OriginalName_Value)
        # However, for OHE, we just look at the prefix before the last underscore, 
        # but safely map it back to original cat_cols.
        # Simple heuristic: if a feature starts with cat__ or starts with same name as a cat col.
        used = set()
        
        # Base mapping from final names to original feature (if missing, identity)
        group_mapping = []
        for i, f_name in enumerate(feat_names):
            base_col = f_name
            # If from our Pipeline (starts with cat__) -> cat__Original_Cataegory -> Original
            # Ensure we check the longest column names first to avoid prefix shadowing
            sorted_cols = sorted(list(cat_cols) + list(num_cols), key=len, reverse=True)
            for c in sorted_cols:
                # Match either prefixed from sklearn ColumnTransformer or direct feature
                if f_name.startswith(f"cat__{c}_") or f_name.startswith(f"num__{c}_") or f_name == f"cat__{c}" or f_name == f"num__{c}" or f_name.startswith(f"{c}_") or f_name == c:
                    base_col = c
                    break
            group_mapping.append(base_col)
            
        unique_bases = []
        for b in group_mapping:
            if b not in unique_bases:
                unique_bases.append(b)
                
        for base in unique_bases:
            # Find indices for this base
            base_indices = [i for i, b in enumerate(group_mapping) if b == base]
            if len(base_indices) == 1:
                grouped_shap.append(shap_values[:, base_indices[0]])
                if hasattr(Xs, "iloc"):
                    grouped_Xs.append(Xs.iloc[:, base_indices[0]].values)
                else:
                    grouped_Xs.append(Xs[:, base_indices[0]])
            else:
                # Group them! For SHAP values, we sum them across the OHE features.
                grouped_shap.append(np.sum(shap_values[:, base_indices], axis=1))
                # For X values of a categorical feature, we extract the original string data if possible!
                # Since Xs is encoded, we just use the original dataset string values to plot color properly.
                if base in X.columns:
                    # original data
                    grouped_Xs.append(X[base].iloc[idx].values)
                else:
                    grouped_Xs.append(np.zeros(shap_values.shape[0]))
                
            grouped_feat_names.append(base)
            
        shap_values = np.column_stack(grouped_shap)
        Xs = pd.DataFrame({k: v for k, v in zip(grouped_feat_names, grouped_Xs)})
        feat_names = grouped_feat_names
        
    return feat_names, Xs, shap_values, idx


def _disambiguate_labels(labels: list[str], raw_names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for lbl, raw in zip(labels, raw_names):
        key = str(lbl)
        if key in seen:
            seen[key] += 1
            out.append(f"{key} ({raw})")
        else:
            seen[key] = 1
            out.append(key)
    return out


def _write_shap_correlation_report(
    *,
    out_expl: str,
    best_model_name: str,
    sel_names: list[str],
    sel_pretty: list[str],
    shap_values_sel: np.ndarray,
    Xs_sel,
    mean_abs_shap: np.ndarray,
    dependence_mode: str | None,
    seed: int,
):
    """Write modern SHAP correlation diagnostics used in applied ML literature.

    - SHAP-SHAP correlation: correlates attribution vectors to detect redundancy / shared effects.
    - SHAP-vs-feature association: Spearman(feature value, SHAP) to summarize directionality.

    NOTE: All correlations are computed on the SHAP sample (<=500 rows by default).
    """
    try:
        import pandas as pd
        from scipy.stats import spearmanr
    except Exception:
        return

    if shap_values_sel is None or len(sel_names) == 0:
        return

    # Ensure ndarray
    vals = np.asarray(shap_values_sel, dtype=float)
    if vals.ndim != 2 or vals.shape[1] != len(sel_names):
        return

    dep_mode_txt = str(dependence_mode or SHAP_DEPENDENCE_MODE or "interventional")

    # SHAP-SHAP correlation matrices (signed and abs) on selected features
    shap_df = pd.DataFrame(vals, columns=sel_names)
    corr_signed = shap_df.corr(method="spearman")
    corr_abs = shap_df.abs().corr(method="spearman")

    # SHAP-vs-feature value association (directionality)
    assoc_rows: list[dict[str, object]] = []
    for j, raw_name in enumerate(sel_names):
        yv = vals[:, j]
        xcol = Xs_sel.iloc[:, j] if hasattr(Xs_sel, "iloc") else Xs_sel[:, j]

        try:
            x_num = pd.to_numeric(xcol, errors="coerce")
            if x_num.notna().any():
                x_arr = x_num.to_numpy(dtype=float)
                feat_type = "numeric"
            else:
                codes, _ = pd.factorize(xcol, sort=False)
                x_arr = codes.astype(float)
                x_arr[codes < 0] = np.nan
                feat_type = "categorical"
        except Exception:
            continue

        mask = np.isfinite(x_arr) & np.isfinite(yv)
        n_eff = int(mask.sum())
        if n_eff < 3:
            continue

        try:
            rho, p = spearmanr(x_arr[mask], yv[mask])
        except Exception:
            continue

        assoc_rows.append(
            {
                "feature": raw_name,
                "pretty": _pretty_label(raw_name),
                "feature_type": feat_type,
                "spearman_r": float(rho) if rho is not None else float("nan"),
                "abs_spearman_r": float(abs(rho)) if rho is not None else float("nan"),
                "p_value": float(p) if p is not None else float("nan"),
                "n": n_eff,
            }
        )

    assoc_df = pd.DataFrame(assoc_rows)
    if not assoc_df.empty:
        assoc_df = assoc_df.sort_values(["abs_spearman_r", "feature"], ascending=[False, True])

    # Save a single Excel report with multiple sheets
    report_path = os.path.join(out_expl, f"{best_model_name}_shap_correlation_report.xlsx")
    meta_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "dependence_mode": dep_mode_txt,
                "seed": int(seed),
                "n_samples": int(vals.shape[0]),
                "n_features": int(vals.shape[1]),
            }
        ]
    )
    fmap_df = pd.DataFrame(
        {
            "feature": list(sel_names),
            "pretty": list(sel_pretty),
            "mean_abs_shap": np.asarray(mean_abs_shap, dtype=float),
        }
    )

    try:
        with pd.ExcelWriter(report_path, engine="openpyxl") as xw:
            meta_df.to_excel(xw, sheet_name="meta", index=False)
            fmap_df.to_excel(xw, sheet_name="feature_map", index=False)
            corr_signed.to_excel(xw, sheet_name="shap_corr_signed")
            corr_abs.to_excel(xw, sheet_name="shap_corr_abs")
            if not assoc_df.empty:
                assoc_df.to_excel(xw, sheet_name="shap_vs_feature", index=False)
    except Exception as _exc:
        LOGGER.warning("SHAP correlation Excel export failed: %s", _exc)

    # Heatmap (signed correlations)
    try:
        import seaborn as sns

        labels_plot = _disambiguate_labels(sel_pretty, sel_names)
        corr_plot = corr_signed.copy()
        corr_plot.index = labels_plot
        corr_plot.columns = labels_plot
        k = len(labels_plot)
        fig_w = min(16.0, max(8.0, 0.45 * k + 2.6))
        fig_h = min(16.0, max(6.8, 0.45 * k + 2.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            corr_plot,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap="coolwarm",
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Spearman ρ"},
            ax=ax,
        )
        ax.set_title(f"SHAP Correlation (Spearman) - {best_model_name}\nmode={dep_mode_txt}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        _save_fig_formats(os.path.join(out_expl, f"{best_model_name}_shap_correlation_heatmap"))
        plt.close(fig)
    except Exception as _exc:
        LOGGER.warning("SHAP correlation heatmap failed: %s", _exc)


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
    dependence_mode: str | None = None,
    cancel_cb=None):
    out_expl = os.path.join(outdir, '3_Manuscript_Figures', best_model_name)
    os.makedirs(out_expl, exist_ok=True)
    if not SHAP_OK:
        with open(os.path.join(out_expl, f"{best_model_name}_shap_warning.txt"), "w", encoding="utf-8") as f:
            f.write("SHAP library not installed.")
        return
    try:
        _raise_if_cancelled(cancel_cb)
        feat_names, Xs, shap_values, _ = _compute_shap(
            best_pipe,
            X,
            num_cols,
            cat_cols,
            seed,
            cancel_cb=cancel_cb,
            dependence_mode=dependence_mode,
        )
        _raise_if_cancelled(cancel_cb)
        # Rank by importance (mean |SHAP|) and drop near-zero variance columns
        imp = np.mean(np.abs(shap_values), axis=0)
        idx_sorted = np.argsort(imp)[::-1]
        if var_thresh is None:
            idx_filtered = list(idx_sorted)
        else:
            # Compute per-feature variance robustly even when Xs contains object/categorical values.
            try:
                if hasattr(Xs, "columns"):
                    var_list = []
                    for col in list(Xs.columns):
                        ser = Xs[col]
                        try:
                            num = pd.to_numeric(ser, errors="coerce")
                            if num.notna().any():
                                var_list.append(float(np.nanvar(num.to_numpy(dtype=float))))
                            else:
                                codes, _ = pd.factorize(ser, sort=False)
                                codes = codes.astype(float)
                                codes[codes < 0] = np.nan
                                var_list.append(float(np.nanvar(codes)))
                        except Exception:
                            var_list.append(0.0)
                    var = np.asarray(var_list, dtype=float)
                else:
                    var = np.asarray(np.var(Xs, axis=0), dtype=float)
            except Exception:
                var = np.zeros_like(imp, dtype=float)

            idx_filtered = [i for i in idx_sorted if float(var[i]) >= float(var_thresh)]
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

        # Select the feature matrix for the chosen features (reused for reports + plots)
        if hasattr(Xs, "iloc"):
            Xs_sel = Xs.iloc[:, top_idx]
        else:
            Xs_sel = Xs[:, top_idx]

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
        except Exception as _exc:
            LOGGER.warning("SHAP values Excel export failed: %s", _exc)
        _raise_if_cancelled(cancel_cb)

        # Correlation report (Spearman SHAP-SHAP + SHAP-vs-feature association)
        try:
            corr_k = min(len(top_idx), 40)
            top_idx_corr = top_idx[:corr_k]
            sel_names_corr = [feat_names[i] for i in top_idx_corr]
            sel_pretty_corr = [feat_pretty_all[i] for i in top_idx_corr]
            if hasattr(Xs, "iloc"):
                Xs_sel_corr = Xs.iloc[:, top_idx_corr]
            else:
                Xs_sel_corr = Xs[:, top_idx_corr]
            _write_shap_correlation_report(
                out_expl=out_expl,
                best_model_name=best_model_name,
                sel_names=sel_names_corr,
                sel_pretty=sel_pretty_corr,
                shap_values_sel=shap_values[:, top_idx_corr],
                Xs_sel=Xs_sel_corr,
                mean_abs_shap=imp[top_idx_corr],
                dependence_mode=dependence_mode,
                seed=seed,
            )
        except Exception as _exc:
            LOGGER.warning("SHAP correlation report failed: %s", _exc)

    # Beeswarm (selected only) with dynamic height and margins
        nfeat = len(top_idx)
        max_label_len = max((len(str(lbl)) for lbl in sel_pretty), default=1)
        height = min(18.0, max(4.6, 0.42 * nfeat + 1.9))
        width = min(15.0, max(9.8, 8.6 + 0.045 * max_label_len))
        plt.figure(figsize=(width, height))
        # Dynamic lower outlier threshold for beeswarm: drop extreme bottom values
        try:
            vals = shap_values[:, top_idx]
            Xs_vals = Xs_sel
            trim_pct = float(SHAP_BEESWARM_TRIM_PCT) if SHAP_BEESWARM_TRIM_PCT is not None else None
            if trim_pct is not None and 0.0 < trim_pct < 50.0:
                flat_vals = np.ravel(vals)
                lower_thresh = float(np.nanpercentile(flat_vals, trim_pct))
                mask = np.all(vals >= lower_thresh, axis=1)
                # guard: if mask prunes almost all rows, relax trimming
                if mask.sum() < max(10, int(0.3 * len(mask))):
                    mask = np.ones(len(mask), dtype=bool)
                vals = vals[mask]
                Xs_vals = Xs_vals.loc[mask] if hasattr(Xs_vals, "loc") else Xs_vals[mask]
            # optional min-cap: if provided, clip values to this lower bound for display stability
            if SHAP_BEESWARM_MIN_CAP is not None:
                try:
                    lb = float(SHAP_BEESWARM_MIN_CAP)
                    vals = np.maximum(vals, lb)
                except Exception as _exc:
                    LOGGER.debug("SHAP_BEESWARM_MIN_CAP clipping skipped: %s", _exc)
        except Exception:
            vals = shap_values[:, top_idx]
            Xs_vals = Xs_sel
        # _qascii is module-level (utils.text.normalize_quotes_ascii or a fallback stub)
        sel_pretty_plot = [_qascii(n) for n in sel_pretty]
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
        except Exception as _exc:
            LOGGER.debug("SHAP beeswarm layout adjustment skipped: %s", _exc)
        _save_fig_formats(os.path.join(out_expl, f"{best_model_name}_shap_summary_beeswarm"))
        plt.close()
        _raise_if_cancelled(cancel_cb)
        # Bar (selected only)
        bar_h = min(14.0, max(3.0, 0.5 * nfeat + 1.4))
        bar_w = min(12.5, max(6.4, 5.6 + 0.04 * max_label_len))
        plt.figure(figsize=(bar_w, bar_h))
        sel_pretty_bar = [_qascii(n) for n in sel_pretty]
        shap.summary_plot(shap_values[:, top_idx], Xs_sel, feature_names=sel_pretty_bar, plot_type="bar", show=False)
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
        except Exception as _exc:
            LOGGER.debug("SHAP bar layout adjustment skipped: %s", _exc)
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
    dependence_mode: str | None = None,
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
        feat_names, Xs, shap_values, idx = _compute_shap(
            best_pipe,
            X,
            num_cols,
            cat_cols,
            seed,
            cancel_cb=cancel_cb,
            dependence_mode=dependence_mode,
        )
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

