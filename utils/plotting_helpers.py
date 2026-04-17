import re
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from config import SAVE_PDF
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
})
import seaborn as sns

from utils.logger import get_logger

# Try LOWESS for smooth trendlines
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False

from utils.text import normalize_text as _norm


from utils.humanize import map_labels
from string import capwords as _capwords
# Centralized quote/backtick normalization
try:
    from utils.text import normalize_quotes_ascii as _qascii
except Exception:
    def _qascii(s: str) -> str:
        return str(s)

# Optional per-feature y-limits for SHAP dependence (kept empty to prefer auto-fit)
FEATURE_YLIMS: Dict[str, Tuple[float, float]] = {}

LOGGER = get_logger(__name__)

# Pretty display names for known columns
DISPLAY_NAME_MAP: Dict[str, str] = {
    _norm('Province'): 'Province',
    _norm('Gender'): 'Gender',
    _norm('Age'): 'Age',
    _norm('Household Size'): 'Household Size',
    _norm('Number of Siblings'): 'Number of Siblings',
    _norm('Birth Order'): 'Birth Order',
    _norm('Place of Residence'): 'Place of Residence',
    _norm('Nationality'): 'Nationality',
    _norm("Mother's Education"): "Mother's Education",
    _norm("Father's Education"): "Father's Education",
    _norm("Parents' Marital Status (Together/Separated)"): "Parents' Marital Status",
    _norm('Household Income'): 'Household Income',
    _norm("Mother's Occupation"): "Mother's Occupation",
    _norm("Father's Occupation"): "Father's Occupation",
    _norm('Province/District'): 'Province/District',
    _norm('Reading Books (Frequency)'): 'Reading Books (Frequency)',
    _norm('TV Time (Daily Hours)'): 'TV Time (Daily Hours)',
    _norm('Mobile Phone (Daily Hours)'): 'Mobile Phone (Daily Hours)',
    _norm('Extracurricular Activity Participation'): 'Extracurricular Activity Participation',
    _norm('Teacher Intervention'): 'Teacher Intervention',
    _norm('Frequency of Bullying Exposure'): 'Frequency of Bullying Exposure',
    _norm('Reporting Bullying to Family'): 'Reporting Bullying to Family',
}


def display_name(raw: str) -> str:
    """Return a publication-friendly display name for a raw column key."""
    k = _norm(raw)
    base = DISPLAY_NAME_MAP.get(k, str(raw).replace('_', ' ').strip())
    # sanitize curly quotes/backticks to ASCII to avoid missing glyphs (centralized)
    try:
        name = _qascii(base)
        # word-based capitalization preserves apostrophes inside words
        name = _capwords(name)
    except Exception:
        name = base
    return name


def _normalize_rule_token(value) -> str:
    txt = str(value if value is not None else "").strip().lower()
    txt = txt.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
    txt = txt.replace("ş", "s").replace("ö", "o").replace("ç", "c")
    return txt


def _map_value_by_rules(value, rule_map: Dict[str, str]) -> object:
    if pd.isna(value):
        return value
    token = _normalize_rule_token(value)
    if token in rule_map:
        return rule_map[token]
    try:
        fval = float(value)
        if np.isfinite(fval) and float(fval).is_integer():
            as_int = _normalize_rule_token(str(int(fval)))
            if as_int in rule_map:
                return rule_map[as_int]
    except Exception:
        LOGGER.exception("Value mapping failed")
    return value


def _rule_sorted_labels(rule_map: Dict[str, str]) -> List[str]:
    def _sort_key(raw_key: str):
        try:
            return (0, float(raw_key))
        except Exception:
            return (1, str(raw_key))

    ordered: List[str] = []
    for src in sorted(rule_map.keys(), key=_sort_key):
        lbl = str(rule_map.get(src, "")).strip()
        if lbl and lbl not in ordered:
            ordered.append(lbl)
    return ordered


# map_labels now imported from utils.humanize to centralize logic


def clip_outliers(values: np.ndarray, limit: Optional[float] = 3.0) -> np.ndarray:
    """Optionally clip numeric array to [-limit, +limit]. If limit is None, return as-is."""
    arr = np.asarray(values, dtype=float)
    if limit is None:
        return arr
    return np.clip(arr, -float(limit), float(limit))


def _group_feature_indices(feat_names: List[str], num_cols: Iterable[str], cat_cols: Iterable[str]) -> Dict[str, List[int]]:
    """
    Build a mapping from raw feature names to list of indices in the processed
    feature array. Numeric features map 1-to-1; categorical features map to all
    their OneHot-encoded columns (prefix `raw_`).
    """
    groups: Dict[str, List[int]] = {c: [] for c in (list(num_cols) + list(cat_cols))}
    # Index numeric
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    for n in num_cols:
        if n in name_to_idx:
            groups[n].append(name_to_idx[n])
    # Index categorical by prefix match
    for c in cat_cols:
        pref = f"{c}_"
        for i, name in enumerate(feat_names):
            if name.startswith(pref):
                groups[c].append(i)
    # Drop empties
    return {k: v for k, v in groups.items() if len(v) > 0}


def _var_of_raw(feature: str, X_raw: pd.DataFrame) -> float:
    s = X_raw[feature]
    try:
        if pd.api.types.is_numeric_dtype(s):
            return float(np.nanvar(pd.to_numeric(s, errors='coerce')))
        # treat categorical as variance of codes
        cat = pd.Categorical(s)
        codes = pd.Series(cat.codes).replace({-1: np.nan})
        return float(np.nanvar(codes))
    except Exception:
        return 0.0


def plot_shap_dependence(
    feature: str,
    shap_values: np.ndarray,
    X_raw: pd.DataFrame,
    feat_names: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    *,
    clip_limit: Optional[float] = None,
    y_limit: Optional[Tuple[float, float]] = None,
    fig_size: Tuple[float, float] = (7, 5),
    color_cmap: str = 'viridis',
    value_label_map: Optional[Dict[str, str]] = None,
    out_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Publication-ready SHAP dependence plot for a raw feature name.

    - categorical/ordinal: boxplot + strip; x shows labels
    - continuous: scatter + LOWESS; x shows raw values
    Applies SHAP clipping and fixed y-axis limits for comparability.
    """
    # ...existing code to locate feature index groups...
    groups = _group_feature_indices(feat_names, num_cols, cat_cols)
    if feature not in groups:
        fkey = _norm(feature)
        candidates = {k: v for k, v in groups.items() if _norm(k) == fkey}
        if not candidates:
            raise KeyError(f"Feature '{feature}' not found in processed feature names.")
        feature = list(candidates.keys())[0]
    idxs = groups[feature]

    # raw and shap arrays
    s_raw = X_raw[feature]
    if feature in cat_cols and len(idxs) > 1:
        s_shap_raw = shap_values[:, idxs].sum(axis=1)
    else:
        s_shap_raw = shap_values[:, idxs[0]]
    # apply optional clipping only for visualization
    s_shap = clip_outliers(s_shap_raw, limit=clip_limit)

    # prepare frame
    dfp = pd.DataFrame({ 'x_raw': s_raw, 'shap': s_shap })

    # detect discrete vs continuous
    is_discrete = False
    if feature in cat_cols or feature in ['Gender']:
        is_discrete = True
    else:
        vals = dfp['x_raw'].dropna().unique()
        if np.all(np.mod(vals, 1) == 0) and len(vals) <= 20:
            is_discrete = True

    fig, ax = plt.subplots(figsize=fig_size)

    if is_discrete:
        applied_rule_map: Dict[str, str] = {}
        if isinstance(value_label_map, dict):
            for src, dst in value_label_map.items():
                src_norm = _normalize_rule_token(src)
                dst_txt = str(dst).strip()
                if src_norm and dst_txt and src_norm not in applied_rule_map:
                    applied_rule_map[src_norm] = dst_txt

        if applied_rule_map:
            mapped = dfp['x_raw'].apply(lambda v: _map_value_by_rules(v, applied_rule_map))
            dfp['x'] = mapped
        else:
            # map labels and preserve defined order (from CATEGORY_LABELS)
            dfp_labeled = map_labels(dfp.rename(columns={'x_raw': feature}))
            dfp['x'] = dfp_labeled[feature]

        # drop categories with no observations
        if pd.api.types.is_categorical_dtype(dfp['x']):
            dfp['x'] = dfp['x'].cat.remove_unused_categories()
        # if not categorical, fall back to sorted unique
        if not pd.api.types.is_categorical_dtype(dfp['x']):
            if applied_rule_map:
                order = _rule_sorted_labels(applied_rule_map)
                for lbl in list(dfp['x'].dropna().unique()):
                    if lbl not in order:
                        order.append(lbl)
            else:
                # For Gender 0/1 ensure Female/Male order
                if feature == 'Gender':
                    order = ['Female', 'Male']
                else:
                    # sort natural order if labels look numeric-like strings (e.g., '0h','1h','2h','3h','4+h' -> custom stays via mapping)
                    try:
                        vals = list(dfp['x'].dropna().unique())
                        vals_sorted = sorted(vals, key=lambda v: float(str(v).split('h')[0].split('+')[0].split('–')[0]) if re.match(r"^[0-9]+", str(v)) else str(v))
                        order = vals_sorted
                    except Exception:
                        order = sorted(dfp['x'].dropna().unique())
            dfp['x'] = pd.Categorical(dfp['x'], categories=order, ordered=True)
        # adjust figure width if many categories
        ncat = len(dfp['x'].cat.categories)
        if ncat > 6:
            fig.set_size_inches(max(fig_size[0], ncat * 0.8), fig_size[1])
        # boxplot
        sns.boxplot(x='x', y='shap', data=dfp, ax=ax,
                    boxprops=dict(facecolor='white', edgecolor='gray'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='gray'), capprops=dict(color='gray'),
                    showcaps=True, fliersize=0)
        # strip points
        sns.stripplot(x='x', y='shap', data=dfp, ax=ax,
                      color='gold', size=4, jitter=True, alpha=0.7)
        ax.set_xlabel(display_name(feature))
        # rotate labels and adjust margins
        ax.tick_params(axis='x', labelrotation=45)
        fig.subplots_adjust(bottom=0.25)
    else:
        # continuous scatter + lowess
        sc = ax.scatter(dfp['x_raw'], dfp['shap'], c=dfp['x_raw'], cmap=color_cmap, s=20, alpha=0.7)
        if _HAS_LOWESS:
            lo = _lowess(dfp['shap'], dfp['x_raw'], return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color='red', linewidth=2)
        else:
            # fallback linear trend
            z = np.polyfit(dfp['x_raw'], dfp['shap'], 1)
            xseq = np.linspace(dfp['x_raw'].min(), dfp['x_raw'].max(), 100)
            ax.plot(xseq, np.polyval(z, xseq), color='red', linewidth=2)
        ax.set_xlabel(display_name(feature))
        # colorbar raw scale
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(display_name(feature))

    # shared styling
    ax.set_ylabel('SHAP value')
    ax.set_title(f"SHAP dependence — {display_name(feature)}")
    # y-limit: fit to data range with a small padding (Q1-style)
    if y_limit is None:
        try:
            y_min = float(np.nanmin(s_shap))
            y_max = float(np.nanmax(s_shap))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                raise ValueError('non-finite')
            # add 5% padding of the span
            span = max(1e-6, y_max - y_min)
            pad = 0.05 * span
            low = y_min - pad
            high = y_max + pad
            # If span is tiny, center around zero with a minimal band
            if span < 0.2:
                half = max(0.1, 0.5 * span + pad)
                ax.set_ylim(-half, half)
            else:
                ax.set_ylim(low, high)
        except Exception:
            # Fallback: symmetric ±2
            ax.set_ylim(-2.0, 2.0)
    else:
        ax.set_ylim(y_limit)
    sns.despine(fig=fig)
    fig.tight_layout()

    # save
    if out_path:
        fig.savefig(out_path + '.png', dpi=300)
        if SAVE_PDF:
            fig.savefig(out_path + '.pdf')
    return fig, ax


def top_raw_features_by_shap(
    shap_values: np.ndarray,
    feat_names: List[str],
    X_raw: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    top_n: Optional[int] = 10,
    var_thresh: Optional[float] = 1e-8,
) -> List[str]:
    """Select top N raw features by mean |SHAP|, grouping OHE columns per categorical feature."""
    groups = _group_feature_indices(feat_names, num_cols, cat_cols)
    # importance per processed col
    imp = np.mean(np.abs(shap_values), axis=0)
    # aggregate by raw feature
    scores: List[Tuple[str, float]] = []
    for raw, idxs in groups.items():
        v = _var_of_raw(raw, X_raw)
        if (var_thresh is None) or (v >= float(var_thresh)):
            scores.append((raw, float(np.sum(imp[idxs]))))
    scores.sort(key=lambda t: t[1], reverse=True)
    if top_n is None or top_n <= 0:
        return [s[0] for s in scores]
    return [s[0] for s in scores[:min(top_n, len(scores))]]


def save_bar(fig_path: str, labels: list[str], values: list[float], title: str, xlabel: str):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_style('whitegrid')
    fig_h = max(3.0, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ypos = np.arange(len(labels))
    bars = ax.barh(ypos, values, color='#64B5F6', edgecolor='#1E88E5', alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.5)
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]
    for y, b, v in zip(ypos, bars, values):
        x = b.get_width()
        off = span * 0.01
        ha = 'left' if x >= 0 else 'right'
        ax.text(x + (off if x >= 0 else -off), y, f"{v:.3f}", va='center', ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
