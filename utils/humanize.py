from typing import Dict

import numpy as np
import pandas as pd
from utils.text import normalize_text as _norm


# Central label mappings for categorical/ordinal variables (normalized keys)
CATEGORY_LABELS: Dict[str, Dict[int, str]] = {
    # Binary
        _norm('Gender'): {0: 'Female', 1: 'Male'},  # 1/2 will be remapped to 0/1 automatically
    # Accept both 0/1 and 1/2 codings by including both in mapping
    _norm("Parents' Marital Status (Together/Separated)"): {0: 'Together', 1: 'Separated'},  # 1/2 handled by remap
    _norm("Parents' Marital Status"): {0: 'Together', 1: 'Separated'},  # fallback when column name lacks parentheses
    # Ordinal scales
    _norm('Grade/Class'): {2: '2', 3: '3', 4: '4'},
    _norm('Reporting Bullying to Family'): {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often', 4: 'Always'},
    _norm('Teacher Intervention'): {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often', 4: 'Always'},
    # Education levels (1-based coding common in dataset)
    _norm("Mother's Education"): {1: 'Illiterate', 2: 'Primary', 3: 'Middle School', 4: 'High School', 5: 'University'},
    _norm("Father's Education"): {1: 'Illiterate', 2: 'Primary', 3: 'Middle School', 4: 'High School', 5: 'University'},
    # Occupation (1=Unemployed, 2=Employed)
    _norm("Mother's Occupation"): {1: 'Employed', 2: 'Unemployed'},  # dataset: 1=Employed, 2=Unemployed
        _norm("Father's Occupation"): {1: 'Employed', 2: 'Unemployed'},  # dataset: 1=Employed, 2=Unemployed
    # Income levels
    _norm('Household Income'): {1: 'Low', 2: 'Medium', 3: 'High'},
    # Time/frequency scales
    _norm('TV Time (Daily Hours)'): {0: '0h', 1: '1h', 2: '2h', 3: '3h', 4: '4+h'},
    _norm('Mobile Phone (Daily Hours)'): {0: '0h', 1: '1h', 2: '2h', 3: '3h', 4: '4+h'},
    _norm('Reading Books (Frequency)'): {0: 'None', 1: '0–2', 2: '2–4', 3: '4–6', 4: '6–8+'},
    # Nationality mapping (1=Turkish, 2=Foreign)
    _norm('Nationality'): {1: 'Turkish', 2: 'Foreign'},
    # Province mapping (1=Urban, 2=Rural)
    _norm('Province/District'): {1: 'Urban', 2: 'Rural'},
}


def _sanitize_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Fix impossible numeric values for certain known columns.

    - Number of Siblings: values < 0 -> 0, large values -> 7+ (as label later)
    - Birth Order: values < 1 -> NaN
    """
    dfc = df.copy()
    # Number of Siblings
    sib_key = _norm('Number of Siblings')
    for col in list(dfc.columns):
        if _norm(col) == sib_key:
            s = pd.to_numeric(dfc[col], errors='coerce')
            s = s.mask(s < 0, 0)
            dfc[col] = s
    # Birth Order
    bo_key = _norm('Birth Order')
    for col in list(dfc.columns):
        if _norm(col) == bo_key:
            s = pd.to_numeric(dfc[col], errors='coerce')
            s = s.mask(s < 1, np.nan)
            dfc[col] = s
    return dfc


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df where known categorical/ordinal columns are converted
    from integer codes to human-readable labels for plotting/reporting.

    This function intentionally does not modify the original df.
    """
    dfp = _sanitize_numeric_ranges(df)
    dfp = dfp.copy()
    for col in list(dfp.columns):
        # Special case: binary collapse for Reporting Bullying to Family
        if _norm(col) == _norm('Reporting Bullying to Family'):
            codes = pd.to_numeric(dfp[col], errors='coerce')
            dfp[col] = pd.Categorical(
                codes.apply(lambda v: 'Yes' if v == 1 else 'No'),
                categories=['Yes', 'No'], ordered=True
            )
            continue
        key = _norm(col)
        mapping = CATEGORY_LABELS.get(key)
        if mapping is not None:
            # normalize to numeric codes first
            codes = pd.to_numeric(dfp[col], errors='coerce')
            mk = sorted({int(k) for k in mapping.keys()})
            present = codes.dropna().astype(int)
            mapped = codes.map(mapping)
            # Explicit binary 1/2 -> 0/1 remap when mapping is 0/1
            if set(mk) == {0, 1} and not present.empty and set(present.unique()).issubset({1, 2}):
                mapped = codes.apply(lambda v: mapping.get(int(v) - 1) if pd.notna(v) else np.nan)
            # Also handle 1/2-coded binaries when mapping keys are 0/1-like labels (e.g., Together/Separated)
            elif set(mk) == {0, 1} and not present.empty and set(present.unique()).issubset({0, 1, 2}):
                mapped = codes.apply(lambda v: mapping.get(int(v) if int(v) in mapping else int(v) - 1) if pd.notna(v) else np.nan)
            # Otherwise, if mapping produced many NaNs, try generic offset alignment
            elif mapped.notna().mean() < 0.6:
                if mk and mk[0] == 0 and mk == list(range(mk[-1] + 1)):
                    if not present.empty:
                        base = int(present.min())
                        mapped = codes.apply(lambda v: mapping.get(int(v) - base) if pd.notna(v) else np.nan)
            dfp[col] = mapped
            # enforce category order by mapping key order
            order = [mapping[k] for k in sorted(mapping.keys()) if k in mapping]
            dfp[col] = pd.Categorical(dfp[col], categories=order, ordered=True)

    # Special bucketing for siblings to '7+' label after mapping
    sib_key = _norm('Number of Siblings')
    for col in list(dfp.columns):
        if _norm(col) == sib_key:
            s = pd.to_numeric(df[col], errors='coerce')
            # categories 0..7+
            labels = [str(i) for i in range(0, 7)] + ['7+']
            cats = pd.Categorical(
                s.apply(lambda v: '7+' if pd.notna(v) and v >= 7 else (str(int(v)) if pd.notna(v) and v >= 0 else np.nan)),
                categories=labels,
                ordered=True,
            )
            dfp[col] = cats
    return dfp
