import pandas as pd
from pandas.api import types as ptypes
import re

def validate_csv_structure(df: pd.DataFrame, min_rows: int = 20):
    """
    Validates the structure and content of a loaded dataset DataFrame.
    Returns a tuple of (critical_errors, warnings).
    """
    critical = []
    warnings = []

    # Basic shape checks
    n_cols = df.shape[1]
    if n_cols < 2:
        critical.append(f"Dataset must have at least 2 columns (found {n_cols}).")
    n_rows = df.shape[0]
    if n_rows < min_rows:
        critical.append(f"Dataset must have at least {min_rows} rows (found {n_rows}).")

    # Column name checks
    cols = list(df.columns)
    if any(str(c).strip() == '' for c in cols):
        critical.append("Some column names are empty.")
    if len(set(cols)) != len(cols):
        critical.append("Column names are not unique.")

    # Cache per-column summaries once (hot path during dataset load)
    try:
        missing_pct_by_col = df.isna().mean()
    except Exception:
        missing_pct_by_col = None
    try:
        nunique_by_col = df.nunique(dropna=True)
    except Exception:
        nunique_by_col = None

    # Missingness and constants
    for col in cols:
        s = df[col]
        try:
            pct_missing = float(missing_pct_by_col[col]) if missing_pct_by_col is not None else float(s.isna().mean())
        except Exception:
            pct_missing = float(s.isna().mean())
        if pct_missing > 0.8:
            warnings.append(
                f"Column '{col}' has {int(pct_missing*100)}% missing values and may lead to unreliable results."
            )
        # constant values ignoring NaNs
        try:
            unique_cnt = int(nunique_by_col[col]) if nunique_by_col is not None else int(s.nunique(dropna=True))
        except Exception:
            unique_cnt = int(s.nunique(dropna=True))

        if unique_cnt <= 1:
            warnings.append(
                f"Column '{col}' contains constant values (no variability) and will be excluded from analysis."
            )

    # Check for at least one numeric column
    numeric_cols = [col for col in cols if ptypes.is_numeric_dtype(df[col])]
    if not numeric_cols:
        critical.append("No numeric columns detected. At least one numeric column is required for modeling.")

    # High-cardinality categorical columns
    for col in cols:
        if not df[col].dropna().empty and not ptypes.is_numeric_dtype(df[col]):
            try:
                unique_cnt = int(nunique_by_col[col]) if nunique_by_col is not None else int(df[col].nunique(dropna=True))
            except Exception:
                unique_cnt = int(df[col].nunique(dropna=True))
            if unique_cnt > 100:
                warnings.append(
                    f"Column '{col}' has high cardinality ({unique_cnt} unique values) which may impact performance."
                )

    # Column name whitespace or unsafe names
    for col in cols:
        col_text = str(col)
        if col_text != col_text.strip():
            warnings.append(
                f"Column name '{col_text}' has leading/trailing whitespace."
            )
        if col_text and col_text[0].isdigit():
            warnings.append(
                f"Column name '{col_text}' starts with a digit which may cause issues in downstream processing."
            )

    # Duplicate row check (hizli olmasi icin ilk 50 bin satira bak ya da hepsi)
    check_df = df if len(df) <= 50000 else df.sample(50000, random_state=42)
    dup_pct = check_df.duplicated().mean()
    if dup_pct > 0.1:
        warnings.append(
            f"{dup_pct*100:.1f}% duplicate rows detected; duplicates may be removed or aggregated."
        )

    # Detect datetime-like columns (quietly)
    for col in cols:
        if df[col].dtype == object:
            # Sadece ilk 100 dolu satiri test edip hiz kazan (O(N) yerine O(1))
            sample_vals = df[col].dropna().head(100)
            if len(sample_vals) > 0:
                try:
                    # format="%Y-%m-%d" tahmini hizli yapmak icin vs. string uzunluguna falan da bakilabilir
                    # ama dogrudan coerce da 100 satir icin anindakidir.
                    parsed = pd.to_datetime(sample_vals, errors='coerce')
                    if parsed.notna().mean() > 0.8:
                        warnings.append(f"Column '{col}' appears to contain datetime values; ensure date features are processed correctly.")
                except Exception:
                    pass

    # Special characters in column names
    for col in cols:
        col_text = str(col)
        if re.search(r"\W", col_text):
            warnings.append(
                f"Column name '{col_text}' contains special characters; consider renaming for consistency."
            )

    # Feature count warning
    if len(cols) > 50:
        warnings.append(
            f"High number of columns ({len(cols)}); performance may be impacted. Consider feature selection."
        )

    return critical, warnings
