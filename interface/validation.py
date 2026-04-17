import pandas as pd

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

    # Missingness and constants
    for col in cols:
        pct_missing = df[col].isna().mean()
        if pct_missing > 0.8:
            warnings.append(
                f"Column '{col}' has {int(pct_missing*100)}% missing values and may lead to unreliable results."
            )
        # constant values ignoring NaNs
        if df[col].dropna().nunique() <= 1:
            warnings.append(
                f"Column '{col}' contains constant values (no variability) and will be excluded from analysis."
            )

    # Check for at least one numeric column
    try:
        import pandas.api.types as ptypes
        numeric_cols = [col for col in cols if ptypes.is_numeric_dtype(df[col])]
        if not numeric_cols:
            critical.append("No numeric columns detected. At least one numeric column is required for modeling.")
    except ImportError:
        pass

    # High-cardinality categorical columns
    for col in cols:
        if not df[col].dropna().empty and not ptypes.is_numeric_dtype(df[col]):
            unique_cnt = df[col].nunique(dropna=True)
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

    # Duplicate row check
    dup_pct = df.duplicated().mean()
    if dup_pct > 0.1:
        warnings.append(
            f"{dup_pct*100:.1f}% duplicate rows detected; duplicates may be removed or aggregated."
        )

    # Detect datetime-like columns (quietly)
    for col in cols:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                parsed = pd.Series([pd.NaT] * len(df))
            if parsed.notna().mean() > 0.8:
                warnings.append(
                    f"Column '{col}' appears to contain datetime values; ensure date features are processed correctly."
                )

    # Special characters in column names
    import re
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
