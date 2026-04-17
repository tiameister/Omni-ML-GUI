import re

with open("data/loader.py", "r", encoding="utf-8") as f:
    text = f.read()

# The current _coerce_numeric_like_object_columns uses a lot of Python string ops across the entire column.
# We can sample the non-nulls or use a vectorized replace much faster.
old_coerce = """def _coerce_numeric_like_object_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    \"\"\"Convert object columns to numeric only when values are predominantly numeric-like.

    This avoids corrupting true categorical text columns while still handling
    comma decimals (e.g., "1,25").
    \"\"\"
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].astype("string")
        non_empty = s.notna() & (s.str.strip() != "")
        if int(non_empty.sum()) == 0:
            continue

        candidate = s.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(candidate, errors="coerce")
        ratio = float(numeric[non_empty].notna().mean())
        if ratio >= threshold:
            df[col] = numeric
    return df"""

new_coerce = """def _coerce_numeric_like_object_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    \"\"\"Convert object columns to numeric only when values are predominantly numeric-like.
    Optimized: Sample-based strict detection (O(1) sampling) to avoid 
    O(N) full-string iterations over purely categorical columns.
    \"\"\"
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return df

    for col in object_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        
        # Optimize: Test a small sample first (100 rows)
        sample = col_data.head(100).astype(str).str.strip()
        sample_candidate = sample.str.replace(",", ".", regex=False)
        sample_num = pd.to_numeric(sample_candidate, errors="coerce")
        
        # If the sample has less than 50% numeric, it's overwhelmingly a text column, skip full conversion
        if sample_num.notna().mean() < 0.5:
            continue
            
        # Full conversion explicitly over C backend if pass
        s_full = col_data.astype(str).str.strip()
        # Drop empties from denominator
        s_full = s_full[s_full != ""]
        if s_full.empty:
            continue
            
        candidate = s_full.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(candidate, errors="coerce")
        
        ratio = numeric.notna().mean()
        if ratio >= threshold:
            # Map back to original dataframe size
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            
    return df"""

text = text.replace(old_coerce, new_coerce)

with open("data/loader.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Optimized loader loops!")
