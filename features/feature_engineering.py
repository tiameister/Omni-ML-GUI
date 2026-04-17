import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def _is_integer_like(series: pd.Series, tol: float = 1e-9) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return False
    return bool(np.all(np.isclose(vals, np.round(vals), atol=tol)))


def _split_numeric_feature_types(
    df: pd.DataFrame, numeric_cols: list[str], discrete_threshold: int
) -> tuple[list[str], list[str]]:
    """Split numeric columns into continuous vs likely discrete coded features.

    Discrete coded features (e.g., gender: 1/2, Likert 1..5) are kept as raw
    columns to preserve interpretability and avoid synthetic polynomial terms.
    """
    continuous_cols: list[str] = []
    discrete_cols: list[str] = []

    for col in numeric_cols:
        ser = pd.to_numeric(df[col], errors="coerce")
        non_null = ser.dropna()
        n_non_null = int(non_null.shape[0])
        nunique = int(non_null.nunique())
        unique_ratio = float(nunique / max(n_non_null, 1))

        likely_discrete = False
        if _is_integer_like(ser):
            likely_discrete = (nunique <= 2) or (
                nunique <= max(2, int(discrete_threshold)) and unique_ratio <= 0.20
            )

        if likely_discrete:
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

    return continuous_cols, discrete_cols


def apply_feature_engineering(
    df: pd.DataFrame,
    output_folder: str = "feature_engineered_dataset",
    degree: int = 2,
    interaction_only: bool = False,
    save_csv: bool = True,
    discrete_threshold: int = 12,
    force_passthrough_cols: list[str] | None = None,
):
    """Applies basic feature engineering.

    - PolynomialFeatures on numeric columns (continuous only)
    - Missing indicator flags for continuous columns

    Saves the engineered DataFrame to output_folder/feature_engineered.csv when save_csv=True.

    Returns (df_engineered, new_feature_cols, categorical_cols)
    """
    if save_csv:
        os.makedirs(output_folder, exist_ok=True)
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    continuous_numeric_cols, discrete_numeric_cols = _split_numeric_feature_types(
        df,
        numeric_cols,
        discrete_threshold=max(2, int(discrete_threshold)),
    )

    passthrough_force = {str(c).strip() for c in (force_passthrough_cols or []) if str(c).strip()}
    if passthrough_force:
        keep_cont: list[str] = []
        for col in continuous_numeric_cols:
            if col in passthrough_force:
                if col not in discrete_numeric_cols:
                    discrete_numeric_cols.append(col)
            else:
                keep_cont.append(col)
        continuous_numeric_cols = keep_cont

    missing_indicator_cols: list[str] = []
    for col in continuous_numeric_cols:
        miss_col = f"{col}_missing"
        df[miss_col] = df[col].isna().astype(int)
        missing_indicator_cols.append(miss_col)

    poly_names: list[str] = []
    if continuous_numeric_cols:
        # Cap features fed into polynomial to prevent MemoryError (O(N^d))
        max_poly_feats = 100
        if len(continuous_numeric_cols) > max_poly_feats:
            import logging
            logging.getLogger(__name__).warning("Too many continuous features. Capping polynomial transformer to top %d by variance.", max_poly_feats)
            # Take highest variance features
            vars_s = df[continuous_numeric_cols].var().sort_values(ascending=False)
            top_cols = vars_s.head(max_poly_feats).index.tolist()
            # Keep the rest as passthrough
            passthrough_cont = [c for c in continuous_numeric_cols if c not in top_cols]
            continuous_numeric_cols = top_cols
            discrete_numeric_cols.extend(passthrough_cont) # Add back as passthrough

        df_num = df[continuous_numeric_cols].fillna(0)
        pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_arr = pf.fit_transform(df_num)
        poly_names = list(pf.get_feature_names_out(continuous_numeric_cols))
        
        # Cast to float32 to save RAM immediately after creation
        poly_arr = poly_arr.astype(np.float32)
        try:
            df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
        except ValueError:
            df_poly = pd.DataFrame(poly_arr, index=df.index)
            df_poly.columns = [f"poly_{i}" for i in range(df_poly.shape[1])]
            poly_names = list(df_poly.columns)
    else:
        df_poly = pd.DataFrame(index=df.index)

    passthrough_cols = [c for c in df.columns if c not in set(continuous_numeric_cols)]
    df_engineered = pd.concat([df_poly, df[passthrough_cols]], axis=1)

    if save_csv:
        fe_path = os.path.join(output_folder, "feature_engineered.csv")
        df_engineered.to_csv(fe_path, index=False, encoding="utf-8-sig")

    new_num_cols = list(poly_names) + list(missing_indicator_cols) + list(discrete_numeric_cols)
    cat_cols = [c for c in df_engineered.columns if c not in set(new_num_cols)]
    return df_engineered, new_num_cols, cat_cols
