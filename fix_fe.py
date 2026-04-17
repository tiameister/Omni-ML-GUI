with open("features/feature_engineering.py", "r", encoding="utf-8") as f:
    text = f.read()

import re

# We must limit PolynomialFeatures so it doesn't cause out-of-memory errors on large datasets with >50 features
old_fe = """    if continuous_numeric_cols:
        df_num = df[continuous_numeric_cols].fillna(0)
        pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_arr = pf.fit_transform(df_num)
        poly_names = list(pf.get_feature_names_out(continuous_numeric_cols))
        try:
            df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
        except ValueError:
            df_poly = pd.DataFrame(poly_arr, index=df.index)
            df_poly.columns = [f"poly_{i}" for i in range(df_poly.shape[1])]
            poly_names = list(df_poly.columns)
    else:
        df_poly = pd.DataFrame(index=df.index)"""

new_fe = """    if continuous_numeric_cols:
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
        df_poly = pd.DataFrame(index=df.index)"""

text = text.replace(old_fe, new_fe)

with open("features/feature_engineering.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Optimized Feature Engineering Memory Bounds!")
