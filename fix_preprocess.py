with open("features/preprocess.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace Custom string conversion with the much faster Native Pandas casting inside the FunctionTransformer execution step,
# removing Python loops from _to_uniform_string
old_str = """def _to_uniform_string(X):
    try:
        return X.astype(str)
    except Exception:
        import pandas as pd
        return pd.DataFrame(X).astype(str)"""

new_str = """def _to_uniform_string(X):
    import pandas as pd
    # Ensure immediate conversion array -> pd.DataFrame -> str natively on C backend
    # This prevents object-by-object fallback on failure
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.astype(str)"""

text = text.replace(old_str, new_str)

# Also for categorical Imputer, most_frequent is wildly slow on text columns because Scikit-learn has to count distinct strings.
# Imputing with a constant 'Missing' string is instantaneous O(1).
old_cat_pipe = """        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(_to_uniform_string, validate=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])"""

new_cat_pipe = """        categorical_pipeline = Pipeline([
            # 'most_frequent' does a costly full-dataset string frequency count. 'Missing' acts as O(1) constant imputation natively.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("to_string", FunctionTransformer(_to_uniform_string, validate=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])"""
text = text.replace(old_cat_pipe, new_cat_pipe)

with open("features/preprocess.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Optimized Categorical Pipeline speeds")
