from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from features.feature_engineering import FeatureEngineeringTransformer


def _to_uniform_string(X):
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.astype(str)

def build_preprocessor(num_cols, cat_cols, ordinal_cols=None, binary_cols=None, fe_enabled=False):
    """
    Returns a ColumnTransformer with:
    - numeric: impute + StandardScaler
    - ordinal: impute + OrdinalEncoder (no scaling)
    - binary: passthrough (no scaling)
    - cat: impute + OneHotEncoder for remaining categorical columns

    Backward-compatible: if ordinal_cols and binary_cols are None, behaves like the previous
    version with numeric + onehot for cat_cols.
    """
    ordinal_cols = list(ordinal_cols) if ordinal_cols is not None else []
    binary_cols = list(binary_cols) if binary_cols is not None else []
    # Remaining categorical columns after excluding ordinal and binary
    cat_other = [c for c in cat_cols if c not in set(ordinal_cols) and c not in set(binary_cols)]

    numeric_pipeline = Pipeline([
        ("fe", FeatureEngineeringTransformer(fe_enabled=fe_enabled)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if ordinal_cols or binary_cols:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # use_unknown value -1 for unseen categories
            ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        categorical_pipeline = Pipeline([
            # 'most_frequent' does a costly full-dataset string frequency count. 'Missing' acts as O(1) constant imputation natively.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("to_string", FunctionTransformer(_to_uniform_string, validate=False)),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.015, max_categories=20, sparse_output=False))
        ])
        from sklearn.pipeline import Pipeline as _Pipeline
        from sklearn.impute import SimpleImputer as _SimpleImputer
        # Impute binary columns before passing through
        binary_pipeline = _Pipeline([
            ("imputer", _SimpleImputer(strategy="most_frequent"))
        ])
        transformers = [
            ("num", numeric_pipeline, num_cols),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("bin", binary_pipeline, binary_cols),
            ("cat", categorical_pipeline, cat_other),
        ]
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        preprocessor.set_output(transform="pandas")
    else:
        categorical_pipeline = Pipeline([
            # 'most_frequent' does a costly full-dataset string frequency count. 'Missing' acts as O(1) constant imputation natively.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("to_string", FunctionTransformer(_to_uniform_string, validate=False)),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.015, max_categories=20, sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ])
        preprocessor.set_output(transform="pandas")

    return preprocessor
