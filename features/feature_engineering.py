import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
import logging

LOGGER = logging.getLogger(__name__)

def _is_integer_like(series: pd.Series, tol: float = 1e-9) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return False
    return bool(np.all(np.isclose(vals, np.round(vals), atol=tol)))

def _split_numeric_feature_types(
    df: pd.DataFrame, numeric_cols: list[str], discrete_threshold: int
) -> tuple[list[str], list[str]]:
    continuous_cols, discrete_cols = [], []
    for col in numeric_cols:
        ser = pd.to_numeric(df[col], errors="coerce")
        non_null = ser.dropna()
        n_non_null = int(non_null.shape[0])
        nunique = int(non_null.nunique())
        unique_ratio = float(nunique / max(n_non_null, 1))

        if _is_integer_like(ser) and ((nunique <= 2) or (nunique <= max(2, int(discrete_threshold)) and unique_ratio <= 0.20)):
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    return continuous_cols, discrete_cols

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that applies polynomial feature expansion.
    Safely calculates variances and missing indicators inside the cross-validation
    loop to completely prevent Data Leakage.
    """
    def __init__(self, degree=2, interaction_only=False, max_poly_feats=100, discrete_threshold=12, fe_enabled=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.max_poly_feats = max_poly_feats
        self.discrete_threshold = discrete_threshold
        self.fe_enabled = fe_enabled

    def fit(self, X, y=None):
        if not self.fe_enabled:
            return self

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        numeric_cols = df.columns.tolist()

        cont_cols, disc_cols = _split_numeric_feature_types(df, numeric_cols, self.discrete_threshold)
        
        self.discrete_cols_ = disc_cols
        self.poly_cols_ = []
        
        if cont_cols:
            if len(cont_cols) > self.max_poly_feats:
                LOGGER.warning("Capping polynomial transformer to top %d continuous features.", self.max_poly_feats)
                # Compute variance inside CV fold safely
                vars_s = df[cont_cols].var().fillna(0).sort_values(ascending=False)
                self.poly_cols_ = vars_s.head(self.max_poly_feats).index.tolist()
                self.discrete_cols_.extend([c for c in cont_cols if c not in self.poly_cols_])
            else:
                self.poly_cols_ = cont_cols
                
            self.pf_ = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=False)
            self.pf_.fit(df[self.poly_cols_].fillna(0))
            
        self.missing_cols_ = []
        for col in self.poly_cols_ + self.discrete_cols_:
            if col in df.columns and df[col].isna().any():
                self.missing_cols_.append(col)
        
        return self

    def transform(self, X):
        if not self.fe_enabled:
            return X
            
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        pieces = []
        # Missing Indicators for continuous cols (fitted only)
        miss_df = pd.DataFrame(index=df.index)
        for col in getattr(self, "missing_cols_", []):
            if col in df.columns:
                miss_df[f"{col}_missing"] = df[col].isna().astype(np.float32)
            else:
                miss_df[f"{col}_missing"] = 0.0
        if not miss_df.empty:
            pieces.append(miss_df)

        # Polynomial Expansion
        if getattr(self, "poly_cols_", []):
            df_poly_input = df[self.poly_cols_] if all(c in df.columns for c in self.poly_cols_) else df.reindex(columns=self.poly_cols_).fillna(0)
            poly_arr = self.pf_.transform(df_poly_input.fillna(0)).astype(np.float32)
            poly_names = list(self.pf_.get_feature_names_out(self.poly_cols_))
            df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
            pieces.append(df_poly)

        # Discrete / Passthrough
        if getattr(self, "discrete_cols_", []):
            disc_input = df[self.discrete_cols_] if all(c in df.columns for c in self.discrete_cols_) else df.reindex(columns=self.discrete_cols_)
            pieces.append(disc_input.astype(np.float32))
        
        # Ensure we capture any remaining columns (e.g. ones that arrived unexpectedly)
        used_cols = set(getattr(self, "poly_cols_", []) + getattr(self, "discrete_cols_", []))
        remaining = [c for c in df.columns if c not in used_cols]
        if remaining:
            pieces.append(df[remaining].astype(np.float32))
            
        if not pieces:
            return pd.DataFrame(index=df.index)
            
        res = pd.concat(pieces, axis=1)
        return res
        
    def get_feature_names_out(self, input_features=None):
        if not self.fe_enabled:
            return np.array(input_features) if input_features is not None else np.array([])
            
        names = []
        # Missing flags
        if hasattr(self, "missing_cols_"):
            for col in self.missing_cols_:
                names.append(f"{col}_missing")
                
        # Polynomial features
        if hasattr(self, "poly_cols_") and self.poly_cols_:
            names.extend(self.pf_.get_feature_names_out(self.poly_cols_))
            
        # Discrete / Passthrough
        if hasattr(self, "discrete_cols_"):
            names.extend(self.discrete_cols_)
            
        # If input_features provided, check for any we missed (e.g. from upstream)
        if input_features is not None:
            used = set(getattr(self, "poly_cols_", []) + getattr(self, "discrete_cols_", []))
            for col in input_features:
                if col not in used:
                    names.append(col)
                    
        return np.array(names)

def apply_feature_engineering(*args, **kwargs):
    """Legacy bypass, logic moved to Pipeline native FeatureEngineeringTransformer"""
    return args[0], list(args[0].columns), []
