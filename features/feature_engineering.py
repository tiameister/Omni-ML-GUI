"""
Feature engineering utilities and transformers for MLTrainer.
Includes custom scikit-learn compatible transformers and helpers.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, KBinsDiscretizer
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
    Scikit-Learn compatible transformer that applies structural feature engineering.
    Now supports Outlier handling (winsorization), Distribution Transform (Yeo-Johnson),
    Missing Indicators, and Polynomial Expansion safely inside the CV fold.
    """
    def __init__(self, fe_enabled=True, config=None):
        self.fe_enabled = fe_enabled
        # NOTE: Keep `config` exactly as passed so sklearn's `clone()` works.
        self.config = config
        cfg = config if config is not None else {}

        # default settings fallbacks
        self.missing_indicators = cfg.get("missing_indicators", True)
        self.outliers = cfg.get("outliers", "winsorize_1_99")
        # Do NOT name this `transform` (it would shadow the .transform() method).
        self.transform_method = cfg.get("transform", "yeo-johnson")
        self.poly_features = cfg.get("poly_features", False)
        self.poly_degree = cfg.get("poly_degree", 2)
        self.poly_max = cfg.get("poly_max", 50)
        self.binning_strategy = cfg.get("binning", "none")  # none, uniform, quantile, kmeans
        self.n_bins = cfg.get("n_bins", 5)
        self.discrete_threshold = 12

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Provide feature name metadata for sklearn's pandas output wrappers.
        self.feature_names_in_ = np.array(df.columns, dtype=object)
        self.n_features_in_ = int(df.shape[1])

        if not self.fe_enabled:
            return self
        numeric_cols = df.columns.tolist()

        cont_cols, disc_cols = _split_numeric_feature_types(df, numeric_cols, self.discrete_threshold)
        
        self.discrete_cols_ = disc_cols
        self.poly_cols_ = []
        self.cont_cols_ = cont_cols
        
        # 1. Outlier Thresholds Calculation
        self.outlier_bounds_ = {}
        if self.outliers in ["winsorize_1_99", "winsorize_5_95"] and cont_cols:
            lower_q = 0.01 if "1_99" in self.outliers else 0.05
            upper_q = 0.99 if "1_99" in self.outliers else 0.95
            for col in cont_cols:
                low = df[col].quantile(lower_q)
                high = df[col].quantile(upper_q)
                self.outlier_bounds_[col] = (low, high)

        # 2. Power Transformation
        self.power_transformer_ = None
        if self.transform_method == "yeo-johnson" and cont_cols:
            df_for_fit = df[cont_cols].copy()
            # Apply outlier bounds to fitting set to prevent skewed fits
            for col, (l, h) in self.outlier_bounds_.items():
                df_for_fit[col] = df_for_fit[col].clip(lower=l, upper=h)
            
            self.power_transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
            # Replace NaNs safely before fitting
            self.power_transformer_.fit(df_for_fit.fillna(df_for_fit.median()))
            
        # 3. Polynomial Expansion
        if self.poly_features and cont_cols:
            if len(cont_cols) > self.poly_max:
                LOGGER.warning("Capping polynomial transformer to top %d continuous features.", self.poly_max)
                vars_s = df[cont_cols].var().fillna(0).sort_values(ascending=False)
                self.poly_cols_ = vars_s.head(self.poly_max).index.tolist()
            else:
                self.poly_cols_ = cont_cols
                
            self.pf_ = PolynomialFeatures(degree=self.poly_degree, interaction_only=False, include_bias=False)
            
            df_poly = df[self.poly_cols_].copy()
            # Simulate the prior steps for the poly fit to behave correctly
            for col, (l, h) in self.outlier_bounds_.items():
                if col in self.poly_cols_: df_poly[col] = df_poly[col].clip(lower=l, upper=h)
            if self.power_transformer_:
                df_poly[self.poly_cols_] = self.power_transformer_.transform(df_poly.fillna(df_poly.median()))[
                    :, [cont_cols.index(c) for c in self.poly_cols_]
                ]

            self.pf_.fit(df_poly.fillna(0))
            
        # 4. Binning/Discretization (Modern alternative to continuous modeling)
        self.binner_ = None
        self.bin_cols_ = []
        if self.binning_strategy in ["uniform", "quantile", "kmeans"] and cont_cols:
            self.bin_cols_ = cont_cols
            strategy_map = {"uniform": "uniform", "quantile": "quantile", "kmeans": "kmeans"}
            self.binner_ = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=strategy_map[self.binning_strategy])
            
            df_bin = df[self.bin_cols_].copy()
            # Apply outlier bounds to fitting set to prevent skewed bins
            for col, (l, h) in self.outlier_bounds_.items():
                df_bin[col] = df_bin[col].clip(lower=l, upper=h)
            
            # Binner cannot handle NaNs during fit, so we impute with median for fit
            self.binner_.fit(df_bin.fillna(df_bin.median()))

        # 5. Missing Indicators
        self.missing_cols_ = []
        if self.missing_indicators:
            for col in cont_cols + disc_cols:
                if col in df.columns and df[col].isna().any():
                    self.missing_cols_.append(col)
        
        return self

    def transform(self, X):
        if not self.fe_enabled:
            return X
            
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df_transformed = df.copy()
        pieces = []

        # 1. Missing Indicators
        miss_df = pd.DataFrame(index=df.index)
        for col in getattr(self, "missing_cols_", []):
            if col in df.columns:
                miss_df[f"{col}_missing"] = df[col].isna().astype(np.float32)
            else:
                miss_df[f"{col}_missing"] = 0.0
        if not miss_df.empty:
            pieces.append(miss_df)

        # Apply transformations to a deep copy of continuous columns
        cont_cols = getattr(self, "cont_cols_", [])
        df_cont = df[cont_cols].copy() if cont_cols else pd.DataFrame(index=df.index)

        if not df_cont.empty:
            # 2. Outliers
            for col, (l, h) in getattr(self, "outlier_bounds_", {}).items():
                if col in df_cont.columns:
                    df_cont[col] = df_cont[col].clip(lower=l, upper=h)
            
            # 3. Power Transform
            if getattr(self, "power_transformer_", None) is not None:
                # We fill na with 0 just for the transform to not crash, though proper interpolation happens later.
                # Actually, the base numeric imputer runs BEFORE this if we arrange pipeline, or AFTER.
                # In current pipeline, FE comes first. Let's fill gracefully for transform, will be preserved or pipeline will fix.
                arr = self.power_transformer_.transform(df_cont.fillna(df_cont.median()))
                # Put back into df_cont but keeping NAs for the real imputer
                mask = df_cont.isna()
                df_cont = pd.DataFrame(arr, columns=df_cont.columns, index=df.index)
                df_cont[mask] = np.nan
            
            pieces.append(df_cont)

        # 4. Polynomial Features
        poly_cols = getattr(self, "poly_cols_", [])
        if poly_cols and hasattr(self, "pf_"):
            df_poly_input = df_cont[poly_cols] if all(c in df_cont.columns for c in poly_cols) else df.reindex(columns=poly_cols)
            poly_arr = self.pf_.transform(df_poly_input.fillna(0)).astype(np.float32)
            poly_names = list(self.pf_.get_feature_names_out(poly_cols))
            df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
            # Remove base features if they were just added to pieces via df_cont (they are duplicated by PolyFeatures)
            # PolynomialFeatures includes original features by default.
            df_poly = df_poly.drop(columns=poly_cols, errors="ignore")
            pieces.append(df_poly)

        # 5. Binning (Discretization)
        if getattr(self, "binner_", None) is not None and getattr(self, "bin_cols_", []):
            bin_cols = self.bin_cols_
            df_bin = df_cont[bin_cols].copy() if all(c in df_cont.columns for c in bin_cols) else df.reindex(columns=bin_cols)
            arr = self.binner_.transform(df_bin.fillna(df_bin.median()))
            # Convert back to DataFrame, put NAs back
            mask = df_bin.isna()
            df_binned = pd.DataFrame(arr, columns=[f"{c}_binned_{self.binning_strategy}" for c in bin_cols], index=df.index)
            # Re-apply NA mask since we just imputed it to run the binner
            df_binned[mask.values] = np.nan
            pieces.append(df_binned)

        # 6. Discrete cols
        disc_cols = getattr(self, "discrete_cols_", [])
        if disc_cols:
            disc_input = df[disc_cols].copy()
            pieces.append(disc_input)
        
        # 7. Fallback/Remaining cols
        used_cols = set(cont_cols + disc_cols)
        remaining = [c for c in df.columns if c not in used_cols]
        if remaining:
            pieces.append(df[remaining].astype(np.float32))
            
        if not pieces:
            return pd.DataFrame(index=df.index)
            
        res = pd.concat(pieces, axis=1)
        # remove duplicate columns securely
        res = res.loc[:, ~res.columns.duplicated()]
        return res
        
    def get_feature_names_out(self, input_features=None):
        if not self.fe_enabled:
            if input_features is not None:
                return np.array(input_features)
            # When sklearn's pandas output wrapper calls with no args, fall back to fitted feature names.
            return np.array(getattr(self, "feature_names_in_", []))
            
        names = []
        if hasattr(self, "missing_cols_"):
            for col in self.missing_cols_:
                names.append(f"{col}_missing")
                
        if hasattr(self, "cont_cols_"):
            names.extend(self.cont_cols_)
                
        if hasattr(self, "poly_cols_") and self.poly_cols_ and hasattr(self, "pf_"):
            poly_names = list(self.pf_.get_feature_names_out(self.poly_cols_))
            for n in poly_names:
                if n not in names: # prevents duplicates of the base cont_cols
                    names.append(n)
            
        if hasattr(self, "binner_") and self.binner_ is not None and getattr(self, "bin_cols_", []):
            for col in self.bin_cols_:
                names.append(f"{col}_binned_{self.binning_strategy}")

        if hasattr(self, "discrete_cols_"):
            names.extend(self.discrete_cols_)
            
        if input_features is not None:
            used = set(getattr(self, "cont_cols_", []) + getattr(self, "discrete_cols_", []))
            for col in input_features:
                if col not in used and col not in names:
                    names.append(col)
                    
        return np.array(names)

def generate_static_fe_dataset(df: pd.DataFrame, config: dict, target_col: str, save_dir: str, filename: str) -> str:
    """
    WARNING: Generates a statically transformed dataset and returns its path.
    Applies FeatureEngineeringTransformer to the features only, merging back the target.
    This causes Data Leakage if the config contains distribution-based techniques 
    (Yeo-Johnson, Outlier percentiles, Target encoding).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    LOGGER.warning("Generating static FE dataset. Beware of Data Leakage (Global transformations applied.)")
    
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[[target_col]].copy()
    else:
        X = df
        y = pd.DataFrame(index=df.index)

    transformer = FeatureEngineeringTransformer(fe_enabled=True, config=config)
    transformer.fit(X)
    X_trans = transformer.transform(X)
    
    transformed_df = pd.concat([X_trans, y], axis=1)
    
    out_path = os.path.join(save_dir, filename)
    transformed_df.to_csv(out_path, index=False)
    return out_path
