"""
Feature engineering utilities and transformers for Omni-ML-GUI.

Core transformer: FeatureEngineeringTransformer
  - Fully sklearn-compatible (BaseEstimator + TransformerMixin)
  - Fit computes ALL statistics on the training fold only — no data leakage
  - Supports: Power Transform (Yeo-Johnson / Log1p), Outlier Winsorization,
    Missing Indicators, Polynomial / Interaction Expansion, Binning
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
import logging

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_integer_like(series: pd.Series, tol: float = 1e-9) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return False
    return bool(np.all(np.isclose(vals, np.round(vals), atol=tol)))


def _split_numeric_feature_types(
    df: pd.DataFrame, numeric_cols: list[str], discrete_threshold: int
) -> tuple[list[str], list[str]]:
    """
    Partition numeric columns into continuous and discrete (ordinal/integer-coded)
    sets.  Discrete columns skip distribution-shaping transforms (power, binning)
    since they carry inherent ordinal meaning.
    """
    continuous_cols, discrete_cols = [], []
    for col in numeric_cols:
        ser = pd.to_numeric(df[col], errors="coerce")
        non_null = ser.dropna()
        n_non_null = int(non_null.shape[0])
        nunique = int(non_null.nunique())
        unique_ratio = float(nunique / max(n_non_null, 1))
        if _is_integer_like(ser) and (
            (nunique <= 2)
            or (nunique <= max(2, int(discrete_threshold)) and unique_ratio <= 0.20)
        ):
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    return continuous_cols, discrete_cols


# ---------------------------------------------------------------------------
# Main transformer
# ---------------------------------------------------------------------------

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer applying structural feature engineering
    inside the cross-validation fold to prevent data leakage.

    All statistics (quantile bounds, power-transform lambdas, bin edges,
    training medians) are fitted on the **training fold only** and then applied
    identically to the validation / test fold.

    Parameters
    ----------
    fe_enabled : bool
        Master switch.  When False the transformer is a pure passthrough.
    config : dict
        Keys understood:
          transform          : "none" | "yeo-johnson" | "log1p"
          outliers           : "none" | "winsorize_1_99" | "winsorize_5_95"
          missing_indicators : bool
          poly_features      : bool
          poly_degree        : int  (2–4)
          poly_max           : int  (max continuous cols fed into poly)
          interaction_only   : bool (True → no power terms, only A*B cross-terms)
          binning            : "none" | "uniform" | "quantile" | "kmeans"
          n_bins             : int  (2–20)
          discrete_threshold : int  (max unique values before treating col as discrete)
    """

    def __init__(self, fe_enabled: bool = True, config: dict | None = None):
        self.fe_enabled = fe_enabled
        # Keep `config` exactly as passed so sklearn's clone() works.
        self.config = config
        cfg = config if config is not None else {}

        self.missing_indicators  = cfg.get("missing_indicators", True)
        self.outliers            = cfg.get("outliers", "winsorize_1_99")
        # Do NOT name this `transform` — it would shadow the .transform() method.
        self.transform_method    = cfg.get("transform", "yeo-johnson")
        self.poly_features       = cfg.get("poly_features", False)
        self.poly_degree         = cfg.get("poly_degree", 2)
        self.poly_max            = cfg.get("poly_max", 50)
        self.interaction_only    = cfg.get("interaction_only", False)
        self.binning_strategy    = cfg.get("binning", "none")
        self.n_bins              = cfg.get("n_bins", 5)
        self.discrete_threshold  = int(cfg.get("discrete_threshold", 12))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.array(df.columns, dtype=object)
        self.n_features_in_ = int(df.shape[1])

        if not self.fe_enabled:
            return self

        numeric_cols = df.columns.tolist()
        cont_cols, disc_cols = _split_numeric_feature_types(
            df, numeric_cols, self.discrete_threshold
        )
        self.cont_cols_   = cont_cols
        self.discrete_cols_ = disc_cols
        self.poly_cols_   = []

        # -------------------------------------------------------------------
        # 1. Store training-fold medians for safe NaN imputation in transform()
        #    Using training medians prevents any leakage of test-set statistics.
        # -------------------------------------------------------------------
        if cont_cols:
            self._train_medians_ = df[cont_cols].median().to_dict()
        else:
            self._train_medians_ = {}

        # -------------------------------------------------------------------
        # 2. Outlier bounds (percentile thresholds from training fold)
        # -------------------------------------------------------------------
        self.outlier_bounds_: dict[str, tuple[float, float]] = {}
        if self.outliers in ("winsorize_1_99", "winsorize_5_95") and cont_cols:
            lower_q = 0.01 if "1_99" in self.outliers else 0.05
            upper_q = 0.99 if "1_99" in self.outliers else 0.95
            for col in cont_cols:
                low  = float(df[col].quantile(lower_q))
                high = float(df[col].quantile(upper_q))
                self.outlier_bounds_[col] = (low, high)

        # Helper: training-fold data after winsorization (used for subsequent fits)
        def _winsorized(df_in: pd.DataFrame) -> pd.DataFrame:
            df_w = df_in.copy()
            for col, (l, h) in self.outlier_bounds_.items():
                if col in df_w.columns:
                    df_w[col] = df_w[col].clip(lower=l, upper=h)
            return df_w

        # -------------------------------------------------------------------
        # 3. Power transformation (Yeo-Johnson or Log1p)
        # -------------------------------------------------------------------
        self.power_transformer_ = None
        self._log1p_shift_: dict[str, float] = {}

        if cont_cols:
            df_for_fit = _winsorized(df[cont_cols])
            df_for_fit = df_for_fit.fillna(
                df_for_fit.median()  # safe: operating on training data only
            )

            if self.transform_method == "yeo-johnson":
                self.power_transformer_ = PowerTransformer(
                    method="yeo-johnson", standardize=False
                )
                self.power_transformer_.fit(df_for_fit)

            elif self.transform_method == "log1p":
                # Learn per-column shift so that x + shift >= 0 on training data.
                # Applied consistently to any incoming data at transform time.
                for col in cont_cols:
                    col_min = float(df_for_fit[col].min())
                    self._log1p_shift_[col] = (
                        max(0.0, -col_min + 1e-8) if col_min < 0 else 0.0
                    )

        # -------------------------------------------------------------------
        # 4. Polynomial / interaction expansion
        # -------------------------------------------------------------------
        if self.poly_features and cont_cols:
            if len(cont_cols) > self.poly_max:
                LOGGER.warning(
                    "Capping polynomial expansion to top %d continuous features "
                    "(by variance) out of %d.",
                    self.poly_max,
                    len(cont_cols),
                )
                vars_s = df[cont_cols].var().fillna(0).sort_values(ascending=False)
                self.poly_cols_ = vars_s.head(self.poly_max).index.tolist()
            else:
                self.poly_cols_ = cont_cols

            self.pf_ = PolynomialFeatures(
                degree=self.poly_degree,
                interaction_only=bool(self.interaction_only),
                include_bias=False,
            )
            # Fit on winsorized + power-transformed training data for best numerical
            # stability of the resulting polynomial features.
            df_poly = _winsorized(df[self.poly_cols_])
            df_poly = df_poly.fillna(df_poly.median())
            if self.power_transformer_ is not None:
                idx = [cont_cols.index(c) for c in self.poly_cols_]
                arr = self.power_transformer_.transform(df_poly.values)
                df_poly = pd.DataFrame(arr, columns=df_poly.columns, index=df_poly.index)
            elif self._log1p_shift_:
                for col in self.poly_cols_:
                    shift = self._log1p_shift_.get(col, 0.0)
                    df_poly[col] = np.log1p(df_poly[col] + shift)
            self.pf_.fit(df_poly.fillna(0))

        # -------------------------------------------------------------------
        # 5. Binning / discretization
        # -------------------------------------------------------------------
        self.binner_: KBinsDiscretizer | None = None
        self.bin_cols_: list[str] = []
        if self.binning_strategy in ("uniform", "quantile", "kmeans") and cont_cols:
            self.bin_cols_ = cont_cols
            self.binner_ = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.binning_strategy,
            )
            df_bin = _winsorized(df[self.bin_cols_])
            self.binner_.fit(df_bin.fillna(df_bin.median()))

        # -------------------------------------------------------------------
        # 6. Missing indicator columns (which cols had missingness in training)
        # -------------------------------------------------------------------
        self.missing_cols_: list[str] = []
        if self.missing_indicators:
            for col in cont_cols + disc_cols:
                if col in df.columns and df[col].isna().any():
                    self.missing_cols_.append(col)

        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X):
        if not self.fe_enabled:
            return X

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # Training medians used for all NaN imputation during transform —
        # guarantees that test-fold statistics never influence the output.
        train_medians = getattr(self, "_train_medians_", {})

        def _fill_with_train_median(df_in: pd.DataFrame) -> pd.DataFrame:
            """Fill NaNs using training-fold medians."""
            df_out = df_in.copy()
            for col in df_out.columns:
                if df_out[col].isna().any():
                    df_out[col] = df_out[col].fillna(train_medians.get(col, 0.0))
            return df_out

        pieces: list[pd.DataFrame] = []

        # -------------------------------------------------------------------
        # 1. Missing Indicators (add before any transformation)
        # -------------------------------------------------------------------
        miss_df = pd.DataFrame(index=df.index)
        for col in getattr(self, "missing_cols_", []):
            if col in df.columns:
                miss_df[f"{col}_missing"] = df[col].isna().astype(np.float32)
            else:
                miss_df[f"{col}_missing"] = 0.0
        if not miss_df.empty:
            pieces.append(miss_df)

        # -------------------------------------------------------------------
        # 2–3. Continuous column processing
        # -------------------------------------------------------------------
        cont_cols = getattr(self, "cont_cols_", [])
        df_cont = df[cont_cols].copy() if cont_cols else pd.DataFrame(index=df.index)

        if not df_cont.empty:
            # 2. Outlier winsorization (clip to training-fold percentile bounds)
            for col, (l, h) in getattr(self, "outlier_bounds_", {}).items():
                if col in df_cont.columns:
                    df_cont[col] = df_cont[col].clip(lower=l, upper=h)

            # 3. Power transform (using training-fold statistics only)
            if getattr(self, "power_transformer_", None) is not None:
                df_for_transform = _fill_with_train_median(df_cont)
                arr = self.power_transformer_.transform(df_for_transform.values)
                na_mask = df_cont.isna()
                df_cont = pd.DataFrame(arr, columns=df_cont.columns, index=df.index)
                df_cont[na_mask] = np.nan

            elif self._log1p_shift_:
                for col in cont_cols:
                    shift = self._log1p_shift_.get(col, 0.0)
                    df_cont[col] = np.log1p(df_cont[col] + shift)

            pieces.append(df_cont)

        # -------------------------------------------------------------------
        # 4. Polynomial / interaction features
        # -------------------------------------------------------------------
        poly_cols = getattr(self, "poly_cols_", [])
        if poly_cols and hasattr(self, "pf_"):
            if all(c in df_cont.columns for c in poly_cols):
                df_poly_input = df_cont[poly_cols].copy()
            else:
                df_poly_input = df.reindex(columns=poly_cols)

            poly_arr = self.pf_.transform(
                _fill_with_train_median(df_poly_input).values
            ).astype(np.float32)
            poly_names = list(self.pf_.get_feature_names_out(poly_cols))
            df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df.index)
            # PolynomialFeatures includes originals; drop them to avoid duplication
            # (they are already in df_cont / pieces).
            df_poly = df_poly.drop(columns=poly_cols, errors="ignore")
            pieces.append(df_poly)

        # -------------------------------------------------------------------
        # 5. Binned features (appended as additional columns, not replacements)
        # -------------------------------------------------------------------
        if getattr(self, "binner_", None) is not None and getattr(self, "bin_cols_", []):
            bin_cols = self.bin_cols_
            if all(c in df_cont.columns for c in bin_cols):
                df_bin = df_cont[bin_cols].copy()
            else:
                df_bin = df.reindex(columns=bin_cols)
            na_mask_bin = df_bin.isna()
            arr = self.binner_.transform(_fill_with_train_median(df_bin).values)
            df_binned = pd.DataFrame(
                arr,
                columns=[f"{c}_binned_{self.binning_strategy}" for c in bin_cols],
                index=df.index,
            )
            df_binned[na_mask_bin.values] = np.nan
            pieces.append(df_binned)

        # -------------------------------------------------------------------
        # 6. Discrete / ordinal columns (passthrough — no shaping)
        # -------------------------------------------------------------------
        disc_cols = getattr(self, "discrete_cols_", [])
        if disc_cols:
            pieces.append(df[disc_cols].copy())

        # -------------------------------------------------------------------
        # 7. Any remaining columns not captured above
        # -------------------------------------------------------------------
        used_cols = set(cont_cols + disc_cols)
        remaining = [c for c in df.columns if c not in used_cols]
        if remaining:
            pieces.append(df[remaining].astype(np.float32))

        if not pieces:
            return pd.DataFrame(index=df.index)

        res = pd.concat(pieces, axis=1)
        res = res.loc[:, ~res.columns.duplicated()]
        return res

    # ------------------------------------------------------------------
    # get_feature_names_out
    # ------------------------------------------------------------------

    def get_feature_names_out(self, input_features=None):
        if not self.fe_enabled:
            if input_features is not None:
                return np.array(input_features)
            return np.array(getattr(self, "feature_names_in_", []))

        names: list[str] = []

        for col in getattr(self, "missing_cols_", []):
            names.append(f"{col}_missing")

        for col in getattr(self, "cont_cols_", []):
            names.append(col)

        poly_cols = getattr(self, "poly_cols_", [])
        if poly_cols and hasattr(self, "pf_"):
            for n in self.pf_.get_feature_names_out(poly_cols):
                if n not in names:
                    names.append(str(n))

        if getattr(self, "binner_", None) is not None:
            for col in getattr(self, "bin_cols_", []):
                names.append(f"{col}_binned_{self.binning_strategy}")

        for col in getattr(self, "discrete_cols_", []):
            names.append(col)

        if input_features is not None:
            used = set(getattr(self, "cont_cols_", []) + getattr(self, "discrete_cols_", []))
            for col in input_features:
                if col not in used and col not in names:
                    names.append(col)

        return np.array(names)

    # ------------------------------------------------------------------
    # Human-readable summary (used by FE Studio preview)
    # ------------------------------------------------------------------

    def describe_pipeline(self) -> list[str]:
        """Return a list of human-readable strings describing active transforms."""
        if not self.fe_enabled:
            return ["Feature Engineering disabled — raw features passed through."]
        lines = []
        if self.transform_method == "yeo-johnson":
            lines.append("Power Transform: Yeo-Johnson (handles zeros and negatives)")
        elif self.transform_method == "log1p":
            lines.append("Power Transform: Log(1+x) with per-column non-negativity shift")
        if self.missing_indicators:
            lines.append("Missing Indicators: binary flags for each column with nulls")
        if self.outliers == "winsorize_1_99":
            lines.append("Outlier Clipping: winsorize at 1st – 99th percentile")
        elif self.outliers == "winsorize_5_95":
            lines.append("Outlier Clipping: winsorize at 5th – 95th percentile")
        if self.binning_strategy != "none":
            lines.append(
                f"Discretization: {self.binning_strategy} binning into {self.n_bins} bins"
            )
        if self.poly_features:
            kind = "interaction-only" if self.interaction_only else f"degree-{self.poly_degree}"
            lines.append(
                f"Polynomial Expansion: {kind}, max {self.poly_max} input features"
            )
        return lines or ["No active transforms."]


# ---------------------------------------------------------------------------
# Static export (academic / inspection use — see data-leakage warning below)
# ---------------------------------------------------------------------------

def generate_static_fe_dataset(
    df: pd.DataFrame,
    config: dict,
    target_col: str | None,
    save_dir: str,
    filename: str,
) -> str:
    """
    Apply FeatureEngineeringTransformer to the full dataset and write a CSV.

    WARNING — Data Leakage
    ----------------------
    All statistics (quantile bounds, power-transform lambdas, bin edges) are
    fitted on the **entire** dataset, not on a training fold.  The resulting
    CSV must NOT be used as-is for cross-validation without understanding that
    the validation rows have already seen the training statistics.

    Use this only for:
      - Inspecting what the transforms produce
      - Publishing a pre-processed dataset alongside a paper (with clear caveats)
      - Sanity-checking column names / shapes

    For rigorous evaluation always use the Dynamic Pipeline (safe) option which
    runs the transformer inside the CV folds.
    """
    os.makedirs(save_dir, exist_ok=True)
    LOGGER.warning(
        "generate_static_fe_dataset: global fit on full dataset — "
        "data leakage risk. See docstring."
    )

    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[[target_col]].copy()
    else:
        X = df.copy()
        y = pd.DataFrame(index=df.index)

    transformer = FeatureEngineeringTransformer(fe_enabled=True, config=config)
    transformer.fit(X)
    X_trans = transformer.transform(X)

    transformed_df = pd.concat([X_trans, y], axis=1)
    out_path = os.path.join(save_dir, filename)
    transformed_df.to_csv(out_path, index=False)
    LOGGER.info(
        "Static FE dataset saved: %s  shape=%s", out_path, tuple(transformed_df.shape)
    )
    return out_path
