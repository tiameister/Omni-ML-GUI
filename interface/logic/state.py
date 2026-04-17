import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from interface.validation import validate_csv_structure
from config import RSTATE
from data.loader import read_dataset_safely
from utils.logger import get_logger


LOGGER = get_logger(__name__)


def _to_uniform_string(X):
    """Cast categorical matrix to uniform string dtype for safe OneHotEncoder input."""
    try:
        return X.astype(str)
    except Exception:
        return pd.DataFrame(X).astype(str)


class AppState:
    def __init__(self):
        self.df = None
        self.target = None
        self.features = None
        self.model_checks = {}     # set by UI on init
        self.fe_enabled = False
        # Optional pre-training Publication Studio profile (naming/value rules, metadata).
        self.studio_profile = {}

    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Load CSV or Excel data with robust parsing and normalization.
        """
        try:
            df, source = read_dataset_safely(path)
            LOGGER.info("GUI loaded dataset '%s' via '%s' shape=%s", path, source, tuple(df.shape))
            return df
        except Exception as exc:
            raise RuntimeError(f"Could not parse dataset: {path}\n{exc}") from exc



    def validate(self, df: pd.DataFrame):
        """
        Run structure/content validation on the DataFrame.
        Returns (critical_errors, warnings).
        """
        return validate_csv_structure(df)

    def set_dataframe(self, df: pd.DataFrame):
        """
        Store the loaded DataFrame in state.
        """
        self.df = df

    def set_features(self, target: str, features: list[str]):
        """
        Define target and feature columns.
        """
        self.target = target
        self.features = features

    def build_preprocessor(self):
        """
        Build a ColumnTransformer: numeric→impute+scale,
        categorical→impute+onehot.
        """
        num_cols = [c for c in self.features if pd.api.types.is_numeric_dtype(self.df[c])]
        cat_cols = [c for c in self.features if c not in num_cols]
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("to_string", FunctionTransformer(_to_uniform_string, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])
