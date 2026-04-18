from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from utils.logger import get_logger


LOGGER = get_logger(__name__)



class AppState:
    def __init__(self):
        self.df = None
        self.target = None
        self.features = None
        self.model_checks = {}     # set by UI on init
        self.fe_enabled = False
        # Optional pre-training Publication Studio profile (naming/value rules, metadata).
        self.studio_profile = {}
        # SSOT: Execution State
        self.selected_models = []
        self.selected_plots = []
        self.cv_mode = "repeated"
        self.cv_folds = 5
        self.persist_outputs = True


    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Load CSV or Excel data with robust parsing and normalization.
        """
        try:
            from data.loader import read_dataset_safely

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
        from interface.validation import validate_csv_structure

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
