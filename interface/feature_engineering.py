"""Compatibility wrapper for feature engineering.

The UI layer used to own the feature engineering implementation. The
implementation is now UI-independent and lives under `features/`.
"""

from features.feature_engineering import apply_feature_engineering

__all__ = ["apply_feature_engineering"]
