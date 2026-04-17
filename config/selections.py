# config/selections.py

DEFAULT_MODELS = [
    "LinearRegression",
    "RidgeCV",
    "RandomForest",
    "HistGB",
    "XGBoost",
    "Lasso",
    "ElasticNet",
    "SVR",
    "KNeighborsRegressor",
    "GradientBoostingRegressor"
]
DEFAULT_PLOTS = ["permutation", "pdp", "shap", "residuals", "actual_vs_predicted", "stats_summary"]
import os

def get_selected_models(selected=None):
    """
    Returns the list of selected models.
    If none provided, returns defaults.
    """
    # Env override: comma-separated list of models
    env_sel = os.environ.get("SELECTED_MODELS")
    if env_sel:
        requested = [m.strip() for m in env_sel.split(',') if m.strip()]
        return [m for m in requested if m in DEFAULT_MODELS]
    if selected is None:
        return DEFAULT_MODELS
    return [m for m in selected if m in DEFAULT_MODELS]
