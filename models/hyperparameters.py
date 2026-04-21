"""Single source of truth for model hyperparameter defaults and UI schemas.

This module centralizes the hyperparameters exposed to the user via the
"Configure" gear button on each model card. The same defaults are consumed by
`models.train.train_and_evaluate` so that:

* The UI always loads with recommended defaults pre-filled.
* The values shown in the UI are the exact values that reach
  `Estimator(**kwargs)` in the training pipeline (no silent hardcoded
  fallbacks override the user selection).

Schema conventions
------------------
Each model maps to an ordered list of parameter specs. A spec is a dict:

    {
        "name": str,          # scikit-learn kwarg name
        "label": str,         # user-facing label
        "type": str,          # one of: int, int_or_none, float, choice, bool
        "default": Any,       # safe, scientifically sensible default
        "tooltip": str,       # short, plain-language explanation
        # numeric types (int / int_or_none / float):
        "min": Number,
        "max": Number,
        "step": Number,
        "none_sentinel": Number,  # int_or_none only (value that maps to None)
        "slider": bool,       # if True, render as slider + readout (numeric only)
        # choice type:
        "choices": list[tuple[str, Any]],  # [(label, value)]
    }

Values flow unchanged into the estimator's constructor kwargs. ``None`` is
supported via ``int_or_none`` (UI uses a sentinel value — by convention
``none_sentinel == min`` so the widget cannot step to an invalid value —
mapped back to ``None`` in :func:`decode_param_value`).
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# NOTE: Keep in sync with the constructors in models/train.py.  The defaults
# below are the ones used when the user does not override anything.

MODEL_PARAM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "LinearRegression": [
        {
            "name": "fit_intercept",
            "label": "Fit Intercept (fit_intercept)",
            "type": "bool",
            "default": True,
            "tooltip": "If enabled, the model estimates an intercept term. "
                       "Disable only when your features are already centered.",
        },
        {
            "name": "positive",
            "label": "Force Positive Coefficients (positive)",
            "type": "bool",
            "default": False,
            "tooltip": "When enabled, constrains all coefficients to be positive. "
                       "Useful for physically constrained problems.",
        },
    ],
    "RidgeCV": [
        {
            "name": "fit_intercept",
            "label": "Fit Intercept (fit_intercept)",
            "type": "bool",
            "default": True,
            "tooltip": "If enabled, the model estimates an intercept term.",
        },
        {
            "name": "cv",
            "label": "Internal CV folds (cv)",
            "type": "int",
            "default": 5,
            "min": 2, "max": 20, "step": 1,
            "tooltip": "Number of folds used by RidgeCV to pick alpha internally.",
        },
    ],
    "RandomForest": [
        {
            "name": "n_estimators",
            "label": "Number of Trees (n_estimators)",
            "type": "int", "slider": True,
            "default": 500,
            "min": 10, "max": 2000, "step": 10,
            "tooltip": "Higher values increase stability but take longer to train.",
        },
        {
            "name": "max_depth",
            "label": "Maximum Tree Depth (max_depth)",
            "type": "int_or_none", "slider": True,
            "default": None,
            # Sentinel == minimum so the UI can't step to an invalid value
            # (sklearn rejects max_depth <= 0). 0 means "Unlimited", 1..50
            # are real depths.
            "none_sentinel": 0,
            "min": 0, "max": 50, "step": 1,
            "tooltip": "Cap on tree depth. 'Unlimited' lets trees grow as "
                       "deep as needed; shallower trees regularize the model.",
        },
        {
            "name": "min_samples_split",
            "label": "Min Samples to Split (min_samples_split)",
            "type": "int",
            "default": 2,
            "min": 2, "max": 50, "step": 1,
            "tooltip": "Minimum samples required to split an internal node.",
        },
        {
            "name": "min_samples_leaf",
            "label": "Min Samples per Leaf (min_samples_leaf)",
            "type": "int",
            "default": 1,
            "min": 1, "max": 50, "step": 1,
            "tooltip": "Minimum samples required at a leaf. Higher values "
                       "reduce variance (smoother model).",
        },
        {
            "name": "max_features",
            "label": "Features per Split (max_features)",
            "type": "choice",
            "default": 1.0,
            "choices": [
                ("All features (1.0)", 1.0),
                ("sqrt", "sqrt"),
                ("log2", "log2"),
                ("50% (0.5)", 0.5),
                ("33% (0.33)", 0.33),
            ],
            "tooltip": "How many features to consider when looking for the "
                       "best split. 'sqrt'/'log2' often help generalization.",
        },
        {
            "name": "bootstrap",
            "label": "Use Bootstrap Samples (bootstrap)",
            "type": "bool",
            "default": True,
            "tooltip": "Whether bootstrap samples are used when building trees.",
        },
    ],
    "HistGB": [
        {
            "name": "learning_rate",
            "label": "Learning Rate (learning_rate)",
            "type": "float",
            "default": 0.1,
            "min": 0.01, "max": 1.0, "step": 0.01,
            "tooltip": "Shrinks the contribution of each tree. Lower values "
                       "generalize better but need more iterations.",
        },
        {
            "name": "max_iter",
            "label": "Max Boosting Iterations (max_iter)",
            "type": "int", "slider": True,
            "default": 100,
            "min": 10, "max": 2000, "step": 10,
            "tooltip": "Maximum number of boosting stages. Early stopping may "
                       "use fewer when training plateaus.",
        },
        {
            "name": "max_depth",
            "label": "Maximum Tree Depth (max_depth)",
            "type": "int_or_none", "slider": True,
            "default": None,
            # Sentinel == minimum (see note on RandomForest.max_depth).
            "none_sentinel": 0,
            "min": 0, "max": 20, "step": 1,
            "tooltip": "Cap on tree depth. 'Unlimited' lets trees grow as "
                       "deep as needed.",
        },
        {
            "name": "min_samples_leaf",
            "label": "Min Samples per Leaf (min_samples_leaf)",
            "type": "int",
            "default": 20,
            "min": 1, "max": 200, "step": 1,
            "tooltip": "Minimum samples per leaf. Higher values regularize.",
        },
        {
            "name": "l2_regularization",
            "label": "L2 Regularization (l2_regularization)",
            "type": "float",
            "default": 0.0,
            "min": 0.0, "max": 5.0, "step": 0.1,
            "tooltip": "L2 regularization strength on leaf values.",
        },
    ],
    "GradientBoostingRegressor": [
        {
            "name": "learning_rate",
            "label": "Learning Rate (learning_rate)",
            "type": "float",
            "default": 0.1,
            "min": 0.01, "max": 1.0, "step": 0.01,
            "tooltip": "Shrinks the contribution of each tree.",
        },
        {
            "name": "n_estimators",
            "label": "Number of Boosting Stages (n_estimators)",
            "type": "int", "slider": True,
            "default": 100,
            "min": 10, "max": 1000, "step": 10,
            "tooltip": "Number of boosting stages to perform.",
        },
        {
            "name": "max_depth",
            "label": "Maximum Tree Depth (max_depth)",
            "type": "int", "slider": True,
            "default": 3,
            "min": 1, "max": 20, "step": 1,
            "tooltip": "Depth of each tree. 3-6 typically works well.",
        },
        {
            "name": "subsample",
            "label": "Row Subsample Fraction (subsample)",
            "type": "float",
            "default": 1.0,
            "min": 0.1, "max": 1.0, "step": 0.05,
            "tooltip": "Fraction of rows used to fit each tree. <1 enables "
                       "stochastic gradient boosting (reduces variance).",
        },
    ],
    "Lasso": [
        {
            "name": "fit_intercept",
            "label": "Fit Intercept (fit_intercept)",
            "type": "bool",
            "default": True,
            "tooltip": "If enabled, the model estimates an intercept term.",
        },
        {
            "name": "cv",
            "label": "Internal CV folds (cv)",
            "type": "int",
            "default": 5,
            "min": 2, "max": 20, "step": 1,
            "tooltip": "Number of folds used by LassoCV to pick alpha.",
        },
        {
            "name": "max_iter",
            "label": "Max Iterations (max_iter)",
            "type": "int", "slider": True,
            "default": 1000,
            "min": 100, "max": 20000, "step": 100,
            "tooltip": "Maximum iterations of the coordinate descent solver.",
        },
    ],
    "ElasticNet": [
        {
            "name": "fit_intercept",
            "label": "Fit Intercept (fit_intercept)",
            "type": "bool",
            "default": True,
            "tooltip": "If enabled, the model estimates an intercept term.",
        },
        {
            "name": "cv",
            "label": "Internal CV folds (cv)",
            "type": "int",
            "default": 5,
            "min": 2, "max": 20, "step": 1,
            "tooltip": "Number of folds used by ElasticNetCV to pick alpha.",
        },
        {
            "name": "max_iter",
            "label": "Max Iterations (max_iter)",
            "type": "int", "slider": True,
            "default": 1000,
            "min": 100, "max": 20000, "step": 100,
            "tooltip": "Maximum iterations of the coordinate descent solver.",
        },
    ],
    "SVR": [
        {
            "name": "kernel",
            "label": "Kernel Function (kernel)",
            "type": "choice",
            "default": "rbf",
            "choices": [
                ("RBF (Gaussian)", "rbf"),
                ("Linear", "linear"),
                ("Polynomial", "poly"),
                ("Sigmoid", "sigmoid"),
            ],
            "tooltip": "How SVR maps features to a higher-dimensional space.",
        },
        {
            "name": "C",
            "label": "Regularization Strength (C)",
            "type": "float",
            "default": 1.0,
            "min": 0.01, "max": 1000.0, "step": 0.1,
            "tooltip": "Inverse of regularization strength; higher C fits "
                       "training data more tightly.",
        },
        {
            "name": "epsilon",
            "label": "Epsilon Tube (epsilon)",
            "type": "float",
            "default": 0.1,
            "min": 0.0, "max": 10.0, "step": 0.05,
            "tooltip": "Width of the epsilon-insensitive tube; errors within "
                       "epsilon incur no penalty.",
        },
        {
            "name": "gamma",
            "label": "Kernel Coefficient (gamma)",
            "type": "choice",
            "default": "scale",
            "choices": [
                ("Scale (recommended)", "scale"),
                ("Auto", "auto"),
            ],
            "tooltip": "Kernel coefficient for RBF, poly and sigmoid kernels.",
        },
    ],
    "KNeighborsRegressor": [
        {
            "name": "n_neighbors",
            "label": "Number of Neighbors (n_neighbors)",
            "type": "int", "slider": True,
            "default": 5,
            "min": 1, "max": 100, "step": 1,
            "tooltip": "Higher k smooths predictions but may underfit.",
        },
        {
            "name": "weights",
            "label": "Neighbor Weighting (weights)",
            "type": "choice",
            "default": "uniform",
            "choices": [
                ("Uniform (all neighbors equal)", "uniform"),
                ("Distance (closer = more weight)", "distance"),
            ],
            "tooltip": "How neighbors contribute to the prediction.",
        },
        {
            "name": "p",
            "label": "Distance Power (p)",
            "type": "choice",
            "default": 2,
            "choices": [("Manhattan (p=1)", 1), ("Euclidean (p=2)", 2)],
            "tooltip": "Power parameter of the Minkowski distance.",
        },
    ],
    "XGBoost": [
        {
            "name": "n_estimators",
            "label": "Number of Trees (n_estimators)",
            "type": "int", "slider": True,
            "default": 800,
            "min": 10, "max": 3000, "step": 10,
            "tooltip": "More trees increase accuracy but slow training.",
        },
        {
            "name": "learning_rate",
            "label": "Learning Rate (learning_rate)",
            "type": "float",
            "default": 0.05,
            "min": 0.005, "max": 1.0, "step": 0.005,
            "tooltip": "Shrinks the contribution of each tree.",
        },
        {
            "name": "max_depth",
            "label": "Maximum Tree Depth (max_depth)",
            "type": "int", "slider": True,
            "default": 6,
            "min": 1, "max": 20, "step": 1,
            "tooltip": "Depth of each tree; deeper trees capture interactions "
                       "but may overfit.",
        },
        {
            "name": "subsample",
            "label": "Row Subsample Fraction (subsample)",
            "type": "float",
            "default": 0.9,
            "min": 0.1, "max": 1.0, "step": 0.05,
            "tooltip": "Fraction of rows used for each tree.",
        },
        {
            "name": "colsample_bytree",
            "label": "Feature Subsample (colsample_bytree)",
            "type": "float",
            "default": 0.9,
            "min": 0.1, "max": 1.0, "step": 0.05,
            "tooltip": "Fraction of features used for each tree.",
        },
        {
            "name": "reg_lambda",
            "label": "L2 Regularization (reg_lambda)",
            "type": "float",
            "default": 1.0,
            "min": 0.0, "max": 10.0, "step": 0.1,
            "tooltip": "L2 regularization on leaf weights.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def has_schema(model_name: str) -> bool:
    """Return True when a hyperparameter schema is registered for ``model_name``."""
    return str(model_name) in MODEL_PARAM_SCHEMAS


def get_param_schema(model_name: str) -> list[dict[str, Any]]:
    """Return a *copy* of the parameter schema list for a model (empty if missing)."""
    return [dict(p) for p in MODEL_PARAM_SCHEMAS.get(str(model_name), [])]


def get_default_hyperparams(model_name: str) -> dict[str, Any]:
    """Return {param_name: default_value} for a given model.

    The values are exactly what the estimator would receive if the user never
    opened the settings dialog. ``None`` is returned for params that map the
    sentinel to None (e.g. ``max_depth=None`` in RandomForest).
    """
    out: dict[str, Any] = {}
    for spec in MODEL_PARAM_SCHEMAS.get(str(model_name), []):
        out[str(spec["name"])] = spec.get("default")
    return out


def decode_param_value(spec: dict[str, Any], raw_value: Any) -> Any:
    """Translate a raw UI value into the value that should reach the estimator.

    * ``int_or_none`` maps ``none_sentinel`` → ``None``.
    * ``choice`` passes the stored ``userData`` through unchanged.
    * Other types cast to the appropriate Python type.
    """
    ptype = str(spec.get("type", ""))
    if ptype == "int_or_none":
        if raw_value is None:
            return None
        sentinel = int(spec.get("none_sentinel", 0))
        try:
            iv = int(raw_value)
        except (TypeError, ValueError):
            return spec.get("default")
        if iv == sentinel:
            return None
        # Out-of-range ints (e.g. a legacy session saved with a previous
        # sentinel convention) collapse to None rather than reaching sklearn
        # as a value it would reject at fit time.
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None and iv < int(lo):
            return None
        if hi is not None and iv > int(hi):
            return None
        return iv
    if ptype == "int":
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return spec.get("default")
    if ptype == "float":
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return spec.get("default")
    if ptype == "bool":
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in ("true", "1", "yes", "on")
        return bool(raw_value)
    if ptype == "choice":
        # Validate against the declared choices; fall back to the default.
        allowed = {v for _lbl, v in spec.get("choices", [])}
        return raw_value if raw_value in allowed else spec.get("default")
    return raw_value


def sanitize_hyperparams(
    model_name: str,
    raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a validated {param: value} dict for ``model_name``.

    * Unknown keys are dropped.
    * Missing keys fall back to the schema default.
    * Values are coerced / validated via :func:`decode_param_value`.

    This function is deliberately conservative: it never raises on malformed
    user input. It produces a dict that is safe to splat into the estimator
    constructor via ``**kwargs``.
    """
    defaults = get_default_hyperparams(model_name)
    if not defaults:
        # No schema registered for this model -> nothing to override.
        return {}

    raw = dict(raw or {})
    out: dict[str, Any] = {}
    for spec in MODEL_PARAM_SCHEMAS.get(str(model_name), []):
        name = str(spec["name"])
        if name in raw:
            out[name] = decode_param_value(spec, raw[name])
        else:
            out[name] = spec.get("default")
    return out


def encode_param_value(spec: dict[str, Any], value: Any) -> Any:
    """Translate a stored value into the raw value the UI widget expects.

    Inverse of :func:`decode_param_value`. Used when re-opening the settings
    dialog so previously saved user choices repopulate correctly. Values that
    fall outside the declared [min, max] range (e.g. legacy sessions) are
    clamped to the sentinel so the dialog opens on "Unlimited" instead of
    silently rounding to an invalid value.
    """
    ptype = str(spec.get("type", ""))
    if ptype == "int_or_none":
        sentinel = int(spec.get("none_sentinel", 0))
        if value is None:
            return sentinel
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return sentinel
        lo = spec.get("min")
        hi = spec.get("max")
        if (lo is not None and iv < int(lo)) or (hi is not None and iv > int(hi)):
            return sentinel
        return iv
    return value


__all__ = [
    "MODEL_PARAM_SCHEMAS",
    "decode_param_value",
    "encode_param_value",
    "get_default_hyperparams",
    "get_param_schema",
    "has_schema",
    "sanitize_hyperparams",
]
