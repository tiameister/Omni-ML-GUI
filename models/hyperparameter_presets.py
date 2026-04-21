"""Configuration presets for model hyperparameters.

These presets give non-expert users a guided starting point. A preset is a
named, fully-resolved set of hyperparameter values layered on top of the
schema defaults declared in :mod:`models.hyperparameters`.

The flow is:

* The :class:`HyperparameterDialog` reads :data:`MODEL_PRESETS` and offers a
  "Configuration Preset" dropdown at the top of the form.
* Selecting a preset calls :func:`resolve_preset` and populates every widget.
* Manual edits afterwards flip the dropdown to "Custom (Manual Tuning)".
* The training runner records the detected preset name (via
  :func:`match_preset`) in ``run_manifest.json`` for provenance.

Design notes
------------
* ``Lasso`` / ``ElasticNet`` in this codebase are actually ``LassoCV`` /
  ``ElasticNetCV``: ``alpha`` is auto-tuned by internal cross-validation, so
  fixed-alpha presets don't cleanly apply. They are intentionally omitted
  here (same rationale as ``LinearRegression`` / ``RidgeCV``).
* Preset bodies are *partial* — only the keys that meaningfully change are
  listed. :func:`resolve_preset` overlays them on the schema defaults so the
  estimator always receives a complete ``**kwargs`` dict.
"""
from __future__ import annotations

from typing import Any

from models.hyperparameters import get_default_hyperparams, sanitize_hyperparams


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

CUSTOM_PRESET_ID = "__custom__"
CUSTOM_PRESET_LABEL = "Custom (Manual Tuning)"


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------
#
# Each entry:
#   {
#       "id":          stable identifier used in logs/manifests,
#       "label":       user-facing dropdown text,
#       "description": one-line rationale shown under the dropdown,
#       "params":      partial override dict; missing keys fall back to the
#                      schema defaults in models/hyperparameters.py.
#   }

MODEL_PRESETS: dict[str, list[dict[str, Any]]] = {
    "RandomForest": [
        {
            "id": "sklearn_default",
            "label": "Default (Scikit-Learn Standard)",
            "description": "Library defaults — fastest baseline.",
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "bootstrap": True,
            },
        },
        {
            "id": "robust",
            "label": "Robust (Noisy / Survey Data)",
            "description": "Flagship social-science preset: stable and well-regularized.",
            "params": {
                "n_estimators": 500,
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "bootstrap": True,
            },
        },
        {
            "id": "small_dataset",
            "label": "Small Dataset (< 500 rows)",
            "description": "Heavier regularization for small-n data to curb overfitting.",
            "params": {
                "n_estimators": 300,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "bootstrap": True,
            },
        },
    ],
    "HistGB": [
        {
            "id": "sklearn_default",
            "label": "Default (Scikit-Learn Standard)",
            "description": "Library defaults.",
            "params": {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_depth": None,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
            },
        },
        {
            "id": "robust",
            "label": "Robust (Controlled Learning)",
            "description": "Lower learning rate with depth cap; trades speed for stability.",
            "params": {
                "learning_rate": 0.05,
                "max_iter": 500,
                "max_depth": 3,
                "min_samples_leaf": 30,
                "l2_regularization": 0.5,
            },
        },
    ],
    "GradientBoostingRegressor": [
        {
            "id": "sklearn_default",
            "label": "Default (Scikit-Learn Standard)",
            "description": "Library defaults.",
            "params": {
                "learning_rate": 0.1,
                "n_estimators": 100,
                "max_depth": 3,
                "subsample": 1.0,
            },
        },
        {
            "id": "robust",
            "label": "Robust (Controlled Learning)",
            "description": "Slower learning, stochastic subsampling; strong regularization.",
            "params": {
                "learning_rate": 0.05,
                "n_estimators": 500,
                "max_depth": 3,
                "subsample": 0.8,
            },
        },
    ],
    "XGBoost": [
        {
            "id": "sklearn_default",
            "label": "Default",
            "description": "XGBoost library defaults.",
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.3,
                "max_depth": 6,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "id": "robust",
            "label": "Robust (Controlled Learning)",
            "description": "Low learning rate + subsampling; a strong default for noisy tabular data.",
            "params": {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
            },
        },
    ],
    "SVR": [
        {
            "id": "sklearn_default",
            "label": "Default (Scikit-Learn Standard)",
            "description": "Library defaults.",
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1,
                "gamma": "scale",
            },
        },
        {
            "id": "robust",
            "label": "Robust (Outlier Tolerant)",
            "description": "Low C and wider epsilon tube; tolerant to noisy targets.",
            "params": {
                "kernel": "rbf",
                "C": 0.1,
                "epsilon": 0.2,
                "gamma": "scale",
            },
        },
        {
            "id": "complex",
            "label": "Complex / Non-Linear",
            "description": "High C with a narrow epsilon tube for intricate patterns.",
            "params": {
                "kernel": "rbf",
                "C": 10.0,
                "epsilon": 0.05,
                "gamma": "scale",
            },
        },
    ],
    "KNeighborsRegressor": [
        {
            "id": "sklearn_default",
            "label": "Default (Scikit-Learn Standard)",
            "description": "Library defaults (k=5, uniform weighting).",
            "params": {
                "n_neighbors": 5,
                "weights": "uniform",
                "p": 2,
            },
        },
        {
            "id": "smooth",
            "label": "Smooth (k=15, distance-weighted)",
            "description": "Larger neighborhood; robust to local noise.",
            "params": {
                "n_neighbors": 15,
                "weights": "distance",
                "p": 2,
            },
        },
        {
            "id": "responsive",
            "label": "Responsive (k=3)",
            "description": "Small, distance-weighted neighborhood; captures fine structure.",
            "params": {
                "n_neighbors": 3,
                "weights": "distance",
                "p": 2,
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def has_presets(model_name: str) -> bool:
    """Return True when at least one preset exists for ``model_name``."""
    return bool(MODEL_PRESETS.get(str(model_name)))


def get_presets(model_name: str) -> list[dict[str, Any]]:
    """Return a deep copy of the preset list for ``model_name`` (empty if none)."""
    out: list[dict[str, Any]] = []
    for preset in MODEL_PRESETS.get(str(model_name), []):
        out.append({
            "id": str(preset["id"]),
            "label": str(preset["label"]),
            "description": str(preset.get("description", "")),
            "params": dict(preset.get("params", {})),
        })
    return out


def get_preset(model_name: str, preset_id: str) -> dict[str, Any] | None:
    """Return a single preset descriptor, or None if ``preset_id`` is unknown."""
    for p in get_presets(model_name):
        if p["id"] == str(preset_id):
            return p
    return None


def resolve_preset(model_name: str, preset_id: str) -> dict[str, Any] | None:
    """Return the fully-resolved ``{param: value}`` dict for a preset.

    Missing keys are filled from :func:`get_default_hyperparams`, and the
    final dict is run through :func:`sanitize_hyperparams` so the values are
    exactly what the estimator would receive at training time. Returns
    ``None`` if the preset id is not registered for the given model.
    """
    preset = get_preset(model_name, preset_id)
    if preset is None:
        return None
    overlay = get_default_hyperparams(model_name)
    overlay.update(preset["params"])
    return sanitize_hyperparams(model_name, overlay)


def match_preset(model_name: str, params: dict[str, Any] | None) -> str | None:
    """Return the preset id whose resolved params equal ``params`` exactly.

    Useful for two things:

    * In the dialog, detecting whether the user's current values correspond
      to a registered preset (so the dropdown can show that preset instead
      of "Custom").
    * At run time, stamping ``run_manifest.json`` with the preset name so
      downstream analyses know which configuration regime produced a run.
    """
    if not params:
        return None
    current = sanitize_hyperparams(model_name, dict(params))
    if not current:
        return None
    for preset in MODEL_PRESETS.get(str(model_name), []):
        candidate = resolve_preset(model_name, preset["id"])
        if candidate == current:
            return str(preset["id"])
    return None


__all__ = [
    "CUSTOM_PRESET_ID",
    "CUSTOM_PRESET_LABEL",
    "MODEL_PRESETS",
    "get_preset",
    "get_presets",
    "has_presets",
    "match_preset",
    "resolve_preset",
]
