"""Application configuration with validated environment overrides."""

import os

from exceptions import ConfigError
from utils.paths import get_versioned_output_folder, resolve_dataset_path


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer, got '{raw}'") from exc

    if value < min_value or value > max_value:
        raise ConfigError(f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


def _to_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a float, got '{raw}'") from exc

    if value < min_value or value > max_value:
        raise ConfigError(f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


VERSION = "1.1.0"
RSTATE = _to_int("RSTATE", 42, 0, 1_000_000_000)

PI_REPEATS = _to_int("PI_REPEATS", 5, 1, 500)
DO_SHAP = _to_bool(os.environ.get("DO_SHAP"), True)
SAVE_PDF = _to_bool(os.environ.get("SAVE_PDF"), False)

OUTPUT_DIR = os.environ.get("OUTPUT_ROOT_DIR", "output").strip() or "output"
RUN_TAG = os.environ.get("RUN_TAG", "").strip() or None

EVAL_PLOTS_ENABLED = _to_bool(os.environ.get("EVAL_PLOTS_ENABLED"), True)

# Dataset path is resolved with environment override first, then project defaults.
DATASET_PATH = str(resolve_dataset_path(strict=False))

# Plotting defaults
SHAP_DEFAULT_YLIM = (-3.0, 3.0)

# SHAP configuration
SHAP_TOP_N = _to_int("SHAP_TOP_N", 10, 1, 5000)
SHAP_VAR_THRESH = _to_float("SHAP_VAR_THRESH", 1e-8, 0.0, 1.0)
FEATURE_NAME_MAP: dict[str, str] = {}
SHAP_ALWAYS_INCLUDE: list[str] = []
SHAP_BEESWARM_TRIM_PCT = _to_float("SHAP_BEESWARM_TRIM_PCT", 1.0, 0.0, 50.0)

# SHAP feature dependence / correlation handling
_shap_dep_raw = os.environ.get("SHAP_DEPENDENCE_MODE", "interventional")
_shap_dep_raw = str(_shap_dep_raw).strip().lower()
if _shap_dep_raw in {"interventional", "independent", "causal"}:
    SHAP_DEPENDENCE_MODE = "interventional"
elif _shap_dep_raw in {"partition", "correlation", "correlated", "grouped"}:
    SHAP_DEPENDENCE_MODE = "partition"
elif _shap_dep_raw in {"tree_path_dependent", "treepath", "legacy"}:
    SHAP_DEPENDENCE_MODE = "tree_path_dependent"
else:
    raise ConfigError(
        "SHAP_DEPENDENCE_MODE must be one of: interventional/independent, partition/correlation, tree_path_dependent"
        f" (got '{_shap_dep_raw}')"
    )

_shap_min_cap_raw = os.environ.get("SHAP_BEESWARM_MIN_CAP")
if _shap_min_cap_raw is None or str(_shap_min_cap_raw).strip() == "":
    SHAP_BEESWARM_MIN_CAP = None
else:
    try:
        SHAP_BEESWARM_MIN_CAP = float(_shap_min_cap_raw)
    except ValueError as exc:
        raise ConfigError(f"SHAP_BEESWARM_MIN_CAP must be numeric, got '{_shap_min_cap_raw}'") from exc

# Cross-validation configuration
CV_MODE = os.environ.get("CV_MODE", "repeated").strip().lower()
if CV_MODE not in {"kfold", "repeated", "nested", "holdout"}:
    raise ConfigError(f"CV_MODE must be one of kfold/repeated/nested/holdout, got '{CV_MODE}'")

CV_FOLDS = _to_int("CV_FOLDS", 5, 2, 20)
CV_REPEATS = _to_int("CV_REPEATS", 3, 1, 50)
NESTED_OUTER_FOLDS = _to_int("NESTED_OUTER_FOLDS", 5, 2, 20)
NESTED_INNER_FOLDS = _to_int("NESTED_INNER_FOLDS", 5, 2, 20)

CV_STRATIFY = os.environ.get("CV_STRATIFY", "none").strip().lower()
if CV_STRATIFY not in {"none", "deciles"}:
    raise ConfigError(f"CV_STRATIFY must be 'none' or 'deciles', got '{CV_STRATIFY}'")

# Permutation importance runtime controls
PI_ONLY_BEST_MODEL = _to_bool(os.environ.get("PI_ONLY_BEST_MODEL"), True)
PI_N_JOBS = _to_int("PI_N_JOBS", -1, -1, 64)
if PI_N_JOBS == 0:
    raise ConfigError("PI_N_JOBS cannot be 0; use -1 for all cores or a positive integer")
PERM_IMPORTANCE_ENABLED = _to_bool(os.environ.get("PERM_IMPORTANCE_ENABLED", os.environ.get("DO_PI")), True)


def get_output_folder(base_name: str) -> str:
    """Return next available versioned output folder path for the given model name."""
    return str(get_versioned_output_folder(base_name=base_name, output_dir=OUTPUT_DIR, run_tag=RUN_TAG))