from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from config import CV_REPEATS, CV_STRATIFY, DO_SHAP, OUTPUT_DIR, RSTATE, RUN_TAG, SHAP_DEPENDENCE_MODE, SHAP_VAR_THRESH
from utils.logger import get_logger
from utils.paths import (
    get_run_model_dir,
    get_run_model_root,
    get_run_root,
    get_run_subdir,
    get_supplements_root,
    get_transient_run_root,
    make_run_id,
    safe_folder_name,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingCallbacks:
    progress: Callable[[int, int], None] | None = None
    plot_progress: Callable[[int, int], None] | None = None
    log: Callable[[str], None] | None = None
    should_cancel: Callable[[], bool] | None = None


def _safe_call(cb, *args, context: str):
    if not callable(cb):
        return
    try:
        cb(*args)
    except Exception:
        # Non-fatal: callback failures should not break training.
        LOGGER.exception("Non-fatal callback failure (%s)", context)


def _raise_if_cancelled(should_cancel):
    if callable(should_cancel) and should_cancel():
        raise RuntimeError("Cancelled by user")


def _project_root_dir() -> str:
    """Return project root; resolves correctly for both source and PyInstaller frozen builds."""
    from utils.paths import get_project_root
    return str(get_project_root())


def _run_optional_script(
    filename: str,
    *,
    env_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
    should_cancel=None,
) -> tuple[bool, str]:
    """Run optionally provided scripts robustly without spawning zombie processes."""
    import subprocess
    import sys
    import os
    from pathlib import Path

    root = _project_root_dir()
    scripts_dir = Path(root) / "scripts"

    raw_name = str(filename or "").strip()
    if not raw_name:
        return False, "Script filename is empty"

    # Block path traversal / absolute paths. Optional scripts must be basenames under ./scripts.
    if "/" in raw_name or "\\" in raw_name:
        return False, f"Invalid script name (must be a basename): {raw_name}"
    if not raw_name.endswith(".py"):
        return False, f"Invalid script name (expected .py): {raw_name}"

    try:
        script_path_obj = (scripts_dir / raw_name).resolve()
        scripts_dir_resolved = scripts_dir.resolve()
        script_path_obj.relative_to(scripts_dir_resolved)
    except Exception:
        return False, f"Invalid script path: scripts/{raw_name}"

    if not script_path_obj.exists():
        return False, f"Script not found: scripts/{raw_name}"
    if not script_path_obj.is_file():
        return False, f"Script is not a file: scripts/{raw_name}"

    script_path = str(script_path_obj)

    env = os.environ.copy()
    # Do not inherit caller-provided PYTHONPATH (prevents module injection).
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = root

    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[str(k)] = str(v)

    # Use context manager to ensure pipes and process are properly finalized
    try:
        with subprocess.Popen(
            [sys.executable, "-X", "faulthandler", script_path],
            cwd=cwd or root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            close_fds=True,
        ) as proc:
            
            stdout_data, stderr_data = [], []
            while True:
                if callable(should_cancel) and should_cancel():
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    return False, f"Cancelled by user while running: {filename}"
                    
                try:
                    # Non-blocking check
                    out, err = proc.communicate(timeout=0.1)
                    if out: stdout_data.append(out.strip())
                    if err: stderr_data.append(err.strip())
                    break
                except subprocess.TimeoutExpired as exc:
                    # Drain partial output to avoid pipe buffers filling up.
                    try:
                        partial_out = getattr(exc, "stdout", None) or getattr(exc, "output", None)
                        partial_err = getattr(exc, "stderr", None)
                        if partial_out:
                            stdout_data.append(str(partial_out).strip())
                        if partial_err:
                            stderr_data.append(str(partial_err).strip())
                    except Exception:
                        pass
                    continue

            return proc.returncode == 0, "\n".join(stdout_data + stderr_data) or f"Exit {proc.returncode}"
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(f"Critical Subprocess Failed: {exc}", exc_info=True)
        return False, f"Exception runtime failure: {exc}"


def run_training(
    *,
    df: pd.DataFrame,
    target: str,
    features: list[str],
    selected_models: list[str],
    selected_plots: list[str],
    cv_mode: str = "repeated",
    cv_folds: int = 5,
    callbacks: TrainingCallbacks | None = None,
    run_id: str | None = None,
    dataset_label: str | None = None,
    persist_outputs: bool = True,
    persist_run_tag: str | None = None,
    fe_enabled: bool = False,
    fe_config: dict[str, object] | None = None,
    feature_value_labels: dict[str, dict[str, str]] | None = None,
    shap_settings: dict[str, object] | None = None,
    optional_scripts: list[tuple[str, str]] | None = None,
):
    """Pure training orchestration: no Qt/UI dependencies.

    Returns: (metrics_df, fitted_models, stats_df, stats_summary_df, out_info)
    """
    from evaluation.explain import (
        generate_regression_stats,
        plot_correlation_matrix,
        plot_feature_importance_heatmap,
        plot_learning_curve,
        plot_predictions_vs_actual,
        plot_qq,
        plot_residual_distribution,
        plot_residuals,
    )
    from evaluation.metrics import get_feature_names_from_pipe, save_model_metrics, save_cv_splits
    from features.preprocess import build_preprocessor
    from models.train import train_and_evaluate

    callbacks = callbacks or TrainingCallbacks()

    if df is None or target is None or features is None:
        raise RuntimeError("df, target, and features are required before training.")


    if target in features:
        features = [f for f in features if f != target]
        _safe_call(callbacks.log, f"WARNING: Target '{target}' was included in feature list! Automatically removed to prevent catastrophic data leakage.", context="log")

    X = df[features].copy()
    y = df[target].copy()


    # Normalize feature_value_labels so downstream SHAP export is stable.
    feature_value_labels_for_outputs: dict[str, dict[str, str]] = {}
    for raw_feature, raw_map in dict(feature_value_labels or {}).items():
        feature_name = str(raw_feature).strip()
        if not feature_name or not isinstance(raw_map, dict):
            continue
        local_map: dict[str, str] = {}
        for raw_src, raw_dst in raw_map.items():
            src = str(raw_src).strip()
            dst = str(raw_dst).strip()
            if not src or not dst:
                continue
            local_map[src] = dst
        if local_map:
            feature_value_labels_for_outputs[feature_name] = local_map

    if not pd.api.types.is_numeric_dtype(y):
        raise RuntimeError("Selected target must be numeric. Please choose a numeric column.")

    nan_count = int(y.isna().sum())
    if nan_count > 0:
        non_na_idx = ~y.isna()
        X = X.loc[non_na_idx]
        y = y.loc[non_na_idx]
        _safe_call(callbacks.log, f"Dropped {nan_count} rows with NaN target values.", context="log")

    def raise_if_cancelled():
        _raise_if_cancelled(callbacks.should_cancel)

    feature_count_before_fe = int(X.shape[1])
    feature_count_after_fe = feature_count_before_fe

    if bool(fe_enabled):
        _safe_call(callbacks.log, "Feature engineering is enabled and will execute inside CV loop safely.", context="log")
        raise_if_cancelled()

    # Build preprocessing pipeline based on effective matrix.
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    
    # Strictly define categoricals and block dimensionality explosion from misclassified numeric identifiers
    cat_cols_raw = [c for c in X.columns if c not in set(num_cols)]
    cat_cols = []
    
    for c in cat_cols_raw:
        unique_count = X[c].nunique()
        # High cardinality safe-guard mask: If an object column has more than 50 unique categories and 
        # its uniqueness ratio is high (e.g. text logs or UUIDs instead of groups), drop it from modelling.
        if unique_count > 50 and (unique_count / max(len(X), 1)) > 0.15:
            _safe_call(callbacks.log, f"WARNING: Dropped high-cardinality categorical '{c}' ({unique_count} distinct values) to prevent dimensionality explosion.", context="log")
            X = X.drop(columns=[c])
        else:
            cat_cols.append(c)

    preproc = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        fe_enabled=bool(fe_enabled),
        fe_config=fe_config,
    )

    def progress_callback(done: int, total: int):
        _safe_call(callbacks.progress, done, total, context="progress")
        raise_if_cancelled()

    _safe_call(callbacks.log, "Training started", context="log")
    _safe_call(callbacks.log, f"Validation: mode={cv_mode}, folds={cv_folds}", context="log")
    if bool(fe_enabled):
        _safe_call(
            callbacks.log,
            f"Feature engineering in-run: enabled ({feature_count_before_fe} -> {feature_count_after_fe}).",
            context="log",
        )

    def model_status(name: str, phase: str):
        _safe_call(callbacks.log, f"{phase.upper()}: {name}", context="log")

    metrics_df, fitted_models = train_and_evaluate(
        X,
        y,
        preproc,
        model_names=selected_models,
        cv_mode=cv_mode,
        cv_folds=cv_folds,
        progress_callback=progress_callback,
        log_callback=callbacks.log,
        model_status_callback=model_status,
    )

    best = str(metrics_df.iloc[0]["model"]) if metrics_df is not None and not metrics_df.empty else None

    if run_id is None or str(run_id).strip() == "":
        prefix = dataset_label or "gui"
        run_id_final = make_run_id(prefix=prefix)
    else:
        run_id_final = safe_folder_name(str(run_id), fallback="run")

    if persist_outputs:
        run_root = get_run_root(run_id_final, output_dir=OUTPUT_DIR, run_tag=persist_run_tag or RUN_TAG)
    else:
        run_root = get_transient_run_root(run_id_final)

    run_outdir = str(run_root)

    out_info: dict[str, object] = {
        "run_id": run_id_final,
        "run_dir": run_outdir,
        "best_model": best,
        "persist_outputs": bool(persist_outputs),
    }

    analysis_root = os.path.join(run_outdir, "1_Overall_Evaluation")
    os.makedirs(analysis_root, exist_ok=True)
    out_info["analysis_dir"] = analysis_root
    model_root = str(get_run_model_root(run_root))
    out_info["model_dir"] = model_root
    supplements_root = str(get_supplements_root(run_root=run_root))
    out_info["supplements_dir"] = supplements_root

    _safe_call(callbacks.log, f"Run output folder: {run_outdir}", context="log")
    _safe_call(
        callbacks.log,
        f"Output persistence: {'enabled' if persist_outputs else 'temporary (not auto-saved)'}",
        context="log",
    )

    if metrics_df is not None and not metrics_df.empty:
        fe_prefix = "feature_engineering_" if fe_enabled else ""
        save_model_metrics(run_outdir, metrics_df, filename_prefix=fe_prefix)
        save_cv_splits(run_outdir, {name: m.get("cv_scores", {}) for name, m in fitted_models.items()})

    # Persist fitted pipelines in a deterministic model subtree.
    try:
        import joblib
        for model_name, payload in dict(fitted_models or {}).items():
            pipe_obj = payload.get("pipe")
            if pipe_obj is None:
                continue
            model_dir = Path(get_run_model_dir(run_root, model_name))
            model_path = model_dir / "model_pipeline.joblib"
            joblib.dump(pipe_obj, model_path)
            cv_scores = payload.get("cv_scores", {}) or {}
            with open(model_dir / "cv_scores.json", "w", encoding="utf-8") as fh:
                json.dump(cv_scores, fh, ensure_ascii=False, indent=2)
    except Exception:
        LOGGER.exception("Failed to persist one or more model artifacts")

    # Provenance
    selection_dir = str(get_run_subdir(run_root, "0_Feature_Selection"))
    out_info["selection_dir"] = selection_dir
    try:
        selected_feats = list(features or [])
        target_col = str(target or "")
        try:
            miss_vars = [c for c in df.columns if df[c].isna().any()]
        except Exception:
            miss_vars = []
        excluded_missing_by_ui = [c for c in miss_vars if c not in selected_feats]
        ui_meta = {
            "target": target_col,
            "selected_features": selected_feats,
            "n_selected": int(len(selected_feats)),
            "variables_with_missing_in_full_df": miss_vars,
            "missing_vars_excluded_by_ui": excluded_missing_by_ui,
        }
        with open(os.path.join(selection_dir, "ui_feature_selection_meta.json"), "w", encoding="utf-8") as f:
            json.dump(ui_meta, f, ensure_ascii=False, indent=2)

        lines = []
        try:
            lines.append(f"Selected features (n={len(selected_feats)}): {', '.join(map(str, selected_feats))}")
        except Exception:
            lines.append(f"Selected features (n={len(selected_feats)}) written to ui_feature_selection_meta.json")
        if excluded_missing_by_ui:
            lines.append("Missing vars excluded by UI: " + ", ".join(map(str, excluded_missing_by_ui)))
        else:
            lines.append("No variables with missingness were excluded by UI.")
        with open(os.path.join(selection_dir, "ui_feature_selection_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        LOGGER.exception("Failed to persist feature selection provenance")

    outdir_by_model: dict[str, str] = {}

    def _get_model_outdir(model_name: str) -> str:
        existing = outdir_by_model.get(model_name)
        if existing:
            return existing
        model_dir = str(get_run_model_dir(run_root, model_name))
        outdir_by_model[model_name] = model_dir
        return model_dir

    stats_df = None
    stats_summary_df = None
    if "LinearRegression" in selected_models and "LinearRegression" in fitted_models:
        try:
            fe_prefix = "feature_engineering_" if fe_enabled else ""
            stats_outdir = _get_model_outdir("LinearRegression")
            stats_df = generate_regression_stats(
                fe_prefix + "LinearRegression",
                fitted_models["LinearRegression"]["pipe"],
                X,
                y,
                stats_outdir,
            )
            try:
                import numpy as np

                y_pred = fitted_models["LinearRegression"]["pipe"].predict(X)
                resid = pd.Series(y.values - np.asarray(y_pred).reshape(-1))
                res_mean = float(resid.mean())
                res_std = float(resid.std(ddof=1))
                res_skew = float(resid.skew())
                res_kurt = float(resid.kurtosis())
                p_norm = None
                try:
                    from scipy.stats import shapiro

                    sample = resid.sample(n=min(5000, len(resid)), random_state=42) if len(resid) > 5000 else resid
                    p_norm = float(shapiro(sample)[1])
                except Exception:
                    try:
                        from statsmodels.stats.stattools import jarque_bera

                        _, pjb, _, _ = jarque_bera(resid)
                        p_norm = float(pjb)
                    except Exception:
                        p_norm = float("nan")
                stats_summary_df = pd.DataFrame(
                    [
                        {
                            "res_mean": res_mean,
                            "res_std": res_std,
                            "res_skew": res_skew,
                            "res_kurtosis": res_kurt,
                            "normality_p": p_norm,
                        }
                    ]
                )
            except Exception:
                stats_summary_df = None
        except Exception as exc:
            _safe_call(callbacks.log, f"Regression stats error: {exc}", context="log")

    if shap_settings is None:
        shap_settings = {}

    try:
        shap_top_n = int(shap_settings.get("top_n", -1))
        shap_var_enabled = str(shap_settings.get("var_enabled", "false")).lower() in ("true", "1", "yes")
        shap_var_thresh = float(shap_settings.get("var_thresh", SHAP_VAR_THRESH)) if shap_var_enabled else None
        shap_always_include_raw = shap_settings.get("always_include", "")
        shap_dep_raw = shap_settings.get("dependence_mode", shap_settings.get("dependence", SHAP_DEPENDENCE_MODE))
        shap_dependence_mode = str(shap_dep_raw or SHAP_DEPENDENCE_MODE or "interventional").strip().lower()
        if shap_dependence_mode not in {"interventional", "partition", "tree_path_dependent"}:
            shap_dependence_mode = str(SHAP_DEPENDENCE_MODE or "interventional").strip().lower()
        if shap_always_include_raw is None:
            shap_always_include = []
        else:
            shap_always_include = [s.strip() for s in str(shap_always_include_raw).split(",") if s.strip()]
    except Exception:
        shap_top_n, shap_var_thresh = -1, None
        shap_always_include = []
        shap_dependence_mode = str(SHAP_DEPENDENCE_MODE or "interventional").strip().lower()

    # Persist run manifest (now includes SHAP configuration for reliability/provenance)
    try:
        run_manifest = {
            "run_id": run_id_final,
            "dataset_label": dataset_label,
            "cv_mode": cv_mode,
            "cv_folds": cv_folds,
            "selected_models": list(selected_models or []),
            "selected_plots": list(selected_plots or []),
            "best_model": best,
            "feature_engineering": bool(fe_enabled),
            "persist_outputs": bool(persist_outputs),
            "shap": {
                "enabled": bool(DO_SHAP),
                "dependence_mode": shap_dependence_mode,
                "top_n": shap_top_n,
                "var_thresh": shap_var_thresh,
                "always_include": list(shap_always_include or []),
            },
        }
        with open(os.path.join(run_outdir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        LOGGER.exception("Failed to write run manifest")

    optional_scripts = list(optional_scripts or [])

    # Build regression stratification labels once so evaluation plots can mirror CV behavior.
    y_strata = None
    try:
        if str(CV_STRATIFY).lower() == "deciles":
            y_series = pd.Series(y)
            y_strata = pd.qcut(
                y_series.rank(method="first"),
                q=10,
                labels=False,
                duplicates="drop",
            )
    except Exception:
        y_strata = None

    def _build_cv_splits(mode: str, *, folds: int) -> list[tuple[list[int], list[int]]] | object:
        """Return either a reusable splitter or an explicit list of (train, test) indices."""
        from sklearn.model_selection import (
            KFold,
            RepeatedKFold,
            RepeatedStratifiedKFold,
            StratifiedKFold,
        )

        n_splits = max(2, int(folds))
        mode_norm = str(mode or "").strip().lower()
        if mode_norm not in {"kfold", "repeated", "nested", "holdout"}:
            mode_norm = "kfold"

        if mode_norm in {"kfold", "nested"}:
            if y_strata is not None:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RSTATE)
                return list(splitter.split(X, y_strata))
            return KFold(n_splits=n_splits, shuffle=True, random_state=RSTATE)

        if mode_norm == "repeated":
            repeats = max(1, int(CV_REPEATS))
            if y_strata is not None:
                splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=RSTATE)
                return list(splitter.split(X, y_strata))
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=RSTATE)
            return list(splitter.split(X, y))

        # holdout does not use CV splits
        return KFold(n_splits=n_splits, shuffle=True, random_state=RSTATE)

    def _compute_eval_predictions(pipe_fit, *, mode: str, folds: int):
        """Return (X_eval, y_eval, y_pred_eval, cv_for_learning_curve)."""
        import numpy as np
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict, train_test_split

        mode_norm = str(mode or "").strip().lower()
        n_splits = max(2, int(folds))

        # Holdout: evaluate on test split only.
        if mode_norm == "holdout":
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RSTATE)
            est = clone(pipe_fit)
            est.fit(X_tr, y_tr)
            y_pred = est.predict(X_te)
            cv_for_lc = _build_cv_splits("kfold", folds=n_splits)
            return X_te, y_te, y_pred, cv_for_lc

        # For CV modes, compute out-of-fold predictions once.
        cv_obj = _build_cv_splits(mode_norm if mode_norm != "nested" else "kfold", folds=n_splits)

        if mode_norm != "repeated":
            y_pred = cross_val_predict(clone(pipe_fit), X, y, cv=cv_obj, n_jobs=-1)
            return X, y, y_pred, cv_obj

        # Repeated CV: average predictions over all test appearances.
        splits = cv_obj if isinstance(cv_obj, list) else list(cv_obj.split(X, y))
        pred_sum = np.zeros(len(y), dtype=float)
        pred_count = np.zeros(len(y), dtype=int)
        for tr_idx, te_idx in splits:
            raise_if_cancelled()
            est = clone(pipe_fit)
            X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            X_te = X.iloc[te_idx] if hasattr(X, "iloc") else X[te_idx]
            est.fit(X_tr, y_tr)
            pred = np.asarray(est.predict(X_te), dtype=float).reshape(-1)
            pred_sum[np.asarray(te_idx, dtype=int)] += pred
            pred_count[np.asarray(te_idx, dtype=int)] += 1

        denom = np.maximum(pred_count, 1)
        y_pred = pred_sum / denom
        return X, y, y_pred, cv_obj

    total_plots = 0
    for model_name in selected_models:
        if "Residuals" in selected_plots:
            total_plots += 1
        if "Residual Distribution" in selected_plots:
            total_plots += 1
        if "Q-Q Plot" in selected_plots:
            total_plots += 1
        if "Correlation Matrix" in selected_plots:
            total_plots += 1
        if "Learning Curve" in selected_plots:
            total_plots += 1
        if "Predictions vs Actual" in selected_plots:
            total_plots += 1
        if ("Feature Importance" in selected_plots) or ("Feature Importance Heatmap" in selected_plots):
            total_plots += 1
        if DO_SHAP and ("SHAP Summary" in selected_plots or "SHAP Dependence" in selected_plots):
            if "SHAP Summary" in selected_plots:
                total_plots += 1
            if "SHAP Dependence" in selected_plots:
                try:
                    n_feats = len(get_feature_names_from_pipe(fitted_models[model_name]["pipe"]))
                except Exception:
                    n_feats = len(X.columns)
                top_dep = n_feats if shap_top_n == -1 else int(shap_top_n)
                total_plots += top_dep

    total_plots += len(optional_scripts)

    plots_done = 0

    def plot_progress_inc():
        nonlocal plots_done
        plots_done += 1
        _safe_call(callbacks.plot_progress, plots_done, total_plots, context="plot_progress")

    _safe_call(callbacks.plot_progress, 0, total_plots, context="plot_progress")

    for model_name in selected_models:
        raise_if_cancelled()
        if model_name in fitted_models:
            _safe_call(callbacks.log, f"Generating plots for {model_name}", context="log")
            pipe = fitted_models[model_name]["pipe"]
            fe_prefix = "feature_engineering_" if fe_enabled else ""

            # Build evaluation-consistent predictions once and reuse across diagnostic plots.
            needs_eval_preds = any(
                p in selected_plots for p in ("Residuals", "Residual Distribution", "Q-Q Plot", "Predictions vs Actual")
            )
            eval_X, eval_y, eval_pred = X, y, None
            if needs_eval_preds:
                try:
                    if str(cv_mode or "").strip().lower() == "nested":
                        nested_oof = fitted_models.get(model_name, {}).get("oof_pred")
                        if nested_oof is not None:
                            eval_X, eval_y, eval_pred = X, y, nested_oof
                        else:
                            eval_X, eval_y, eval_pred, _cv_for_lc_unused = _compute_eval_predictions(
                                pipe,
                                mode=cv_mode,
                                folds=cv_folds,
                            )
                    else:
                        eval_X, eval_y, eval_pred, _cv_for_lc_unused = _compute_eval_predictions(
                            pipe,
                            mode=cv_mode,
                            folds=cv_folds,
                        )
                except Exception as exc:
                    _safe_call(callbacks.log, f"[Warning] Could not compute evaluation predictions: {exc}", context="log")
                    eval_X, eval_y, eval_pred = X, y, None

            try:
                lc_mode = "kfold" if str(cv_mode).strip().lower() == "holdout" else cv_mode
                cv_for_lc = _build_cv_splits(lc_mode, folds=int(cv_folds or 5))
            except Exception:
                cv_for_lc = 5

            model_outdir: str | None = None

            def _outdir() -> str:
                nonlocal model_outdir
                if model_outdir is None:
                    model_outdir = _get_model_outdir(model_name)
                    _safe_call(callbacks.log, f"Generating plots into: {model_outdir}", context="log")
                return model_outdir

            if "Residuals" in selected_plots:
                raise_if_cancelled()
                plot_residuals(fe_prefix + model_name, pipe, eval_X, eval_y, _outdir(), preds=eval_pred)
                plot_progress_inc()

            if "Residual Distribution" in selected_plots:
                raise_if_cancelled()
                plot_residual_distribution(
                    fe_prefix + model_name,
                    pipe,
                    eval_X,
                    eval_y,
                    _outdir(),
                    preds=eval_pred,
                )
                plot_progress_inc()

            if "Q-Q Plot" in selected_plots:
                raise_if_cancelled()
                plot_qq(fe_prefix + model_name, pipe, eval_X, eval_y, _outdir(), preds=eval_pred)
                plot_progress_inc()

            if "Correlation Matrix" in selected_plots:
                raise_if_cancelled()
                plot_correlation_matrix(fe_prefix + model_name, X, y, _outdir())
                plot_progress_inc()

            if "Learning Curve" in selected_plots:
                raise_if_cancelled()
                plot_learning_curve(fe_prefix + model_name, pipe, X, y, _outdir(), cv=cv_for_lc)
                plot_progress_inc()

            if "Predictions vs Actual" in selected_plots:
                raise_if_cancelled()
                plot_predictions_vs_actual(
                    fe_prefix + model_name,
                    pipe,
                    eval_X,
                    eval_y,
                    _outdir(),
                    preds=eval_pred,
                )
                plot_progress_inc()

            if ("Feature Importance" in selected_plots) or ("Feature Importance Heatmap" in selected_plots):
                raise_if_cancelled()
                plot_feature_importance_heatmap(fe_prefix + model_name, pipe, X, y, _outdir())
                plot_progress_inc()

            if DO_SHAP and ("SHAP Summary" in selected_plots or "SHAP Dependence" in selected_plots):
                try:
                    raise_if_cancelled()
                    num_cols2 = X.select_dtypes(include=["number"]).columns.tolist()
                    cat_cols2 = X.select_dtypes(exclude=["number"]).columns.tolist()

                    if "SHAP Summary" in selected_plots:
                        raise_if_cancelled()
                        from evaluation.explain import generate_shap_summary

                        tn = None if shap_top_n == -1 else int(shap_top_n)
                        vt = None if shap_var_thresh is None else float(shap_var_thresh)
                        generate_shap_summary(
                            fe_prefix + model_name,
                            pipe,
                            X,
                            num_cols2,
                            cat_cols2,
                            _outdir(),
                            top_n=tn,
                            var_thresh=vt,
                            dependence_mode=shap_dependence_mode,
                            cancel_cb=callbacks.should_cancel,
                        )
                        plot_progress_inc()

                    if "SHAP Dependence" in selected_plots:
                        raise_if_cancelled()
                        from evaluation.explain import generate_shap_dependence

                        tn = None if shap_top_n == -1 else int(shap_top_n)
                        vt = None if shap_var_thresh is None else float(shap_var_thresh)
                        try:
                            n_feats = len(get_feature_names_from_pipe(pipe))
                        except Exception:
                            n_feats = len(X.columns)
                        eff_k = n_feats if tn is None else int(tn)
                        generate_shap_dependence(
                            fe_prefix + model_name,
                            pipe,
                            X,
                            num_cols2,
                            cat_cols2,
                            _outdir(),
                            top_n=tn if tn is not None else n_feats,
                            seed=42,
                            var_thresh=vt,
                            always_include=shap_always_include,
                            feature_value_labels=feature_value_labels_for_outputs,
                            dependence_mode=shap_dependence_mode,
                            cancel_cb=callbacks.should_cancel,
                        )
                        for _ in range(int(eff_k)):
                            plot_progress_inc()
                except Exception as exc:
                    if "cancelled by user" in str(exc).strip().lower():
                        raise
                    _safe_call(
                        callbacks.log,
                        f"[Warning] SHAP plots skipped for {model_name}: {exc}",
                        context="log",
                    )

    if optional_scripts:
        _safe_call(callbacks.log, f"Running {len(optional_scripts)} extra analysis task(s)...", context="log")
        _safe_call(callbacks.log, f"Extra analysis output folder: {analysis_root}", context="log")

    for label, script_filename in optional_scripts:
        raise_if_cancelled()
        _safe_call(callbacks.log, f"START: Extra analysis - {label}", context="log")

        env_overrides = {
            "OUTPUT_ROOT_DIR": run_outdir,
            "RUN_TAG": "",
            "MLTRAINER_RUN_ROOT": run_outdir,
            "MLTRAINER_ANALYSIS_ROOT": analysis_root,
            "MLTRAINER_SUPPLEMENTS_ROOT": supplements_root,
        }

        ok, output = _run_optional_script(
            script_filename,
            env_overrides=env_overrides,
            should_cancel=callbacks.should_cancel,
        )

        if (not ok) and ("cancelled by user" in str(output).strip().lower()):
            raise RuntimeError("Cancelled by user")

        if ok:
            _safe_call(callbacks.log, f"DONE: Extra analysis - {label}", context="log")
        else:
            _safe_call(callbacks.log, f"[Warning] Extra analysis failed: {label}", context="log")

        if callable(callbacks.log):
            lines = [ln for ln in str(output).splitlines() if ln.strip()]
            for ln in lines[-4:]:
                _safe_call(callbacks.log, f"  {ln}", context="log")

        plot_progress_inc()


    try:
        import gc
        import platform
        from datetime import datetime
        metadata_export = {
            "experiment_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_label": dataset_label or "unknown",
            "dataset_rows": int(len(df)),
            "dataset_cols": int(len(df.columns)),
            "target_variable": target,
            "input_features": features,
            "cv_strategy": {
                "mode": cv_mode,
                "folds": cv_folds,
            },
            "models_selected": selected_models,
            "system_info": {
                "os": platform.system(),
                "python_version": platform.python_version()
            },
            "results": metrics_df.to_dict(orient="records") if not metrics_df.empty else []
        }
        
        json_path = os.path.join(run_outdir, "experiment_metadata.json")
        with open(json_path, "w", encoding="utf-8") as meta_f:
            json.dump(metadata_export, meta_f, indent=4, ensure_ascii=False)
        
        LOGGER.info(f"Experiment metadata exported successfully to {json_path}")
        
        # Hard memory sweep for long GUI sessions
        del metadata_export
        gc.collect()
        
    except Exception as meta_exc:
        LOGGER.warning(f"Could not generate experiment metadata JSON: {meta_exc}")

    try:
        for root_dir, dirs, files in os.walk(run_outdir, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root_dir, d)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
    except Exception:
        LOGGER.exception("Failed to cleanup empty directories")

    return metrics_df, fitted_models, stats_df, stats_summary_df, out_info
