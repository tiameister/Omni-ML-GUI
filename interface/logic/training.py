import os
import json
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.metrics import get_feature_names_from_pipe

from models.train import train_and_evaluate
from evaluation.metrics import save_model_metrics, dump_permutation
from interface.feature_engineering import apply_feature_engineering
from interface.logic.state import AppState
from evaluation.explain import (
    explain_with_shap,
    generate_pdp,
    generate_regression_stats,
    plot_residuals,
    plot_residual_distribution,
    plot_qq,
    plot_correlation_matrix,
    plot_learning_curve,
    plot_feature_importance_heatmap,
    plot_predictions_vs_actual,
)
from utils.helpers import ensure_outdir
from utils.paths import (
    get_run_root,
    get_transient_run_root,
    make_run_id,
    get_run_subdir,
    get_run_model_dir,
    safe_folder_name,
)
from config import OUTPUT_DIR, RUN_TAG, DO_SHAP, SHAP_TOP_N, SHAP_VAR_THRESH
from PyQt6.QtCore import QSettings

from shap import Explainer
from shap import summary_plot

RSTATE = 42
PI_REPEATS = 5


def _project_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_optional_scripts(selected_items: list[str]) -> list[tuple[str, str]]:
    """Return [(label, filename)] for selected optional script options."""
    try:
        from interface.widgets.checkboxes import get_optional_script_label_map
        label_map = get_optional_script_label_map()
    except Exception:
        label_map = {}

    resolved: list[tuple[str, str]] = []
    for label in selected_items:
        filename = label_map.get(label)
        if filename:
            resolved.append((label, filename))

    # Keep first occurrence only
    seen: set[str] = set()
    dedup: list[tuple[str, str]] = []
    for label, filename in resolved:
        if filename in seen:
            continue
        seen.add(filename)
        dedup.append((label, filename))
    return dedup


def _run_optional_script(
    filename: str,
    *,
    env_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
    should_cancel=None,
) -> tuple[bool, str]:
    """Run a script file from scripts/ as a separate Python process.
    Returns (ok, combined_output).

    env_overrides can be used to route script outputs into the current run folder.
    """
    root = _project_root_dir()
    script_path = os.path.join(root, "scripts", filename)
    if not os.path.exists(script_path):
        return False, f"Script not found: scripts/{filename}"

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing_pp else f"{root}{os.pathsep}{existing_pp}"

    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[str(k)] = str(v)

    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-X", "faulthandler", script_path],
            cwd=cwd or root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        while True:
            if callable(should_cancel) and should_cancel():
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                except Exception:
                    stdout, stderr = "", ""
                parts = []
                if stdout and stdout.strip():
                    parts.append(stdout.strip())
                if stderr and stderr.strip():
                    parts.append(stderr.strip())
                detail = "\n".join(parts).strip()
                suffix = f"\n{detail}" if detail else ""
                return False, f"Cancelled by user while running scripts/{filename}{suffix}"

            try:
                stdout, stderr = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue

        parts = []
        if stdout and stdout.strip():
            parts.append(stdout.strip())
        if stderr and stderr.strip():
            parts.append(stderr.strip())
        out = "\n".join(parts).strip()
        if not out:
            out = f"exit code {proc.returncode}"
        return proc.returncode == 0, out
    except Exception as e:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return False, str(e)


def run_training(
    state,
    selected_models,
    selected_plots,
    cv_mode='repeated',
    cv_folds=5,
    external_progress_cb=None,
    external_plot_progress_cb=None,
    external_log_cb=None,
    should_cancel=None,
    run_id: str | None = None,
    dataset_label: str | None = None,
    persist_outputs: bool = True,
    persist_run_tag: str | None = None,
    feature_value_labels: dict[str, dict[str, str]] | None = None,
):
    """
    Train models, update status bar and result box, and return metrics and fitted models.
    """
    # Validate state
    if state.df is None or state.target is None or state.features is None:
        raise RuntimeError("State requires dataframe, target, and features before training.")
    # Prepare data
    X = state.df[state.features].copy()
    y = state.df[state.target].copy()
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
    # Validate target is numeric and has no NaNs
    if not pd.api.types.is_numeric_dtype(y):
        raise RuntimeError("Selected target must be numeric. Please choose a numeric column.")
    # Drop rows with NaN in target and inform user
    nan_count = y.isna().sum()
    if nan_count > 0:
        non_na_idx = ~y.isna()
        X = X.loc[non_na_idx]
        y = y.loc[non_na_idx]
        if callable(external_log_cb):
            try:
                external_log_cb(f"Dropped {nan_count} rows with NaN target values.")
            except Exception:
                pass
    def raise_if_cancelled():
        if callable(should_cancel) and should_cancel():
            raise RuntimeError("Cancelled by user")

    feature_count_before_fe = int(X.shape[1])
    feature_count_after_fe = feature_count_before_fe
    if bool(getattr(state, "fe_enabled", False)):
        raise_if_cancelled()
        if callable(external_log_cb):
            try:
                external_log_cb(
                    f"Feature engineering started for training matrix ({feature_count_before_fe} base features)."
                )
            except Exception:
                pass
        X_engineered, _new_num_cols, _cat_cols = apply_feature_engineering(
            X,
            output_folder="feature_engineered_dataset",
            save_csv=False,
            force_passthrough_cols=list(feature_value_labels_for_outputs.keys()),
        )
        X = X_engineered
        feature_count_after_fe = int(X.shape[1])
        if callable(external_log_cb):
            try:
                external_log_cb(
                    f"Feature engineering completed: {feature_count_before_fe} -> {feature_count_after_fe} features."
                )
            except Exception:
                pass
        raise_if_cancelled()

    # Build preprocessing pipeline from the effective feature matrix used in this run.
    preproc_state = AppState()
    preproc_state.set_dataframe(X)
    preproc_state.set_features(str(getattr(state, "target", "target")), list(X.columns))
    preproc = preproc_state.build_preprocessor()

    # Define progress callback (forwards to external)
    def progress_callback(done, total):
        if callable(external_progress_cb):
            try:
                external_progress_cb(done, total)
            except Exception:
                pass
        raise_if_cancelled()

    # Train models
    if callable(external_log_cb):
        try:
            external_log_cb("Training started")
            external_log_cb(f"Validation: mode={cv_mode}, folds={cv_folds}")
            if bool(getattr(state, "fe_enabled", False)):
                external_log_cb(
                    f"Feature engineering in-run: enabled ({feature_count_before_fe} -> {feature_count_after_fe})."
                )
        except Exception:
            pass
    def model_status(name, phase):
        if callable(external_log_cb):
            try:
                external_log_cb(f"{phase.upper()}: {name}")
            except Exception:
                pass
    metrics_df, fitted_models = train_and_evaluate(
        X, y, preproc,
        model_names=selected_models,
        cv_mode=cv_mode,
        cv_folds=cv_folds,
        progress_callback=progress_callback,
        log_callback=external_log_cb,
        model_status_callback=model_status
    )

    # --- Output management (UX-first): one folder per training run ---
    best = str(metrics_df.iloc[0]["model"]) if not metrics_df.empty else None

    # Create a stable run folder (so outputs do not spread into random versioned folders).
    # If caller did not provide run_id, generate one.
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

    if callable(external_log_cb):
        try:
            external_log_cb(f"Run output folder: {run_outdir}")
            external_log_cb(f"Output persistence: {'enabled' if persist_outputs else 'temporary (not auto-saved)'}")
        except Exception:
            pass

    # Save run-wide metrics (all models) under the run folder.
    if metrics_df is not None and not metrics_df.empty:
        fe_prefix = "feature_engineering_" if getattr(state, "fe_enabled", False) else ""
        save_model_metrics(run_outdir, metrics_df, filename_prefix=fe_prefix)

    # Persist UI feature selection for provenance under run/selection/
    selection_dir = str(get_run_subdir(run_root, "0_Feature_Selection"))
    out_info["selection_dir"] = selection_dir
    try:
        selected_feats = list(getattr(state, 'features', []))
        target_col = str(getattr(state, 'target', ''))
        try:
            miss_vars = [c for c in state.df.columns if state.df[c].isna().any()]
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
    except Exception as e:
        if callable(external_log_cb):
            try:
                external_log_cb(f"UI feature selection meta save failed: {e}")
            except Exception:
                pass

    # Save a tiny run manifest for traceability (does not affect UX if ignored)
    try:
        run_manifest = {
            "run_id": run_id_final,
            "dataset_label": dataset_label,
            "cv_mode": cv_mode,
            "cv_folds": cv_folds,
            "selected_models": list(selected_models or []),
            "selected_plots": list(selected_plots or []),
            "best_model": best,
            "feature_engineering": bool(getattr(state, "fe_enabled", False)),
            "persist_outputs": bool(persist_outputs),
        }
        with open(os.path.join(run_outdir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Model output folders are created lazily (only when we actually write files).
    outdir_by_model: dict[str, str] = {}

    def _get_model_outdir(model_name: str) -> str:
        existing = outdir_by_model.get(model_name)
        if existing:
            return existing
        # This creates: <run>/models/<ModelName>/
        model_dir = str(get_run_model_dir(run_root, model_name))
        outdir_by_model[model_name] = model_dir
        return model_dir

    # Generate detailed regression statistics for LinearRegression
    stats_df = None
    stats_summary_df = None
    if "LinearRegression" in selected_models and "LinearRegression" in fitted_models:
        try:
            fe_prefix = "feature_engineering_" if getattr(state, "fe_enabled", False) else ""
            stats_outdir = _get_model_outdir("LinearRegression")
            stats_df = generate_regression_stats(
                fe_prefix + "LinearRegression",
                fitted_models["LinearRegression"]["pipe"],
                X, y, stats_outdir
            )
            # Compute residuals summary
            try:
                import numpy as np
                y_pred = fitted_models["LinearRegression"]["pipe"].predict(X)
                resid = pd.Series(y.values - np.asarray(y_pred).reshape(-1))
                res_mean = float(resid.mean())
                res_std = float(resid.std(ddof=1))
                res_skew = float(resid.skew())
                res_kurt = float(resid.kurtosis())
                # Normality p-value via Shapiro or JB fallback
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
                        p_norm = float('nan')
                stats_summary_df = pd.DataFrame([{
                    'res_mean': res_mean,
                    'res_std': res_std,
                    'res_skew': res_skew,
                    'res_kurtosis': res_kurt,
                    'normality_p': p_norm
                }])
            except Exception:
                stats_summary_df = None
            # UI thread will render stats_df and summary
        except Exception as e:
            if callable(external_log_cb):
                try:
                    external_log_cb(f"Regression stats error: {e}")
                except Exception:
                    pass

    # Load runtime SHAP settings from QSettings (fallback to config defaults)
    try:
        _settings = QSettings()
        # Default behavior: analyze all features (store -1 sentinel); we convert to None when calling SHAP
        shap_top_n = int(_settings.value('shap/top_n', -1))
        shap_var_enabled = str(_settings.value('shap/var_enabled', 'false')).lower() in ("true","1","yes")
        shap_var_thresh = float(_settings.value('shap/var_thresh', SHAP_VAR_THRESH)) if shap_var_enabled else None
        shap_always_include_raw = _settings.value('shap/always_include', '')
        if shap_always_include_raw is None:
            shap_always_include = []
        else:
            shap_always_include = [s.strip() for s in str(shap_always_include_raw).split(',') if s.strip()]
    except Exception:
        shap_top_n, shap_var_thresh = -1, None
        shap_always_include = []

    optional_scripts = _resolve_optional_scripts(selected_plots)

    # Compute total number of plots to generate (best effort) and initialize plot progress
    total_plots = 0
    for model_name in selected_models:
        # Count only those we will actually attempt
        if "Residuals" in selected_plots:
            total_plots += 1
        if "Residual Distribution" in selected_plots:
            total_plots += 1
        if "Q-Q Plot" in selected_plots: total_plots += 1
        if "Correlation Matrix" in selected_plots: total_plots += 1
        if "Learning Curve" in selected_plots: total_plots += 1
        if "Predictions vs Actual" in selected_plots: total_plots += 1
        if ("Feature Importance" in selected_plots) or ("Feature Importance Heatmap" in selected_plots):
            total_plots += 1
        if DO_SHAP and ("SHAP Summary" in selected_plots or "SHAP Dependence" in selected_plots):
            # Count SHAP based on which parts are selected
            if "SHAP Summary" in selected_plots:
                total_plots += 1
            if "SHAP Dependence" in selected_plots:
                # If top_n is -1 (all), estimate with transformed feature count; else use top_n
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
        if callable(external_plot_progress_cb):
            try:
                external_plot_progress_cb(plots_done, total_plots)
            except Exception:
                pass

    if callable(external_plot_progress_cb):
        try:
            external_plot_progress_cb(0, total_plots)
        except Exception:
            pass

    # Generate additional plots per selected options
    for model_name in selected_models:
        raise_if_cancelled()
        if model_name in fitted_models:
            if callable(external_log_cb):
                try:
                    external_log_cb(f"Generating plots for {model_name}")
                except Exception:
                    pass
            pipe = fitted_models[model_name]["pipe"]
            fe_prefix = "feature_engineering_" if getattr(state, "fe_enabled", False) else ""

            # Lazily create the model folder only if we actually generate at least one artifact.
            model_outdir: str | None = None

            def _outdir() -> str:
                nonlocal model_outdir
                if model_outdir is None:
                    model_outdir = _get_model_outdir(model_name)
                    if callable(external_log_cb):
                        try:
                            external_log_cb(f"Generating plots into: {model_outdir}")
                        except Exception:
                            pass
                return model_outdir

            # Residuals scatter plot
            if "Residuals" in selected_plots:
                raise_if_cancelled()
                plot_residuals(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Residual distribution histogram
            if "Residual Distribution" in selected_plots:
                raise_if_cancelled()
                plot_residual_distribution(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Q-Q plot
            if "Q-Q Plot" in selected_plots:
                raise_if_cancelled()
                plot_qq(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Correlation matrix heatmap
            if "Correlation Matrix" in selected_plots:
                raise_if_cancelled()
                plot_correlation_matrix(
                    fe_prefix + model_name,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Learning curve
            if "Learning Curve" in selected_plots:
                raise_if_cancelled()
                plot_learning_curve(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Predictions vs Actual
            if "Predictions vs Actual" in selected_plots:
                raise_if_cancelled()
                plot_predictions_vs_actual(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # Feature importance
            if ("Feature Importance" in selected_plots) or ("Feature Importance Heatmap" in selected_plots):
                raise_if_cancelled()
                plot_feature_importance_heatmap(
                    fe_prefix + model_name,
                    pipe,
                    X,
                    y,
                    _outdir(),
                ); plot_progress_inc()
            # SHAP analyses
            if DO_SHAP and ("SHAP Summary" in selected_plots or "SHAP Dependence" in selected_plots):
                try:
                    raise_if_cancelled()
                    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
                    # Do summary if selected
                    if "SHAP Summary" in selected_plots:
                        raise_if_cancelled()
                        from evaluation.explain import generate_shap_summary
                        tn = None if shap_top_n == -1 else int(shap_top_n)
                        vt = None if shap_var_thresh is None else float(shap_var_thresh)
                        generate_shap_summary(
                            fe_prefix + model_name,
                            pipe,
                            X,
                            num_cols,
                            cat_cols,
                            _outdir(),
                            top_n=tn,
                            var_thresh=vt,
                            cancel_cb=should_cancel,
                        )
                        plot_progress_inc()
                    # Do dependence if selected
                    if "SHAP Dependence" in selected_plots:
                        raise_if_cancelled()
                        from evaluation.explain import generate_shap_dependence
                        tn = None if shap_top_n == -1 else int(shap_top_n)
                        vt = None if shap_var_thresh is None else float(shap_var_thresh)
                        # Count for progress must be numeric; determine effective K here
                        try:
                            n_feats = len(get_feature_names_from_pipe(pipe))
                        except Exception:
                            n_feats = len(X.columns)
                        eff_k = n_feats if tn is None else int(tn)
                        generate_shap_dependence(
                            fe_prefix + model_name,
                            pipe,
                            X,
                            num_cols,
                            cat_cols,
                            _outdir(),
                            top_n=tn if tn is not None else n_feats,
                            seed=RSTATE,
                            var_thresh=vt,
                            always_include=shap_always_include,
                            feature_value_labels=feature_value_labels_for_outputs,
                            cancel_cb=should_cancel,
                        )
                        for _ in range(int(eff_k)):
                            plot_progress_inc()
                except Exception as e:
                    if "cancelled by user" in str(e).strip().lower():
                        raise
                    if callable(external_log_cb):
                        try:
                            external_log_cb(f"[Warning] SHAP plots skipped for {model_name}: {e}")
                        except Exception:
                            pass

    if optional_scripts and callable(external_log_cb):
        try:
            external_log_cb(f"Running {len(optional_scripts)} extra analysis task(s)...")
            external_log_cb(f"Extra analysis output folder: {analysis_root}")
        except Exception:
            pass

    for label, script_filename in optional_scripts:
        raise_if_cancelled()

        if callable(external_log_cb):
            try:
                external_log_cb(f"START: Extra analysis - {label}")
            except Exception:
                pass

        env_overrides = {
            # Keep scripts bound to the current run and a dedicated analysis subtree.
            "OUTPUT_ROOT_DIR": run_outdir,
            "RUN_TAG": "",
            "MLTRAINER_RUN_ROOT": run_outdir,
            "MLTRAINER_ANALYSIS_ROOT": analysis_root,
        }

        ok, output = _run_optional_script(
            script_filename,
            env_overrides=env_overrides,
            should_cancel=should_cancel,
        )

        if (not ok) and ("cancelled by user" in str(output).strip().lower()):
            raise RuntimeError("Cancelled by user")

        if callable(external_log_cb):
            try:
                if ok:
                    external_log_cb(f"DONE: Extra analysis - {label}")
                else:
                    external_log_cb(f"[Warning] Extra analysis failed: {label}")
                lines = [ln for ln in str(output).splitlines() if ln.strip()]
                for ln in lines[-4:]:
                    external_log_cb(f"  {ln}")
            except Exception:
                pass

        plot_progress_inc()

    # --- Clean up empty directories in the analysis folder to improve UX ---
    try:
        import shutil
        for root_dir, dirs, files in os.walk(run_outdir, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root_dir, d)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
    except Exception:
        pass

    return metrics_df, fitted_models, stats_df, stats_summary_df, out_info
