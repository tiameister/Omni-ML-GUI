from __future__ import annotations

"""UI adapter for training.

This module keeps the UI-facing `run_training(...)` API stable, but delegates
heavy orchestration to the UI-independent core layer.
"""

from core.training_runner import TrainingCallbacks, run_training as run_training_core


def _resolve_optional_scripts(selected_items: list[str]) -> list[tuple[str, str]]:
    """Return [(label, filename)] for selected optional script options.

    NOTE: The mapping lives in the UI layer (checkbox definitions). The core
    runner receives the resolved list so it doesn't import UI modules.
    """
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

    seen: set[str] = set()
    dedup: list[tuple[str, str]] = []
    for label, filename in resolved:
        if filename in seen:
            continue
        seen.add(filename)
        dedup.append((label, filename))
    return dedup


def run_training(
    state,
    selected_models,
    selected_plots,
    cv_mode="repeated",
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
    shap_settings: dict[str, object] | None = None,
):
    """UI entrypoint: runs training based on current UI state.

    Returns: (metrics_df, fitted_models, stats_df, stats_summary_df, out_info)
    """
    if state is None or getattr(state, "df", None) is None:
        raise RuntimeError("State requires dataframe before training.")
    if getattr(state, "target", None) is None or getattr(state, "features", None) is None:
        raise RuntimeError("State requires target and features before training.")

    callbacks = TrainingCallbacks(
        progress=external_progress_cb,
        plot_progress=external_plot_progress_cb,
        log=external_log_cb,
        should_cancel=should_cancel,
    )

    optional_scripts = _resolve_optional_scripts(list(selected_plots or []))

    return run_training_core(
        df=state.df,
        target=str(state.target),
        features=list(state.features),
        selected_models=list(selected_models or []),
        selected_plots=list(selected_plots or []),
        cv_mode=str(cv_mode or "repeated"),
        cv_folds=int(cv_folds or 5),
        callbacks=callbacks,
        run_id=run_id,
        dataset_label=dataset_label,
        persist_outputs=bool(persist_outputs),
        persist_run_tag=persist_run_tag,
        fe_enabled=bool(getattr(state, "fe_enabled", False)),
        fe_config=getattr(state, "fe_config", None),
        feature_value_labels=feature_value_labels,
        shap_settings=shap_settings,
        optional_scripts=optional_scripts,
    )
