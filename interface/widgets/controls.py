from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QProgressBar, QTextEdit,
    QTabWidget, QScrollArea, QTableWidget, QGroupBox, QFrame, QGridLayout, QListWidget,
    QAbstractItemView, QListWidgetItem
)
from interface.widgets.checkboxes import create_model_checkboxes, create_plot_checkboxes
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from utils.localization import tr


class NoWheelComboBox(QComboBox):
    """Disable wheel changes to avoid accidental value edits while scrolling."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Disable wheel changes to avoid accidental value edits while scrolling."""

    def wheelEvent(self, event):
        event.ignore()


def _make_kpi_card(title: str, value: str):
    card = QFrame()
    card.setObjectName("kpiCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(1)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("kpiTitle")
    value_lbl = QLabel(value)
    value_lbl.setObjectName("kpiValue")

    layout.addWidget(title_lbl)
    layout.addWidget(value_lbl)
    return card, title_lbl, value_lbl


def apply_translations(w):
    """Apply translatable static texts to the control surface."""

    w.load_button.setText(tr("controls.buttons.load_dataset", default="Load Dataset"))
    w.preview_button.setText(tr("controls.buttons.preview", default="Preview"))
    w.info_button.setText(tr("controls.buttons.about_guide", default="About / Guide"))
    w.vars_button.setText(tr("controls.buttons.select_variables", default="Select Variables"))
    w.fe_checkbox.setText(tr("controls.feature_engineering.enable", default="Enable Feature Engineering"))
    w.fe_checkbox.setToolTip(
        tr(
            "controls.feature_engineering.tooltip",
            default="No heavy processing runs on toggle. Engineered features are generated in the background when training starts.",
        )
    )

    current_mode = w.cv_mode_combo.currentData()
    w.cv_mode_combo.blockSignals(True)
    try:
        w.cv_mode_combo.clear()
        w.cv_mode_combo.addItem(tr("controls.validation.repeated_kfold", default="Repeated K-Fold (Recommended)"), "repeated")
        w.cv_mode_combo.addItem(tr("controls.validation.kfold", default="K-Fold"), "kfold")
        w.cv_mode_combo.addItem(tr("controls.validation.nested", default="Nested CV (Thorough)"), "nested")
        w.cv_mode_combo.addItem(tr("controls.validation.holdout", default="Hold-Out (Fast)"), "holdout")
        idx = w.cv_mode_combo.findData(current_mode)
        w.cv_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
    finally:
        w.cv_mode_combo.blockSignals(False)
    w.cv_mode_combo.setToolTip(
        tr(
            "controls.validation.combo_tooltip",
            default="Use click-to-select. Mouse wheel is disabled to prevent accidental changes.",
        )
    )
    w.cv_spin.setToolTip(
        tr(
            "controls.validation.folds_tooltip",
            default="Folds value is changed by arrows or keyboard. Mouse wheel is disabled.",
        )
    )
    w.cv_folds_label.setText(tr("controls.validation.folds", default="Folds:"))
    w.cv_validation_label.setText(tr("controls.validation.label", default="Validation:"))

    w.train_button.setText(tr("controls.buttons.start_queue_training", default="Start / Queue Training"))
    w.persist_output_checkbox.setText(tr("controls.buttons.save_outputs_auto", default="Save outputs automatically"))
    w.persist_output_checkbox.setToolTip(
        tr(
            "controls.buttons.save_outputs_auto_tooltip",
            default="When disabled, outputs are kept temporary and can be saved manually after training.",
        )
    )
    w.cancel_button.setText(tr("controls.buttons.cancel", default="Cancel"))

    w.data_hint_label.setText(tr("controls.workflow.step1_hint", default="Start your workflow by loading a CSV or Excel dataset. The system will parse your data and generate a preview before you continue."))

    w.config_hint_label.setText(
        tr("controls.workflow.step2_hint", default="Assign your Target and Features, and verify cross validation strategies.")
    )

    w.model_hint_label.setText(
        tr("controls.workflow.step3_hint", default="Select which machine learning models you want to include in the evaluation phase.")
    )
    w.open_models_panel_btn.setText(tr("controls.buttons.choose_models", default="Choose Models..."))
    w.open_models_panel_btn.setToolTip(
        tr("controls.workflow.step3_button_tip", default="Open the model selection popup.")
    )

    w.run_hint_label.setText(
        tr("controls.workflow.step4_hint", default="Initialize the training pipeline. Live monitoring and runtime context will stream below.")
    )
    w.customize_plots_btn.setText(tr("controls.buttons.customize_plots", default="Customize Plots..."))
    w.shap_settings_btn.setText(tr("controls.buttons.shap_settings", default="SHAP Settings..."))
    w.open_output_btn.setText(tr("controls.buttons.open_output_folder", default="Open Output Folder"))
    w.reset_session_btn.setText(tr("controls.buttons.reset_session", default="Reset Session"))

    w.progress_title_label.setText(tr("controls.progress.title", default="Execution Monitor"))
    w.progress_train_label.setText(tr("controls.progress.training", default="Training"))
    w.progress_plot_label.setText(tr("controls.progress.plots", default="Plots and analyses"))
    w.feedback_focus_label.setText(
        tr(
            "controls.feedback.template",
            default="Now: {now}\nNext: {next}\nBlockers: {blockers}",
            now=tr("controls.feedback.now_ready", default="Ready"),
            next=tr("controls.feedback.next_load_dataset", default="Load dataset"),
            blockers=tr("controls.feedback.blockers_none", default="None"),
        )
    )

    # Added missing translated labels for initialization/language switch
    w.step1_title.setText(tr("controls.workflow.step1_title", default="Overview & Dataset"))
    w.step2_title.setText(tr("controls.workflow.step2_title", default="Variables & Validation Setup"))
    w.step3_title.setText(tr("controls.workflow.step3_title", default="Model Pool Setup"))
    w.step4_title.setText(tr("controls.workflow.step4_title", default="Execution & Monitoring"))
    
    w.data_info_label.setText(tr("status.no_dataset_loaded", default="No dataset loaded yet."))
    w.selection_label.setText(tr("status.target_not_selected_features_zero", default="Target: not selected | Features: 0"))
    w.model_summary_label.setText(tr("status.no_model_selected", default="No model selected yet."))
    w.progress_phase_label.setText(tr("status.idle", default="Idle"))
    w.progress_timing_label.setText(tr("status.elapsed_eta_default", default="Elapsed: -- | ETA: --"))
    w.status_label.setText(tr("status.ready_begin", default="Ready. Load a dataset to begin."))
    w.results_save_status.setText(tr("results.save_status.not_saved", default="Run not saved"))

    if hasattr(w, "feedback_event_label"):
        w.feedback_event_label.setText(
            tr(
                "controls.feedback.details_template",
                default="Latest: {event}\nJobs: {jobs}",
                event=tr("controls.feedback.latest_none", default="No recent event"),
                jobs=tr("controls.feedback.jobs_idle", default="No active jobs"),
            )
        )

    w.step_tabs.setTabText(0, tr("controls.tabs.step1", default="1. Dataset"))
    w.step_tabs.setTabText(1, tr("controls.tabs.step2", default="2. Variables"))
    w.step_tabs.setTabText(2, tr("controls.tabs.step3", default="3. Models"))
    if w.step_tabs.count() > 3:
        w.step_tabs.setTabText(3, tr("controls.tabs.step4", default="4. Train"))

    w.kpi_dataset_title.setText(tr("controls.kpi.dataset", default="Dataset"))
    w.kpi_target_title.setText(tr("controls.kpi.target", default="Target"))
    w.kpi_run_title.setText(tr("controls.kpi.run", default="Run"))

    w.result_box.setPlaceholderText(
        tr("controls.results.training_summary_placeholder", default="Training summary and model ranking will appear here.")
    )
    w.stats_box.setPlaceholderText(
        tr("controls.results.stats_placeholder", default="Statistical diagnostics will appear here.")
    )
    w.log_box.setPlaceholderText(
        tr("controls.results.logs_placeholder", default="Execution logs will appear here.")
    )
    w.results_empty_label.setText(
        tr(
            "controls.results.empty",
            default="No training result yet. Start training to unlock Results Hub.",
        )
    )
    w.results_save_button.setText(tr("controls.results.save_this_run", default="Save This Run"))
    w.results_save_button.setToolTip(
        tr("controls.results.save_tooltip", default="Persist current temporary run outputs into output/runs.")
    )
    w.results_summary_title.setText(tr("controls.results.summary_title", default="Results Summary"))
    w.results_decision_title.setText(tr("controls.results.decision_title", default="Run Decision Snapshot"))
    w.results_decision_best_label.setText(tr("controls.results.decision_best", default="Best model:"))
    w.results_decision_metrics_label.setText(tr("controls.results.decision_metrics", default="Critical metrics:"))
    w.results_decision_confidence_label.setText(tr("controls.results.decision_confidence", default="Confidence:"))
    w.results_decision_next_label.setText(tr("controls.results.decision_next", default="Next action:"))
    w.results_decision_best_value.setText(tr("common.not_available_short", default="-"))
    w.results_decision_metrics_value.setText(tr("common.not_available_short", default="-"))
    w.results_decision_confidence_value.setText(tr("common.not_available_short", default="-"))
    w.results_decision_next_value.setText(tr("common.not_available_short", default="-"))
    w.results_summary_text.setPlaceholderText(
        tr("controls.results.best_model_placeholder", default="Best model summary will appear here after training.")
    )

    w.figures_model_label.setText(tr("controls.results.filters.model", default="Model:"))
    if w.figures_model_filter.count() > 0 and w.figures_model_filter.itemData(0) == "all":
        w.figures_model_filter.setItemText(0, tr("controls.results.filters.all_models", default="All models"))

    w.figures_group_label.setText(tr("controls.results.filters.group", default="Group:"))
    if w.figures_category_filter.count() > 0 and w.figures_category_filter.itemData(0) == "all":
        w.figures_category_filter.setItemText(0, tr("controls.results.filters.all_groups", default="All groups"))

    w.shap_model_label.setText(tr("controls.results.filters.model", default="Model:"))
    if w.shap_model_filter.count() > 0 and w.shap_model_filter.itemData(0) == "all":
        w.shap_model_filter.setItemText(0, tr("controls.results.filters.all_models", default="All models"))

    w.results_tabs.setTabText(0, tr("controls.results.tabs.summary", default="Summary"))
    w.results_tabs.setTabText(1, tr("controls.results.tabs.tables", default="Tables"))
    w.results_tabs.setTabText(2, tr("controls.results.tabs.figures", default="Figures"))
    w.results_tabs.setTabText(3, tr("controls.results.tabs.shap", default="SHAP"))

    w.jobs_hint_label.setText(
        tr(
            "controls.jobs.hint",
            default="Queued, running and failed jobs are tracked here. Use retry for failed jobs.",
        )
    )
    w.jobs_run_next_btn.setText(tr("controls.jobs.run_next", default="Run Next"))
    w.jobs_retry_failed_btn.setText(tr("controls.jobs.retry_failed", default="Retry Failed"))
    w.jobs_clear_finished_btn.setText(tr("controls.jobs.clear_finished", default="Clear Finished"))
    w.jobs_table.setHorizontalHeaderLabels(
        [
            tr("controls.jobs.columns.id", default="ID"),
            tr("controls.jobs.columns.status", default="Status"),
            tr("controls.jobs.columns.models", default="Models"),
            tr("controls.jobs.columns.cv", default="CV"),
            tr("controls.jobs.columns.created", default="Created"),
            tr("controls.jobs.columns.elapsed", default="Elapsed"),
            tr("controls.jobs.columns.message", default="Message"),
        ]
    )

    w.dev_console_dialog.setWindowTitle(tr("controls.dialogs.dev_console", default="Developer & Activity Console"))
    w.dev_tabs.setTabText(0, tr("controls.dialogs.activity", default="Activity"))
    w.dev_tabs.setTabText(1, tr("controls.dialogs.notifications", default="Notifications"))
    w.dev_tabs.setTabText(2, tr("controls.dialogs.jobs", default="Jobs"))
    w.results_dialog.setWindowTitle(tr("controls.dialogs.results_hub", default="Results Hub"))


def build_layout():
    # Main container and horizontal layout
    w = QWidget()
    main_layout = QHBoxLayout(w)
    main_layout.setContentsMargins(10, 8, 10, 10)
    main_layout.setSpacing(8)

    # Controls shared by center panel
    w.load_button = QPushButton("Load Dataset")
    w.load_button.setObjectName("accentButton")
    # Info row next to Load Data: file info + preview
    info_row = QHBoxLayout()
    w.data_info_label = QLabel("No dataset loaded yet.")
    w.data_info_label.setObjectName("hintLabel")
    w.data_info_label.setWordWrap(True)
    w.preview_button = QPushButton("Preview")
    w.preview_button.setObjectName("ghostButton")
    w.preview_button.setMinimumWidth(92)
    w.preview_button.setEnabled(False)
    info_row.addWidget(w.data_info_label)
    info_row.addStretch()
    info_row.addWidget(w.preview_button)
    w.info_button = QPushButton("About / Guide")
    w.info_button.setObjectName("ghostButton")
    w.vars_button = QPushButton("Select Variables")
    w.vars_button.setObjectName("accentButton")
    w.vars_button.setEnabled(False)
    
    w.studio_btn = QPushButton("Publication Studio")
    w.studio_btn.setObjectName("actionButton")  # We can style it normally
    w.studio_btn.setEnabled(False)
    w.studio_btn.setToolTip("Configure publication-ready names for outputs.")
    
    w.selection_label = QLabel("Target: not selected | Features: 0")
    w.selection_label.setObjectName("hintLabel")
    w.selection_label.setWordWrap(True)
    w.fe_checkbox = QCheckBox("Enable Feature Engineering")
    w.fe_checkbox.setEnabled(False)
    w.fe_checkbox.setToolTip("Create and use engineered features before training.")

    # CV controls
    cv_layout = QGridLayout()
    cv_layout.setHorizontalSpacing(8)
    cv_layout.setVerticalSpacing(6)
    w.cv_mode_combo = NoWheelComboBox()
    w.cv_mode_combo.addItem("Repeated K-Fold (Recommended)", "repeated")
    w.cv_mode_combo.addItem("K-Fold", "kfold")
    w.cv_mode_combo.addItem("Nested CV (Thorough)", "nested")
    w.cv_mode_combo.addItem("Hold-Out (Fast)", "holdout")
    w.cv_mode_combo.setToolTip("Use click-to-select. Mouse wheel is disabled to prevent accidental changes.")
    w.cv_spin = NoWheelSpinBox(); w.cv_spin.setMinimum(2); w.cv_spin.setValue(5)
    w.cv_spin.setToolTip("Folds value is changed by arrows or keyboard. Mouse wheel is disabled.")
    w.cv_folds_label = QLabel("Folds:")
    w.cv_validation_label = QLabel("Validation:")
    cv_layout.addWidget(w.cv_validation_label, 0, 0)
    cv_layout.addWidget(w.cv_mode_combo, 0, 1, 1, 3)
    cv_layout.addWidget(w.cv_folds_label, 1, 2)
    cv_layout.addWidget(w.cv_spin, 1, 3)
    cv_layout.setColumnStretch(1, 1)

    w.train_button = QPushButton("Start / Queue Training")
    w.train_button.setObjectName("trainButton")
    w.persist_output_checkbox = QCheckBox("Save outputs automatically")
    w.persist_output_checkbox.setChecked(False)
    w.persist_output_checkbox.setToolTip("When disabled, outputs are kept temporary and can be saved manually after training.")
    w.cancel_button = QPushButton("Cancel")
    w.cancel_button.setEnabled(False)
    w.cancel_button.setVisible(False)
    w.train_button.setEnabled(False)
    w.progress_bar = QProgressBar(); w.progress_bar.setVisible(False)
    w.plot_progress_bar = QProgressBar(); w.plot_progress_bar.setVisible(False)

    # Checkboxes
    w.model_checks, model_group = create_model_checkboxes()
    w.plot_checks, plot_group = create_plot_checkboxes()
    w.model_summary_label = QLabel("No model selected yet.")
    w.model_summary_label.setObjectName("hintLabel")
    model_group.setVisible(False)

    # Attach hidden widgets to main layout to prevent GC
    main_layout.addWidget(model_group)

    # Center panel: Controls + Plots
    center_panel = QWidget(); center_panel.setObjectName("workPanel")
    center_layout = QVBoxLayout(center_panel)
    center_layout.setContentsMargins(8, 8, 8, 8)
    center_layout.setSpacing(10)

    center_scroll = QScrollArea()
    center_scroll.setObjectName("centerScroll")
    center_scroll.setWidgetResizable(True)
    center_scroll.setFrameShape(QFrame.Shape.NoFrame)
    center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    center_content = QWidget()
    center_content.setObjectName("centerContent")
    center_content_layout = QVBoxLayout(center_content)
    center_content_layout.setContentsMargins(0, 0, 0, 0)
    center_content_layout.setSpacing(10)

    w.data_card = QFrame()
    w.data_card.setObjectName("workflowCard")
    data_card_layout = QVBoxLayout(w.data_card)
    data_card_layout.setContentsMargins(24, 24, 24, 24)
    data_card_layout.setSpacing(8)
    
    w.step1_title = QLabel("Overview & Dataset")
    w.step1_title.setObjectName("sectionTitle")
    data_card_layout.addWidget(w.step1_title)
    
    w.data_hint_label = QLabel("Start your workflow by loading a CSV or Excel dataset. The system will parse your data and generate a preview before you continue.")
    w.data_hint_label.setObjectName("hintLabel")
    w.data_hint_label.setWordWrap(True)
    data_card_layout.addWidget(w.data_hint_label)
    
    data_card_layout.addSpacing(16)
    
    action_row = QHBoxLayout()
    action_row.addWidget(w.load_button)
    action_row.addStretch()
    data_card_layout.addLayout(action_row)
    
    data_card_layout.addSpacing(12)
    
    w.info_card = QFrame()
    w.info_card.setObjectName("summaryCard")
    info_card_layout = QVBoxLayout(w.info_card)
    info_card_layout.setContentsMargins(16, 16, 16, 16)
    info_card_layout.addLayout(info_row)
    
    data_card_layout.addWidget(w.info_card)

    w.config_card = QFrame()
    w.config_card.setObjectName("workflowCard")
    config_card_layout = QVBoxLayout(w.config_card)
    config_card_layout.setContentsMargins(24, 24, 24, 24)
    config_card_layout.setSpacing(8)
    
    w.step2_title = QLabel("Variables & Validation Setup")
    w.step2_title.setObjectName("sectionTitle")
    config_card_layout.addWidget(w.step2_title)
    
    w.config_hint_label = QLabel("Assign your Target and Features, and verify cross validation strategies.")
    w.config_hint_label.setObjectName("hintLabel")
    w.config_hint_label.setWordWrap(True)
    config_card_layout.addWidget(w.config_hint_label)
    
    config_card_layout.addSpacing(16)
    
    vars_row = QHBoxLayout()
    vars_row.setSpacing(12)
    vars_row.addWidget(w.vars_button)
    vars_row.addWidget(w.studio_btn)
    vars_row.addStretch()
    
    config_card_layout.addLayout(vars_row)
    
    config_card_layout.addSpacing(12)
    
    w.config_summary_card = QFrame()
    w.config_summary_card.setObjectName("summaryCard")
    config_summary_layout = QVBoxLayout(w.config_summary_card)
    config_summary_layout.setContentsMargins(16, 16, 16, 16)
    config_summary_layout.setSpacing(12)
    
    config_summary_layout.addWidget(w.selection_label)
    config_summary_layout.addWidget(w.fe_checkbox)
    
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    separator.setStyleSheet("border: none; border-top: 1px solid #EBEBEE;")
    config_summary_layout.addWidget(separator)
    
    config_summary_layout.addLayout(cv_layout)
    
    config_card_layout.addWidget(w.config_summary_card)

    w.model_card = QFrame()
    w.model_card.setObjectName("workflowCard")
    model_card_layout = QVBoxLayout(w.model_card)
    model_card_layout.setContentsMargins(24, 24, 24, 24)
    model_card_layout.setSpacing(8)
    
    w.step3_title = QLabel("Model Pool Setup")
    w.step3_title.setObjectName("sectionTitle")
    model_card_layout.addWidget(w.step3_title)
    
    w.model_hint_label = QLabel("Select which machine learning models you want to include in the evaluation phase.")
    w.model_hint_label.setObjectName("hintLabel")
    w.model_hint_label.setWordWrap(True)
    model_card_layout.addWidget(w.model_hint_label)
    
    model_card_layout.addSpacing(16)
    
    w.open_models_panel_btn = QPushButton("Choose Models...")
    w.open_models_panel_btn.setObjectName("accentButton")
    w.open_models_panel_btn.setToolTip("Open the model selection popup.")
    w.open_models_panel_btn.setEnabled(False)
    
    model_action_row = QHBoxLayout()
    model_action_row.addWidget(w.open_models_panel_btn)
    model_action_row.addStretch()
    model_card_layout.addLayout(model_action_row)
    
    model_card_layout.addSpacing(12)
    
    w.model_badge_card = QFrame()
    w.model_badge_card.setObjectName("badgeCard")
    model_badge_layout = QHBoxLayout(w.model_badge_card)
    model_badge_layout.setContentsMargins(12, 10, 12, 10)
    w.model_summary_label.setObjectName("badgeLabel")
    model_badge_layout.addWidget(w.model_summary_label)
    model_badge_layout.addStretch()
    model_card_layout.addWidget(w.model_badge_card)

    w.run_card = QFrame()
    w.run_card.setObjectName("workflowCard")
    run_card_layout = QVBoxLayout(w.run_card)
    run_card_layout.setContentsMargins(24, 24, 24, 24)
    run_card_layout.setSpacing(8)
    
    w.step4_title = QLabel("Execution & Monitoring")
    w.step4_title.setObjectName("sectionTitle")
    run_card_layout.addWidget(w.step4_title)
    
    w.run_hint_label = QLabel("Initialize the training pipeline. Live monitoring and runtime context will stream below.")
    w.run_hint_label.setObjectName("hintLabel")
    w.run_hint_label.setWordWrap(True)
    run_card_layout.addWidget(w.run_hint_label)

    run_card_layout.addSpacing(16)

    row_train = QHBoxLayout()
    row_train.addWidget(w.train_button)
    row_train.setSpacing(12)
    row_train.addWidget(w.train_button)
    row_train.addWidget(w.cancel_button)
    row_train.addStretch()
    run_card_layout.addLayout(row_train)
    
    run_card_layout.addSpacing(12)

    w.run_summary_card = QFrame()
    w.run_summary_card.setObjectName("summaryCard")    
    run_summary_layout = QVBoxLayout(w.run_summary_card)
    run_summary_layout.setContentsMargins(16, 16, 16, 16)
    run_summary_layout.setSpacing(8)
    
    run_summary_layout.addWidget(w.persist_output_checkbox)
    
    w.runtime_hint_label = QLabel("Estimated runtime: waiting for model selection")
    w.runtime_hint_label.setObjectName("hintLabel")
    run_summary_layout.addWidget(w.runtime_hint_label)
    
    w.plot_summary_label = QLabel("")
    w.plot_summary_label.setObjectName("hintLabel")
    w.plot_summary_label.setWordWrap(True)
    run_summary_layout.addWidget(w.plot_summary_label)

    run_card_layout.addWidget(w.run_summary_card)

    # Hidden action buttons: exposed via top menu for a cleaner workflow surface
    w.customize_plots_btn = QPushButton("Customize Plots…")
    w.shap_settings_btn = QPushButton("SHAP Settings…")
    w.customize_plots_btn.setVisible(False)
    w.shap_settings_btn.setVisible(False)

    w.open_output_btn = QPushButton("Open Output Folder")
    w.open_output_btn.setObjectName("ghostButton")
    w.reset_session_btn = QPushButton("Reset Session")
    w.reset_session_btn.setObjectName("ghostButton")
    w.open_output_btn.setVisible(False)
    w.reset_session_btn.setVisible(False)

    # Progress and status (hidden until used)
    w.progress_panel = QFrame()
    w.progress_panel.setObjectName("progressPanel")
    progress_layout = QVBoxLayout(w.progress_panel)
    progress_layout.setContentsMargins(10, 8, 10, 8)
    progress_layout.setSpacing(6)

    top_row = QHBoxLayout()
    w.progress_title_label = QLabel("Execution Monitor")
    w.progress_title_label.setObjectName("progressTitle")
    w.progress_phase_label = QLabel("Idle")
    w.progress_phase_label.setObjectName("hintLabel")
    top_row.addWidget(w.progress_title_label)
    top_row.addStretch()
    top_row.addWidget(w.progress_phase_label)
    progress_layout.addLayout(top_row)

    train_row = QHBoxLayout()
    w.progress_train_label = QLabel("Training")
    w.progress_train_label.setObjectName("hintLabel")
    w.progress_train_stats_label = QLabel("0/0 (0%)")
    w.progress_train_stats_label.setObjectName("progressValue")
    train_row.addWidget(w.progress_train_label)
    train_row.addStretch()
    train_row.addWidget(w.progress_train_stats_label)
    progress_layout.addLayout(train_row)

    w.progress_bar.setObjectName("trainProgressBar")
    w.progress_bar.setTextVisible(False)
    progress_layout.addWidget(w.progress_bar)

    plot_row = QHBoxLayout()
    w.progress_plot_label = QLabel("Plots and analyses")
    w.progress_plot_label.setObjectName("hintLabel")
    w.progress_plot_stats_label = QLabel("0/0 (0%)")
    w.progress_plot_stats_label.setObjectName("progressValue")
    plot_row.addWidget(w.progress_plot_label)
    plot_row.addStretch()
    plot_row.addWidget(w.progress_plot_stats_label)
    progress_layout.addLayout(plot_row)

    w.plot_progress_bar.setObjectName("plotProgressBar")
    w.plot_progress_bar.setTextVisible(False)
    progress_layout.addWidget(w.plot_progress_bar)

    w.progress_timing_label = QLabel("Elapsed: -- | ETA: --")
    w.progress_timing_label.setObjectName("hintLabel")
    progress_layout.addWidget(w.progress_timing_label)

    w.progress_panel.setVisible(False)
    run_card_layout.addWidget(w.progress_panel)

    w.status_label = QLabel("Ready. Load a dataset to begin.")
    w.status_label.setObjectName("statusPill")
    w.status_label.setWordWrap(True)
    w.status_label.setVisible(True)
    run_card_layout.addWidget(w.status_label)

    w.feedback_focus_label = QLabel("Now: Ready\nNext: Load dataset\nBlockers: None")
    w.feedback_focus_label.setObjectName("hintLabel")
    w.feedback_focus_label.setWordWrap(True)
    run_card_layout.addWidget(w.feedback_focus_label)

    w.feedback_event_label = QLabel("Latest: No recent event\nJobs: No active jobs")
    w.feedback_event_label.setObjectName("hintLabel")
    w.feedback_event_label.setWordWrap(True)
    run_card_layout.addWidget(w.feedback_event_label)

    w.step_tabs = QTabWidget()
    w.step_tabs.setObjectName("workflowTabs")
    w.step_tabs.setDocumentMode(True)

    step_data = QWidget(); step_data_lay = QVBoxLayout(step_data)
    step_data_lay.setContentsMargins(0, 0, 0, 0)
    step_data_lay.setSpacing(8)
    step_data_lay.addWidget(w.data_card)
    step_data_lay.addStretch()

    step_config = QWidget(); step_config_lay = QVBoxLayout(step_config)
    step_config_lay.setContentsMargins(0, 0, 0, 0)
    step_config_lay.setSpacing(8)
    step_config_lay.addWidget(w.config_card)
    step_config_lay.addStretch()

    step_model = QWidget(); step_model_lay = QVBoxLayout(step_model)
    step_model_lay.setContentsMargins(0, 0, 0, 0)
    step_model_lay.setSpacing(8)
    step_model_lay.addWidget(w.model_card)
    step_model_lay.addStretch()

    step_train = QWidget(); step_train_lay = QVBoxLayout(step_train)
    step_train_lay.setContentsMargins(0, 0, 0, 0)
    step_train_lay.setSpacing(8)
    step_train_lay.addWidget(w.run_card)
    step_train_lay.addStretch()

    w.step_tabs.addTab(step_data, "1. Dataset")
    w.step_tabs.addTab(step_config, "2. Variables")
    w.step_tabs.addTab(step_model, "3. Models")
    w.step_tabs.addTab(step_train, "4. Train")
    center_content_layout.addWidget(w.step_tabs)

    # Keep the toolbox instance alive by adding it to the layout but hidden; the dialog owns the detailed UI
    center_content_layout.addWidget(plot_group)
    plot_group.setVisible(False)
    center_content_layout.addStretch()

    center_scroll.setWidget(center_content)
    center_layout.addWidget(center_scroll)
    w.center_scroll = center_scroll

    # Right panel: Output tabs
    right_panel = QWidget(); right_panel.setObjectName("resultPanel")
    right_layout = QVBoxLayout(right_panel)

    kpi_row = QHBoxLayout()
    ds_card, w.kpi_dataset_title, w.kpi_dataset_value = _make_kpi_card("Dataset", "Not loaded")
    tg_card, w.kpi_target_title, w.kpi_target_value = _make_kpi_card("Target", "Not selected")
    rn_card, w.kpi_run_title, w.kpi_run_value = _make_kpi_card("Run", "Idle")
    kpi_row.addWidget(ds_card)
    kpi_row.addWidget(tg_card)
    kpi_row.addWidget(rn_card)
    right_layout.addLayout(kpi_row)

    # Results and statistics text areas and tables
    w.result_box = QTextEdit(); w.result_box.setReadOnly(True)
    w.result_box.setPlaceholderText("Training summary and model ranking will appear here.")
    w.stats_box = QTextEdit(); w.stats_box.setReadOnly(True)
    w.stats_box.setPlaceholderText("Statistical diagnostics will appear here.")
    w.log_box = QTextEdit(); w.log_box.setReadOnly(True)
    w.log_box.setPlaceholderText("Execution logs will appear here.")
    # Compact tables
    w.metrics_table = QTableWidget(); w.metrics_table.setObjectName("metricsTable")
    w.stats_table = QTableWidget(); w.stats_table.setObjectName("statsTable")
    w.metrics_table.setSortingEnabled(True)
    w.stats_table.setSortingEnabled(True)
    w.metrics_table.setAlternatingRowColors(True)
    w.stats_table.setAlternatingRowColors(True)
    w.metrics_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    w.stats_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    w.metrics_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    w.stats_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    w.metrics_table.verticalHeader().setVisible(False)
    w.stats_table.verticalHeader().setVisible(False)
    w.metrics_table.horizontalHeader().setStretchLastSection(True)
    w.stats_table.horizontalHeader().setStretchLastSection(True)
    # Monospace font and no wrap for readability
    mono = QFont("Consolas", 10)
    w.result_box.setFont(mono)
    w.result_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    w.stats_box.setFont(mono)
    w.stats_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    tabs = QTabWidget()
    tabs.setDocumentMode(True)

    # Results Hub
    results_tab = QWidget()
    results_layout = QVBoxLayout(results_tab)
    results_layout.setContentsMargins(0, 0, 0, 0)
    results_layout.setSpacing(8)

    w.results_empty_label = QLabel("No training result yet. Start training to unlock Results Hub.")
    w.results_empty_label.setObjectName("hintLabel")
    w.results_empty_label.setWordWrap(True)
    results_layout.addWidget(w.results_empty_label)

    w.results_save_row = QWidget()
    save_row_layout = QHBoxLayout(w.results_save_row)
    save_row_layout.setContentsMargins(0, 0, 0, 0)
    save_row_layout.setSpacing(6)
    w.results_save_button = QPushButton("Save This Run")
    w.results_save_button.setObjectName("accentButton")
    w.results_save_button.setToolTip("Persist current temporary run outputs into output/runs.")
    w.results_save_status = QLabel("Run not saved")
    w.results_save_status.setObjectName("hintLabel")
    w.results_save_status.setWordWrap(True)
    save_row_layout.addWidget(w.results_save_button)
    save_row_layout.addWidget(w.results_save_status, 1)
    results_layout.addWidget(w.results_save_row)

    w.results_decision_card = QFrame()
    w.results_decision_card.setObjectName("decisionCard")
    decision_layout = QGridLayout(w.results_decision_card)
    decision_layout.setContentsMargins(10, 8, 10, 8)
    decision_layout.setHorizontalSpacing(10)
    decision_layout.setVerticalSpacing(4)
    w.results_decision_title = QLabel("Run Decision Snapshot")
    w.results_decision_title.setObjectName("sectionTitle")
    w.results_decision_best_label = QLabel("Best model:")
    w.results_decision_best_value = QLabel("-")
    w.results_decision_metrics_label = QLabel("Critical metrics:")
    w.results_decision_metrics_value = QLabel("-")
    w.results_decision_confidence_label = QLabel("Confidence:")
    w.results_decision_confidence_value = QLabel("-")
    w.results_decision_next_label = QLabel("Next action:")
    w.results_decision_next_value = QLabel("-")

    decision_layout.addWidget(w.results_decision_title, 0, 0, 1, 2)
    decision_layout.addWidget(w.results_decision_best_label, 1, 0)
    decision_layout.addWidget(w.results_decision_best_value, 1, 1)
    decision_layout.addWidget(w.results_decision_metrics_label, 2, 0)
    decision_layout.addWidget(w.results_decision_metrics_value, 2, 1)
    decision_layout.addWidget(w.results_decision_confidence_label, 3, 0)
    decision_layout.addWidget(w.results_decision_confidence_value, 3, 1)
    decision_layout.addWidget(w.results_decision_next_label, 4, 0)
    decision_layout.addWidget(w.results_decision_next_value, 4, 1)
    results_layout.addWidget(w.results_decision_card)

    w.results_tabs = QTabWidget()
    w.results_tabs.setDocumentMode(True)

    summary_tab = QWidget()
    summary_layout = QVBoxLayout(summary_tab)
    summary_layout.setContentsMargins(0, 0, 0, 0)
    summary_layout.setSpacing(6)
    w.results_summary_title = QLabel("Results Summary")
    w.results_summary_title.setObjectName("sectionTitle")

    w.results_summary_text = QTextEdit()
    w.results_summary_text.setReadOnly(True)
    w.results_summary_text.setPlaceholderText("Best model summary will appear here after training.")
    summary_layout.addWidget(w.results_summary_title)
    summary_layout.addWidget(w.results_summary_text)

    tables_tab = QWidget()
    tables_layout = QVBoxLayout(tables_tab)
    tables_layout.setContentsMargins(0, 0, 0, 0)
    tables_layout.setSpacing(6)
    tables_layout.addWidget(w.metrics_table)
    tables_layout.addWidget(w.stats_table)

    figures_tab = QWidget()
    figures_layout = QVBoxLayout(figures_tab)
    figures_layout.setContentsMargins(0, 0, 0, 0)
    figures_layout.setSpacing(6)
    figures_filter_row = QHBoxLayout()
    w.figures_model_label = QLabel("Model:")
    figures_filter_row.addWidget(w.figures_model_label)
    w.figures_model_filter = NoWheelComboBox()
    w.figures_model_filter.addItem("All models", userData="all")
    figures_filter_row.addWidget(w.figures_model_filter, 1)
    w.figures_group_label = QLabel("Group:")
    figures_filter_row.addWidget(w.figures_group_label)
    w.figures_category_filter = NoWheelComboBox()
    w.figures_category_filter.addItem("All groups", userData="all")
    figures_filter_row.addWidget(w.figures_category_filter, 1)
    figures_layout.addLayout(figures_filter_row)
    w.figures_list = QListWidget()
    w.figures_list.setObjectName("resultsFigureList")
    w.figures_list.setAlternatingRowColors(True)
    w.figures_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    w.figures_img = QLabel()
    w.figures_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w.figures_img.setMinimumHeight(220)
    w.figures_img.setScaledContents(False)
    figures_layout.addWidget(w.figures_list)
    figures_layout.addWidget(w.figures_img)

    shap_tab = QWidget()
    shap_layout = QVBoxLayout(shap_tab)
    shap_layout.setContentsMargins(0, 0, 0, 0)
    shap_layout.setSpacing(6)
    shap_filter_row = QHBoxLayout()
    w.shap_model_label = QLabel("Model:")
    shap_filter_row.addWidget(w.shap_model_label)
    w.shap_model_filter = NoWheelComboBox()
    w.shap_model_filter.addItem("All models", userData="all")
    shap_filter_row.addWidget(w.shap_model_filter, 1)
    shap_layout.addLayout(shap_filter_row)
    w.shap_list = QListWidget()
    w.shap_list.setObjectName("resultsShapList")
    w.shap_list.setAlternatingRowColors(True)
    w.shap_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    w.shap_img = QLabel()
    w.shap_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w.shap_img.setMinimumHeight(220)
    w.shap_img.setScaledContents(False)
    shap_layout.addWidget(w.shap_list)
    shap_layout.addWidget(w.shap_img)

    w.results_tabs.addTab(summary_tab, "Summary")
    w.results_tabs.addTab(tables_tab, "Tables")
    w.results_tabs.addTab(figures_tab, "Figures")
    w.results_tabs.addTab(shap_tab, "SHAP")
    results_layout.addWidget(w.results_tabs)

    console_tab = QWidget(); ct_layout = QVBoxLayout(console_tab); ct_layout.addWidget(w.log_box)
    notifications_tab = QWidget(); nt_layout = QVBoxLayout(notifications_tab)
    w.notifications_list = QListWidget()
    w.notifications_list.setObjectName("notificationsList")
    w.notifications_list.setAlternatingRowColors(True)
    w.notifications_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
    nt_layout.addWidget(w.notifications_list)

    jobs_tab = QWidget(); jt_layout = QVBoxLayout(jobs_tab)
    w.jobs_hint_label = QLabel("Queued, running and failed jobs are tracked here. Use retry for failed jobs.")
    w.jobs_hint_label.setObjectName("hintLabel")
    w.jobs_hint_label.setWordWrap(True)
    jt_layout.addWidget(w.jobs_hint_label)

    jobs_actions_row = QHBoxLayout()
    w.jobs_run_next_btn = QPushButton("Run Next")
    w.jobs_run_next_btn.setObjectName("ghostButton")
    w.jobs_retry_failed_btn = QPushButton("Retry Failed")
    w.jobs_retry_failed_btn.setObjectName("ghostButton")
    w.jobs_clear_finished_btn = QPushButton("Clear Finished")
    w.jobs_clear_finished_btn.setObjectName("ghostButton")
    jobs_actions_row.addWidget(w.jobs_run_next_btn)
    jobs_actions_row.addWidget(w.jobs_retry_failed_btn)
    jobs_actions_row.addWidget(w.jobs_clear_finished_btn)
    jobs_actions_row.addStretch()
    jt_layout.addLayout(jobs_actions_row)

    w.jobs_table = QTableWidget()
    w.jobs_table.setObjectName("jobsTable")
    w.jobs_table.setColumnCount(7)
    w.jobs_table.setHorizontalHeaderLabels(["ID", "Status", "Models", "CV", "Created", "Elapsed", "Message"])
    w.jobs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    w.jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    w.jobs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    w.jobs_table.setAlternatingRowColors(True)
    w.jobs_table.verticalHeader().setVisible(False)
    w.jobs_table.horizontalHeader().setStretchLastSection(True)
    jt_layout.addWidget(w.jobs_table)
    w.results_tab = results_tab
    w.notifications_tab = notifications_tab
    w.jobs_tab = jobs_tab
    right_layout.addWidget(results_tab)
    
    from PyQt6.QtWidgets import QDialog
    w.dev_console_dialog = QDialog(w)
    w.dev_console_dialog.setWindowTitle(tr("controls.dialogs.dev_console", default="Developer & Activity Console"))
    w.dev_console_dialog.resize(800, 600)
    dev_layout = QVBoxLayout(w.dev_console_dialog)
    w.dev_tabs = QTabWidget()
    w.dev_tabs.addTab(console_tab, "Activity")
    w.dev_tabs.addTab(notifications_tab, "Notifications")
    w.dev_tabs.addTab(jobs_tab, "Jobs")
    dev_layout.addWidget(w.dev_tabs)


    # Assemble main layout
    center_panel.setMinimumWidth(360)
    w.center_panel = center_panel
    main_layout.addWidget(center_panel)

    # Convert right_panel to a standalone dialog
    w.results_dialog = QDialog(w)
    w.results_dialog.setWindowTitle(tr("controls.dialogs.results_hub", default="Results Hub"))
    w.results_dialog.resize(900, 700)
    dialog_layout = QVBoxLayout(w.results_dialog)
    dialog_layout.setContentsMargins(0, 0, 0, 0)
    dialog_layout.addWidget(right_panel)

    apply_translations(w)
    w.apply_translations = lambda: apply_translations(w)
    return w

