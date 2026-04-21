from interface.widgets.apple_helpers import create_apple_settings_row
from PySide6.QtWidgets import (
    QApplication,
    QStyle,
    QTableView,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QProgressBar, QTextEdit,
    QTabWidget, QScrollArea, QTableWidget, QFrame, QGridLayout, QListWidget, QAbstractItemView, QSizePolicy, QAbstractSpinBox, QFormLayout, QStackedWidget
)
from interface.widgets.checkboxes import create_model_checkboxes, create_plot_checkboxes
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QFontDatabase
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
    w.run_hint_label.setText(
        tr("controls.workflow.step4_hint", default="Start training to populate Summary, Tables, Figures and SHAP.")
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
    w.step2_title.setText(tr("controls.workflow.step2_title", default="Variables"))
    w.step3_title.setText(tr("controls.workflow.step3_title", default="Models"))
    w.step4_title.setText(tr("controls.workflow.step4_title", default="Train"))
    
    if hasattr(w, "data_empty_title"):
        w.data_empty_title.setText(tr("controls.dataset.empty_title", default="No dataset loaded"))
    if hasattr(w, "data_empty_subtitle"):
        w.data_empty_subtitle.setText(
            tr(
                "controls.dataset.empty_subtitle",
                default="Load a CSV or Excel file to begin.",
            )
        )
    w.data_info_label.setText(tr("status.no_dataset_loaded", default="No dataset loaded yet."))
    w.selection_label.setText(tr("status.variables_pending", default="0 Features Selected (Target pending)"))
    try:
        w.model_summary_label.setText(
            tr(
                "status.training_queue_header",
                default="Training Queue ({selected}/{total})",
                selected=0,
                total=len(getattr(w, "model_checks", {}) or {}),
            )
        )
    except Exception as e:
        import logging
        logging.warning(f"Failed to set model_summary_label: {e}")
        w.model_summary_label.setText(tr("status.no_model_selected", default="No model selected yet."))
    w.progress_phase_label.setText(tr("status.idle", default="Idle"))
    w.progress_timing_label.setText(tr("status.elapsed_eta_default", default="Elapsed: -- | ETA: --"))
    # Mid-page status line removed (footer already communicates state).
    try:
        w.status_label.setText("")
        w.status_label.setVisible(False)
    except Exception as e:
        import logging
        logging.debug(f"Failed to set status_label: {e}")

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

    w.log_box.setPlaceholderText(
        tr("controls.results.logs_placeholder", default="Execution logs will appear here.")
    )
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


def build_layout():
    # Main container (row + footer)
    w = QWidget()
    outer_layout = QVBoxLayout(w)
    # Full-bleed footer: remove global margins; individual panels own padding.
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(0)

    content_row = QWidget()
    main_layout = QHBoxLayout(content_row)
    main_layout.setContentsMargins(16, 16, 16, 16)
    main_layout.setSpacing(16)
    outer_layout.addWidget(content_row, 1)

    # Controls shared by center panel
    w.load_button = QPushButton("Load Dataset")
    w.load_button.setObjectName("accentButton")

    # Add basic icons (uses platform-native glyphs; can be swapped to SVG later)
    try:
        w.load_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        w.load_button.setIconSize(QSize(16, 16))
    except Exception:
        pass

    w.data_info_label = QLabel("No dataset loaded yet.")
    w.data_info_label.setObjectName("hintLabel")
    w.data_info_label.setWordWrap(True)

    w.preview_button = QPushButton("Preview")
    w.preview_button.setObjectName("ghostButton")
    w.preview_button.setMinimumWidth(92)
    w.preview_button.setEnabled(False)
    try:
        w.preview_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        w.preview_button.setIconSize(QSize(16, 16))
    except Exception as e:
        import logging
        logging.debug(f"Failed to set preview_button icon: {e}")

    w.info_button = QPushButton("About / Guide")
    w.info_button.setObjectName("ghostButton")
    w.vars_button = QPushButton("Select Variables")
    w.vars_button.setObjectName("accentButton")
    w.vars_button.setEnabled(False)
    
    w.studio_btn = QPushButton("Publication Studio")
    w.studio_btn.setObjectName("actionButton")
    w.studio_btn.setEnabled(False)
    w.studio_btn.setToolTip("Select variables to unlock Publication Studio.")
    
    # We will alias selection_label to the subtitle object later so the badge goes into the text organically.
    w.fe_checkbox = QCheckBox("Enable Feature Engineering")
    w.fe_checkbox.setEnabled(False)
    w.fe_checkbox.setToolTip("Create and use engineered features before training.")

    # CV controls (compact, card-friendly)
    w.cv_mode_combo = NoWheelComboBox()
    w.cv_mode_combo.addItem("Repeated K-Fold (Recommended)", "repeated")
    w.cv_mode_combo.addItem("K-Fold", "kfold")
    w.cv_mode_combo.addItem("Nested CV (Thorough)", "nested")
    w.cv_mode_combo.addItem("Hold-Out (Fast)", "holdout")
    w.cv_mode_combo.setToolTip("Use click-to-select. Mouse wheel is disabled to prevent accidental changes.")
    w.cv_mode_combo.setObjectName("cvMethodCombo")
    w.cv_mode_combo.setMinimumWidth(250)
    w.cv_mode_combo.setMaximumWidth(300)
    w.cv_spin = NoWheelSpinBox(); w.cv_spin.setMinimum(2); w.cv_spin.setValue(5)
    w.cv_spin.setObjectName("cvFoldsSpin")
    try:
        w.cv_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    except Exception:
        pass
    w.cv_spin.setMinimumWidth(60)
    w.cv_spin.setMaximumWidth(60)
    w.cv_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w.cv_spin.setToolTip("Folds value is changed by arrows or keyboard. Mouse wheel is disabled.")
    w.cv_spin.setObjectName("cvFoldsSpin")
    w.cv_folds_label = QLabel("Folds:")
    w.cv_validation_label = QLabel("Validation:")

    w.train_button = QPushButton("Start / Queue Training")
    w.train_button.setObjectName("trainButton")
    w.persist_output_checkbox = QCheckBox("Save outputs automatically")
    w.persist_output_checkbox.setObjectName("persistOutputCheckbox")
    w.persist_output_checkbox.setChecked(False)
    w.persist_output_checkbox.setToolTip("When disabled, outputs are kept temporary and can be saved manually after training.")
    w.cancel_button = QPushButton("Cancel")
    w.cancel_button.setEnabled(False)
    w.cancel_button.setVisible(False)
    w.train_button.setEnabled(False)
    w.progress_bar = QProgressBar(); w.progress_bar.setVisible(False)
    w.plot_progress_bar = QProgressBar(); w.plot_progress_bar.setVisible(False)

    # Model picker (inline)
    w.model_checks, model_group = create_model_checkboxes()
    w.model_picker = model_group
    w.plot_checks, plot_group = create_plot_checkboxes()
    w.model_summary_label = QLabel("No model selected yet.")
    w.model_summary_label.setObjectName("hintLabel")
    # Disabled until variables are selected (enabled by controller logic)
    model_group.setEnabled(False)

    # Center panel: Controls + Plots
    center_panel = QWidget(); center_panel.setObjectName("workPanel")
    center_layout = QVBoxLayout(center_panel)
    center_layout.setContentsMargins(16, 16, 16, 16)
    center_layout.setSpacing(16)

    center_scroll = QScrollArea()
    center_scroll.setObjectName("centerScroll")
    center_scroll.setWidgetResizable(True)
    center_scroll.setFrameShape(QFrame.Shape.NoFrame)
    center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    center_content = QWidget()
    center_content.setObjectName("centerContent")
    center_content_layout = QVBoxLayout(center_content)
    center_content_layout.setContentsMargins(0, 0, 0, 0)
    center_content_layout.setSpacing(16)

    w.data_card = QFrame()
    w.data_card.setObjectName("workflowCard")
    w.data_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
    action_row.setSpacing(8)
    action_row.addWidget(w.load_button)
    action_row.addWidget(w.preview_button)
    action_row.addStretch(1)
    data_card_layout.addLayout(action_row)
    
    data_card_layout.addSpacing(12)
    
    w.info_card = QFrame()
    w.info_card.setObjectName("summaryCard")
    info_card_layout = QVBoxLayout(w.info_card)
    info_card_layout.setContentsMargins(16, 16, 16, 16)

    # Empty state (shown until a dataset is loaded)
    w.data_empty_state = QWidget()
    w.data_empty_state.setObjectName("emptyState")
    empty_layout = QVBoxLayout(w.data_empty_state)
    empty_layout.setContentsMargins(0, 0, 0, 0)
    empty_layout.setSpacing(8)

    w.data_empty_icon = QLabel()
    w.data_empty_icon.setObjectName("emptyStateIcon")
    w.data_empty_icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    try:
        pix = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon).pixmap(48, 48)
        w.data_empty_icon.setPixmap(pix)
    except Exception as e:
        import logging
        logging.debug(f"Failed to set data_empty_icon pixmap: {e}")

    w.data_empty_title = QLabel("No dataset loaded")
    w.data_empty_title.setObjectName("emptyStateTitle")
    w.data_empty_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    w.data_empty_subtitle = QLabel("Load a CSV or Excel file to begin.")
    w.data_empty_subtitle.setObjectName("emptyStateSubtitle")
    w.data_empty_subtitle.setWordWrap(True)
    w.data_empty_subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    empty_layout.addWidget(w.data_empty_icon)
    empty_layout.addWidget(w.data_empty_title)
    empty_layout.addWidget(w.data_empty_subtitle)

    # Loaded state (shown after a dataset is loaded)
    w.data_loaded_state = QWidget()
    w.data_loaded_state.setObjectName("loadedState")
    loaded_layout = QHBoxLayout(w.data_loaded_state)
    loaded_layout.setContentsMargins(0, 0, 0, 0)
    loaded_layout.setSpacing(8)
    loaded_layout.addWidget(w.data_info_label, 1)
    w.data_loaded_state.setVisible(False)

    info_card_layout.addWidget(w.data_empty_state)
    info_card_layout.addWidget(w.data_loaded_state)
    
    data_card_layout.addWidget(w.info_card)
    data_card_layout.addStretch(1)

    w.config_card = QFrame()
    w.config_card.setObjectName("workflowCard")
    w.config_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
    
    # Variables card
    w.variables_card = QFrame()
    w.variables_card.setObjectName("appleCard")
    variables_layout = QVBoxLayout(w.variables_card)
    variables_layout.setContentsMargins(0, 0, 0, 0)
    variables_layout.setSpacing(0)

    # 1. Row: Select Variables
    row1, w.vars_target_title, w.vars_target_subtitle = create_apple_settings_row(
        right_widget=w.vars_button,
        title_text=tr("controls.variables.row1_title", default="Target & Features"),
        subtitle_text=tr("controls.variables.row1_subtitle", default="Requires dataset to be loaded first"),
        show_bottom_line=True
    )
    
    # Selection badge goes into the subtitle dynamically
    w.selection_label = w.vars_target_subtitle

    variables_layout.addWidget(row1)

    # 2. Row: Publication Studio
    row2, w.vars_studio_title, w.vars_studio_subtitle = create_apple_settings_row(
        right_widget=w.studio_btn,
        title_text=tr("controls.variables.row2_title", default="Publication Studio"),
        subtitle_text=tr("controls.variables.row2_subtitle", default="Configure professional names for reports"),
        show_bottom_line=True
    )
    variables_layout.addWidget(row2)

    # 3. Row: Feature Engineering
    w.fe_checkbox.setText("")  # Remove text, make it a simple toggle
    w.fe_checkbox.setObjectName("toggleSwitch")
    
    # We will add a Setup/Config button below it or next to it.
    w.fe_setup_btn = QPushButton("Customize...")
    w.fe_setup_btn.setObjectName("actionButton")
    w.fe_setup_btn.setMinimumWidth(80)
    w.fe_setup_btn.setVisible(False) # show when checked
    
    fe_controls_lay = QHBoxLayout()
    fe_controls_lay.setContentsMargins(0,0,0,0)
    fe_controls_lay.addWidget(w.fe_setup_btn)
    fe_controls_lay.addWidget(w.fe_checkbox)
    
    fe_control_widget = QWidget()
    fe_control_widget.setLayout(fe_controls_lay)
    
    row3, w.fe_title, w.fe_subtitle = create_apple_settings_row(
        right_widget=fe_control_widget,
        title_text=tr("controls.variables.row3_title", default="Feature Engineering"),
        subtitle_text=tr("controls.variables.row3_subtitle", default="Automatically imputes missing values and scales numeric columns"),
        show_bottom_line=False
    )
    variables_layout.addWidget(row3)

    # 4. Row: CV Method
    w.cv_mode_combo.setMinimumWidth(200)
    w.cv_mode_combo.setMaximumWidth(200)
    # 4. Row: CV Method
    w.cv_card = QFrame()
    w.cv_card.setObjectName("summaryCard")
    w.cv_card.setMaximumWidth(850)
    cv_card_layout = QVBoxLayout(w.cv_card)
    cv_card_layout.setContentsMargins(0, 0, 0, 0)
    cv_card_layout.setSpacing(0)

    row4, w.cv_method_title, w.cv_method_subtitle = create_apple_settings_row(
        right_widget=w.cv_mode_combo,
        title_text=tr("controls.validation.row1_title", default="Validation Method"),
        subtitle_text=tr("controls.validation.row1_subtitle", default="Select cross-validation strategy"),
        show_bottom_line=True
    )
    cv_card_layout.addWidget(row4)

    # 5. Row: CV Folds
    row5, w.cv_folds_title, w.cv_folds_subtitle = create_apple_settings_row(
        right_widget=w.cv_spin,
        title_text=tr("controls.validation.row2_title", default="Number of Folds"),
        subtitle_text=tr("controls.validation.row2_subtitle", default="For K-Fold validation splits"),
        show_bottom_line=False
    )
    cv_card_layout.addWidget(row5)

    w.variables_card.setMaximumWidth(850)
    config_card_layout.addWidget(w.variables_card)
    config_card_layout.addSpacing(16)
    config_card_layout.addWidget(w.cv_card)
    config_card_layout.addStretch(1)

    w.model_card = QFrame()
    w.model_card.setObjectName("workflowCard")
    w.model_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

    model_body = QHBoxLayout()
    model_body.setSpacing(16)
    model_group.setObjectName("modelPicker")
    model_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    model_body.addWidget(model_group, 2)
    
    w.model_badge_card = QFrame()
    w.model_badge_card.setObjectName("badgeCard")
    model_badge_layout = QHBoxLayout(w.model_badge_card)
    model_badge_layout.setContentsMargins(12, 10, 12, 10)
    w.model_summary_label.setObjectName("badgeLabel")
    model_badge_layout.addWidget(w.model_summary_label)
    model_badge_layout.addStretch()

    w.model_selected_list = QListWidget()
    w.model_selected_list.setObjectName("modelSelectedList")
    w.model_selected_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    try:
        w.model_selected_list.setSpacing(2)
    except Exception as e:
        import logging
        logging.debug(f"Failed to set model_selected_list spacing: {e}")

    w.model_sidebar = QFrame()
    w.model_sidebar.setObjectName("modelSidebar")
    model_sidebar_layout = QVBoxLayout(w.model_sidebar)
    model_sidebar_layout.setContentsMargins(4, 6, 4, 4)
    model_sidebar_layout.setSpacing(10)
    model_sidebar_layout.addWidget(w.model_badge_card)
    model_sidebar_layout.addWidget(w.model_selected_list, 1)

    model_body.addWidget(w.model_sidebar, 1)
    model_card_layout.addLayout(model_body, 1)
    model_card_layout.addStretch(1)

    w.run_card = QFrame()
    w.run_card.setObjectName("workflowCard")
    w.run_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    run_card_layout = QVBoxLayout(w.run_card)
    run_card_layout.setContentsMargins(24, 24, 24, 24)
    run_card_layout.setSpacing(10)
    
    w.step4_title = QLabel("Execution & Monitoring")
    w.step4_title.setObjectName("sectionTitle")
    run_card_layout.addWidget(w.step4_title)
    
    w.run_hint_label = QLabel("Start training to populate Summary, Tables, Figures and SHAP.")
    w.run_hint_label.setObjectName("hintLabel")
    w.run_hint_label.setWordWrap(True)
    run_card_layout.addWidget(w.run_hint_label)

    run_card_layout.addSpacing(16)

    w.runtime_hint_label = QLabel("Estimated time: waiting for model selection")
    w.runtime_hint_label.setObjectName("hintLabel")

    # Keep the legacy status label for controller compatibility, but don't
    # consume vertical space in Step 4 (footer already communicates status).
    w.status_label = QLabel("")
    w.status_label.setObjectName("trainBlockerLabel")
    w.status_label.setWordWrap(True)
    w.status_label.setProperty("severity", "neutral")
    w.status_label.setVisible(False)

    # Explicit setup/active state switch prevents partial UI overlap regressions.
    w.train_stage_stack = QStackedWidget()
    w.train_stage_stack.setObjectName("trainStageStack")

    setup_stage = QWidget()
    setup_layout = QVBoxLayout(setup_stage)
    setup_layout.setContentsMargins(0, 0, 0, 0)
    setup_layout.setSpacing(12)

    setup_hero = QWidget()
    setup_hero_layout = QVBoxLayout(setup_hero)
    setup_hero_layout.setContentsMargins(24, 6, 24, 6)
    setup_hero_layout.setSpacing(6)
    setup_hero_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    w.train_button.setMinimumHeight(40)
    w.train_button.setMinimumWidth(150)
    setup_hero_layout.addWidget(w.train_button, 0, Qt.AlignmentFlag.AlignHCenter)
    setup_hero_layout.addWidget(w.runtime_hint_label, 0, Qt.AlignmentFlag.AlignHCenter)
    setup_layout.addWidget(setup_hero)

    active_stage = QWidget()
    active_layout = QVBoxLayout(active_stage)
    active_layout.setContentsMargins(0, 0, 0, 0)
    active_layout.setSpacing(8)

    # Group training controls in a frame for visual clarity
    train_controls_frame = QFrame()
    train_controls_frame.setObjectName("trainControlsFrame")
    train_controls_layout = QHBoxLayout(train_controls_frame)
    train_controls_layout.setContentsMargins(8, 8, 8, 8)
    train_controls_layout.setSpacing(10)
    train_controls_layout.addWidget(w.persist_output_checkbox)
    train_controls_layout.addStretch(1)
    train_controls_layout.addWidget(w.cancel_button)
    active_layout.addWidget(train_controls_frame)

    # Plot/analysis summary shown below the controls row (populated dynamically by the app).
    w.plot_summary_label = QLabel("")
    w.plot_summary_label.setObjectName("hintLabel")
    w.plot_summary_label.setWordWrap(True)
    active_layout.addWidget(w.plot_summary_label)

    w.progress_panel = QFrame()
    w.progress_panel.setObjectName("progressPanel")
    progress_layout = QVBoxLayout(w.progress_panel)
    progress_layout.setContentsMargins(10, 8, 10, 8)
    progress_layout.setSpacing(5)

    # Divider is part of progress_panel so it only appears when training is active.
    _progress_divider = QFrame()
    _progress_divider.setFrameShape(QFrame.Shape.HLine)
    _progress_divider.setFrameShadow(QFrame.Shadow.Sunken)
    _progress_divider.setStyleSheet("margin-top: 4px; margin-bottom: 8px; background: #e0e4ea;")
    progress_layout.addWidget(_progress_divider)

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
    active_layout.addWidget(w.progress_panel)

    w.train_stage_stack.addWidget(setup_stage)
    w.train_stage_stack.addWidget(active_stage)
    w.train_stage_stack.setCurrentWidget(setup_stage)
    w.train_stage_setup = setup_stage
    w.train_stage_active = active_stage
    run_card_layout.addWidget(w.train_stage_stack, 0)

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
    w.feedback_focus_label = QLabel("Now: Ready\nNext: Load dataset\nBlockers: None")
    w.feedback_focus_label.setObjectName("footerLabel")
    w.feedback_focus_label.setWordWrap(True)

    w.feedback_event_label = QLabel("Latest: No recent event\nJobs: No active jobs")
    w.feedback_event_label.setObjectName("footerLabel")
    w.feedback_event_label.setWordWrap(True)

    w.step_tabs = QTabWidget()
    w.step_tabs.setObjectName("workflowTabs")
    # Let QSS own the look (avoid native "utility" tab chrome)
    w.step_tabs.setDocumentMode(False)

    step_data = QWidget(); step_data_lay = QVBoxLayout(step_data)
    step_data_lay.setContentsMargins(0, 0, 0, 0)
    step_data_lay.setSpacing(8)
    step_data_lay.addWidget(w.data_card, 1)

    step_config = QWidget(); step_config_lay = QVBoxLayout(step_config)
    step_config_lay.setContentsMargins(0, 0, 0, 0)
    step_config_lay.setSpacing(8)
    step_config_lay.addWidget(w.config_card, 1)

    step_model = QWidget(); step_model_lay = QVBoxLayout(step_model)
    step_model_lay.setContentsMargins(0, 0, 0, 0)
    step_model_lay.setSpacing(8)
    step_model_lay.addWidget(w.model_card, 1)

    step_train = QWidget(); step_train_lay = QVBoxLayout(step_train)
    step_train_lay.setContentsMargins(0, 0, 0, 0)
    step_train_lay.setSpacing(8)
    step_train_lay.addWidget(w.run_card, 1)

    w.step_tabs.addTab(step_data, "1. Dataset")
    w.step_tabs.addTab(step_config, "2. Variables")
    w.step_tabs.addTab(step_model, "3. Models")
    w.step_tabs.addTab(step_train, "4. Train")
    center_content_layout.addWidget(w.step_tabs, 1)

    # Keep the toolbox instance alive by adding it to the layout but hidden; the dialog owns the detailed UI
    center_content_layout.addWidget(plot_group)
    plot_group.setVisible(False)

    center_scroll.setWidget(center_content)
    center_layout.addWidget(center_scroll)
    w.center_scroll = center_scroll

    # Results Hub now lives directly in Step 4 (single-column UX).
    kpi_row = QHBoxLayout()
    ds_card, w.kpi_dataset_title, w.kpi_dataset_value = _make_kpi_card("Dataset", "Not loaded")
    tg_card, w.kpi_target_title, w.kpi_target_value = _make_kpi_card("Target", "Not selected")
    rn_card, w.kpi_run_title, w.kpi_run_value = _make_kpi_card("Run", "Idle")
    kpi_row.addWidget(ds_card)
    kpi_row.addWidget(tg_card)
    kpi_row.addWidget(rn_card)
    # Keep KPI widgets owned/alive for controller compatibility, but hide from Step 4 UX.
    ds_card.setVisible(False)
    tg_card.setVisible(False)
    rn_card.setVisible(False)

    # Logs and result tables
    w.log_box = QTextEdit(); w.log_box.setReadOnly(True)
    w.log_box.setPlaceholderText("Execution logs will appear here.")
    # Compact tables
    w.metrics_table = QTableView(); w.metrics_table.setObjectName("metricsTable")
    w.stats_table = QTableView(); w.stats_table.setObjectName("statsTable")
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
    try:
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(10)
    except Exception:
        mono = QFont("Monospace", 10)
    w.log_box.setFont(mono)
    w.log_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

    # Results Hub
    results_tab = QWidget()
    results_layout = QVBoxLayout(results_tab)
    results_layout.setContentsMargins(0, 0, 0, 0)
    results_layout.setSpacing(6)
    results_layout.addLayout(kpi_row)

    w.results_tabs = QTabWidget()
    w.results_tabs.setDocumentMode(False)

    summary_tab = QWidget()
    summary_layout = QVBoxLayout(summary_tab)
    summary_layout.setContentsMargins(0, 0, 0, 0)
    summary_layout.setSpacing(6)
    w.results_summary_text = QTextEdit()
    w.results_summary_text.setReadOnly(True)
    w.results_summary_text.setPlaceholderText("Best model summary will appear here after training.")
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
    w.figures_img.setMinimumHeight(180)
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
    w.shap_img.setMinimumHeight(180)
    w.shap_img.setScaledContents(False)
    shap_layout.addWidget(w.shap_list)
    shap_layout.addWidget(w.shap_img)

    w.results_tabs.addTab(summary_tab, "Summary")
    w.results_tabs.addTab(tables_tab, "Tables")
    w.results_tabs.addTab(figures_tab, "Figures")
    w.results_tabs.addTab(shap_tab, "SHAP")
    w.results_tabs.setMinimumHeight(260)
    results_layout.addWidget(w.results_tabs, 1)
    w.results_tabs.setEnabled(False)

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
    run_card_layout.addWidget(results_tab, 1)

    from PySide6.QtWidgets import QDialog
    w.dev_console_dialog = QDialog(w)
    w.dev_console_dialog.setWindowTitle(tr("controls.dialogs.dev_console", default="Developer & Activity Console"))
    w.dev_console_dialog.resize(800, 600)
    dev_layout = QVBoxLayout(w.dev_console_dialog)
    w.dev_tabs = QTabWidget()
    w.dev_tabs.addTab(console_tab, "Activity")
    w.dev_tabs.addTab(notifications_tab, "Notifications")
    w.dev_tabs.addTab(jobs_tab, "Jobs")
    dev_layout.addWidget(w.dev_tabs)

    # Single-column main layout: workflow tabs + integrated Train results hub.
    center_panel.setMinimumWidth(360)
    center_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    w.center_panel = center_panel
    main_layout.addWidget(center_panel)

    # ── Footer bar ──────────────────────────────────────────────────────────
    w.footer_bar = QFrame()
    w.footer_bar.setObjectName("footerBar")
    footer_layout = QHBoxLayout(w.footer_bar)
    footer_layout.setContentsMargins(16, 10, 16, 10)
    footer_layout.setSpacing(16)
    footer_layout.addWidget(w.feedback_focus_label, 1)
    footer_layout.addWidget(w.feedback_event_label, 1)
    outer_layout.addWidget(w.footer_bar, 0)

    # NOTE: No inline w.setStyleSheet() here.
    # All visual styling is owned exclusively by interface/style/style.qss loaded
    # via theme_manager.  Inline stylesheets on parent widgets override application-
    # level QSS selectors for every descendant, which was the third root cause of
    # the recurring layout regression.

    apply_translations(w)
    w.apply_translations = lambda: apply_translations(w)
    return w

