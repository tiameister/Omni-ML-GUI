from __future__ import annotations

import os
import sys
import time
import json
import html
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QDialog,
    QListWidgetItem,
    QTableWidgetItem,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStyle,
)
from PySide6.QtGui import QPalette, QColor, QPixmap, QIcon, QShortcut, QKeySequence, QAction
from PySide6.QtCore import (
    Qt,
    QUrl,
    QSettings,
    QCoreApplication,
    QTimer,
    QThread,
    Signal,
    QObject,
    QMetaObject,
    Q_ARG,
    Slot,
)
from PySide6.QtGui import QDesktopServices

from interface.widgets.startup import StartupDialog
from interface.logic.theme import theme_manager
from interface.logic.state import AppState
from interface.widgets.controls import build_layout
from interface.widgets.header import create_header
from interface.widgets.dialogs import (
    DataPreviewDialog,
    ColumnSelectionDialog,
    ModelSelectionDialog,
    PlotSelectionDialog,
    ShapSettingsDialog,
    AboutDialog,
    CommandPaletteDialog,
    PublicationExportDialog,
)
from interface.widgets.fe_studio import FeatureEngineeringStudioDialog
from interface.widgets.checkboxes import get_plot_pages, get_optional_script_label_map
from data.file_types import SUPPORTED_DATASET_EXTENSIONS
from config import OUTPUT_DIR, RUN_TAG
from utils.logger import get_logger
from utils.paths import EVALUATION_DIR, LEGACY_EVALUATION_DIR, LEGACY_FEATURE_SELECTION_DIR, LEGACY_MANUSCRIPT_DIR, MANUSCRIPT_DIR, get_output_root, get_project_root, safe_folder_name
from utils.text import normalize_quotes_ascii
from utils.localization import i18n, tr

if TYPE_CHECKING:
    import pandas as pd


LOGGER = get_logger(__name__)
UI_LAYOUT_VERSION = 5


_PANDAS = None


def _pd():
    """
    Lazy-import pandas to keep GUI startup fast.
    Returns:
        pandas module
    """
    global _PANDAS
    if _PANDAS is None:
        import pandas as _pandas

        _PANDAS = _pandas
    return _PANDAS


class _TrainWorker(QObject):
    """
    Worker class for running model training in a background thread.
    Progress/log marshals to the GUI via QMetaObject; completion uses pending data +
    invokeMethod (reliable with PySide; nested Python slots + Signal(object,...) are not).
    """

    def __init__(
        self,
        effective_state,
        selected_models,
        selected_plots,
        cv_mode,
        cv_folds,
        should_cancel_fn,
        run_id,
        dataset_label,
        persist_outputs,
        feature_value_labels,
        shap_settings,
        gui_app=None,
        parent=None,
    ):
        super().__init__(parent)
        self._gui_app = gui_app
        self.effective_state = effective_state
        self.selected_models = selected_models
        self.selected_plots = selected_plots
        self.cv_mode = cv_mode
        self.cv_folds = cv_folds
        self.should_cancel_fn = should_cancel_fn
        self.run_id = run_id
        self.dataset_label = dataset_label
        self.persist_outputs = persist_outputs
        self.feature_value_labels = feature_value_labels
        self.shap_settings = shap_settings

    def run(self):
        t0 = time.time()
        try:
            # Import lazily so GUI startup doesn't pay sklearn/pandas cost.
            from interface.logic.training import run_training as run_training_ui

            gui = self._gui_app

            # sklearn/joblib may call these callbacks from pool threads (not this QThread).
            # Emitting QObject signals from arbitrary threads is unsafe; marshal to the GUI
            # thread via QMetaObject.invokeMethod (requires @Slot targets on the main window).
            def _safe_progress(current: int, total: int) -> None:
                if gui is None:
                    return
                QMetaObject.invokeMethod(
                    gui,
                    "_update_progress",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, int(current)),
                    Q_ARG(int, int(total)),
                )

            def _safe_plot_progress(current: int, total: int) -> None:
                if gui is None:
                    return
                QMetaObject.invokeMethod(
                    gui,
                    "_update_plot_progress",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, int(current)),
                    Q_ARG(int, int(total)),
                )

            def _safe_log(msg: str) -> None:
                if gui is None:
                    return
                QMetaObject.invokeMethod(
                    gui,
                    "_append_log",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, str(msg)),
                )

            metrics_df, fitted_models, stats_df, stats_summary_df, out_info = run_training_ui(
                self.effective_state,
                self.selected_models,
                self.selected_plots,
                cv_mode=self.cv_mode,
                cv_folds=self.cv_folds,
                external_progress_cb=_safe_progress,
                external_plot_progress_cb=_safe_plot_progress,
                external_log_cb=_safe_log,
                should_cancel=self.should_cancel_fn,
                run_id=self.run_id,
                dataset_label=self.dataset_label,
                persist_outputs=self.persist_outputs,
                feature_value_labels=self.feature_value_labels,
                shap_settings=self.shap_settings,
            )
            elapsed = time.time() - t0
            if gui is not None:
                gui._pending_training_result = (
                    metrics_df,
                    fitted_models,
                    stats_df,
                    stats_summary_df,
                    out_info,
                    elapsed,
                )
                QMetaObject.invokeMethod(
                    gui,
                    "_on_training_worker_finished",
                    Qt.ConnectionType.QueuedConnection,
                )
        except Exception as e:
            LOGGER.exception("Training worker failed")
            gui = self._gui_app
            if gui is not None:
                gui._pending_training_error = str(e)
                QMetaObject.invokeMethod(
                    gui,
                    "_on_training_worker_error",
                    Qt.ConnectionType.QueuedConnection,
                )


class _DatasetLoadWorker(QObject):
    finished = Signal(object, object, object)  # returns (df, error_msg, path)

    def __init__(self, load_callable, path):
        super().__init__()
        self.load_callable = load_callable
        self.path = path

    def run(self):
        try:
            df = self.load_callable(self.path)
            self.finished.emit(df, None, self.path)
        except Exception as e:
            self.finished.emit(None, str(e), self.path)


class MLTrainerApp(QMainWindow):
    """
    Main application window for the ML Trainer GUI.
    Manages UI state, user actions, session recovery, and job execution.
    """
    def __init__(self):
        super().__init__()
        self._snapshot_guard = True
        self._language_listener = self._on_language_changed

        self._run_started_at: float | None = None
        self._plot_started_at: float | None = None
        self._event_log: list[str] = []
        self._jobs: list[dict] = []
        self._next_job_id_value = 1
        self._active_job_id: int | None = None
        self._launch_job_id: int | None = None
        self._cancelled = False
        # Re-entrancy guard for training-thread spawn (see _on_train).
        self._training_launch_in_progress = False
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._snapshot_limit = 80
        self._current_dataset_path: str | None = None
        self._previous_exit_was_clean = True
        self._latest_run_info: dict | None = None
        self._latest_result_dir: str | None = None
        self._latest_result_saved: bool = False
        self._latest_metrics_df: pd.DataFrame | None = None
        self._latest_stats_summary_df: pd.DataFrame | None = None
        self._figures_map: dict[str, str] = {}
        self._shap_map: dict[str, str] = {}
        self._figure_records: list[dict] = []
        self._shap_records: list[dict] = []
        self._feedback_now = tr("controls.feedback.now_ready", default="Ready")
        self._feedback_next = tr("controls.feedback.next_load_dataset", default="Load dataset")
        self._feedback_blockers = tr("controls.feedback.blockers_none", default="None")
        self._feedback_latest_event = tr("controls.feedback.latest_none", default="No recent event")
        self._feedback_jobs = tr("controls.feedback.jobs_idle", default="No active jobs")
        self._has_prompted_for_studio = False
        # Training worker stores results here; completion runs on the GUI thread via
        # QMetaObject.invokeMethod (avoids PySide issues with QueuedConnection + nested slots).
        self._pending_training_result: tuple | None = None
        self._pending_training_error: str | None = None

        self._restore_language_preference()
        self._restore_theme_preference()

        self._setup_window()
        self._mark_session_dirty()
        self.state = AppState()
        self.header = create_header(
            tr("app.title", default="Machine Learning Trainer"),
            subtitle=tr("app.subtitle", default="Build, train and evaluate regression models")
        )
        self.controls = build_layout()
        self._assemble_ui()
        self._load_stylesheet()

        self._model_summary_timer = QTimer(self)
        self._model_summary_timer.setSingleShot(True)
        self._model_summary_timer.setInterval(80)
        self._model_summary_timer.timeout.connect(self._refresh_model_summary)

        self._runtime_hint_timer = QTimer(self)
        self._runtime_hint_timer.setSingleShot(True)
        self._runtime_hint_timer.setInterval(80)
        self._runtime_hint_timer.timeout.connect(self._update_runtime_hint)

        self._connect_signals()
        self._init_job_manager_ui()
        self._restore_event_log()
        self._setup_menu_bar()
        self._restore_window_state()
        self._refresh_plot_check_states_from_settings()
        self._refresh_model_summary()
        self._on_cv_mode_changed(self.controls.cv_mode_combo.currentText())
        self._set_guided_step_state(has_data=False, has_variables=False, has_models=False)
        self._go_to_step(0)
        self._sync_menu_actions()
        i18n.add_listener(self._language_listener)
        self._apply_translations()
        self._snapshot_guard = False
        self._maybe_recover_previous_session()
        self._set_train_stage("setup")
        self._push_notification("info", tr("notifications.app_ready", default="Application ready. Load a dataset to start."))
        self._capture_snapshot(clear_redo=True)

    def _restore_language_preference(self):
        settings = QSettings()
        saved_lang = str(settings.value("ui/language", "")).strip().lower()
        if saved_lang and saved_lang in i18n.get_supported_languages():
            i18n.set_language(saved_lang)

    def _apply_translations(self):
        self.setWindowTitle(tr("app.title", default="Machine Learning Trainer"))

        if hasattr(self.header, "titleLabel"):
            self.header.titleLabel.setText(tr("app.title", default="Machine Learning Trainer"))
        if hasattr(self.header, "subtitleLabel"):
            self.header.subtitleLabel.setText(tr("app.subtitle", default="Build, train and evaluate regression models"))
        if hasattr(self.header, "globalInfoButton"):
            self.header.globalInfoButton.setText(tr("header.info_short", default="i"))
            self.header.globalInfoButton.setToolTip(tr("header.info_tooltip", default="About and guide"))
        if hasattr(self.header, "toggleModelsButton"):
            self.header.toggleModelsButton.setText(tr("header.models", default="Models"))
            self.header.toggleModelsButton.setToolTip(tr("header.models_tooltip", default="Open model selection"))

        if hasattr(self.controls, "apply_translations"):
            self.controls.apply_translations()

        self._retranslate_menu_bar()
        self._sync_language_actions()

        self._update_variable_selection_ui()
        self._refresh_model_summary()
        self._refresh_plot_check_states_from_settings()
        self._sync_results_ui_state()
        self._on_cv_mode_changed(self.controls.cv_mode_combo.currentText())
        self._set_feedback_focus()
        self._refresh_feedback_context()
        self._refresh_train_primary_action()
        self._update_header_density()

    def _on_language_changed(self):
        self._apply_translations()


    def _restore_theme_preference(self):
        settings = QSettings()
        saved_theme = str(settings.value("ui/theme", "system")).strip().lower()
        self._change_theme(saved_theme)

    def _change_theme(self, theme_name: str):
        from PySide6.QtWidgets import QApplication
        theme_manager.load_theme(QApplication.instance(), theme_name)
        self._sync_theme_actions()

    def _sync_theme_actions(self):
        if not hasattr(self, "act_theme_light"):
            return
        current = theme_manager.current_theme
        self.act_theme_light.blockSignals(True)
        self.act_theme_dark.blockSignals(True)
        self.act_theme_system.blockSignals(True)
        
        self.act_theme_light.setChecked(current == "light")
        self.act_theme_dark.setChecked(current == "dark")
        self.act_theme_system.setChecked(current == "system")
        
        self.act_theme_light.blockSignals(False)
        self.act_theme_dark.blockSignals(False)
        self.act_theme_system.blockSignals(False)

    def _change_language(self, lang_code: str):
        code = str(lang_code or "").strip().lower()
        if not code:
            return
        if i18n.set_language(code):
            settings = QSettings()
            settings.setValue("ui/language", code)
            self.statusBar().showMessage(tr("status.language_changed", default="Language changed."))
            self._push_notification("info", tr("notifications.language_changed", default="Language preference updated."))

    def _sync_language_actions(self):
        if not hasattr(self, "act_lang_en") or not hasattr(self, "act_lang_tr"):
            return
        current = i18n.get_language()
        self.act_lang_en.blockSignals(True)
        self.act_lang_tr.blockSignals(True)
        try:
            self.act_lang_en.setChecked(current == "en")
            self.act_lang_tr.setChecked(current == "tr")
        finally:
            self.act_lang_en.blockSignals(False)
            self.act_lang_tr.blockSignals(False)

    def _setup_window(self):
        self.setWindowTitle(tr("app.title", default="Machine Learning Trainer"))
        self.resize(1400, 860)
        self.setMinimumSize(980, 640)
        self.setAcceptDrops(True)
        # macOS: restore native window chrome feel.
        if sys.platform == "darwin":
            try:
                self.setUnifiedTitleAndToolBarOnMac(True)
            except Exception:
                pass
            try:
                self.setDocumentMode(True)
            except Exception:
                pass

    @staticmethod
    def _clear_table(table) -> None:
        if table is None:
            return
        # QTableView path
        if hasattr(table, "setModel"):
            try:
                table.setModel(None)
            except Exception:
                pass
        # QTableWidget path
        if hasattr(table, "clear"):
            try:
                table.clear()
            except Exception:
                pass
        if hasattr(table, "setRowCount"):
            try:
                table.setRowCount(0)
            except Exception:
                pass
        if hasattr(table, "setColumnCount"):
            try:
                table.setColumnCount(0)
            except Exception:
                pass

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "--"
        total = int(round(seconds))
        mins, secs = divmod(total, 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def _set_feedback_focus(
        self,
        now: str | None = None,
        next_step: str | None = None,
        blockers: str | None = None,
    ):
        if now is not None:
            self._feedback_now = str(now).strip() or tr("controls.feedback.now_ready", default="Ready")
        if next_step is not None:
            self._feedback_next = str(next_step).strip() or tr(
                "controls.feedback.next_load_dataset", default="Load dataset"
            )
        if blockers is not None:
            self._feedback_blockers = str(blockers).strip() or tr(
                "controls.feedback.blockers_none", default="None"
            )

        if hasattr(self.controls, "feedback_focus_label"):
            self.controls.feedback_focus_label.setText(
                tr(
                    "controls.feedback.template",
                    default="Now: {now}\nNext: {next}\nBlockers: {blockers}",
                    now=self._feedback_now,
                    next=self._feedback_next,
                    blockers=self._feedback_blockers,
                )
            )
        self._refresh_feedback_context()

    def _refresh_feedback_context(self):
        queued = sum(1 for j in self._jobs if str(j.get("status", "")) == "Queued")
        running = sum(1 for j in self._jobs if str(j.get("status", "")) == "Running")
        failed = sum(1 for j in self._jobs if str(j.get("status", "")) in {"Failed", "Cancelled"})

        if running > 0:
            self._feedback_jobs = tr(
                "controls.feedback.jobs_running",
                default="Running: {running} | Queued: {queued} | Failed: {failed}",
                running=running,
                queued=queued,
                failed=failed,
            )
        elif queued > 0 or failed > 0:
            self._feedback_jobs = tr(
                "controls.feedback.jobs_queue_state",
                default="Queued: {queued} | Failed: {failed}",
                queued=queued,
                failed=failed,
            )
        else:
            self._feedback_jobs = tr("controls.feedback.jobs_idle", default="No active jobs")

        if hasattr(self.controls, "feedback_event_label"):
            self.controls.feedback_event_label.setText(
                tr(
                    "controls.feedback.details_template",
                    default="Latest: {event}\nJobs: {jobs}",
                    event=self._feedback_latest_event,
                    jobs=self._feedback_jobs,
                )
            )

    def _assemble_ui(self):
        central = self.controls
        # Build a vertical container so header sits on top, content below
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        container = QWidget()
        container.setObjectName("appCanvas")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(16)
        vbox.addWidget(self.header)
        vbox.addWidget(central)
        self.setCentralWidget(container)
        self.statusBar().showMessage(tr("status.ready_start", default="Ready. Load a dataset to start."))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_header_density()


    def _load_stylesheet(self):
        # Migrated to ThemeManager logic via self._restore_theme_preference()
        pass


    def _connect_signals(self):
        c = self.controls
        c.load_button.clicked.connect(self._on_load)
        c.vars_button.clicked.connect(self._on_edit_vars)
        if hasattr(c, "studio_btn"):
            c.studio_btn.clicked.connect(self._open_pretraining_publication_studio)
        c.train_button.clicked.connect(self._on_train)
        c.cancel_button.clicked.connect(self._on_cancel)
        c.info_button.clicked.connect(self._on_info)
        c.open_output_btn.clicked.connect(self._on_open_output_folder)
        c.reset_session_btn.clicked.connect(self._on_reset_session)
        # Header's global info
        if hasattr(self.header, "globalInfoButton"):
            self.header.globalInfoButton.clicked.connect(self._on_about)
        if hasattr(self.header, "toggleModelsButton"):
            self.header.toggleModelsButton.clicked.connect(lambda _checked=False: self._open_models_panel())
        # Feature engineering is toggled in UI and applied during training.
        c.fe_checkbox.toggled.connect(self._on_toggle_feature_engineering)
        # Preview button
        c.preview_button.clicked.connect(self._on_preview)
        c.fe_setup_btn.clicked.connect(self._on_fe_settings)
        # Customize plots
        c.customize_plots_btn.clicked.connect(self._on_customize_plots)
        # SHAP settings
        c.shap_settings_btn.clicked.connect(self._on_shap_settings)
        c.results_save_button.clicked.connect(self._on_save_current_run)
        c.results_retrain_button.clicked.connect(self._on_train)
        c.figures_list.currentItemChanged.connect(self._on_figure_item_changed)
        c.figures_model_filter.currentIndexChanged.connect(self._apply_figure_filters)
        c.figures_category_filter.currentIndexChanged.connect(self._apply_figure_filters)
        c.shap_list.currentItemChanged.connect(self._on_shap_item_changed)
        c.shap_model_filter.currentIndexChanged.connect(self._apply_shap_filters)
        # CV mode behavior
        c.cv_mode_combo.currentTextChanged.connect(self._on_cv_mode_changed)
        # Model summary refresh
        for chk in c.model_checks.values():
            chk.toggled.connect(self._refresh_model_summary_debounced)
            chk.toggled.connect(self._on_user_setting_changed)

        # Per-model gear / settings buttons: open HyperparameterDialog so users
        # can override sklearn defaults for the targeted model only.
        settings_buttons = getattr(c.model_picker, "model_settings_buttons", {}) or {}
        for model_name, btn in settings_buttons.items():
            if btn is None:
                continue
            btn.clicked.connect(
                lambda _checked=False, n=model_name: self._open_hyperparameter_dialog(n)
            )
        for chk in c.plot_checks.values():
            chk.toggled.connect(self._update_runtime_hint_debounced)
            chk.toggled.connect(self._on_user_setting_changed)

        c.cv_mode_combo.currentIndexChanged.connect(self._on_user_setting_changed)
        c.cv_spin.valueChanged.connect(self._on_user_setting_changed)

        if hasattr(c, "step_tabs"):
            c.step_tabs.currentChanged.connect(
                lambda _idx: self._sync_step_tab_titles(
                    has_data=self.state.df is not None,
                    has_variables=bool(self.state.target is not None and self.state.features),
                    has_models=any(chk.isChecked() for chk in c.model_checks.values()),
                )
            )

        if hasattr(c, "jobs_run_next_btn"):
            c.jobs_run_next_btn.clicked.connect(self._start_next_queued_job)
        if hasattr(c, "jobs_retry_failed_btn"):
            c.jobs_retry_failed_btn.clicked.connect(self._retry_failed_jobs)
        if hasattr(c, "jobs_clear_finished_btn"):
            c.jobs_clear_finished_btn.clicked.connect(self._clear_finished_jobs)

        self._init_shortcuts()

    def _init_shortcuts(self):
        self._shortcut_load = QShortcut(QKeySequence("Ctrl+O"), self)
        self._shortcut_load.activated.connect(self._on_load)

        self._shortcut_train = QShortcut(QKeySequence("Ctrl+Return"), self)
        self._shortcut_train.activated.connect(self._on_train)

        self._shortcut_vars = QShortcut(QKeySequence("Ctrl+L"), self)
        self._shortcut_vars.activated.connect(self._on_edit_vars)

        self._shortcut_command_palette = QShortcut(QKeySequence("Ctrl+Shift+P"), self)
        self._shortcut_command_palette.activated.connect(self._open_command_palette)

    def _setup_menu_bar(self):
        mb = self.menuBar()
        try:
            # macOS: use system menu bar (Apple-like). Other OSes: keep in-window menu.
            mb.setNativeMenuBar(sys.platform == "darwin")
        except Exception:
            pass

        self.menu_file = mb.addMenu("")
        self.act_file_open = QAction(self)
        self.act_file_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_file_open.triggered.connect(self._on_load)
        self.menu_file.addAction(self.act_file_open)

        self.act_file_preview = QAction(self)
        self.act_file_preview.triggered.connect(self._on_preview)
        self.menu_file.addAction(self.act_file_preview)

        self.menu_file.addSeparator()
        self.act_file_exit = QAction(self)
        self.act_file_exit.triggered.connect(self.close)
        self.menu_file.addAction(self.act_file_exit)

        self.menu_edit = mb.addMenu("")
        self.act_edit_variables = QAction(self)
        self.act_edit_variables.setShortcut(QKeySequence("Ctrl+L"))
        self.act_edit_variables.triggered.connect(self._on_edit_vars)
        self.menu_edit.addAction(self.act_edit_variables)

        self.act_edit_feature_engineering = QAction(self)
        self.act_edit_feature_engineering.setCheckable(True)
        self.act_edit_feature_engineering.toggled.connect(lambda checked: self.controls.fe_checkbox.setChecked(bool(checked)))
        self.menu_edit.addAction(self.act_edit_feature_engineering)

        self.act_edit_publication_studio = QAction(self)
        self.act_edit_publication_studio.triggered.connect(self._open_pretraining_publication_studio)
        self.menu_edit.addAction(self.act_edit_publication_studio)

        self.menu_edit.addSeparator()
        self.act_edit_undo = QAction(self)
        self.act_edit_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_edit_undo.triggered.connect(self._undo_snapshot)
        self.menu_edit.addAction(self.act_edit_undo)

        self.act_edit_redo = QAction(self)
        self.act_edit_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self.act_edit_redo.triggered.connect(self._redo_snapshot)
        self.menu_edit.addAction(self.act_edit_redo)

        self.menu_view = mb.addMenu("")
        self.act_view_command_palette = QAction(self)
        self.act_view_command_palette.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.act_view_command_palette.triggered.connect(self._open_command_palette)
        self.menu_view.addAction(self.act_view_command_palette)

        self.act_view_notifications = QAction(self)
        self.act_view_notifications.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.act_view_notifications.triggered.connect(self._open_notification_center)
        self.menu_view.addAction(self.act_view_notifications)

        self.act_view_jobs = QAction(self)
        self.act_view_jobs.setShortcut(QKeySequence("Ctrl+Shift+J"))
        self.act_view_jobs.triggered.connect(self._open_job_manager)
        self.menu_view.addAction(self.act_view_jobs)

        self.menu_view.addSeparator()
        self.act_view_model_pool = QAction(self)
        self.act_view_model_pool.triggered.connect(lambda _checked=False: self._open_models_panel())
        self.menu_view.addAction(self.act_view_model_pool)


        self.menu_view.addSeparator()
        self.act_view_step1 = QAction(self)
        self.act_view_step1.triggered.connect(lambda: self._go_to_step(0))
        self.act_view_step2 = QAction(self)
        self.act_view_step2.triggered.connect(lambda: self._go_to_step(1))
        self.act_view_step3 = QAction(self)
        self.act_view_step3.triggered.connect(lambda: self._go_to_step(2))
        self.act_view_step4 = QAction(self)
        self.act_view_step4.triggered.connect(lambda: self._go_to_step(3))
        self.menu_view.addAction(self.act_view_step1)
        self.menu_view.addAction(self.act_view_step2)
        self.menu_view.addAction(self.act_view_step3)
        self.menu_view.addAction(self.act_view_step4)

        self.menu_run = mb.addMenu("")
        self.act_run_start = QAction(self)
        self.act_run_start.setShortcut(QKeySequence("Ctrl+Return"))
        self.act_run_start.triggered.connect(self._on_train)
        self.menu_run.addAction(self.act_run_start)

        self.act_run_cancel = QAction(self)
        self.act_run_cancel.triggered.connect(self._on_cancel)
        self.menu_run.addAction(self.act_run_cancel)

        self.menu_run.addSeparator()
        self.act_run_next_queued = QAction(self)
        self.act_run_next_queued.triggered.connect(self._start_next_queued_job)
        self.menu_run.addAction(self.act_run_next_queued)

        self.act_run_retry_failed = QAction(self)
        self.act_run_retry_failed.triggered.connect(self._retry_failed_jobs)
        self.menu_run.addAction(self.act_run_retry_failed)

        self.act_run_clear_finished = QAction(self)
        self.act_run_clear_finished.triggered.connect(self._clear_finished_jobs)
        self.menu_run.addAction(self.act_run_clear_finished)

        self.menu_settings = mb.addMenu("")
        self.act_settings_open_model_pool = QAction(self)
        self.act_settings_open_model_pool.triggered.connect(lambda _checked=False: self._open_models_panel())
        self.menu_settings.addAction(self.act_settings_open_model_pool)

        self.act_settings_customize_plots = QAction(self)
        self.act_settings_customize_plots.triggered.connect(self._on_customize_plots)
        self.menu_settings.addAction(self.act_settings_customize_plots)

        self.act_settings_shap = QAction(self)
        self.act_settings_shap.triggered.connect(self._on_shap_settings)
        self.menu_settings.addAction(self.act_settings_shap)


        self.menu_settings_theme = self.menu_settings.addMenu("")
        from PySide6.QtGui import QActionGroup
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)

        self.act_theme_system = QAction(self)
        self.act_theme_system.setCheckable(True)
        self.act_theme_system.triggered.connect(lambda checked=False: self._change_theme("system") if checked else None)
        self.theme_action_group.addAction(self.act_theme_system)
        self.menu_settings_theme.addAction(self.act_theme_system)

        self.act_theme_light = QAction(self)
        self.act_theme_light.setCheckable(True)
        self.act_theme_light.triggered.connect(lambda checked=False: self._change_theme("light") if checked else None)
        self.theme_action_group.addAction(self.act_theme_light)
        self.menu_settings_theme.addAction(self.act_theme_light)

        self.act_theme_dark = QAction(self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_dark.triggered.connect(lambda checked=False: self._change_theme("dark") if checked else None)
        self.theme_action_group.addAction(self.act_theme_dark)
        self.menu_settings_theme.addAction(self.act_theme_dark)

        self.menu_settings_language = self.menu_settings.addMenu("")

        self.language_action_group = QActionGroup(self)
        self.language_action_group.setExclusive(True)

        self.act_lang_en = QAction(self)
        self.act_lang_en.setCheckable(True)
        self.act_lang_en.triggered.connect(lambda checked=False: self._change_language("en") if checked else None)
        self.language_action_group.addAction(self.act_lang_en)
        self.menu_settings_language.addAction(self.act_lang_en)

        self.act_lang_tr = QAction(self)
        self.act_lang_tr.setCheckable(True)
        self.act_lang_tr.triggered.connect(lambda checked=False: self._change_language("tr") if checked else None)
        self.language_action_group.addAction(self.act_lang_tr)
        self.menu_settings_language.addAction(self.act_lang_tr)

        self.menu_settings.addSeparator()
        self.act_settings_open_output = QAction(self)
        self.act_settings_open_output.triggered.connect(self._on_open_output_folder)
        self.menu_settings.addAction(self.act_settings_open_output)

        self.act_settings_reset = QAction(self)
        self.act_settings_reset.triggered.connect(self._on_reset_session)
        self.menu_settings.addAction(self.act_settings_reset)

        self.act_settings_restore_checkpoint = QAction(self)
        self.act_settings_restore_checkpoint.triggered.connect(self._restore_last_checkpoint)
        self.menu_settings.addAction(self.act_settings_restore_checkpoint)

        self.menu_help = mb.addMenu("")
        self.act_help_guide = QAction(self)
        self.act_help_guide.triggered.connect(self._on_info)
        self.menu_help.addAction(self.act_help_guide)

        self.act_help_about = QAction(self)
        self.act_help_about.triggered.connect(self._on_about)
        self.menu_help.addAction(self.act_help_about)

        self._retranslate_menu_bar()

    def _retranslate_menu_bar(self):
        self.menu_file.setTitle(tr("menu.file", default="File"))
        self.act_file_open.setText(tr("menu.file_open_dataset", default="Open Dataset..."))
        self.act_file_preview.setText(tr("menu.file_preview_dataset", default="Preview Dataset"))
        self.act_file_exit.setText(tr("menu.file_exit", default="Exit"))

        self.menu_edit.setTitle(tr("menu.edit", default="Edit"))
        self.act_edit_variables.setText(tr("menu.edit_select_variables", default="Select Variables"))
        self.act_edit_feature_engineering.setText(tr("menu.edit_enable_fe", default="Enable Feature Engineering"))
        self.act_edit_publication_studio.setText(tr("menu.edit_publication_studio", default="Publication Studio (Pre-training)..."))
        self.act_edit_undo.setText(tr("menu.edit_undo", default="Undo Last Change"))
        self.act_edit_redo.setText(tr("menu.edit_redo", default="Redo Last Change"))

        self.menu_view.setTitle(tr("menu.view", default="View"))
        self.act_view_command_palette.setText(tr("menu.view_command_palette", default="Command Palette..."))
        self.act_view_notifications.setText(tr("menu.view_notifications", default="Notification Center"))
        self.act_view_jobs.setText(tr("menu.view_jobs", default="Job Manager"))
        self.act_view_model_pool.setText(tr("menu.view_model_selection", default="Model Selection..."))
        self.act_view_step1.setText(tr("menu.view_step1", default="Go to Step 1: Dataset"))
        self.act_view_step2.setText(tr("menu.view_step2", default="Go to Step 2: Variables"))
        self.act_view_step3.setText(tr("menu.view_step3", default="Go to Step 3: Models"))
        self.act_view_step4.setText(tr("menu.view_step4", default="Go to Step 4: Train"))

        self.menu_run.setTitle(tr("menu.run", default="Run"))
        self.act_run_start.setText(tr("menu.run_start", default="Start Training"))
        self.act_run_cancel.setText(tr("menu.run_cancel", default="Cancel Current Run"))
        self.act_run_next_queued.setText(tr("menu.run_next_queued", default="Run Next Queued Job"))
        self.act_run_retry_failed.setText(tr("menu.run_retry_failed", default="Retry Failed Jobs"))
        self.act_run_clear_finished.setText(tr("menu.run_clear_finished", default="Clear Finished Jobs"))

        self.menu_settings.setTitle(tr("menu.settings", default="Settings"))
        self.act_settings_open_model_pool.setText(tr("menu.settings_model_selection", default="Model Selection..."))
        self.act_settings_customize_plots.setText(tr("menu.settings_plot_analysis", default="Plot and Analysis Settings..."))
        self.act_settings_shap.setText(tr("menu.settings_shap", default="SHAP Settings..."))

        self.menu_settings_theme.setTitle(tr("menu.theme", default="Appearance"))
        self.act_theme_system.setText(tr("menu.theme_system", default="System (Auto)"))
        self.act_theme_light.setText(tr("menu.theme_light", default="Light"))
        self.act_theme_dark.setText(tr("menu.theme_dark", default="Dark"))
        self.menu_settings_language.setTitle(tr("menu.language", default="Language"))

        self.act_lang_en.setText(tr("menu.lang_en", default="English"))
        self.act_lang_tr.setText(tr("menu.lang_tr", default="Turkish"))
        self.act_settings_open_output.setText(tr("menu.settings_open_output", default="Open Output Folder"))
        self.act_settings_reset.setText(tr("menu.settings_reset_session", default="Reset Session"))
        self.act_settings_restore_checkpoint.setText(tr("menu.settings_restore_checkpoint", default="Restore Last Checkpoint"))

        self.menu_help.setTitle(tr("menu.help", default="Help"))
        self.act_help_guide.setText(tr("help.open_guide", default="Open Guide"))
        self.act_help_about.setText(tr("help.about", default="About"))

    def _build_command_palette_commands(self) -> list[dict]:
        c = self.controls
        has_data = self.state.df is not None
        has_variables = self.state.target is not None and bool(self.state.features)
        has_models = any(chk.isChecked() for chk in c.model_checks.values())
        train_running = self._active_job_id is not None
        queued_jobs = sum(1 for j in self._jobs if j.get("status") == "Queued")
        failed_jobs = sum(1 for j in self._jobs if j.get("status") in {"Failed", "Cancelled"})
        finished_jobs = sum(1 for j in self._jobs if j.get("status") in {"Completed", "Failed", "Cancelled"})
        checkpoint_available = self._load_recovery_checkpoint() is not None

        grp_file = tr("command_palette.groups.file", default="File")
        grp_edit = tr("command_palette.groups.edit", default="Edit")
        grp_view = tr("command_palette.groups.view", default="View")
        grp_run = tr("command_palette.groups.run", default="Run")
        grp_settings = tr("command_palette.groups.settings", default="Settings")
        grp_help = tr("command_palette.groups.help", default="Help")

        result_visible = True
        fe_state = tr("command_palette.state.enabled", default="enabled") if c.fe_checkbox.isChecked() else tr("command_palette.state.disabled", default="disabled")
        run_title = (
            tr("command_palette.run.queue_current", default="Queue Current Configuration")
            if train_running
            else tr("command_palette.run.start_training", default="Start Training")
        )
        run_desc = (
            tr("command_palette.run.queue_desc", default="Queue a new job with current selections.")
            if train_running
            else tr("command_palette.run.start_desc", default="Run selected models with current validation settings.")
        )
        undo_count = max(len(self._undo_stack) - 1, 0)
        redo_count = len(self._redo_stack)

        return [
            {
                "group": grp_file,
                "title": tr("command_palette.file.open_dataset", default="Open Dataset"),
                "description": tr("command_palette.file.open_dataset_desc", default="Load CSV or Excel dataset from disk."),
                "keywords": "open load dataset file csv excel",
                "enabled": True,
                "callback": self._on_load,
            },
            {
                "group": grp_file,
                "title": tr("command_palette.file.preview_dataset", default="Preview Dataset"),
                "description": tr("command_palette.file.preview_dataset_desc", default="Open data preview dialog for the loaded dataset."),
                "keywords": "preview data table",
                "enabled": has_data,
                "callback": self._on_preview,
            },
            {
                "group": grp_edit,
                "title": tr("command_palette.edit.select_variables", default="Select Variables"),
                "description": tr("command_palette.edit.select_variables_desc", default="Choose target and feature columns."),
                "keywords": "target features columns variables",
                "enabled": has_data,
                "callback": self._on_edit_vars,
            },
            {
                "group": grp_edit,
                "title": tr("command_palette.edit.toggle_fe", default="Toggle Feature Engineering ({state})", state=fe_state),
                "description": tr("command_palette.edit.toggle_fe_desc", default="Enable or disable feature engineering before training."),
                "keywords": "feature engineering toggle",
                "enabled": c.fe_checkbox.isEnabled(),
                "callback": lambda: c.fe_checkbox.setChecked(not c.fe_checkbox.isChecked()),
            },
            {
                "group": grp_edit,
                "title": tr("command_palette.edit.publication_studio_setup", default="Open Publication Studio (Pre-training)"),
                "description": tr(
                    "command_palette.edit.publication_studio_setup_desc",
                    default="Optional step after variable selection: define publication labels before training outputs are generated.",
                ),
                "keywords": "publication studio pretraining rename labels variables",
                "enabled": has_variables,
                "callback": self._open_pretraining_publication_studio,
            },
            {
                "group": grp_edit,
                "title": tr("command_palette.edit.undo", default="Undo Last Change ({count})", count=undo_count),
                "description": tr("command_palette.edit.undo_desc", default="Restore previous configuration snapshot."),
                "keywords": "undo revert snapshot",
                "enabled": undo_count > 0,
                "callback": self._undo_snapshot,
            },
            {
                "group": grp_edit,
                "title": tr("command_palette.edit.redo", default="Redo Last Change ({count})", count=redo_count),
                "description": tr("command_palette.edit.redo_desc", default="Reapply configuration snapshot that was undone."),
                "keywords": "redo forward snapshot",
                "enabled": redo_count > 0,
                "callback": self._redo_snapshot,
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.open_model_selection", default="Open Model Selection"),
                "description": tr("command_palette.view.open_model_selection_desc", default="Go to Step 3: Models."),
                "keywords": "model selection models step 3",
                "enabled": True,
                "callback": self._open_models_panel,
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.open_results", default="Open Results (Step 4)"),
                "description": tr("command_palette.view.open_results_desc", default="Jump to Step 4 where results are embedded."),
                "keywords": "results train step 4",
                "enabled": True,
                "callback": lambda: self._go_to_step(3),
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.open_notifications", default="Open Notification Center"),
                "description": tr("command_palette.view.open_notifications_desc", default="Focus the notifications tab and show recent events."),
                "keywords": "notifications events alerts",
                "enabled": True,
                "callback": self._open_notification_center,
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.open_job_manager", default="Open Job Manager"),
                "description": tr("command_palette.view.open_job_manager_desc", default="Focus the jobs tab and inspect queue status."),
                "keywords": "jobs queue manager",
                "enabled": True,
                "callback": self._open_job_manager,
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.step1", default="Go to Step 1: Dataset"),
                "description": tr("command_palette.view.step1_desc", default="Focus guided step for loading dataset."),
                "keywords": "step dataset workflow",
                "enabled": True,
                "callback": lambda: self._go_to_step(0),
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.step2", default="Go to Step 2: Variables"),
                "description": tr("command_palette.view.step2_desc", default="Focus guided step for variable configuration."),
                "keywords": "step variables workflow",
                "enabled": has_data,
                "callback": lambda: self._go_to_step(1),
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.step3", default="Go to Step 3: Models"),
                "description": tr("command_palette.view.step3_desc", default="Focus guided step for model selection."),
                "keywords": "step models workflow",
                "enabled": has_variables,
                "callback": lambda: self._go_to_step(2),
            },
            {
                "group": grp_view,
                "title": tr("command_palette.view.step4", default="Go to Step 4: Train"),
                "description": tr("command_palette.view.step4_desc", default="Focus guided step for training and analysis."),
                "keywords": "step train workflow",
                "enabled": has_variables and has_models,
                "callback": lambda: self._go_to_step(3),
            },
            {
                "group": grp_run,
                "title": run_title,
                "description": run_desc,
                "keywords": "run train queue models",
                "enabled": c.train_button.isEnabled(),
                "callback": self._on_train,
            },
            {
                "group": grp_run,
                "title": tr("command_palette.run.cancel", default="Cancel Current Run"),
                "description": tr("command_palette.run.cancel_desc", default="Request cancellation for the active training job."),
                "keywords": "cancel stop run training",
                "enabled": train_running,
                "callback": self._on_cancel,
            },
            {
                "group": grp_run,
                "title": tr("command_palette.run.next_queued", default="Run Next Queued Job ({count})", count=queued_jobs),
                "description": tr("command_palette.run.next_queued_desc", default="Start the next queued job when idle."),
                "keywords": "run next queued",
                "enabled": (queued_jobs > 0) and (not train_running),
                "callback": self._start_next_queued_job,
            },
            {
                "group": grp_run,
                "title": tr("command_palette.run.retry_failed", default="Retry Failed Jobs ({count})", count=failed_jobs),
                "description": tr("command_palette.run.retry_failed_desc", default="Re-queue jobs with Failed or Cancelled status."),
                "keywords": "retry failed cancelled",
                "enabled": failed_jobs > 0,
                "callback": self._retry_failed_jobs,
            },
            {
                "group": grp_run,
                "title": tr("command_palette.run.clear_finished", default="Clear Finished Jobs ({count})", count=finished_jobs),
                "description": tr("command_palette.run.clear_finished_desc", default="Remove completed/failed/cancelled jobs from manager."),
                "keywords": "clear finished jobs",
                "enabled": finished_jobs > 0,
                "callback": self._clear_finished_jobs,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.plot_analysis", default="Plot and Analysis Settings"),
                "description": tr("command_palette.settings.plot_analysis_desc", default="Choose output plots and optional extra analyses."),
                "keywords": "plots analyses settings",
                "enabled": c.customize_plots_btn.isEnabled(),
                "callback": self._on_customize_plots,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.shap", default="SHAP Settings"),
                "description": tr("command_palette.settings.shap_desc", default="Configure SHAP explainability options."),
                "keywords": "shap explainability settings",
                "enabled": c.shap_settings_btn.isEnabled(),
                "callback": self._on_shap_settings,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.publication_helper_setup", default="Publication Studio (Pre-training)"),
                "description": tr(
                    "command_palette.settings.publication_helper_setup_desc",
                    default="Define variable/label naming rules before training outputs are generated.",
                ),
                "keywords": "publication studio pretraining rename labels",
                "enabled": has_variables,
                "callback": self._open_pretraining_publication_studio,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.open_output", default="Open Output Folder"),
                "description": tr("command_palette.settings.open_output_desc", default="Open generated outputs in file explorer."),
                "keywords": "output folder exports open",
                "enabled": c.open_output_btn.isEnabled(),
                "callback": self._on_open_output_folder,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.reset_session", default="Reset Session"),
                "description": tr("command_palette.settings.reset_session_desc", default="Clear current dataset, selections and visible outputs."),
                "keywords": "reset session clear",
                "enabled": c.reset_session_btn.isEnabled(),
                "callback": self._on_reset_session,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.restore_checkpoint", default="Restore Last Checkpoint"),
                "description": tr("command_palette.settings.restore_checkpoint_desc", default="Recover latest saved configuration and job state checkpoint."),
                "keywords": "restore checkpoint recovery",
                "enabled": checkpoint_available,
                "callback": self._restore_last_checkpoint,
            },
            {
                "group": grp_help,
                "title": tr("help.open_guide", default="Open Guide"),
                "description": tr("command_palette.help.open_guide_desc", default="Open user guide PDF if available."),
                "keywords": "guide info help",
                "enabled": True,
                "callback": self._on_info,
            },
            {
                "group": grp_help,
                "title": tr("help.about", default="About"),
                "description": tr("command_palette.help.about_desc", default="Show application and build information."),
                "keywords": "about version",
                "enabled": True,
                "callback": self._on_about,
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.lang_en", default="Switch Language: English"),
                "description": tr("command_palette.settings.lang_en_desc", default="Change application language to English."),
                "keywords": "language english",
                "enabled": i18n.get_language() != "en",
                "callback": lambda: self._change_language("en"),
            },
            {
                "group": grp_settings,
                "title": tr("command_palette.settings.lang_tr", default="Switch Language: Turkish"),
                "description": tr("command_palette.settings.lang_tr_desc", default="Change application language to Turkish."),
                "keywords": "language turkish turkce",
                "enabled": i18n.get_language() != "tr",
                "callback": lambda: self._change_language("tr"),
            },
        ]

    def _open_command_palette(self):
        self._sync_menu_actions()
        dlg = CommandPaletteDialog(self._build_command_palette_commands(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        command = dlg.get_selected_command()
        if not command:
            return

        callback = command.get("callback")
        if not callable(callback):
            return

        try:
            callback()
            cmd_title = str(command.get("title", tr("common.unknown", default="Unknown")))
            msg = tr("status.command_executed", default="Command executed: {title}", title=cmd_title)
            self.statusBar().showMessage(msg)
            self._push_notification("info", msg)
        except Exception:
            LOGGER.exception("Command palette execution failed for %s", command.get("title"))

    def _open_notification_center(self):
        c = self.controls
        if hasattr(c, "dev_console_dialog"):
            c.dev_console_dialog.show()
            if hasattr(c, "dev_tabs") and hasattr(c, "notifications_tab"):
                idx = c.dev_tabs.indexOf(c.notifications_tab)
                if idx >= 0:
                    c.dev_tabs.setCurrentIndex(idx)
        self.statusBar().showMessage(tr("status.notification_center_opened", default="Notification Center opened"))

    def _open_job_manager(self):
        c = self.controls
        if hasattr(c, "dev_console_dialog"):
            c.dev_console_dialog.show()
            if hasattr(c, "dev_tabs") and hasattr(c, "jobs_tab"):
                idx = c.dev_tabs.indexOf(c.jobs_tab)
                if idx >= 0:
                    c.dev_tabs.setCurrentIndex(idx)
        self.statusBar().showMessage(tr("status.job_manager_opened", default="Job Manager opened"))

    def _init_job_manager_ui(self):
        c = self.controls
        if not hasattr(c, "jobs_table"):
            return
        c.jobs_table.setRowCount(0)
        self._refresh_job_table()

    def _job_by_id(self, job_id: int | None):
        if job_id is None:
            return None
        for job in self._jobs:
            if int(job.get("id", -1)) == int(job_id):
                return job
        return None

    @staticmethod
    def _summarize_models(model_names: list[str]) -> str:
        if not model_names:
            return "-"
        preview = ", ".join(model_names[:2])
        extra = len(model_names) - 2
        if extra > 0:
            return f"{preview} +{extra}"
        return preview

    @staticmethod
    def _format_cv_label(mode: str, folds: int) -> str:
        if mode == "holdout":
            return "holdout"
        return f"{mode}/{folds}"

    def _refresh_job_table(self):
        c = self.controls
        if not hasattr(c, "jobs_table"):
            return

        tbl = c.jobs_table
        prev_signals = tbl.blockSignals(True)
        prev_updates = None
        try:
            tbl.setSortingEnabled(False)
            try:
                prev_updates = tbl.updatesEnabled()
                tbl.setUpdatesEnabled(False)
            except Exception:
                prev_updates = None

            tbl.setRowCount(len(self._jobs))
            status_colors = {
                "Queued": QColor("#2E4D67"),
                "Running": QColor("#125284"),
                "Completed": QColor("#1E633B"),
                "Failed": QColor("#8D1E1E"),
                "Cancelled": QColor("#8A6200"),
            }

            for row, job in enumerate(self._jobs):
                created = time.strftime(
                    "%H:%M:%S", time.localtime(float(job.get("created_at", time.time())))
                )
                elapsed_val = job.get("elapsed")
                if elapsed_val is None and job.get("status") == "Running":
                    started = job.get("started_at")
                    elapsed_val = (time.time() - float(started)) if started else None

                values = [
                    str(job.get("id", "")),
                    str(job.get("status", "")),
                    self._summarize_models(list(job.get("selected_models", []))),
                    self._format_cv_label(str(job.get("cv_mode", "repeated")), int(job.get("cv_folds", 5))),
                    created,
                    self._format_duration(float(elapsed_val)) if elapsed_val is not None else "--",
                    str(job.get("message", "")),
                ]

                for col, val in enumerate(values):
                    item = QTableWidgetItem(val)
                    if col == 1:
                        item.setForeground(
                            status_colors.get(str(job.get("status", "")), QColor("#2E4D67"))
                        )
                    tbl.setItem(row, col, item)

            try:
                tbl.resizeColumnsToContents()
            except Exception:
                LOGGER.exception("Failed to resize columns in jobs UI")
        finally:
            try:
                tbl.setSortingEnabled(True)
            except Exception:
                pass
            try:
                if prev_updates is not None:
                    tbl.setUpdatesEnabled(prev_updates)
            except Exception:
                pass
            tbl.blockSignals(prev_signals)

        self._refresh_feedback_context()
        self._save_recovery_checkpoint()

    def _open_model_selection_dialog(self, prompt_for_training: bool = False) -> bool:
        c = self.controls
        model_names = list(c.model_checks.keys())
        selected_models = [name for name, chk in c.model_checks.items() if chk.isChecked()]

        dlg = ModelSelectionDialog(model_names=model_names, selected_models=selected_models, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False

        chosen = set(dlg.get_selected_models())
        prev_guard = self._snapshot_guard
        self._snapshot_guard = True
        try:
            for name, chk in c.model_checks.items():
                chk.setChecked(name in chosen)
        finally:
            self._snapshot_guard = prev_guard

        self._refresh_model_summary()
        self._sync_menu_actions()
        self._capture_snapshot(clear_redo=True)

        if chosen:
            has_data = self.state.df is not None
            has_variables = self.state.target is not None and bool(self.state.features)
            self._set_guided_step_state(has_data=has_data, has_variables=has_variables, has_models=True)
            self.statusBar().showMessage(
                tr("status.models_selected", default="{count} model selected", count=len(chosen))
            )
            if has_variables:
                c.status_label.setText(
                    tr(
                        "status.models_selected_next_step",
                        default="Models selected. Continue with Step 4: Start Training.",
                    )
                )
                self._go_to_step(3)
            self._set_feedback_focus(
                now=tr("status.models_selected", default="{count} model selected", count=len(chosen)),
                next_step=tr("status.feedback.next_start_training", default="Start training"),
                blockers=tr("controls.feedback.blockers_none", default="None"),
            )
            if prompt_for_training:
                self._push_notification(
                    "info",
                    tr(
                        "notifications.model_selection_completed",
                        default="Model selection completed ({count} model).",
                        count=len(chosen),
                    ),
                )
            return True

        QMessageBox.warning(
            self,
            tr("dialogs.training.no_models_title", default="No Models Selected"),
            tr("dialogs.training.no_models_message", default="Please select at least one model."),
        )
        return False

    def _collect_training_request(self):
        c = self.controls
        if self.state.target is None or not self.state.features:
            self._set_feedback_focus(
                now=tr("status.feedback.now_variables_missing", default="Step 2 is incomplete"),
                next_step=tr("status.feedback.next_select_variables", default="Select target and features"),
                blockers=tr("status.feedback.blocker_variables", default="Target/features are not selected"),
            )
            QMessageBox.information(
                self,
                tr("dialogs.training.select_variables_title", default="Select Variables"),
                tr(
                    "dialogs.training.select_variables_message",
                    default="Please complete Step 2 and select target/features before training.",
                ),
            )
            self._go_to_step(1)
            return None

        selected_models = [name for name, chk in c.model_checks.items() if chk.isChecked()]
        if not selected_models:
            self._go_to_step(2)
            self._set_feedback_focus(
                now=tr("status.feedback.now_model_required", default="Model selection is required"),
                next_step=tr("status.feedback.next_choose_model", default="Choose at least one model"),
                blockers=tr("status.feedback.blocker_no_model", default="No model selected"),
            )
            self.statusBar().showMessage(
                tr(
                    "status.opening_model_selection",
                    default="No model selected. Opening model selection...",
                )
            )
            opened = self._open_model_selection_dialog(prompt_for_training=True)
            if not opened:
                self._set_feedback_focus(
                    now=tr("status.feedback.now_training_blocked", default="Training is blocked"),
                    next_step=tr("status.feedback.next_open_model_selection", default="Open model selection and choose a model"),
                    blockers=tr("status.feedback.blocker_no_model", default="No model selected"),
                )
                self._push_notification(
                    "warning",
                    tr("notifications.training_blocked_no_models", default="Training blocked: no models selected."),
                )
                return None
            selected_models = [name for name, chk in c.model_checks.items() if chk.isChecked()]

        try:
            selected_plots = [name for name, chk in c.plot_checks.items() if chk.isChecked()]
        except RuntimeError:
            settings = QSettings()
            pages, _ = get_plot_pages()
            script_labels = set(get_optional_script_label_map().keys())
            selected_plots = []
            for title, items in pages.items():
                for name in items:
                    val = settings.value(f"plots/{title}/{name}", None)
                    default_checked = name not in script_labels
                    if (default_checked and val is None) or str(val).lower() in ("true", "1", "yes"):
                        selected_plots.append(name)
            selected_plots = list(dict.fromkeys(selected_plots))

        cv_mode = c.cv_mode_combo.currentData() if c.cv_mode_combo.currentData() else 'repeated'
        cv_folds = int(c.cv_spin.value())
        # Only carry overrides for models that are actually selected so the
        # saved job / manifest stays tight.
        all_overrides = dict(getattr(self.state, "model_hyperparams", {}) or {})
        model_hyperparams = {
            name: dict(params)
            for name, params in all_overrides.items()
            if name in selected_models and isinstance(params, dict) and params
        }
        return {
            "selected_models": selected_models,
            "selected_plots": selected_plots,
            "cv_mode": cv_mode,
            "cv_folds": cv_folds,
            "studio_profile": self._studio_profile_data(),
            "model_hyperparams": model_hyperparams,
            "persist_outputs": bool(c.persist_output_checkbox.isChecked()) if hasattr(c, "persist_output_checkbox") else True,
        }

    def _create_job(self, request: dict, status: str = "Queued", message: str = "Waiting in queue") -> dict:
        job = {
            "id": int(self._next_job_id_value),
            "status": str(status),
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed": None,
            "attempt": 1,
            "selected_models": list(request.get("selected_models", [])),
            "selected_plots": list(request.get("selected_plots", [])),
            "cv_mode": str(request.get("cv_mode", "repeated")),
            "cv_folds": int(request.get("cv_folds", 5)),
            "studio_profile": dict(request.get("studio_profile", {}) or {}),
            "model_hyperparams": {
                str(name): dict(params)
                for name, params in dict(request.get("model_hyperparams", {}) or {}).items()
                if isinstance(params, dict)
            },
            "persist_outputs": bool(request.get("persist_outputs", False)),
            "message": str(message),
            "error": "",
        }
        self._next_job_id_value += 1
        self._jobs.append(job)
        self._refresh_job_table()
        return job

    def _refresh_train_primary_action(self):
        c = self.controls
        is_running = self._active_job_id is not None

        if is_running:
            button_text = tr("controls.buttons.queue_training", default="Queue Training")
            button_tip = tr(
                "controls.buttons.queue_training_tooltip",
                default="Queue current configuration while another job is running.",
            )
            menu_text = tr("menu.run_queue_current", default="Queue Current Configuration")
        else:
            button_text = tr("controls.buttons.start_training", default="Start Training")
            button_tip = tr(
                "controls.buttons.start_training_tooltip",
                default="Start training with the current configuration.",
            )
            menu_text = tr("menu.run_start", default="Start Training")

        c.train_button.setText(button_text)
        c.train_button.setToolTip(button_tip)
        if hasattr(self, "act_run_start"):
            self.act_run_start.setText(menu_text)

    def _set_train_stage(self, stage: str):
        c = self.controls
        if not hasattr(c, "train_stage_stack"):
            return
        if stage == "active":
            c.train_stage_stack.setVisible(True)
            c.train_stage_stack.setCurrentWidget(c.train_stage_active)
            return
        if stage == "results":
            c.train_stage_stack.setVisible(False)
            c.progress_panel.setVisible(False)
            c.cancel_button.setVisible(False)
            return
        c.train_stage_stack.setVisible(True)
        c.train_stage_stack.setCurrentWidget(c.train_stage_setup)
        c.progress_panel.setVisible(False)
        c.cancel_button.setVisible(False)

    def _set_controls_for_run_state(self, running: bool):
        c = self.controls
        has_data = self.state.df is not None
        has_variables = self.state.target is not None and bool(self.state.features)
        has_models = any(chk.isChecked() for chk in c.model_checks.values())

        if running:
            # Hard lock: while training, every input that could mutate
            # the configuration of the *running* job (or accidentally
            # spawn a second one) is disabled. Cancel + Open-Output stay
            # enabled so the user is never trapped.
            c.load_button.setEnabled(False)
            c.vars_button.setEnabled(False)
            if hasattr(c, "studio_btn"):
                c.studio_btn.setEnabled(False)
            c.fe_checkbox.setEnabled(False)
            c.preview_button.setEnabled(False)
            for chk in c.model_checks.values():
                chk.setEnabled(False)
            for chk in c.plot_checks.values():
                chk.setEnabled(False)
            c.cv_mode_combo.setEnabled(False)
            c.cv_spin.setEnabled(False)
            c.train_button.setEnabled(False)
            if hasattr(c, "model_picker"):
                c.model_picker.setEnabled(False)
            c.info_button.setEnabled(True)
            c.customize_plots_btn.setEnabled(False)
            c.shap_settings_btn.setEnabled(False)
            c.open_output_btn.setEnabled(True)
            c.reset_session_btn.setEnabled(False)
        else:
            c.load_button.setEnabled(True)
            c.vars_button.setEnabled(has_data)
            if hasattr(c, "studio_btn"):
                c.studio_btn.setEnabled(has_variables)
            c.fe_checkbox.setEnabled(has_data)
            c.preview_button.setEnabled(has_data)
            c.train_button.setEnabled(has_variables and has_models)
            if hasattr(c, "model_picker"):
                c.model_picker.setEnabled(has_variables)
            c.info_button.setEnabled(True)
            c.customize_plots_btn.setEnabled(True)
            c.shap_settings_btn.setEnabled(True)
            c.open_output_btn.setEnabled(True)
            c.reset_session_btn.setEnabled(True)
            for chk in c.model_checks.values():
                chk.setEnabled(True)
            for chk in c.plot_checks.values():
                chk.setEnabled(True)
            c.cv_mode_combo.setEnabled(True)
            c.cv_spin.setEnabled((c.cv_mode_combo.currentData() or "repeated") != "holdout")

        self._refresh_train_primary_action()

    def _start_next_queued_job(self):
        if self._active_job_id is not None:
            self.statusBar().showMessage(
                tr("status.job_already_running", default="A job is already running. New jobs remain queued.")
            )
            return

        next_job = None
        for job in self._jobs:
            if job.get("status") == "Queued":
                next_job = job
                break

        if next_job is None:
            self.statusBar().showMessage(tr("status.no_queued_jobs", default="No queued jobs to run."))
            self._sync_menu_actions()
            return

        # SSOT Fix: Stop applying job metadata back to UI controls!
        # Just launch it directly from the job dict without overriding user's active screen
        self._launch_job_id = int(next_job.get("id"))
        self._on_train()

    def _retry_failed_jobs(self):
        count = 0
        for job in self._jobs:
            if job.get("status") in {"Failed", "Cancelled"}:
                job["status"] = "Queued"
                job["attempt"] = int(job.get("attempt", 1)) + 1
                job["started_at"] = None
                job["finished_at"] = None
                job["elapsed"] = None
                job["message"] = tr("jobs.retry_queued", default="Retry queued")
                job["error"] = ""
                count += 1

        if count == 0:
            self.statusBar().showMessage(tr("status.no_failed_jobs", default="No failed or cancelled jobs to retry."))
            return

        self._refresh_job_table()
        retry_msg = tr("status.retry_queued", default="Retry queued for {count} job(s).", count=count)
        self._push_notification("info", retry_msg)
        self.statusBar().showMessage(retry_msg)
        if self._active_job_id is None:
            self._start_next_queued_job()
        self._sync_menu_actions()

    def _clear_finished_jobs(self):
        before = len(self._jobs)
        self._jobs = [
            j for j in self._jobs
            if j.get("status") not in {"Completed", "Failed", "Cancelled"}
        ]
        cleared = before - len(self._jobs)
        if cleared <= 0:
            self.statusBar().showMessage(tr("status.no_finished_jobs", default="No finished jobs to clear."))
            return

        self._refresh_job_table()
        clear_msg = tr("status.cleared_finished_jobs", default="Cleared {count} finished job(s).", count=cleared)
        self._push_notification("info", clear_msg)
        self.statusBar().showMessage(clear_msg)
        self._sync_menu_actions()

    @staticmethod
    def _snapshot_signature(snapshot: dict) -> str:
        try:
            return json.dumps(snapshot, sort_keys=True, ensure_ascii=True)
        except Exception:
            return str(snapshot)

    def _compose_snapshot(self) -> dict:
        c = self.controls
        selected_models = sorted([name for name, chk in c.model_checks.items() if chk.isChecked()])
        selected_plots = sorted([name for name, chk in c.plot_checks.items() if chk.isChecked()])
        cv_mode = c.cv_mode_combo.currentData() if c.cv_mode_combo.currentData() else "repeated"
        step_index = c.step_tabs.currentIndex() if hasattr(c, "step_tabs") else 0
        return {
            "has_data": bool(self.state.df is not None),
            "dataset_path": self._current_dataset_path,
            "target": str(self.state.target) if self.state.target is not None else None,
            "features": list(self.state.features or []),
            "studio_profile": dict(getattr(self.state, "studio_profile", {}) or {}),
            "selected_models": selected_models,
            "selected_plots": selected_plots,
            "cv_mode": str(cv_mode),
            "cv_folds": int(c.cv_spin.value()),
            "fe_enabled": bool(c.fe_checkbox.isChecked()),
            "model_hyperparams": {
                str(name): dict(params)
                for name, params in dict(getattr(self.state, "model_hyperparams", {}) or {}).items()
                if isinstance(params, dict) and params
            },
            "step_index": int(step_index),
        }

    def _capture_snapshot(self, clear_redo: bool = True):
        if self._snapshot_guard:
            return

        snapshot = self._compose_snapshot()
        signature = self._snapshot_signature(snapshot)
        if self._undo_stack and signature == self._snapshot_signature(self._undo_stack[-1]):
            self._sync_menu_actions()
            return

        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._snapshot_limit:
            self._undo_stack = self._undo_stack[-self._snapshot_limit:]
        if clear_redo:
            self._redo_stack.clear()
        self._save_recovery_checkpoint(snapshot)
        self._sync_menu_actions()

    def _on_user_setting_changed(self, *_args):
        self._capture_snapshot(clear_redo=True)

    def _studio_profile_data(self, profile: dict | None = None) -> dict:
        if isinstance(profile, dict):
            return dict(profile)
        raw = getattr(self.state, "studio_profile", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _studio_column_rename_map(self, profile: dict | None = None) -> dict[str, str]:
        prof = self._studio_profile_data(profile)
        rules = prof.get("naming_rules", [])
        if not isinstance(rules, list) or not rules:
            return {}

        selected = self._build_publication_selected_variables()
        selected_by_norm: dict[str, str] = {}
        for raw_name in selected:
            key = self._normalize_publication_term(raw_name)
            if key and key not in selected_by_norm:
                selected_by_norm[key] = raw_name

        rename_map: dict[str, str] = {}
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            src = str(rule.get("source", "")).strip()
            dst = str(rule.get("target", "")).strip()
            if not src or not dst:
                continue
            src_key = self._normalize_publication_term(src)
            selected_name = selected_by_norm.get(src_key)
            if not selected_name:
                continue
            if self._normalize_publication_term(selected_name) == self._normalize_publication_term(dst):
                continue
            rename_map[selected_name] = dst
        return rename_map

    def _studio_mapped_name(self, raw_name: str, profile: dict | None = None) -> str:
        txt = str(raw_name or "").strip()
        if not txt:
            return ""
        rename_map = self._studio_column_rename_map(profile)
        mapped = rename_map.get(txt, txt)
        return normalize_quotes_ascii(str(mapped))

    def _studio_value_rules_for_training(
        self,
        profile: dict | None = None,
        rename_map: dict[str, str] | None = None,
        allowed_scopes: tuple[str, ...] = ("all", "figures"),
    ) -> list[dict]:
        prof = self._studio_profile_data(profile)
        rules = prof.get("value_rules", [])
        if not isinstance(rules, list) or not rules:
            return []

        allowed_scope_set = {str(s).strip().lower() for s in allowed_scopes if str(s).strip()}
        if not allowed_scope_set:
            allowed_scope_set = {"all", "figures"}

        rename_map = dict(rename_map or {})
        rename_map_by_norm: dict[str, str] = {}
        for src, dst in rename_map.items():
            src_norm = self._normalize_publication_term(src)
            if src_norm and src_norm not in rename_map_by_norm:
                rename_map_by_norm[src_norm] = str(dst)

        normalized_rules: list[dict] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            src = str(rule.get("source", "")).strip()
            dst = str(rule.get("target", "")).strip()
            if not src or not dst:
                continue

            scope = str(rule.get("scope", "all") or "all").strip().lower()
            if scope not in {"all", "tables", "figures"}:
                continue
            if scope not in allowed_scope_set:
                continue

            column = str(rule.get("column", "")).strip()
            column_norms: set[str] = set()
            if column:
                column_norm = self._normalize_publication_term(column)
                if column_norm:
                    column_norms.add(column_norm)
                    mapped = rename_map_by_norm.get(column_norm)
                    if mapped:
                        mapped_norm = self._normalize_publication_term(mapped)
                        if mapped_norm:
                            column_norms.add(mapped_norm)

            src_norm = self._normalize_publication_term(src)
            if not src_norm:
                continue
            normalized_rules.append(
                {
                    "source_norm": src_norm,
                    "target": dst,
                    "column_norms": column_norms,
                }
            )
        return normalized_rules

    def _studio_feature_value_label_maps(
        self,
        feature_columns: list[str],
        normalized_rules: list[dict],
    ) -> dict[str, dict[str, str]]:
        if not feature_columns or not normalized_rules:
            return {}

        out: dict[str, dict[str, str]] = {}
        for col_name in feature_columns:
            col_txt = str(col_name).strip()
            if not col_txt:
                continue
            col_norm = self._normalize_publication_term(col_txt)
            if not col_norm:
                continue

            local_map: dict[str, str] = {}
            for rule in normalized_rules:
                if not isinstance(rule, dict):
                    continue
                col_norms = set(rule.get("column_norms", set()) or set())
                # Figure-level value labels must be column-specific to avoid
                # leaking generic codes (e.g., 1/2) into unrelated features.
                if not col_norms:
                    continue
                if col_norm not in col_norms:
                    continue

                src_norm = str(rule.get("source_norm", "")).strip()
                dst = str(rule.get("target", "")).strip()
                if not src_norm or not dst:
                    continue
                if src_norm not in local_map:
                    local_map[src_norm] = dst

            if local_map:
                out[col_txt] = local_map
        return out

    def _build_effective_training_state(self, profile: dict | None = None) -> tuple[AppState, dict]:
        rename_map = self._studio_column_rename_map(profile)
        value_rules = self._studio_value_rules_for_training(profile, rename_map=rename_map)

        base_runtime = {
            "rename_map": {},
            "value_rule_count": 0,
            "value_columns": 0,
            "feature_value_labels": {},
            "fe_enabled": bool(self.state.fe_enabled),
            "feature_count_before_fe": 0,
            "feature_count_after_fe": 0,
        }

        if self.state.df is None:
            return self.state, base_runtime

        try:
            pd = _pd()
            if not isinstance(self.state.df, pd.DataFrame):
                return self.state, base_runtime
        except Exception:
            return self.state, base_runtime

        raw_target = str(self.state.target or "").strip()
        raw_features = [str(f).strip() for f in list(self.state.features or []) if str(f).strip()]
        if not raw_target or not raw_features:
            return self.state, base_runtime

        mapped_target = rename_map.get(raw_target, raw_target)
        mapped_features = [rename_map.get(f, f) for f in raw_features]

        if not rename_map and not value_rules and not bool(self.state.fe_enabled):
            return self.state, {
                "rename_map": {},
                "value_rule_count": 0,
                "value_columns": 0,
                "feature_value_labels": {},
                "fe_enabled": False,
                "feature_count_before_fe": len(mapped_features),
                "feature_count_after_fe": len(mapped_features),
            }

        combined_norm = [self._normalize_publication_term(mapped_target)]
        combined_norm.extend(self._normalize_publication_term(x) for x in mapped_features)
        if len(combined_norm) != len(set(combined_norm)):
            raise RuntimeError(
                tr(
                    "dialogs.publication_helper.training_name_collision",
                    default="Studio variable labels create duplicate training names. Please keep target/features unique.",
                )
            )

        effective_df = self.state.df.copy(deep=True)
        applied_map = {}
        for src, dst in rename_map.items():
            if src in effective_df.columns and src != dst:
                applied_map[src] = dst
        if applied_map:
            effective_df = effective_df.rename(columns=applied_map)

        if mapped_target not in effective_df.columns:
            raise RuntimeError(
                tr(
                    "dialogs.publication_helper.training_target_missing",
                    default="Mapped target column was not found in the training dataset.",
                )
            )

        missing_features = [f for f in mapped_features if f not in effective_df.columns]
        if missing_features:
            raise RuntimeError(
                tr(
                    "dialogs.publication_helper.training_features_missing",
                    default="Some mapped feature columns were not found: {names}",
                    names=", ".join(missing_features[:8]),
                )
            )

        feature_value_labels = self._studio_feature_value_label_maps(mapped_features, value_rules)

        effective_state = AppState()
        effective_state.set_dataframe(effective_df)
        effective_state.set_features(mapped_target, mapped_features)
        effective_state.fe_enabled = bool(self.state.fe_enabled)
        try:
            effective_state.model_checks = {
                str(name): bool(chk.isChecked())
                for name, chk in getattr(self.controls, "model_checks", {}).items()
            }
        except Exception:
            effective_state.model_checks = dict(self.state.model_checks)
        effective_state.studio_profile = self._studio_profile_data(profile)
        return effective_state, {
            "rename_map": applied_map,
            "value_rule_count": sum(len(v) for v in feature_value_labels.values()),
            "value_columns": len(feature_value_labels),
            "feature_value_labels": feature_value_labels,
            "fe_enabled": bool(self.state.fe_enabled),
            "feature_count_before_fe": len(mapped_features),
            "feature_count_after_fe": len(mapped_features),
        }

    def _update_variable_selection_ui(self):
        c = self.controls
        has_data = self.state.df is not None
        has_models = any(chk.isChecked() for chk in c.model_checks.values())
        if not has_data:
            c.vars_button.setEnabled(False)
            if hasattr(c, "vars_target_subtitle"):
                c.vars_target_subtitle.setText(tr("controls.variables.row1_subtitle", default="Requires dataset to be loaded first"))
            if hasattr(c, "studio_btn"):
                c.studio_btn.setEnabled(False)
            c.preview_button.setEnabled(False)
            c.fe_checkbox.blockSignals(True)
            c.fe_checkbox.setChecked(False)
            c.fe_checkbox.blockSignals(False)
            c.fe_checkbox.setEnabled(False)
            try:
                c.selection_label.setVisible(False)
            except Exception:
                pass
            self._set_step2_selection_badge_state("blocked")
            c.kpi_target_value.setText(tr("status.kpi.not_selected", default="Not selected"))
            c.train_button.setEnabled(False)
            if hasattr(c, "model_picker"):
                c.model_picker.setEnabled(False)
            if hasattr(c, "studio_btn"):
                c.studio_btn.setToolTip(tr("controls.studio.disabled_no_data", default="Load a dataset first."))
            self._set_guided_step_state(has_data=False, has_variables=False, has_models=False)
            return

        c.vars_button.setEnabled(True)
        if hasattr(c, "vars_target_subtitle"):
            c.vars_target_subtitle.setText(tr("controls.variables.row1_subtitle_ready", default="Click to select target regression variable"))
        c.preview_button.setEnabled(True)
        c.fe_checkbox.setEnabled(True)
        try:
            c.selection_label.setVisible(True)
        except Exception:
            pass

        if self.state.target is None or not self.state.features:
            if hasattr(c, "studio_btn"):
                c.studio_btn.setEnabled(False)
                c.studio_btn.setToolTip(tr("controls.studio.disabled_no_vars", default="Select target and features to unlock Publication Studio."))
            c.selection_label.setText(
                tr(
                    "status.variables_pending",
                    default="0 Features Selected (Target pending)",
                )
            )
            self._set_step2_selection_badge_state("pending")
            c.selection_label.setToolTip("")
            c.kpi_target_value.setText(tr("status.kpi.not_selected", default="Not selected"))
            c.train_button.setEnabled(False)
            if hasattr(c, "model_picker"):
                c.model_picker.setEnabled(False)
            self._set_guided_step_state(has_data=True, has_variables=False, has_models=False)
            return

        target_disp = self._studio_mapped_name(str(self.state.target))
        feat_names = [self._studio_mapped_name(str(f)) for f in self.state.features]
        preview_n = 4
        feat_preview = ", ".join(feat_names[:preview_n])
        suffix = (
            tr("status.selection_more_suffix", default=", +{count} more", count=len(feat_names) - preview_n)
            if len(feat_names) > preview_n
            else ""
        )
        c.selection_label.setText(
            tr(
                "status.selection_target_features",
                default="Target: {target} | Features: {count} selected",
                target=target_disp,
                count=len(feat_names),
            )
            + (f" ({feat_preview}{suffix})" if feat_preview else "")
        )
        c.selection_label.setToolTip(
            tr("status.selected_features_tooltip", default="Selected features:\n")
            + (", ".join(feat_names) if feat_names else tr("common.none", default="none"))
        )
        
        if hasattr(c, "vars_target_subtitle"):
            c.vars_target_subtitle.setText(f"Target: {target_disp} \n{len(feat_names)} Features selected")

        c.kpi_target_value.setText(target_disp)
        self._set_step2_selection_badge_state("ready")
        c.train_button.setEnabled(has_models)
        if hasattr(c, "studio_btn"):
            c.studio_btn.setEnabled(True)
            c.studio_btn.setToolTip(tr("controls.studio.tooltip", default="Configure publication-ready names for outputs."))
        if hasattr(c, "model_picker"):
            c.model_picker.setEnabled(True)
        self._set_guided_step_state(has_data=True, has_variables=True, has_models=has_models)

    def _set_step2_selection_badge_state(self, state: str):
        c = self.controls
        if not hasattr(c, "selection_label"):
            return
        try:
            c.selection_label.setProperty("state", str(state))
            c.selection_label.style().unpolish(c.selection_label)
            c.selection_label.style().polish(c.selection_label)
        except Exception:
            pass

    def _apply_snapshot(self, snapshot: dict):
        prev_guard = self._snapshot_guard
        self._snapshot_guard = True
        try:
            c = self.controls

            # Feature engineering checkbox is applied without side effects in undo/redo.
            fe_enabled = bool(snapshot.get("fe_enabled", False))
            c.fe_checkbox.blockSignals(True)
            c.fe_checkbox.setChecked(fe_enabled)
            c.fe_checkbox.blockSignals(False)
            self.state.fe_enabled = fe_enabled

            snap_path = snapshot.get("dataset_path")
            if isinstance(snap_path, str) and snap_path.strip():
                self._current_dataset_path = snap_path.strip()
            elif not snapshot.get("has_data"):
                self._current_dataset_path = None

            selected_models = set(snapshot.get("selected_models", []))
            for name, chk in c.model_checks.items():
                chk.setChecked(name in selected_models)

            selected_plots = set(snapshot.get("selected_plots", []))
            for name, chk in c.plot_checks.items():
                chk.setChecked(name in selected_plots)

            target_mode = str(snapshot.get("cv_mode", "repeated"))
            idx = 0
            for i in range(c.cv_mode_combo.count()):
                if c.cv_mode_combo.itemData(i) == target_mode:
                    idx = i
                    break
            c.cv_mode_combo.setCurrentIndex(idx)
            c.cv_spin.setValue(int(snapshot.get("cv_folds", c.cv_spin.value())))

            if self.state.df is not None:
                cols = set(self.state.df.columns)
                target = snapshot.get("target")
                feats = [f for f in snapshot.get("features", []) if f in cols and f != target]
                if target in cols:
                    self.state.target = target
                    self.state.features = feats
                else:
                    self.state.target = None
                    self.state.features = None
            else:
                self.state.target = None
                self.state.features = None

            raw_profile = snapshot.get("studio_profile", {})
            if isinstance(raw_profile, dict):
                self.state.studio_profile = dict(raw_profile)
            else:
                self.state.studio_profile = {}

            raw_hparams = snapshot.get("model_hyperparams", {})
            if isinstance(raw_hparams, dict):
                self.state.model_hyperparams = {
                    str(name): dict(params)
                    for name, params in raw_hparams.items()
                    if isinstance(params, dict)
                }
            else:
                self.state.model_hyperparams = {}

            self._update_variable_selection_ui()

            self._refresh_model_summary()
            self._update_runtime_hint()

            if hasattr(c, "step_tabs"):
                step_idx = int(snapshot.get("step_index", c.step_tabs.currentIndex()))
                self._go_to_step(step_idx)
        finally:
            self._snapshot_guard = prev_guard

        self._sync_menu_actions()

    def _undo_snapshot(self):
        if len(self._undo_stack) <= 1:
            self.statusBar().showMessage(tr("status.nothing_to_undo", default="Nothing to undo."))
            return

        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        target = self._undo_stack[-1]
        self._apply_snapshot(target)
        undo_msg = tr("status.undo_applied", default="Undo applied.")
        self.statusBar().showMessage(undo_msg)
        self._push_notification("info", undo_msg)

    def _redo_snapshot(self):
        if not self._redo_stack:
            self.statusBar().showMessage(tr("status.nothing_to_redo", default="Nothing to redo."))
            return

        snapshot = self._redo_stack.pop()
        self._undo_stack.append(snapshot)
        self._apply_snapshot(snapshot)
        redo_msg = tr("status.redo_applied", default="Redo applied.")
        self.statusBar().showMessage(redo_msg)
        self._push_notification("info", redo_msg)

    @staticmethod
    def _event_level_from_line(line: str) -> str:
        txt = str(line)
        if " ERROR:" in txt:
            return "ERROR"
        if " WARNING:" in txt:
            return "WARNING"
        if " SUCCESS:" in txt:
            return "SUCCESS"
        return "INFO"

    def _append_notification_item(self, line: str):
        c = self.controls
        if not hasattr(c, "notifications_list"):
            return
        level = self._event_level_from_line(line)
        item = QListWidgetItem(line)
        if level == "ERROR":
            item.setForeground(QColor("#8D1E1E"))
        elif level == "WARNING":
            item.setForeground(QColor("#8A6200"))
        elif level == "SUCCESS":
            item.setForeground(QColor("#1E633B"))
        else:
            item.setForeground(QColor("#2E4D67"))
        c.notifications_list.addItem(item)
        c.notifications_list.scrollToBottom()

    def _restore_event_log(self):
        settings = QSettings()
        raw = settings.value("ui/eventLog", [])
        if raw is None:
            rows = []
        elif isinstance(raw, str):
            rows = [raw] if raw.strip() else []
        else:
            try:
                rows = [str(x) for x in list(raw)]
            except Exception:
                rows = []

        self._event_log = rows[-250:]
        c = self.controls
        if hasattr(c, "notifications_list"):
            c.notifications_list.clear()
            for line in self._event_log:
                self._append_notification_item(str(line))

        if self._event_log:
            latest = str(self._event_log[-1])
            self._feedback_latest_event = latest.split(": ", 1)[1] if ": " in latest else latest
        else:
            self._feedback_latest_event = tr("controls.feedback.latest_none", default="No recent event")
        self._refresh_feedback_context()

    def _mark_session_dirty(self):
        settings = QSettings()
        try:
            clean_val = str(settings.value("session/cleanExit", "true")).strip().lower()
            self._previous_exit_was_clean = clean_val in ("1", "true", "yes", "on")
        except Exception:
            self._previous_exit_was_clean = True

        try:
            last_path = str(settings.value("ui/lastDatasetPath", "")).strip()
            self._current_dataset_path = last_path if last_path else None
        except Exception:
            self._current_dataset_path = None

        settings.setValue("session/cleanExit", False)
        settings.setValue("session/lastLaunchAt", int(time.time()))

    def _mark_session_clean(self):
        settings = QSettings()
        settings.setValue("session/cleanExit", True)
        settings.setValue("session/lastCloseAt", int(time.time()))

    def _save_recovery_checkpoint(self, snapshot: dict | None = None):
        if self._snapshot_guard:
            return
        try:
            snap = snapshot if isinstance(snapshot, dict) else self._compose_snapshot()
            jobs_payload: list[dict] = []
            for job in self._jobs[-160:]:
                if not isinstance(job, dict):
                    continue
                jobs_payload.append({
                    "id": int(job.get("id", 0)),
                    "status": str(job.get("status", "Queued")),
                    "created_at": float(job.get("created_at", time.time())),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "elapsed": job.get("elapsed"),
                    "attempt": int(job.get("attempt", 1)),
                    "selected_models": list(job.get("selected_models", [])),
                    "selected_plots": list(job.get("selected_plots", [])),
                    "cv_mode": str(job.get("cv_mode", "repeated")),
                    "cv_folds": int(job.get("cv_folds", 5)),
                    "studio_profile": dict(job.get("studio_profile", {}) or {}),
                    "model_hyperparams": {
                        str(name): dict(params)
                        for name, params in dict(job.get("model_hyperparams", {}) or {}).items()
                        if isinstance(params, dict)
                    },
                    "persist_outputs": bool(job.get("persist_outputs", False)),
                    "message": str(job.get("message", "")),
                    "error": str(job.get("error", "")),
                })

            payload = {
                "saved_at": time.time(),
                "snapshot": snap,
                "jobs": jobs_payload,
                "next_job_id": int(self._next_job_id_value),
            }
            settings = QSettings()
            settings.setValue("recovery/checkpointJson", json.dumps(payload, ensure_ascii=False))
            settings.setValue("recovery/checkpointAt", int(payload["saved_at"]))
        except Exception:
            LOGGER.exception("Failed to save recovery checkpoint")

    def _load_recovery_checkpoint(self):
        settings = QSettings()
        raw = settings.value("recovery/checkpointJson", "")
        if not raw:
            return None

        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            payload = json.loads(str(raw))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _restore_jobs_from_checkpoint(self, jobs_payload):
        restored: list[dict] = []
        if isinstance(jobs_payload, list):
            for raw in jobs_payload:
                if not isinstance(raw, dict):
                    continue
                try:
                    status = str(raw.get("status", "Queued"))
                    message = str(raw.get("message", ""))
                    if status == "Running":
                        status = "Cancelled"
                        message = tr("jobs.interrupted_shutdown", default="Interrupted by unexpected shutdown")

                    restored.append({
                        "id": int(raw.get("id", 0)),
                        "status": status,
                        "created_at": float(raw.get("created_at", time.time())),
                        "started_at": raw.get("started_at"),
                        "finished_at": raw.get("finished_at"),
                        "elapsed": raw.get("elapsed"),
                        "attempt": int(raw.get("attempt", 1)),
                        "selected_models": list(raw.get("selected_models", [])),
                        "selected_plots": list(raw.get("selected_plots", [])),
                        "cv_mode": str(raw.get("cv_mode", "repeated")),
                        "cv_folds": int(raw.get("cv_folds", 5)),
                        "studio_profile": dict(raw.get("studio_profile", {}) or {}),
                        "model_hyperparams": {
                            str(name): dict(params)
                            for name, params in dict(raw.get("model_hyperparams", {}) or {}).items()
                            if isinstance(params, dict)
                        },
                        "persist_outputs": bool(raw.get("persist_outputs", False)),
                        "message": message,
                        "error": str(raw.get("error", "")),
                    })
                except Exception:
                    continue

        self._jobs = restored
        self._active_job_id = None
        self._launch_job_id = None
        self._cancelled = False
        self._next_job_id_value = (max((int(j.get("id", 0)) for j in restored), default=0) + 1)
        # Re-entrancy guard for the training-thread spawn path. The main
        # train button is also disabled while running, but a frustrated
        # user firing the keyboard shortcut or menu action repeatedly
        # could still race the worker setup; this flag turns that race
        # into a no-op.
        self._training_launch_in_progress = False
        self._refresh_job_table()

    def _apply_loaded_dataset_state(self, df: pd.DataFrame, path: str, *, notify: bool = True, capture_state: bool = True):
        self.state.set_dataframe(df)
        self.state.target = None
        self.state.features = None
        self.state.studio_profile = {}

        c = self.controls
        c.vars_button.setEnabled(True)
        c.train_button.setEnabled(False)
        c.fe_checkbox.setEnabled(True)
        if hasattr(c, "model_picker"):
            c.model_picker.setEnabled(False)

        abs_path = os.path.abspath(path)
        self._current_dataset_path = abs_path
        fname = os.path.basename(abs_path)
        c.data_info_label.setText(
            tr(
                "status.dataset_loaded_info",
                default="{filename} - {rows:,} rows x {cols:,} columns",
                filename=fname,
                rows=df.shape[0],
                cols=df.shape[1],
            )
        )
        if hasattr(c, "data_empty_state"):
            c.data_empty_state.setVisible(False)
        if hasattr(c, "data_loaded_state"):
            c.data_loaded_state.setVisible(True)
        c.selection_label.setText(tr("status.target_not_selected_features_zero", default="Target: not selected | Features: 0"))
        c.selection_label.setToolTip("")
        c.preview_button.setEnabled(True)
        c.status_label.setText(tr("status.dataset_loaded_next_step", default="Dataset loaded. Continue with Step 2: Select Variables."))
        c.kpi_dataset_value.setText(f"{df.shape[0]:,} x {df.shape[1]:,}")
        c.kpi_target_value.setText(tr("status.kpi.not_selected", default="Not selected"))
        c.kpi_run_value.setText(tr("status.kpi.ready", default="Ready"))
        self._set_feedback_focus(
            now=tr("status.feedback.now_dataset_loaded", default="Dataset loaded"),
            next_step=tr("status.feedback.next_select_variables", default="Select target and features"),
            blockers=tr("controls.feedback.blockers_none", default="None"),
        )
        self.statusBar().showMessage(tr("status.loaded_path", default="Loaded: {path}", path=abs_path))

        settings = QSettings()
        settings.setValue("ui/lastDatasetPath", abs_path)

        self._set_guided_step_state(has_data=True, has_variables=False, has_models=False)
        self._go_to_step(1)
        self._update_runtime_hint()
        self._sync_menu_actions()

        if notify:
            self._push_notification(
                "success",
                tr(
                    "notifications.dataset_loaded",
                    default="Dataset loaded: {filename} ({rows:,}x{cols:,})",
                    filename=fname,
                    rows=df.shape[0],
                    cols=df.shape[1],
                ),
            )

        if capture_state:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._capture_snapshot(clear_redo=True)

    def _load_dataset_for_recovery(self, path: str) -> bool:
        try:
            df = self.state.load_dataset(path)
        except Exception:
            LOGGER.exception("Recovery dataset load failed for %s", path)
            return False

        crit, warns = self.state.validate(df)
        if crit:
            return False

        self._apply_loaded_dataset_state(df, path, notify=False, capture_state=False)
        if warns:
            self._push_notification(
                "warning",
                tr("notifications.recovered_dataset_warnings", default="Recovered dataset has {count} warning(s).", count=len(warns)),
            )
        return True

    def _restore_from_checkpoint_payload(self, payload: dict):
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
        dataset_loaded = False
        dataset_path = snapshot.get("dataset_path") if isinstance(snapshot, dict) else None
        if isinstance(dataset_path, str) and dataset_path and os.path.exists(dataset_path):
            dataset_loaded = self._load_dataset_for_recovery(dataset_path)
        elif isinstance(snapshot, dict) and snapshot.get("has_data"):
            self._push_notification(
                "warning",
                tr("notifications.checkpoint_dataset_missing", default="Checkpoint expected a dataset, but the file is not available."),
            )

        self._restore_jobs_from_checkpoint(payload.get("jobs", []))

        if isinstance(snapshot, dict) and snapshot:
            self._apply_snapshot(snapshot)

        self._undo_stack.clear()
        self._redo_stack.clear()
        self._capture_snapshot(clear_redo=True)
        return dataset_loaded

    def _maybe_recover_previous_session(self):
        if self._previous_exit_was_clean:
            return

        payload = self._load_recovery_checkpoint()
        if not payload:
            self._push_notification(
                "warning",
                tr("notifications.no_recovery_checkpoint", default="Previous session did not close cleanly. No recovery checkpoint found."),
            )
            return

        saved_at = payload.get("saved_at")
        try:
            when_text = time.strftime("%H:%M:%S", time.localtime(float(saved_at))) if saved_at else tr("common.unknown_time", default="unknown time")
        except Exception:
            when_text = tr("common.unknown_time", default="unknown time")

        answer = QMessageBox.question(
            self,
            tr("dialogs.recovery.recover_session_title", default="Recover Session"),
            tr(
                "dialogs.recovery.recover_session_message",
                default="Previous session ended unexpectedly.\nRestore the last checkpoint from {when}?",
                when=when_text,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._push_notification("info", tr("notifications.recovery_skipped", default="Recovery skipped by user."))
            return
        dataset_loaded = self._restore_from_checkpoint_payload(payload)
        if dataset_loaded:
            self._push_notification("success", tr("notifications.recovered_success", default="Previous session recovered from checkpoint."))
        else:
            self._push_notification("warning", tr("notifications.checkpoint_restored_partial", default="Checkpoint restored partially."))

    def _restore_last_checkpoint(self):
        payload = self._load_recovery_checkpoint()
        if not payload:
            QMessageBox.information(
                self,
                tr("dialogs.recovery.checkpoint_title", default="Checkpoint"),
                tr("dialogs.recovery.no_checkpoint", default="No recovery checkpoint was found."),
            )
            return

        answer = QMessageBox.question(
            self,
            tr("dialogs.recovery.restore_title", default="Restore Checkpoint"),
            tr(
                "dialogs.recovery.restore_message",
                default="Restore the latest saved checkpoint now?\nThis will replace current configuration and queue state.",
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        dataset_loaded = self._restore_from_checkpoint_payload(payload)
        if dataset_loaded:
            restored_msg = tr("notifications.checkpoint_restored", default="Checkpoint restored.")
            self._push_notification("success", restored_msg)
            self.statusBar().showMessage(restored_msg)
        else:
            partial_msg = tr("notifications.checkpoint_restored_partial", default="Checkpoint restored partially.")
            self._push_notification("warning", partial_msg)
            self.statusBar().showMessage(partial_msg)

    def _push_notification(self, level: str, message: str, **kwargs):
        level_norm = str(level or "info").upper()
        if level_norm not in {"INFO", "SUCCESS", "WARNING", "ERROR"}:
            level_norm = "INFO"
        clean_message = str(message).strip()
        if not clean_message:
            return

        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {level_norm}: {clean_message}"
        self._event_log.append(line)
        self._event_log = self._event_log[-250:]
        self._feedback_latest_event = clean_message
        self._append_notification_item(line)
        self._refresh_feedback_context()

    def _sync_menu_actions(self):
        c = self.controls
        has_variables = self.state.target is not None and bool(self.state.features)
        is_running = self._active_job_id is not None
        queued_jobs = sum(1 for j in self._jobs if j.get("status") == "Queued")
        failed_jobs = sum(1 for j in self._jobs if j.get("status") in {"Failed", "Cancelled"})
        finished_jobs = sum(1 for j in self._jobs if j.get("status") in {"Completed", "Failed", "Cancelled"})
        undo_count = max(len(self._undo_stack) - 1, 0)
        redo_count = len(self._redo_stack)
        checkpoint_available = self._load_recovery_checkpoint() is not None
        self._refresh_train_primary_action()

        def _set_checked(action_name: str, state: bool):
            act = getattr(self, action_name, None)
            if act is None:
                return
            act.blockSignals(True)
            try:
                act.setChecked(state)
            finally:
                act.blockSignals(False)

        _set_checked("act_view_model_pool", False)
        _set_checked("act_edit_feature_engineering", bool(c.fe_checkbox.isChecked()))

        if hasattr(self, "act_file_preview"):
            self.act_file_preview.setEnabled(c.preview_button.isEnabled())
        if hasattr(self, "act_edit_variables"):
            self.act_edit_variables.setEnabled(c.vars_button.isEnabled())
        if hasattr(self, "act_edit_feature_engineering"):
            self.act_edit_feature_engineering.setEnabled(c.fe_checkbox.isEnabled())
        if hasattr(self, "act_edit_publication_studio"):
            self.act_edit_publication_studio.setEnabled(has_variables)
        if hasattr(self, "act_edit_undo"):
            self.act_edit_undo.setEnabled(undo_count > 0)
        if hasattr(self, "act_edit_redo"):
            self.act_edit_redo.setEnabled(redo_count > 0)
        if hasattr(self, "act_run_start"):
            self.act_run_start.setEnabled(c.train_button.isEnabled())
        if hasattr(self, "act_run_cancel"):
            self.act_run_cancel.setEnabled(is_running and c.cancel_button.isEnabled())
        if hasattr(self, "act_run_next_queued"):
            self.act_run_next_queued.setEnabled((queued_jobs > 0) and (not is_running))
        if hasattr(self, "act_run_retry_failed"):
            self.act_run_retry_failed.setEnabled(failed_jobs > 0)
        if hasattr(self, "act_run_clear_finished"):
            self.act_run_clear_finished.setEnabled(finished_jobs > 0)
        if hasattr(self, "act_settings_customize_plots"):
            self.act_settings_customize_plots.setEnabled(c.customize_plots_btn.isEnabled())
        if hasattr(self, "act_settings_shap"):
            self.act_settings_shap.setEnabled(c.shap_settings_btn.isEnabled())
        if hasattr(self, "act_settings_open_output"):
            self.act_settings_open_output.setEnabled(c.open_output_btn.isEnabled())
        if hasattr(self, "act_settings_reset"):
            self.act_settings_reset.setEnabled(c.reset_session_btn.isEnabled())
        if hasattr(self, "act_settings_restore_checkpoint"):
            self.act_settings_restore_checkpoint.setEnabled(checkpoint_available)

        if hasattr(c, "jobs_run_next_btn"):
            c.jobs_run_next_btn.setEnabled((queued_jobs > 0) and (not is_running))
        if hasattr(c, "jobs_retry_failed_btn"):
            c.jobs_retry_failed_btn.setEnabled(failed_jobs > 0)
        if hasattr(c, "jobs_clear_finished_btn"):
            c.jobs_clear_finished_btn.setEnabled(finished_jobs > 0)

    def _sync_results_ui_state(self):
        c = self.controls
        has_result = bool(self._latest_result_dir)
        is_running = self._active_job_id is not None

        c.results_tabs.setEnabled(has_result)
        c.results_save_row.setVisible(has_result)
        c.results_save_button.setEnabled(has_result and (not self._latest_result_saved))
        c.results_retrain_button.setEnabled(not is_running)
        c.results_summary_text.setVisible(True)
        if not has_result:
            # Clear content so placeholder text becomes visible.
            c.results_summary_text.clear()
            c.results_save_status.setText(tr("results.save_status.not_saved", default="Run not saved"))
            c.results_save_status.setStyleSheet("")
            if not is_running:
                self._set_train_stage("setup")
        elif self._latest_result_saved:
            c.results_save_status.setText(tr("results.save_status.saved", default="✓ Saved to output/runs"))
            c.results_save_status.setStyleSheet("color: #1f7a3f; font-weight: 650;")
            if not is_running:
                self._set_train_stage("results")
        else:
            c.results_save_status.setText(
                tr("results.save_status.temporary", default="Temporary run: click Save Results to persist")
            )
            c.results_save_status.setStyleSheet("")
            if not is_running:
                self._set_train_stage("results")

    def _discover_result_images(self, run_dir: str):
        figure_records: list[dict] = []
        shap_records: list[dict] = []
        if not run_dir or not os.path.isdir(run_dir):
            return figure_records, shap_records

        for root, _dirs, files in os.walk(run_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, run_dir).replace("\\", "/")
                lower_rel = rel_path.lower()
                parts = [p for p in rel_path.split("/") if p]

                model = "Run"
                group = parts[0] if parts else "general"
                if len(parts) >= 3 and parts[0].lower() == "models":
                    model = parts[1]
                    group = parts[2]

                rec = {
                    "rel": rel_path,
                    "path": full_path,
                    "model": model,
                    "group": group,
                }

                if "shap" in lower_rel:
                    shap_records.append(rec)
                else:
                    figure_records.append(rec)

        figure_records.sort(key=lambda r: str(r.get("rel", "")))
        shap_records.sort(key=lambda r: str(r.get("rel", "")))
        return figure_records, shap_records

    def _set_combo_items(self, combo, values: list[tuple[str, str]], all_label: str):
        prev = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(all_label, userData="all")
            for label, data in values:
                combo.addItem(label, userData=data)
            idx = combo.findData(prev)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _refresh_figure_filter_options(self):
        c = self.controls
        if not hasattr(c, "figures_model_filter") or not hasattr(c, "figures_category_filter"):
            return

        model_values = sorted({str(r.get("model", "Run")) for r in self._figure_records})
        group_values = sorted({str(r.get("group", "general")) for r in self._figure_records})

        self._set_combo_items(
            c.figures_model_filter,
            [(m, m) for m in model_values],
            all_label=tr("controls.results.filters.all_models", default="All models"),
        )
        self._set_combo_items(
            c.figures_category_filter,
            [(g.replace("_", " ").title(), g) for g in group_values],
            all_label=tr("controls.results.filters.all_groups", default="All groups"),
        )

    def _refresh_shap_filter_options(self):
        c = self.controls
        if not hasattr(c, "shap_model_filter"):
            return

        model_values = sorted({str(r.get("model", "Run")) for r in self._shap_records})
        self._set_combo_items(
            c.shap_model_filter,
            [(m, m) for m in model_values],
            all_label=tr("controls.results.filters.all_models", default="All models"),
        )

    def _apply_figure_filters(self):
        c = self.controls
        selected_model = c.figures_model_filter.currentData() if hasattr(c, "figures_model_filter") else "all"
        selected_group = c.figures_category_filter.currentData() if hasattr(c, "figures_category_filter") else "all"

        filtered = {}
        for rec in self._figure_records:
            model = str(rec.get("model", "Run"))
            group = str(rec.get("group", "general"))
            if selected_model not in (None, "all") and model != str(selected_model):
                continue
            if selected_group not in (None, "all") and group != str(selected_group):
                continue
            filtered[str(rec.get("rel", ""))] = str(rec.get("path", ""))

        self._figures_map = filtered
        self._fill_image_list(c.figures_list, self._figures_map)
        if not self._figures_map:
            c.figures_img.clear()

    def _apply_shap_filters(self):
        c = self.controls
        selected_model = c.shap_model_filter.currentData()

        filtered = {}
        for rec in self._shap_records:
            model = str(rec.get("model", "Run"))
            if selected_model not in (None, "all") and model != str(selected_model):
                continue
            filtered[str(rec.get("rel", ""))] = str(rec.get("path", ""))

        self._shap_map = filtered
        self._fill_image_list(c.shap_list, self._shap_map)
        if not self._shap_map:
            c.shap_img.clear()

    def _fill_image_list(self, list_widget, mapping: dict[str, str]):
        list_widget.clear()
        for rel in mapping.keys():
            list_widget.addItem(rel)
        if list_widget.count() > 0:
            list_widget.setCurrentRow(0)

    def _load_image_to_label(self, path: str | None, label):
        if not path or not os.path.exists(path):
            label.clear()
            return
        pm = QPixmap(path)
        if pm.isNull():
            label.clear()
            return
        target_w = max(label.width(), 200)
        target_h = max(label.height(), 180)
        scaled = pm.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def _populate_results_gallery(self, run_dir: str):
        c = self.controls
        self._figure_records, self._shap_records = self._discover_result_images(run_dir)
        self._refresh_figure_filter_options()
        self._refresh_shap_filter_options()
        self._apply_figure_filters()
        self._apply_shap_filters()

    def _on_figure_item_changed(self, current, _previous):
        if current is None:
            self.controls.figures_img.clear()
            return
        rel = current.text()
        path = self._figures_map.get(rel)
        self._load_image_to_label(path, self.controls.figures_img)

    def _on_shap_item_changed(self, current, _previous):
        if current is None:
            self.controls.shap_img.clear()
            return
        rel = current.text()
        path = self._shap_map.get(rel)
        self._load_image_to_label(path, self.controls.shap_img)

    def _build_results_summary_html(self, metrics_df, best_model: str) -> str:
        if metrics_df is None or metrics_df.empty:
            return tr("results.no_metrics", default="No metrics to display.")

        best_row = metrics_df.iloc[0]
        metric_priority = [
            "accuracy", "f1", "precision", "recall", "auc", "roc_auc",
            "r2", "rmse", "mae", "mape", "mse",
        ]

        def _normalized(name: str) -> str:
            return str(name).strip().lower().replace("-", "_").replace(" ", "_")

        numeric_metric_cols: list[str] = []
        for col in metrics_df.columns:
            norm = _normalized(str(col))
            if norm in {"model", "validation_mode", "trainingtime", "training_time"}:
                continue
            try:
                float(best_row.get(col))
                numeric_metric_cols.append(str(col))
            except Exception:
                continue

        def _metric_rank(col: str) -> tuple[int, str]:
            norm = _normalized(col)
            for idx, token in enumerate(metric_priority):
                if token in norm:
                    return (idx, norm)
            return (len(metric_priority) + 1, norm)

        numeric_metric_cols = sorted(numeric_metric_cols, key=_metric_rank)
        display_metric_cols = numeric_metric_cols[:6]

        cols: list[str] = []
        if "model" in metrics_df.columns:
            cols.append("model")
        cols.extend(display_metric_cols)
        if "validation_mode" in metrics_df.columns:
            cols.append("validation_mode")
        if not cols:
            cols = list(metrics_df.columns[: min(7, len(metrics_df.columns))])

        top_df = metrics_df.loc[:, cols].head(8).copy()
        model_count = int(len(metrics_df.index))

        def _fmt(v):
            try:
                fv = float(v)
                return f"{fv:.4f}"
            except Exception:
                return str(v)

        chips = [f"{col}: {_fmt(best_row.get(col))}" for col in display_metric_cols[:4]]
        chips_html = " | ".join(html.escape(x) for x in chips) if chips else html.escape(
            tr("results.summary.metrics_unavailable", default="Key metrics unavailable in this run output.")
        )

        header = "".join(
            "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #d6dde6;'>"
            f"{html.escape(str(c))}</th>"
            for c in top_df.columns
        )
        rows = []
        for _idx, row in top_df.iterrows():
            cells = "".join(
                "<td style='padding:6px 8px; border-bottom:1px solid #edf1f5;'>"
                f"{html.escape(_fmt(val))}</td>"
                for val in row.tolist()
            )
            rows.append(f"<tr>{cells}</tr>")
        rows_html = "".join(rows)

        return (
            f"<h3 style='margin:0 0 10px 0; font-size:15px;'>"
            f"{html.escape(tr('results.best_model_prefix', default='Best model: {model}', model=best_model or '-'))}"
            f"</h3>"
            f"<p style='margin:0 0 8px 0; font-size:13px; opacity:0.95;'>{html.escape(tr('results.summary.models_evaluated', default='Models evaluated: {count}', count=model_count))}</p>"
            f"<p style='margin:0 0 14px 0; font-size:13px; opacity:0.95;'>{chips_html}</p>"
            "<p style='margin:0 0 10px 0; font-size:12px; opacity:0.85;'>Top candidates by validation metrics</p>"
            "<table style='border-collapse:collapse; width:100%;'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )

    def _render_results_from_run(self, metrics_df, stats_summary_df, out_info):
        c = self.controls
        self._latest_metrics_df = None
        self._latest_stats_summary_df = None
        try:
            pd = _pd()
            self._latest_metrics_df = metrics_df.copy(deep=True) if isinstance(metrics_df, pd.DataFrame) else None
            self._latest_stats_summary_df = (
                stats_summary_df.copy(deep=True) if isinstance(stats_summary_df, pd.DataFrame) else None
            )
        except Exception:
            self._latest_metrics_df = None
            self._latest_stats_summary_df = None

        if metrics_df is not None:
            if not metrics_df.empty:
                best_model = str(metrics_df.iloc[0]["model"]) if "model" in metrics_df.columns else ""
                summary_text = self._build_results_summary_html(metrics_df, best_model)
                c.results_summary_text.setHtml(summary_text)
            else:
                no_metrics = tr("results.no_metrics", default="No metrics to display.")
                c.results_summary_text.setPlainText(no_metrics)

        try:
            self._fill_table(c.metrics_table, metrics_df)
        except Exception:
            LOGGER.exception("Table fill failed for metrics_table")
        if stats_summary_df is not None:
            try:
                self._fill_table(c.stats_table, stats_summary_df)
            except Exception:
                LOGGER.exception("Table fill failed for stats_table")

        run_dir = ""
        persist_outputs = False
        try:
            if isinstance(out_info, dict):
                run_dir = str(out_info.get("run_dir") or "")
                persist_outputs = bool(out_info.get("persist_outputs", False))
        except Exception:
            pass

        self._latest_run_info = out_info if isinstance(out_info, dict) else None
        self._latest_result_dir = run_dir if run_dir and os.path.isdir(run_dir) else None
        self._latest_result_saved = bool(persist_outputs)

        if self._latest_result_dir:
            self._populate_results_gallery(self._latest_result_dir)
            run_paths = self._resolve_run_paths(self._latest_result_dir, out_info)

            best = str(metrics_df.iloc[0]["model"]) if metrics_df is not None and not metrics_df.empty else None
            if best and hasattr(c, "shap_model_filter"):
                idx_best = c.shap_model_filter.findData(best)
                if idx_best >= 0:
                    c.shap_model_filter.setCurrentIndex(idx_best)
                    self._apply_shap_filters()

            fe_prefix = "feature_engineering_" if getattr(self.state, "fe_enabled", False) else ""
            # R² bar chart is saved under canonical evaluation directory.
            r2_base = f"{fe_prefix}metrics_R2_cv.png" if fe_prefix else "metrics_R2_cv.png"
            r2_parent = run_paths.get("metrics") or run_paths.get("evaluation_legacy") or self._latest_result_dir
            r2_png = os.path.join(r2_parent, r2_base)
            if os.path.exists(r2_png):
                self._load_image_to_label(r2_png, c.figures_img)

            if best:
                # generate_shap_summary saves under:
                #   <run_dir>/models/<safe_model_name>/<figures_dir>/<shap_key>/
                # where shap_key = fe_prefix + model_name.
                shap_model_key = fe_prefix + best
                shap_png = os.path.join(
                    run_paths.get("models_root", os.path.join(self._latest_result_dir, "models")),
                    safe_folder_name(best, fallback="model"),
                    MANUSCRIPT_DIR,
                    shap_model_key,
                    f"{shap_model_key}_shap_summary_beeswarm.png",
                )
                if not os.path.exists(shap_png):
                    shap_png = os.path.join(
                        run_paths.get("models_root", os.path.join(self._latest_result_dir, "models")),
                        safe_folder_name(best, fallback="model"),
                        run_paths.get("figures_legacy", LEGACY_MANUSCRIPT_DIR),
                        shap_model_key,
                        f"{shap_model_key}_shap_summary_beeswarm.png",
                    )
                if os.path.exists(shap_png):
                    self._load_image_to_label(shap_png, c.shap_img)

        self._sync_results_ui_state()

    def _resolve_run_paths(self, run_dir: str, out_info: object) -> dict[str, str]:
        """Resolve run subdirectories from runtime info first, then manifest."""
        resolved: dict[str, str] = {}
        try:
            if isinstance(out_info, dict):
                raw_map = out_info.get("path_map")
                if isinstance(raw_map, dict):
                    resolved.update({str(k): str(v) for k, v in raw_map.items() if v})
        except Exception:
            pass
        try:
            manifest_path = os.path.join(str(run_dir or ""), "run_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh) or {}
                manifest_paths = manifest.get("paths")
                if isinstance(manifest_paths, dict):
                    resolved.update({str(k): str(v) for k, v in manifest_paths.items() if v})
        except Exception:
            LOGGER.exception("Failed to resolve run paths from manifest")

        run_dir_s = str(run_dir or "")
        if run_dir_s:
            resolved.setdefault("metrics", os.path.join(run_dir_s, EVALUATION_DIR))
            resolved.setdefault("evaluation_legacy", os.path.join(run_dir_s, LEGACY_EVALUATION_DIR))
            resolved.setdefault("feature_selection_legacy", os.path.join(run_dir_s, LEGACY_FEATURE_SELECTION_DIR))
            resolved.setdefault("figures_legacy", os.path.join(run_dir_s, LEGACY_MANUSCRIPT_DIR))
            resolved.setdefault("models_root", os.path.join(run_dir_s, "models"))
            resolved.setdefault("supplements_root", os.path.join(run_dir_s, "analysis", "supplements"))
        return resolved

    def _on_save_current_run(self):
        if not self._latest_result_dir or not os.path.isdir(self._latest_result_dir):
            QMessageBox.information(
                self,
                tr("results.save_run_title", default="Save Run"),
                tr("results.no_run_to_save", default="No result run is available to save."),
            )
            return

        if self._latest_result_saved:
            QMessageBox.information(
                self,
                tr("results.save_run_title", default="Save Run"),
                tr("results.already_saved", default="Current run is already saved in output/runs."),
            )
            return

        out_info = self._latest_run_info if isinstance(self._latest_run_info, dict) else {}
        run_id = str(out_info.get("run_id") or os.path.basename(self._latest_result_dir))
        target_root = get_output_root(output_dir=OUTPUT_DIR, run_tag=RUN_TAG)
        target_runs = os.path.join(str(target_root), "runs")
        os.makedirs(target_runs, exist_ok=True)

        safe_run_id = safe_folder_name(run_id, fallback="run")
        canonical_dir = os.path.join(target_runs, safe_run_id)
        src_real = os.path.realpath(self._latest_result_dir)
        dst_real = os.path.realpath(canonical_dir)
        dest_dir = canonical_dir

        # New algorithm writes directly to canonical output/runs. Copy only
        # for legacy runs that came from a non-canonical location.
        if src_real != dst_real:
            if os.path.exists(dest_dir):
                import time
                dest_dir = os.path.join(target_runs, f"{safe_run_id}_{int(time.time())}")
            try:
                import shutil
                shutil.copytree(self._latest_result_dir, dest_dir)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    tr("results.save_run_title", default="Save Run"),
                    tr("results.save_failed", default="Run could not be saved:\n{error}", error=e),
                )
                return

        self._latest_result_saved = True
        if isinstance(self._latest_run_info, dict):
            self._latest_run_info["persist_outputs"] = True
            self._latest_run_info["saved_dir"] = dest_dir
        self._mark_run_manifest_saved(dest_dir)

        self._sync_results_ui_state()
        self.statusBar().showMessage(tr("status.run_saved", default="Run saved: {path}", path=dest_dir))
        self._push_notification("success", tr("notifications.run_saved_to", default="Run saved to {path}", path=dest_dir))

    def _mark_run_manifest_saved(self, run_dir: str):
        try:
            manifest_path = os.path.join(str(run_dir or ""), "run_manifest.json")
            if not os.path.exists(manifest_path):
                return
            with open(manifest_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return
            payload["persist_outputs"] = True
            payload["saved_via_ui"] = True
            payload["saved_at"] = int(time.time())
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            LOGGER.exception("Failed to update run manifest save metadata")

    @staticmethod
    def _normalize_publication_term(value) -> str:
        txt = normalize_quotes_ascii(str(value if value is not None else "")).strip().lower()
        txt = txt.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
        txt = txt.replace("ş", "s").replace("ö", "o").replace("ç", "c")
        return txt

    def _build_publication_selected_variables(self) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()

        target = str(self.state.target or "").strip()
        if target:
            selected.append(target)
            seen.add(target.lower())

        for feat in list(self.state.features or []):
            name = str(feat).strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            selected.append(name)
            seen.add(key)
        return selected

    def _build_publication_column_profile(self) -> dict:
        profile: dict[str, dict] = {}

        selected_variables = self._build_publication_selected_variables()
        df = self.state.df
        if df is None:
            return profile

        try:
            pd = _pd()
            if not isinstance(df, pd.DataFrame) or df.empty:
                return profile
        except Exception:
            return profile

        for col_name in selected_variables:
            if col_name not in df.columns:
                continue

            series = df[col_name]
            rec = profile.setdefault(
                col_name,
                {
                    "dtype": str(series.dtype),
                    "tables": ["training_data"],
                    "sample_values": [],
                },
            )

            try:
                non_null = series.dropna()
            except Exception:
                non_null = series

            samples = []
            try:
                uniques = pd.unique(non_null.astype(object))
                for raw in list(uniques)[:20]:
                    txt = normalize_quotes_ascii(str(raw)).strip()
                    if not txt:
                        continue
                    samples.append(txt)
            except Exception:
                pass

            existing = [str(x) for x in rec.get("sample_values", [])]
            seen_samples = {x.lower() for x in existing}
            for s in samples:
                low = s.lower()
                if low in seen_samples:
                    continue
                existing.append(s)
                seen_samples.add(low)
                if len(existing) >= 20:
                    break
            rec["sample_values"] = existing

        return profile

    def _build_publication_target_options(self) -> list[str]:
        options: list[str] = []
        seen: set[str] = set()

        for name in self._build_publication_selected_variables():
            key = name.lower()
            if key in seen:
                continue
            options.append(name)
            seen.add(key)

        return options

    def _set_guided_step_state(self, has_data: bool, has_variables: bool, has_models: bool):
        c = self.controls
        if not hasattr(c, "step_tabs"):
            return
        c.step_tabs.setTabEnabled(0, True)
        c.step_tabs.setTabEnabled(1, bool(has_data))
        c.step_tabs.setTabEnabled(2, bool(has_variables))
        if c.step_tabs.count() > 3:
            c.step_tabs.setTabEnabled(3, bool(has_variables) and bool(has_models))

        # Keep Step 4 warnings polite and local (near the disabled action).
        if hasattr(c, "status_label") and (not getattr(c, "cancel_button", None) or not c.cancel_button.isVisible()):
            try:
                if not has_data:
                    c.status_label.setProperty("severity", "warn")
                    c.status_label.setText(
                        tr(
                            "status.train_requires_dataset",
                            default="⚠️ Please complete Step 1 (Dataset) before training.",
                        )
                    )
                else:
                    c.status_label.setProperty("severity", "neutral")
            except Exception:
                pass
        self._sync_step_tab_titles(has_data=has_data, has_variables=has_variables, has_models=has_models)

    def _sync_step_tab_titles(self, *, has_data: bool, has_variables: bool, has_models: bool):
        c = self.controls
        if not hasattr(c, "step_tabs"):
            return

        c.step_tabs.setTabText(0, tr("controls.tabs.step1", default="1. Dataset"))
        c.step_tabs.setTabText(1, tr("controls.tabs.step2", default="2. Variables"))
        c.step_tabs.setTabText(2, tr("controls.tabs.step3", default="3. Models"))
        if c.step_tabs.count() > 3:
            c.step_tabs.setTabText(3, tr("controls.tabs.step4", default="4. Train"))

        try:
            ok_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        except Exception:
            ok_icon = QIcon()

        try:
            # Only show checkmarks for *past* steps that are validated.
            # This prevents a "time-travel" checkmark from appearing on future tabs.
            current_idx = int(c.step_tabs.currentIndex())
            c.step_tabs.setTabIcon(0, ok_icon if (has_data and current_idx > 0) else QIcon())
            c.step_tabs.setTabIcon(1, ok_icon if (has_variables and current_idx > 1) else QIcon())
            c.step_tabs.setTabIcon(2, ok_icon if (has_models and current_idx > 2) else QIcon())
            if c.step_tabs.count() > 3:
                c.step_tabs.setTabIcon(3, QIcon())
        except Exception:
            pass

    def _go_to_step(self, index: int):
        c = self.controls
        if not hasattr(c, "step_tabs"):
            return
        try:
            bounded = max(0, min(int(index), c.step_tabs.count() - 1))
            c.step_tabs.setCurrentIndex(bounded)
        except Exception:
            pass

    def _open_models_panel(self):
        self._go_to_step(2)
        self.statusBar().showMessage(tr("status.models_opened", default="Models"))

    def _restore_window_state(self):
        settings = QSettings()
        raw_layout_version = settings.value("ui/layoutVersion", 0)
        try:
            layout_version = int(raw_layout_version)
        except Exception:
            layout_version = 0
            LOGGER.warning("Invalid ui/layoutVersion value '%s'; falling back to default layout.", raw_layout_version)

        if layout_version == UI_LAYOUT_VERSION:
            geometry = settings.value("ui/geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)

        self._update_header_density()

    def _update_header_density(self):
        is_compact = self.width() < 1180
        is_tight = self.width() < 980
        if hasattr(self.header, "titleLabel"):
            title_font = self.header.titleLabel.font()
            title_font.setPointSize(16 if is_compact else 20)
            self.header.titleLabel.setFont(title_font)
        if hasattr(self.header, "subtitleLabel"):
            sub_font = self.header.subtitleLabel.font()
            sub_font.setPointSize(9 if is_compact else 10)
            self.header.subtitleLabel.setFont(sub_font)
            if is_tight:
                self.header.subtitleLabel.setText(
                    tr("app.subtitle_compact", default="Build | Train | Evaluate")
                )
            else:
                self.header.subtitleLabel.setText(
                    tr("app.subtitle", default="Build, train and evaluate regression models")
                )
            self.header.subtitleLabel.setVisible(self.width() >= 860)
        if hasattr(self.header, "setMaximumHeight"):
            self.header.setMaximumHeight(92 if is_compact else 112)

    def closeEvent(self, event):
        settings = QSettings()
        settings.setValue("ui/geometry", self.saveGeometry())
        settings.setValue("ui/layoutVersion", UI_LAYOUT_VERSION)
        settings.setValue("ui/eventLog", self._event_log[-250:])
        self._save_recovery_checkpoint()
        self._mark_session_clean()
        i18n.remove_listener(self._language_listener)
        super().closeEvent(event)

    def _refresh_model_summary_debounced(self):
        try:
            self._model_summary_timer.start()
        except Exception:
            self._refresh_model_summary()

    def _update_runtime_hint_debounced(self):
        try:
            self._runtime_hint_timer.start()
        except Exception:
            self._update_runtime_hint()

    def _refresh_model_summary(self):
        c = self.controls
        selected = [name for name, chk in c.model_checks.items() if chk.isChecked()]
        total = len(c.model_checks)
        has_data = self.state.df is not None
        has_variables = self.state.target is not None and bool(self.state.features)
        has_models = bool(selected)

        c.model_summary_label.setText(
            tr(
                "status.training_queue_header",
                default="Training Queue ({selected}/{total})",
                selected=len(selected),
                total=total,
            )
        )

        if hasattr(c, "model_picker"):
            c.model_picker.setEnabled(has_variables)

        if hasattr(c, "model_selected_list"):
            try:
                lst = c.model_selected_list
                lst.clear()

                # Import model descriptions
                try:
                    from models.model_descriptions import MODEL_DESCRIPTIONS
                except Exception:
                    MODEL_DESCRIPTIONS = {}

                for name in sorted(selected):
                    item = QListWidgetItem("")
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(6, 2, 6, 2)
                    row_layout.setSpacing(6)

                    label = QLabel(str(name))
                    # Set tooltip to model description if available
                    desc = MODEL_DESCRIPTIONS.get(str(name), str(name))
                    label.setToolTip(desc)
                    item.setToolTip(desc)

                    btn = QPushButton("×")
                    btn.setObjectName("removeChip")
                    btn.setToolTip(tr("controls.models.remove", default="Remove"))
                    btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    try:
                        btn.setMinimumSize(24, 24)
                    except Exception:
                        pass
                    try:
                        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    except Exception:
                        pass
                    btn.clicked.connect(lambda _checked=False, n=name: c.model_checks.get(n).setChecked(False) if c.model_checks.get(n) else None)

                    row_layout.addWidget(label, 1)
                    row_layout.addWidget(btn, 0)

                    item.setSizeHint(row.sizeHint())
                    lst.addItem(item)
                    lst.setItemWidget(item, row)
            except Exception:
                pass

        c.train_button.setEnabled(has_variables and has_models)

        if not c.cancel_button.isVisible():
            if has_variables:
                if has_models:
                    c.status_label.setText(
                        tr("status.models_selected_next_step", default="Models selected. Continue with Step 4: Start Training.")
                    )
                else:
                    c.status_label.setText(
                        tr("status.variables_selected_next_step", default="Variables selected. Continue with Step 3: Choose Models.")
                    )

        self._set_guided_step_state(has_data=has_data, has_variables=has_variables, has_models=has_models)
        self._update_runtime_hint()

    def _on_cv_mode_changed(self, _text: str):
        c = self.controls
        mode = c.cv_mode_combo.currentData()
        is_holdout = (mode == 'holdout')
        c.cv_spin.setEnabled(not is_holdout)
        c.cv_folds_label.setEnabled(not is_holdout)

        # Avoid overriding key workflow messages while the user is training.
        if c.cancel_button.isVisible():
            self._update_runtime_hint()
            return

        mode_text = tr("status.holdout_fast", default="Hold-Out (fast run)") if is_holdout else c.cv_mode_combo.currentText()
        self.statusBar().showMessage(tr("status.validation_mode", default="Validation mode: {mode}", mode=mode_text))

        if self.state.df is None:
            c.status_label.setText(tr("status.ready_begin", default="Ready. Load a dataset to begin."))
        elif self.state.target is None or not self.state.features:
            c.status_label.setText(tr("status.dataset_loaded_next_step", default="Dataset loaded. Continue with Step 2: Select Variables."))
        else:
            has_models = any(chk.isChecked() for chk in c.model_checks.values())
            if has_models:
                c.status_label.setText(
                    tr("status.models_selected_next_step", default="Models selected. Continue with Step 4: Start Training.")
                )
            else:
                c.status_label.setText(
                    tr("status.variables_selected_next_step", default="Variables selected. Continue with Step 3: Choose Models.")
                )
        self._update_runtime_hint()

    def _update_runtime_hint(self):
        c = self.controls
        selected_models = [name for name, chk in c.model_checks.items() if chk.isChecked()]
        if not selected_models:
            c.runtime_hint_label.setText(
                tr("status.runtime_select_model", default="Estimated time: select at least one model")
            )
            try:
                c.runtime_hint_label.setToolTip("")
            except Exception:
                pass
            return

        mode = c.cv_mode_combo.currentData() or 'repeated'
        base_seconds = {
            'holdout': 8,
            'kfold': 22,
            'repeated': 38,
            'nested': 90,
        }.get(mode, 38)
        est_seconds = base_seconds * len(selected_models)

        selected_plots = [name for name, chk in c.plot_checks.items() if chk.isChecked()]
        script_labels = set(get_optional_script_label_map().keys())
        selected_scripts = [name for name in selected_plots if name in script_labels]
        selected_core_plots = [name for name in selected_plots if name not in script_labels]

        if any('SHAP' in p for p in selected_core_plots):
            est_seconds = int(est_seconds * 1.25)
        if getattr(self.state, 'fe_enabled', False):
            est_seconds = int(est_seconds * 1.12)
        if selected_scripts:
            # Extra analyses are post-run helpers and can add notable runtime.
            est_seconds += 20 * len(selected_scripts)

        mins, secs = divmod(max(est_seconds, 1), 60)
        if mins > 0:
            txt = tr(
                "status.runtime_estimate_min_short",
                default="Estimated time: ~{mins}m {secs}s",
                mins=mins,
                secs=secs,
            )
        else:
            txt = tr(
                "status.runtime_estimate_sec_short",
                default="Estimated time: ~{secs}s",
                secs=secs,
            )
        c.runtime_hint_label.setText(txt)

        try:
            detail = tr(
                "status.runtime_estimate_tooltip",
                default="Models: {models}\nValidation: {mode}\nExtra analysis tasks: {extra}",
                models=len(selected_models),
                mode=str(c.cv_mode_combo.currentText() or ""),
                extra=len(selected_scripts),
            )
            c.runtime_hint_label.setToolTip(detail)
        except Exception:
            pass

    def _refresh_plot_check_states_from_settings(self):
        """Sync the sidebar plot checkboxes with persisted QSettings, in case they were changed in the dialog."""
        from PySide6.QtCore import QSettings
        settings = QSettings()
        script_labels = set(get_optional_script_label_map().keys())
        legacy_titles = {
            "Model Fit": "📊 Model Fit",
            "Diagnostics": "🔍 Diagnostics",
            "Explainability": "📈 Explainability",
        }
        # We need the page titles; reuse helper
        from interface.widgets.checkboxes import get_plot_pages
        pages, _ = get_plot_pages()
        for title, items in pages.items():
            for name in items:
                if name in self.controls.plot_checks:
                    chk = self.controls.plot_checks[name]
                    val = settings.value(f"plots/{title}/{name}", None)
                    if val is None and title in legacy_titles:
                        val = settings.value(f"plots/{legacy_titles[title]}/{name}", None)
                    if val is None:
                        chk.setChecked(name not in script_labels)
                    else:
                        chk.setChecked(str(val).lower() in ("true", "1", "yes"))
        # Also update compact summary with selected plot sections and extra analysis count.
        pages, _ = get_plot_pages()
        chosen_sections = []
        selected_script_count = 0
        for title, items in pages.items():
            for name in items:
                val = settings.value(f"plots/{title}/{name}", None)
                if val is None and title in legacy_titles:
                    val = settings.value(f"plots/{legacy_titles[title]}/{name}", None)
                default_checked = name not in script_labels
                is_checked = default_checked if val is None else str(val).lower() in ("true", "1", "yes")
                if is_checked:
                    if name in script_labels:
                        selected_script_count += 1
                        continue
                    chosen_sections.append(title)
                    break
        # dedupe while keeping order
        seen = set(); dedup = []
        for s in chosen_sections:
            if s not in seen:
                dedup.append(s); seen.add(s)
        self.controls.plot_summary_label.setText(
            tr(
                "status.plot_summary",
                default="Plots: {plots} | Extra analyses: {count}",
                plots=(", ".join(dedup) if dedup else tr("common.none", default="none")),
                count=selected_script_count,
            )
        )
        self._update_runtime_hint()

    def _on_customize_plots(self):
        dlg = PlotSelectionDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._refresh_plot_check_states_from_settings()
            self.statusBar().showMessage(
                tr("status.plot_settings_saved", default="Plot and analysis settings saved.")
            )

    def _on_shap_settings(self):
        dlg = ShapSettingsDialog(self)
        dlg.exec()

    def _open_hyperparameter_dialog(self, model_name: str):
        """Open the per-model hyperparameter dialog and persist the result.

        Values are stored in ``self.state.model_hyperparams`` (SSOT) and passed
        directly to the estimator at training time (see training_runner and
        models.train). Unchanged models keep using sklearn defaults.
        """
        from interface.widgets.hyperparameter_dialog import HyperparameterDialog
        from models.hyperparameters import has_schema

        if not has_schema(model_name):
            QMessageBox.information(
                self,
                tr("dialogs.hyperparameters.title", default="Configure: {model}", model=model_name),
                tr(
                    "dialogs.hyperparameters.no_schema",
                    default="This model has no user-configurable hyperparameters.",
                ),
            )
            return

        current = dict(getattr(self.state, "model_hyperparams", {}).get(model_name, {}))
        dlg = HyperparameterDialog(model_name, current=current, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            values = dlg.values()
            if not hasattr(self.state, "model_hyperparams") or self.state.model_hyperparams is None:
                self.state.model_hyperparams = {}
            self.state.model_hyperparams[model_name] = values
            self._on_user_setting_changed()
            self._push_notification(
                "info",
                tr(
                    "notifications.hyperparameters_saved",
                    default="Custom hyperparameters saved for {model}.",
                    model=model_name,
                ),
            )
            self.statusBar().showMessage(
                tr(
                    "status.hyperparameters_saved",
                    default="Hyperparameters updated for {model}.",
                    model=model_name,
                )
            )
            self._refresh_model_summary_debounced()

    def _on_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    @staticmethod
    def _is_supported_dataset_file(path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in SUPPORTED_DATASET_EXTENSIONS

    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            if url.isLocalFile() and self._is_supported_dataset_file(url.toLocalFile()):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if self._is_supported_dataset_file(local_path):
                    self._load_dataset_path(local_path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _load_dataset_path(self, path: str):
        self._push_notification("info", tr("notifications.loading_dataset", default="Loading dataset in background..."), timeout=3000)
        self.statusBar().showMessage(tr("status.loading_data", default="Loading dataset..."))
        self.setEnabled(False)  # Lock UI
        
        self._load_thread = QThread(self)
        self._load_worker = _DatasetLoadWorker(self.state.load_dataset, path)
        self._load_worker.moveToThread(self._load_thread)
        
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_dataset_load_finished)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.start()

    def _on_dataset_load_finished(self, df, error_msg, path):
        self.setEnabled(True)
        self.statusBar().clearMessage()
        
        if error_msg is not None:
            QMessageBox.critical(self, tr("dialogs.load_error.title", default="Load Error"), str(error_msg))
            LOGGER.exception(f"Failed loading dataset from {path}")
            self._push_notification("error", tr("notifications.dataset_load_failed", default=f"Dataset load failed: {os.path.basename(path)}"))
            return
            
        dlg = DataPreviewDialog(df, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            crit, warns = self.state.validate(df)
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("dialogs.validation_error.title", default="Validation Error"),
                tr(
                    "dialogs.validation_error.message",
                    default="Dataset validation failed unexpectedly.\n{error}",
                    error=e,
                ),
            )
            LOGGER.exception("Dataset validation crashed for %s", path)
            self._push_notification(
                "error",
                tr(
                    "notifications.dataset_validation_crashed",
                    default="Dataset validation failed unexpectedly.",
                ),
            )
            return

        if crit:
            QMessageBox.critical(self, tr("dialogs.invalid_dataset.title", default="Invalid Dataset"), "\n".join(crit))
            self._push_notification(
                "error",
                tr("notifications.dataset_validation_failed", default="Dataset validation failed (critical issues found)."),
            )
            return
        if warns:
            QMessageBox.warning(self, tr("dialogs.warnings.title", default="Warnings"), "\n".join(warns))
            self._push_notification(
                "warning",
                tr("notifications.dataset_loaded_with_warnings", default="Dataset loaded with {count} warning(s).", count=len(warns)),
            )
        self._apply_loaded_dataset_state(df, path, notify=True, capture_state=True)

    def _on_load(self):
        settings = QSettings()
        last_path = str(settings.value("ui/lastDatasetPath", ""))
        start_dir = os.path.dirname(last_path) if last_path else ""
        dataset_glob = " ".join(f"*{ext}" for ext in SUPPORTED_DATASET_EXTENSIONS)
        file_filter = (
            f"Dataset Files ({dataset_glob});;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xlsm *.xls *.xlsb)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("dialogs.select_dataset.title", default="Select Dataset"),
            start_dir,
            file_filter,
        )
        if not path:
            return
        self._load_dataset_path(path)

    def _on_preview(self):
        if self.state.df is None:
            QMessageBox.information(
                self,
                tr("dialogs.no_data.title", default="No Data"),
                tr("dialogs.no_data.load_first", default="Load a dataset first."),
            )
            return
        dlg = DataPreviewDialog(self.state.df, self)
        dlg.exec()

    def _on_edit_vars(self):
        df = self.state.df
        if df is None:
            self._set_feedback_focus(
                now=tr("status.feedback.now_dataset_required", default="Dataset is required"),
                next_step=tr("controls.feedback.next_load_dataset", default="Load dataset"),
                blockers=tr("status.feedback.blocker_no_dataset", default="No dataset loaded"),
            )
            QMessageBox.information(
                self,
                tr("dialogs.no_data.title", default="No Data"),
                tr("dialogs.no_data.load_first", default="Load a dataset first."),
            )
            return
        dlg = ColumnSelectionDialog(df, self,
            initial_target=self.state.target,
            initial_features=self.state.features
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            t, feats = dlg.get_selection()
            self.state.set_features(t, feats)
            target_disp = self._studio_mapped_name(str(t))
            feat_names = [self._studio_mapped_name(str(f)) for f in feats]
            preview_n = 4
            feat_preview = ", ".join(feat_names[:preview_n])
            suffix = (
                tr("status.selection_more_suffix", default=", +{count} more", count=len(feat_names) - preview_n)
                if len(feat_names) > preview_n
                else ""
            )
            self.controls.selection_label.setText(
                tr(
                    "status.selection_target_features",
                    default="Target: {target} | Features: {count} selected",
                    target=target_disp,
                    count=len(feat_names),
                )
                + (f" ({feat_preview}{suffix})" if feat_preview else "")
            )
            self.controls.selection_label.setToolTip(
                tr("status.selected_features_tooltip", default="Selected features:\n")
                + (", ".join(feat_names) if feat_names else tr("common.none", default="none"))
            )
            self.controls.kpi_target_value.setText(target_disp)
            self._set_step2_selection_badge_state("ready")
            has_models = any(chk.isChecked() for chk in self.controls.model_checks.values())
            has_variables = bool(t and feats)
            
            self.controls.train_button.setEnabled(has_models and has_variables)
            if hasattr(self.controls, "model_picker"):
                self.controls.model_picker.setEnabled(has_variables)
            if hasattr(self.controls, "studio_btn"):
                self.controls.studio_btn.setEnabled(has_variables)
            self.controls.status_label.setText(
                tr("status.variables_selected_next_step", default="Variables selected. Continue with Step 3: Choose Models.")
            )
            self._set_feedback_focus(
                now=tr("status.feedback.now_variables_ready", default="Variables selected"),
                next_step=tr("status.feedback.next_choose_model", default="Choose at least one model"),
                blockers=tr("controls.feedback.blockers_none", default="None"),
            )
            self._set_guided_step_state(has_data=True, has_variables=True, has_models=has_models)
            # self._go_to_step(2)  # Avoid automatic jump so user can configure Feature Engineering in Step 2
            self._update_runtime_hint()
            self._sync_menu_actions()
            self._push_notification(
                "success",
                tr(
                    "notifications.variables_selected",
                    default="Variables selected: target={target}, features={count}",
                    target=target_disp,
                    count=len(feat_names),
                ),
            )

            # Sadece değişkenler ilk kez seçildiğinde sor
            if not self._has_prompted_for_studio:
                reply = QMessageBox.question(
                    self,
                    tr("publication.prompt.title", default="Publication Studio"),
                    tr("publication.prompt.msg", default="Would you like to open Publication Studio to configure naming rules and variable metadata now?\n\n(You can also access it later using the 'Publication Studio' button)"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._open_pretraining_publication_studio()
                self._has_prompted_for_studio = True

            self._capture_snapshot(clear_redo=True)

    def _open_pretraining_publication_studio(self) -> bool:
        if self.state.df is None or self.state.target is None or not self.state.features:
            return False

        profile = self._studio_profile_data()
        default_dir = os.path.join(str(get_output_root(output_dir=OUTPUT_DIR, run_tag=RUN_TAG)), "publication_ready")
        dlg = PublicationExportDialog(
            assets=[],
            default_output_dir=default_dir,
            parent=self,
            column_profile=self._build_publication_column_profile(),
            target_options=self._build_publication_target_options(),
            default_target=str(self.state.target or ""),
            selected_variables=self._build_publication_selected_variables(),
            setup_mode=True,
            initial_naming_rules=list(profile.get("naming_rules", [])),
            initial_value_rules=list(profile.get("value_rules", [])),
            initial_format_rules=list(profile.get("format_rules", [])),
        )

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False

        payload = dlg.get_setup_payload()
        self.state.studio_profile = dict(payload or {})
        self._update_variable_selection_ui()
        self._update_runtime_hint()
        self._sync_menu_actions()
        self._push_notification(
            "success",
            tr(
                "notifications.studio_profile_updated",
                default="Publication Studio settings saved and will be applied in the next training run.",
            ),
        )
        self._capture_snapshot(clear_redo=True)
        return True

    @Slot()
    def _on_training_worker_finished(self):
        """Runs on the GUI thread after training (see _TrainWorker.run)."""
        pack = self._pending_training_result
        self._pending_training_result = None
        if pack is None:
            return
        metrics_df, _fitted_models, stats_df, stats_summary_df, out_info, elapsed = pack
        c = self.controls
        c.progress_phase_label.setText(tr("status.completed", default="Completed"))
        c.progress_timing_label.setText(
            tr(
                "status.elapsed_eta_zero",
                default="Elapsed: {elapsed} | ETA: 0s",
                elapsed=self._format_duration(elapsed),
            )
        )
        c.cancel_button.setEnabled(False)
        c.cancel_button.setVisible(False)
        self._set_controls_for_run_state(False)
        self._set_train_stage("results")
        self._on_cv_mode_changed(c.cv_mode_combo.currentText())
        self.statusBar().showMessage(tr("status.training_complete_sec", default="Training complete in {seconds:.1f}s", seconds=elapsed))
        c.kpi_run_value.setText(tr("status.kpi.done_sec", default="Done ({seconds:.1f}s)", seconds=elapsed))
        self._set_feedback_focus(
            now=tr("status.feedback.now_training_completed", default="Training completed"),
            next_step=tr("status.feedback.next_review_results", default="Review decision summary and diagnostics"),
            blockers=tr("controls.feedback.blockers_none", default="None"),
        )

        try:
            self._render_results_from_run(metrics_df, stats_summary_df, out_info)
            c.results_tabs.setCurrentIndex(0)
        except Exception:
            LOGGER.exception("Failed to render results from training run")

        self._thread.quit()
        self._thread.wait()

        try:
            total_plots = c.plot_progress_bar.maximum()
            if total_plots > 0:
                c.plot_progress_bar.setValue(total_plots)
                c.plot_progress_bar.setVisible(True)
                self.statusBar().showMessage(tr("status.plots_ready", default="Plots and extra analyses are ready"))
            c.progress_train_stats_label.setText(
                tr("status.done_duration", default="done ({duration})", duration=self._format_duration(elapsed))
            )
            c.progress_plot_stats_label.setText(
                f"{c.plot_progress_bar.value()}/{max(c.plot_progress_bar.maximum(), 1)} (100%)"
                if c.plot_progress_bar.maximum() > 0 else "0/0 (0%)"
            )
        except Exception:
            LOGGER.exception("Failed updating completion UI")

        running_job = self._job_by_id(self._active_job_id)
        if running_job is not None:
            running_job["status"] = "Completed"
            running_job["finished_at"] = time.time()
            running_job["elapsed"] = float(elapsed)
            running_job["message"] = tr(
                "jobs.completed_in",
                default="Completed in {duration}",
                duration=self._format_duration(elapsed),
            )
            running_job["error"] = ""

        self._active_job_id = None
        self._cancelled = False
        self._refresh_job_table()
        self._sync_menu_actions()
        self._push_notification(
            "success",
            tr("notifications.training_completed", default="Training completed in {seconds:.1f}s.", seconds=elapsed),
        )
        self._go_to_step(3)
        if any(j.get("status") == "Queued" for j in self._jobs):
            self._start_next_queued_job()

    @Slot()
    def _on_training_worker_error(self):
        """Runs on the GUI thread when training raises (see _TrainWorker.run)."""
        msg = self._pending_training_error
        self._pending_training_error = None
        if msg is None:
            return
        c = self.controls
        elapsed = (time.monotonic() - self._run_started_at) if self._run_started_at else None
        was_cancelled = self._cancelled or "cancelled by user" in str(msg).strip().lower()

        c.progress_phase_label.setText(
            tr("status.cancelled", default="Cancelled") if was_cancelled else tr("status.failed", default="Failed")
        )
        c.progress_timing_label.setText(
            tr(
                "status.elapsed_eta_unknown",
                default="Elapsed: {elapsed} | ETA: --",
                elapsed=self._format_duration(elapsed),
            )
        )
        c.cancel_button.setEnabled(False)
        c.cancel_button.setVisible(False)
        self._set_controls_for_run_state(False)
        self._set_train_stage("setup")
        self._on_cv_mode_changed(c.cv_mode_combo.currentText())
        c.kpi_run_value.setText(
            tr("status.cancelled", default="Cancelled") if was_cancelled else tr("status.failed", default="Failed")
        )
        self._set_feedback_focus(
            now=(
                tr("status.feedback.now_cancelled", default="Run cancelled")
                if was_cancelled
                else tr("status.feedback.now_failed", default="Run failed")
            ),
            next_step=(
                tr("status.feedback.next_start_training", default="Start training")
                if was_cancelled
                else tr("status.feedback.next_retry_when_ready", default="Adjust settings and retry")
            ),
            blockers=(
                tr("controls.feedback.blockers_none", default="None")
                if was_cancelled
                else tr("status.feedback.blocker_training_failed", default="Training error requires attention")
            ),
        )

        if not was_cancelled:
            QMessageBox.warning(self, tr("dialogs.training_error.title", default="Training Error"), msg)

        self._thread.quit()
        self._thread.wait()

        running_job = self._job_by_id(self._active_job_id)
        if running_job is not None:
            running_job["status"] = "Cancelled" if was_cancelled else "Failed"
            running_job["finished_at"] = time.time()
            running_job["elapsed"] = float(elapsed) if elapsed is not None else None
            running_job["message"] = (
                tr("jobs.cancelled_by_user", default="Cancelled by user")
                if was_cancelled
                else tr("jobs.run_failed", default="Run failed")
            )
            running_job["error"] = str(msg)

        self._active_job_id = None
        self._cancelled = False
        self._refresh_job_table()
        self._sync_menu_actions()

        if was_cancelled:
            self._push_notification("warning", tr("notifications.training_cancelled", default="Training cancelled by user."))
            self.statusBar().showMessage(
                tr("status.current_job_cancelled_waiting", default="Current job cancelled. Queued jobs are waiting.")
            )
        else:
            self._push_notification("error", tr("notifications.training_failed", default="Training failed: {error}", error=msg))
            if any(j.get("status") == "Queued" for j in self._jobs):
                self._start_next_queued_job()

    def _on_train(self):
        c = self.controls

        # Re-entrancy guard: collapse multi-click / shortcut-spam into a
        # single execution. The flag is released at the end of this
        # method, so it only protects the spawn window, not the run.
        if self._training_launch_in_progress:
            return
        self._training_launch_in_progress = True
        try:
            self._on_train_impl()
        finally:
            self._training_launch_in_progress = False

    def _on_train_impl(self):
        c = self.controls

        # SSOT Queue Execution: If launching a queued job, bypass current UI state!
        if getattr(self, "_launch_job_id", None) is not None:
            queued_job = self._job_by_id(self._launch_job_id)
            if queued_job:
                request = {
                    "selected_models": queued_job.get("selected_models", []),
                    "selected_plots": queued_job.get("selected_plots", []),
                    "cv_mode": queued_job.get("cv_mode", "repeated"),
                    "cv_folds": queued_job.get("cv_folds", 5),
                    "studio_profile": queued_job.get("studio_profile", {}),
                    "model_hyperparams": queued_job.get("model_hyperparams", {}),
                    "persist_outputs": queued_job.get("persist_outputs", False),
                }
            else:
                request = self._collect_training_request()
        else:
            request = self._collect_training_request()
        if request is None:
            self._launch_job_id = None
            return

        # Strict pre-flight validation runs in the GUI thread to fail
        # fast and produce a friendly dialog instead of a worker-thread
        # stack trace if the dataset is incompatible with training.
        from core.data_validation import validate_training_input
        validation_report = validate_training_input(
            self.state.df, self.state.target, self.state.features
        )
        if validation_report.is_blocking:
            QMessageBox.critical(
                self,
                tr("dialogs.training_validation.blocking_title", default="Cannot Start Training"),
                validation_report.render(),
            )
            self._launch_job_id = None
            return
        if validation_report.has_warnings:
            proceed = QMessageBox.warning(
                self,
                tr("dialogs.training_validation.warning_title", default="Confirm Training"),
                tr(
                    "dialogs.training_validation.warning_body",
                    default="Pre-training check returned the following items:\n\n{report}\n\nContinue?",
                    report=validation_report.render(),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if proceed != QMessageBox.StandardButton.Yes:
                self._launch_job_id = None
                return

        if self._active_job_id is not None:
            queued_job = self._create_job(
                request,
                status="Queued",
                message=tr("jobs.queued_while_running", default="Queued while another job is running"),
            )
            queued_msg = tr("status.job_queued", default="Job #{id} queued", id=queued_job["id"])
            self.statusBar().showMessage(queued_msg)
            self._set_feedback_focus(
                now=tr("status.feedback.now_job_queued", default="Job queued"),
                next_step=tr("status.feedback.next_wait_running", default="Wait for active run to finish"),
                blockers=tr("status.feedback.blocker_active_job", default="Another job is currently running"),
            )
            self._push_notification("info", tr("notifications.job_queued", default="Job #{id} queued.", id=queued_job["id"]))
            self._sync_menu_actions()
            return

        selected_models = list(request["selected_models"])
        selected_plots = list(request["selected_plots"])
        cv_mode = str(request["cv_mode"])
        cv_folds = int(request["cv_folds"])
        request_hparams = {
            str(name): dict(params)
            for name, params in dict(request.get("model_hyperparams", {}) or {}).items()
            if isinstance(params, dict)
        }
        persist_outputs = bool(request.get("persist_outputs", c.persist_output_checkbox.isChecked() if hasattr(c, "persist_output_checkbox") else True))

        active_job = self._job_by_id(self._launch_job_id)
        if active_job is None or active_job.get("status") != "Queued":
            active_job = self._create_job(
                request,
                status="Queued",
                message=tr("jobs.preparing_to_run", default="Preparing to run"),
            )
        self._launch_job_id = None

        if not request_hparams and active_job is not None:
            request_hparams = {
                str(name): dict(params)
                for name, params in dict(active_job.get("model_hyperparams", {}) or {}).items()
                if isinstance(params, dict)
            }

        job_studio_profile = dict(active_job.get("studio_profile", {}) or {})
        try:
            effective_state, studio_runtime = self._build_effective_training_state(job_studio_profile)
        except Exception as e:
            err_msg = str(e)
            active_job["status"] = "Failed"
            active_job["finished_at"] = time.time()
            active_job["elapsed"] = 0.0
            active_job["message"] = tr(
                "jobs.studio_profile_invalid",
                default="Studio profile is invalid for training",
            )
            active_job["error"] = err_msg
            self._refresh_job_table()
            self._sync_menu_actions()
            self._push_notification(
                "error",
                tr(
                    "notifications.studio_profile_invalid",
                    default="Studio profile could not be applied: {error}",
                    error=err_msg,
                ),
            )
            QMessageBox.warning(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                err_msg,
            )
            if any(j.get("status") == "Queued" for j in self._jobs):
                self._start_next_queued_job()
            return

        active_job["status"] = "Running"
        active_job["started_at"] = time.time()
        active_job["finished_at"] = None
        active_job["elapsed"] = None
        active_job["message"] = tr(
            "jobs.running_attempt",
            default="Running attempt {attempt}",
            attempt=int(active_job.get("attempt", 1)),
        )
        active_job["error"] = ""
        self._active_job_id = int(active_job.get("id"))
        self._go_to_step(3)
        self._refresh_job_table()
        self._push_notification(
            "info",
            tr(
                "notifications.job_started",
                default="Job #{id} started with {models} model(s), CV={cv_mode}, folds={folds}",
                id=active_job["id"],
                models=len(selected_models),
                cv_mode=cv_mode,
                folds=(cv_folds if cv_mode != "holdout" else "n/a"),
            ),
        )

        # Prepare UI
        self._run_started_at = time.monotonic()
        self._plot_started_at = None
        self._set_train_stage("active")
        c.progress_panel.setVisible(True)
        c.progress_phase_label.setText(tr("status.phase_training", default="Phase 1/2: Model Training"))
        c.progress_timing_label.setText(tr("status.elapsed_eta_calculating", default="Elapsed: 0s | ETA: calculating..."))
        c.progress_train_stats_label.setText("0/0 (0%)")
        c.progress_plot_stats_label.setText("0/0 (0%)")
        c.progress_bar.setValue(0)
        c.progress_bar.setMaximum(1)
        c.progress_bar.setVisible(True)
        c.plot_progress_bar.setValue(0)
        c.plot_progress_bar.setMaximum(1)
        c.plot_progress_bar.setVisible(True)
        c.cancel_button.setVisible(True)
        c.cancel_button.setEnabled(True)
        # Results are only shown after run completion for cleaner UX.
        
        self._latest_run_info = None
        self._latest_result_dir = None
        self._latest_result_saved = False
        self._latest_metrics_df = None
        self._latest_stats_summary_df = None
        self._figures_map = {}
        self._shap_map = {}
        self._figure_records = []
        self._shap_records = []
        self._clear_table(c.metrics_table)
        self._clear_table(c.stats_table)
        c.figures_img.clear(); c.shap_img.clear()
        c.figures_list.clear()
        c.shap_list.clear()
        self._refresh_figure_filter_options()
        self._refresh_shap_filter_options()
        c.results_summary_text.clear()
        self._sync_results_ui_state()
        c.status_label.setText(tr("status.job_running", default="Job #{id} running...", id=active_job["id"]))
        c.kpi_run_value.setText(tr("status.kpi.running_job", default="Running #{id}", id=active_job["id"]))
        self._set_feedback_focus(
            now=tr("status.feedback.now_training_running", default="Training is running"),
            next_step=tr("status.feedback.next_monitor_progress", default="Monitor progress and wait for completion"),
            blockers=tr("controls.feedback.blockers_none", default="None"),
        )
        c.log_box.clear()
        rename_count = len(dict(studio_runtime.get("rename_map", {}) or {}))
        value_rule_count = int(studio_runtime.get("value_rule_count", 0) or 0)
        value_columns = int(studio_runtime.get("value_columns", 0) or 0)
        fe_enabled_for_run = bool(studio_runtime.get("fe_enabled", False))
        fe_before = int(studio_runtime.get("feature_count_before_fe", 0) or 0)
        feature_value_labels = dict(studio_runtime.get("feature_value_labels", {}) or {})
        if rename_count > 0 or value_rule_count > 0:
            self._append_log(
                tr(
                    "status.studio_profile_applied_detailed",
                    default="Studio profile applied: {rename_count} variable rename(s), {value_rule_count} value-label rule(s), {value_columns} feature label map(s).",
                    rename_count=rename_count,
                    value_rule_count=value_rule_count,
                    value_columns=value_columns,
                )
            )
        if fe_enabled_for_run:
            self._append_log(
                tr(
                    "status.feature_engineering_scheduled_for_training",
                    default="Feature engineering enabled. It will run during training in background (worker thread). Starting features: {before}.",
                    before=fe_before,
                )
            )
        self._set_controls_for_run_state(True)
        self._sync_menu_actions()

        # Create worker and thread
        self._cancelled = False

        # Build a stable per-run output id (so outputs don't scatter into versioned model folders).
        try:
            qsettings = QSettings()
            ds_path = str(qsettings.value("ui/lastDatasetPath", ""))
        except Exception:
            ds_path = ""
        dataset_label = os.path.splitext(os.path.basename(ds_path))[0] if ds_path else "dataset"
        from utils.paths import make_run_id
        run_id = make_run_id(prefix=f"job{active_job['id']}_{dataset_label}")

        # Read SHAP settings once in the UI thread (core runner must stay Qt-free)
        try:
            qsettings = QSettings()
            shap_settings = {
                "top_n": qsettings.value("shap/top_n", -1),
                "var_enabled": qsettings.value("shap/var_enabled", "false"),
                "var_thresh": qsettings.value("shap/var_thresh", ""),
                "always_include": qsettings.value("shap/always_include", ""),
                "dependence_mode": qsettings.value("shap/dependence_mode", "interventional"),
            }
        except Exception:
            shap_settings = {}

        # Safe cleanup of any previous dead thread to plug memory leaks
        try:
            if getattr(self, '_thread', None) is not None:
                # If the C++ object isn't deleted, this works. Otherwise it raises RuntimeError.
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
        except RuntimeError:
            pass # C++ object already deleted by deleteLater
            
        self._thread = None
        self._worker = None
        
        # Inject the hyperparameters into the effective_state so the training
        # adapter picks them up via state.model_hyperparams. This keeps the
        # worker signature stable while still honoring queued-job overrides.
        try:
            effective_state.model_hyperparams = dict(request_hparams)
        except Exception:
            LOGGER.exception("Could not attach model_hyperparams to effective_state")

        self._thread = QThread(self)
        self._worker = _TrainWorker(
            effective_state, selected_models, selected_plots, cv_mode, cv_folds,
            lambda: getattr(self, '_cancelled', False),
            run_id, dataset_label, persist_outputs, feature_value_labels, shap_settings,
            gui_app=self,
        )
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._thread.deleteLater)
        # Do NOT connect deleteLater/thread.quit to worker signals before the main-thread
        # completion handlers. deleteLater runs synchronously during emit; an early
        # thread.quit is queued before QueuedConnection slots — the worker can be destroyed
        # and Qt drops the still-queued UI slots, so results never render.
        self._thread.started.connect(self._worker.run)
        # Progress/log UI updates are driven via QMetaObject.invokeMethod from the worker
        # (see _TrainWorker.run) so callbacks are safe when joblib calls them from pool threads.
        # Training completion uses _pending_training_result + _on_training_worker_finished (invokeMethod).

        self._thread.start()

    def _on_open_output_folder(self):
        output_root = get_output_root(output_dir=OUTPUT_DIR, run_tag=RUN_TAG)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_root)))
        self.statusBar().showMessage(tr("status.opened_output_folder", default="Opened output folder: {path}", path=output_root))

    def _on_reset_session(self):
        answer = QMessageBox.question(
            self,
            tr("dialogs.reset_session.title", default="Reset Session"),
            tr("dialogs.reset_session.message", default="This will clear loaded data and visible results. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.state = AppState()
        self._current_dataset_path = None
        self._jobs = []
        self._active_job_id = None
        self._launch_job_id = None
        self._cancelled = False
        self._latest_run_info = None
        self._latest_result_dir = None
        self._latest_result_saved = False
        self._latest_metrics_df = None
        self._latest_stats_summary_df = None
        self._figures_map = {}
        self._shap_map = {}
        self._figure_records = []
        self._shap_records = []
        c = self.controls

        c.vars_button.setEnabled(False)
        c.train_button.setEnabled(False)
        c.fe_checkbox.setChecked(False)
        c.fe_checkbox.setEnabled(False)
        c.preview_button.setEnabled(False)
        if hasattr(c, "model_picker"):
            c.model_picker.setEnabled(False)
        c.cancel_button.setEnabled(False)
        c.cancel_button.setVisible(False)
        self._set_train_stage("setup")
        c.progress_phase_label.setText(tr("status.idle", default="Idle"))
        c.progress_train_stats_label.setText("0/0 (0%)")
        c.progress_plot_stats_label.setText("0/0 (0%)")
        c.progress_timing_label.setText(tr("status.elapsed_eta_default", default="Elapsed: -- | ETA: --"))
        c.progress_bar.setValue(0)
        c.progress_bar.setMaximum(1)
        c.plot_progress_bar.setValue(0)
        c.plot_progress_bar.setMaximum(1)

        c.data_info_label.setText(tr("status.no_dataset_loaded", default="No dataset loaded yet."))
        if hasattr(c, "data_loaded_state"):
            c.data_loaded_state.setVisible(False)
        if hasattr(c, "data_empty_state"):
            c.data_empty_state.setVisible(True)
        c.selection_label.setText(tr("status.target_not_selected_features_zero", default="Target: not selected | Features: 0"))
        c.selection_label.setToolTip("")
        c.status_label.setText(tr("status.session_reset_load_begin", default="Session reset. Load a dataset to begin."))

        c.log_box.clear()
        c.results_summary_text.clear()
        c.figures_img.clear()
        c.shap_img.clear()
        c.figures_list.clear()
        c.shap_list.clear()
        self._refresh_figure_filter_options()
        self._refresh_shap_filter_options()
        self._clear_table(c.metrics_table)
        self._clear_table(c.stats_table)

        c.kpi_dataset_value.setText(tr("status.kpi.not_loaded", default="Not loaded"))
        c.kpi_target_value.setText(tr("status.kpi.not_selected", default="Not selected"))
        c.kpi_run_value.setText(tr("status.idle", default="Idle"))
        self._set_feedback_focus(
            now=tr("controls.feedback.now_ready", default="Ready"),
            next_step=tr("controls.feedback.next_load_dataset", default="Load dataset"),
            blockers=tr("controls.feedback.blockers_none", default="None"),
        )

        self._set_guided_step_state(has_data=False, has_variables=False, has_models=False)
        self._go_to_step(0)
        self._refresh_model_summary()
        self._refresh_plot_check_states_from_settings()
        self._sync_results_ui_state()
        self._refresh_job_table()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._sync_menu_actions()
        self.statusBar().showMessage(tr("status.session_reset", default="Session reset"))
        LOGGER.info("GUI session reset by user")
        self._push_notification("info", tr("notifications.session_reset_completed", default="Session reset completed."))
        self._capture_snapshot(clear_redo=True)

    def _on_info(self):
        # Open info.pdf with default PDF viewer.
        # get_project_root() resolves to PyInstaller's _MEIPASS in frozen
        # builds, so the bundled PDF is found regardless of install mode.
        pdf_path = str(get_project_root() / "info.pdf")
        if os.path.exists(pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
        else:
            # Fallback to built-in About dialog when a local guide is absent.
            self._on_about()

    @Slot(int, int)
    def _update_progress(self, current, total):
        c = self.controls
        c.progress_bar.setMaximum(total)
        c.progress_bar.setValue(current)
        if total > 0:
            pct = int((current / max(total, 1)) * 100)
            elapsed = (time.monotonic() - self._run_started_at) if self._run_started_at else None
            eta = None
            if current > 0 and elapsed is not None:
                eta = (elapsed / max(current, 1)) * max(total - current, 0)
            c.progress_phase_label.setText(tr("status.phase_training", default="Phase 1/2: Model Training"))
            c.progress_train_stats_label.setText(f"{current}/{total} ({pct}%)")
            c.progress_timing_label.setText(
                tr(
                    "status.elapsed_eta_dynamic",
                    default="Elapsed: {elapsed} | ETA: {eta}",
                    elapsed=self._format_duration(elapsed),
                    eta=self._format_duration(eta),
                )
            )
            c.status_label.setText(
                tr("status.training_progress", default="Training progress: {current}/{total} ({pct}%)", current=current, total=total, pct=pct)
            )
            c.kpi_run_value.setText(tr("status.kpi.running_pct", default="Running ({pct}%)", pct=pct))

    @Slot(int, int)
    def _update_plot_progress(self, current, total):
        c = self.controls
        if total <= 0:
            return
        if self._plot_started_at is None:
            self._plot_started_at = time.monotonic()
        elapsed_plot = time.monotonic() - self._plot_started_at
        eta_plot = (elapsed_plot / max(current, 1)) * max(total - current, 0) if current > 0 else None
        c.plot_progress_bar.setMaximum(total)
        c.plot_progress_bar.setVisible(True)
        c.plot_progress_bar.setValue(current)
        pct = int((current / max(total, 1)) * 100)
        c.progress_phase_label.setText(tr("status.phase_plots", default="Phase 2/2: Plots and Analyses"))
        c.progress_plot_stats_label.setText(f"{current}/{total} ({pct}%)")
        c.progress_timing_label.setText(
            tr(
                "status.elapsed_plot_eta",
                default="Elapsed: {elapsed} | Plot ETA: {eta}",
                elapsed=self._format_duration(time.monotonic() - self._run_started_at if self._run_started_at else None),
                eta=self._format_duration(eta_plot),
            )
        )
        c.kpi_run_value.setText(tr("status.kpi.plotting_pct", default="Plotting ({pct}%)", pct=pct))
        self.statusBar().showMessage(
            tr("status.generating_plots", default="Generating plots... {current}/{total} ({pct}%)", current=current, total=total, pct=pct)
        )

    @Slot(str)
    def _append_log(self, text: str):
        c = self.controls
        c.log_box.append(text)
        # Update mini status label when receiving START/DONE logs
        if text.startswith("START:") or text.startswith("Start"):
            c.status_label.setText(text)
            self.statusBar().showMessage(text)
        elif text.startswith("DONE:") or text.startswith("[Done]"):
            c.status_label.setText(text)
            self.statusBar().showMessage(text)
        elif text.startswith("[Warning]") or text.startswith("Warning"):
            c.status_label.setText(text)
            self.statusBar().showMessage(text)

    def _on_cancel(self):
        self._cancelled = True
        self.controls.kpi_run_value.setText(tr("status.cancelling", default="Cancelling"))
        self.controls.progress_phase_label.setText(tr("status.cancelling_dots", default="Cancelling..."))
        self.controls.status_label.setText(tr("status.cancelling_current_run", default="Cancelling current run..."))
        self._set_feedback_focus(
            now=tr("status.feedback.now_cancelling", default="Cancelling current run"),
            next_step=tr("status.feedback.next_wait_cancel", default="Wait for cancellation to complete"),
            blockers=tr("status.feedback.blocker_cancel_in_progress", default="Cancellation in progress"),
        )
        self.statusBar().showMessage(tr("status.cancelling_dots", default="Cancelling..."))
        running_job = self._job_by_id(self._active_job_id)
        if running_job is not None:
            running_job["message"] = tr("jobs.cancellation_requested", default="Cancellation requested")
            self._refresh_job_table()
        self._push_notification("warning", tr("notifications.cancellation_requested", default="Cancellation requested for current run."))

    def _on_fe_settings(self):
        c = self.controls
        dlg = FeatureEngineeringStudioDialog(self.state.fe_config, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = dlg.get_config()
            if new_config != self.state.fe_config:
                self.state.fe_config = new_config
                self._capture_snapshot(clear_redo=True)

            if dlg.export_requested:
                from features.feature_engineering import generate_static_fe_dataset
                from PySide6.QtWidgets import QMessageBox
                import os

                if self.state.df is None:
                    QMessageBox.warning(
                        self,
                        tr("error.dataset_load", default="Dataset Load Error"),
                        "No dataset loaded. Load a dataset before exporting.",
                    )
                    return

                df = self.state.df
                target = getattr(self.state, "target", None)
                dataset_path = getattr(self.state, "dataset_path", None) or ""
                out_dir = os.path.dirname(dataset_path) if dataset_path else os.getcwd()
                base_name = (
                    os.path.splitext(os.path.basename(dataset_path))[0]
                    if dataset_path
                    else "dataset"
                )
                out_file = f"{base_name}_engineered.csv"
                
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    out_path = generate_static_fe_dataset(df, self.state.fe_config, target, out_dir, out_file)
                    self.statusBar().showMessage(f"Static Engineered Dataset Generated: {out_file}", 6000)
                    self._push_notification("success", f"Static feature engineering file generated: {out_file}", title="FE Success")
                    
                    # Switch immediately to the generated dataset.
                    self._load_dataset_path(out_path)
                    
                    # Turn off 'Feature Engineering' checkbox since it's now statically embedded
                    if hasattr(self.controls, "fe_checkbox"):
                        self.controls.fe_checkbox.setChecked(False)
                        
                except Exception as e:
                    QMessageBox.critical(self, "Export Failed", f"Static Dataset generation failed:\n{e}")
                finally:
                    QApplication.restoreOverrideCursor()

    def _on_toggle_feature_engineering(self, checked: bool):
        # Feature engineering is applied in the training pipeline.
        self.state.fe_enabled = bool(checked)
        self._sync_menu_actions()
        self._update_runtime_hint()
        c = self.controls
        
        if hasattr(c, "fe_setup_btn"):
            c.fe_setup_btn.setVisible(checked)
            
        if not checked:
            self.statusBar().showMessage(tr("status.feature_engineering_disabled", default="Feature Engineering disabled"))
            self._push_notification("info", tr("notifications.feature_engineering_disabled", default="Feature engineering disabled."))
            self._capture_snapshot(clear_redo=True)
            return
        if self.state.df is None:
            QMessageBox.warning(
                self,
                tr("dialogs.no_data.title", default="No Data"),
                tr("dialogs.feature_engineering.load_dataset_first", default="Load a dataset before enabling Feature Engineering."),
            )
            # auto-uncheck if no data
            self.controls.fe_checkbox.setChecked(False)
            self._push_notification(
                "warning",
                tr("notifications.feature_engineering_dataset_required", default="Feature engineering cannot start without a dataset."),
            )
            return

        self.statusBar().showMessage(
            tr(
                "status.feature_engineering_enabled_training_phase",
                default="Feature Engineering enabled. It will be applied during training.",
            )
        )
        self._push_notification(
            "info",
            tr(
                "notifications.feature_engineering_enabled_training_phase",
                default="Feature engineering enabled and scheduled for the next training run.",
            ),
        )
        self._capture_snapshot(clear_redo=True)

    def _fill_table(self, table, df: pd.DataFrame):
        """Feed a table with MVC architecture to avoid UI freezing."""
        from interface.widgets.models import PandasTableModel
        try:
            if df is None or df.empty:
                if hasattr(table, "setModel"):
                    table.setModel(None)
                elif hasattr(table, "clear"):
                    table.clear()
                    table.setRowCount(0)
                    table.setColumnCount(0)
                return
            
            # Using QAbstractTableModel specifically to prevent UI freezing
            # caused by manual QTableWidgetItem loops! (MVC over GUI widgets)
            model = PandasTableModel(df, parent=table)
            
            if hasattr(table, "setModel"):
                table.setModel(model)
                if hasattr(table, "resizeColumnsToContents"):
                    table.resizeColumnsToContents()
            else:
                LOGGER.error("Cannot fill table: Does not support MVC setModel")
        except Exception as e:
            LOGGER.error(f"Failed to populate MVC table with dataframe: {e}", exc_info=True)


def run_app():
    app = QApplication(sys.argv)
    # Configure QSettings scope for deterministic persistence
    QCoreApplication.setOrganizationName("MLTrainer")
    QCoreApplication.setOrganizationDomain("local.mltrainer")
    QCoreApplication.setApplicationName("MLTrainerApp")

    # Install process-wide exception hooks immediately. After this point any
    # unhandled exception — main thread, worker thread, or queued slot —
    # surfaces as a clean modal instead of vanishing into a non-existent
    # console (critical for frozen .exe builds).
    from interface.widgets.error_dialog import install_global_exception_handlers
    install_global_exception_handlers()
    # Use INI format on Windows to avoid registry surprises
    try:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    except Exception:
        pass

    try:
        saved_lang = str(QSettings().value("ui/language", "")).strip().lower()
        if saved_lang in i18n.get_supported_languages():
            i18n.set_language(saved_lang)
    except Exception:
        pass

    # Typography: prefer Inter (if installed), otherwise fall back to system fonts.
    from PySide6.QtGui import QFont, QFontDatabase
    try:
        if "Inter" in QFontDatabase.families():
            app.setFont(QFont("Inter", 10))
        elif sys.platform == "darwin":
            app.setFont(QFont(".AppleSystemUIFont", 10))
        else:
            app.setFont(QFont("Segoe UI", 10))
    except Exception:
        pass
    # Set application icon.
    # get_project_root() resolves to PyInstaller's _MEIPASS in frozen
    # builds, so the bundled icon is located the same way as in source mode.
    icon_path = get_project_root() / "images" / "fau.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    # Avoid forcing a non-native palette on macOS.
    if sys.platform != "darwin":
        app.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#EEF2F6"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#18212B"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#005EA8"))
        app.setPalette(pal)

    # show splash/startup
    try:
        show_startup = str(QSettings().value("ui/show_startup", "true")).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        show_startup = True
    if show_startup:
        splash = StartupDialog()
        if splash.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)

    win = MLTrainerApp()
    win.show()
    sys.exit(app.exec())
