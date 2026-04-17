from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHBoxLayout, QPushButton,
    QComboBox, QScrollArea, QWidget, QCheckBox, QDialogButtonBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QRadioButton, QHeaderView, QAbstractItemView,
    QSizePolicy, QGridLayout, QFrame, QListWidget, QListWidgetItem, QFileDialog,
    QTabWidget, QTextEdit
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QMessageBox
import json
import os
import re
from pathlib import Path

from interface.widgets.checkboxes import (
    get_plot_pages,
    apply_settings_to_checks,
    get_optional_script_label_map,
    get_recommended_optional_script_labels,
)
from utils.localization import tr
from utils.text import normalize_quotes_ascii as _qascii

# Small helpers to persist/restore dialog geometry consistently
def _restore_geometry(widget: QDialog, key: str) -> None:
    try:
        s = QSettings()
        ba = s.value(key)
        if ba:
            widget.restoreGeometry(ba)
    except Exception:
        pass


def _save_geometry(widget: QDialog, key: str) -> None:
    try:
        s = QSettings()
        s.setValue(key, widget.saveGeometry())
        try:
            s.sync()
        except Exception:
            pass
    except Exception:
        pass

class DataPreviewDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.data_preview.title", default="Data Preview"))
        self.resize(980, 620)
        self.setMinimumSize(700, 420)
        self.df = df
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(8)

        # Controls row for quick viewport selection
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel(tr("dialogs.data_preview.rows", default="Rows:")))
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, max(1, len(df)))
        self.rows_spin.setValue(min(50, len(df)))
        ctrl.addWidget(self.rows_spin)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel(tr("dialogs.data_preview.columns", default="Columns:")))
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, max(1, len(df.columns)))
        self.cols_spin.setValue(min(30, len(df.columns)))
        ctrl.addWidget(self.cols_spin)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        self.note = QLabel("")
        self.note.setObjectName("hintLabel")
        lay.addWidget(self.note)

        tbl = QTableWidget(0, 0)
        self.tbl = tbl

        # Small usability touches
        tbl.setAlternatingRowColors(True)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSortingEnabled(True)
        tbl.setWordWrap(False)
        tbl.verticalHeader().setVisible(False)
        hh = tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        tbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(tbl)

        # Populate initial viewport and hook controls
        def _populate():
            try:
                n = max(1, min(int(self.rows_spin.value()), len(self.df)))
            except Exception:
                n = min(50, len(self.df))

            try:
                n_cols = max(1, min(int(self.cols_spin.value()), len(self.df.columns)))
            except Exception:
                n_cols = min(30, len(self.df.columns))

            display_cols = list(self.df.columns[:n_cols])
            self.note.setText(
                tr(
                    "dialogs.data_preview.note",
                    default="Showing first {rows:,} rows and {cols:,} of {total_cols:,} columns (dataset: {total_rows:,} x {total_cols:,})",
                    rows=n,
                    cols=n_cols,
                    total_rows=len(self.df),
                    total_cols=len(self.df.columns),
                )
            )

            self.tbl.setSortingEnabled(False)
            self.tbl.clear()
            self.tbl.setColumnCount(len(display_cols))
            self.tbl.setHorizontalHeaderLabels([_qascii(c) for c in display_cols])
            self.tbl.setRowCount(n)
            for i, row in enumerate(self.df.head(n)[display_cols].itertuples(index=False)):
                for j, val in enumerate(row):
                    self.tbl.setItem(i, j, QTableWidgetItem(str(val)))
            self.tbl.setSortingEnabled(True)

        _populate()
        self.rows_spin.valueChanged.connect(lambda _v: _populate())
        self.cols_spin.valueChanged.connect(lambda _v: _populate())

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        # Restore and persist geometry
        _restore_geometry(self, 'dialogs/DataPreview/geometry')
        btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/DataPreview/geometry'))
        btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/DataPreview/geometry'))

class ColumnSelectionDialog(QDialog):
    def __init__(self, df, parent=None, initial_target=None, initial_features=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.column_selection.title", default="Select Target and Features"))
        self.resize(760, 640)
        self.setMinimumSize(620, 480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(8)

        lay.addWidget(QLabel(tr("dialogs.column_selection.target", default="Target:")))
        # Show normalized display text but store raw name in userData
        self.target = QComboBox()
        for col in df.columns:
            self.target.addItem(_qascii(col), userData=col)
        if initial_target in df.columns:
            # set by userData
            try:
                for i in range(self.target.count()):
                    if self.target.itemData(i) == initial_target:
                        self.target.setCurrentIndex(i)
                        break
            except Exception:
                pass
        lay.addWidget(self.target)

        # Features header row with filter
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(tr("dialogs.column_selection.features", default="Features:")))
        hdr.addStretch()
        self.filter_edit = QLineEdit(); self.filter_edit.setPlaceholderText(tr("dialogs.column_selection.filter", default="Filter..."))
        self.filter_edit.setClearButtonEnabled(True)
        hdr.addWidget(self.filter_edit)
        lay.addLayout(hdr)
        self.filter_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self.filter_shortcut.activated.connect(self.filter_edit.setFocus)

        # Action buttons arranged for smaller widths
        actions = QGridLayout()
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(6)
        btn_sel_all = QPushButton(tr("dialogs.column_selection.select_all", default="Select All"))
        btn_sel_none = QPushButton(tr("dialogs.column_selection.deselect_all", default="Deselect All"))
        btn_sel_numeric = QPushButton(tr("dialogs.column_selection.only_numeric", default="Only Numeric"))
        btn_sel_categ = QPushButton(tr("dialogs.column_selection.only_categorical", default="Only Categorical"))
        actions.addWidget(btn_sel_all, 0, 0)
        actions.addWidget(btn_sel_none, 0, 1)
        actions.addWidget(btn_sel_numeric, 1, 0)
        actions.addWidget(btn_sel_categ, 1, 1)
        lay.addLayout(actions)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("hintLabel")
        lay.addWidget(self.summary_label)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget(); v = QVBoxLayout(cont)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        self.feats = {}
        for col in df.columns:
            # Render normalized text but keep mapping by raw name
            chk = QCheckBox(_qascii(col))
            if initial_features and col in initial_features:
                chk.setChecked(True)
            chk.stateChanged.connect(self._update_ok)
            self.feats[col] = chk
            v.addWidget(chk)
        v.addStretch()
        scroll.setWidget(cont); lay.addWidget(scroll)

        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)
        self.target.currentIndexChanged.connect(self._sync_target_with_features)
        self._sync_target_with_features()
        self._update_ok()

        # Wire filter and bulk select actions
        def apply_filter(text: str):
            t = text.strip().lower()
            for name, cb in self.feats.items():
                cb.setVisible(t in name.lower() if t else True)
            self._update_ok()
        self.filter_edit.textChanged.connect(apply_filter)

        def set_all(state: bool):
            for cb in self.feats.values():
                if cb.isVisible() and cb.isEnabled():
                    cb.setChecked(state)
        btn_sel_all.clicked.connect(lambda: set_all(True))
        btn_sel_none.clicked.connect(lambda: set_all(False))

        # Select only numeric (excludes the current target if present among features)
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)
        categ_cols = set(df.select_dtypes(exclude=["number"]).columns)
        def select_numeric():
            tgt = self.target.currentData()
            for name, cb in self.feats.items():
                if not cb.isVisible():
                    continue
                if name == tgt:
                    cb.setChecked(False)
                else:
                    cb.setChecked(name in numeric_cols)
        btn_sel_numeric.clicked.connect(select_numeric)

        def select_categorical():
            tgt = self.target.currentData()
            for name, cb in self.feats.items():
                if not cb.isVisible():
                    continue
                if name == tgt:
                    cb.setChecked(False)
                else:
                    cb.setChecked(name in categ_cols)
        btn_sel_categ.clicked.connect(select_categorical)

        # Restore and persist geometry
        _restore_geometry(self, 'dialogs/ColumnSelection/geometry')
        self.btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/ColumnSelection/geometry'))
        self.btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/ColumnSelection/geometry'))

    def _sync_target_with_features(self):
        tgt = self.target.currentData()
        for raw_name, cb in self.feats.items():
            is_target = (raw_name == tgt)
            cb.setEnabled(not is_target)
            if is_target:
                cb.setChecked(False)
                cb.setToolTip(tr("dialogs.column_selection.target_tooltip", default="Target variable cannot be selected as a feature."))
            else:
                cb.setToolTip("")
        self._update_ok()

    def _update_ok(self):
        checked = [name for name, cb in self.feats.items() if cb.isChecked() and cb.isEnabled()]
        visible = [name for name, cb in self.feats.items() if cb.isVisible()]
        ok = len(checked) > 0
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        self.summary_label.setText(
            tr(
                "dialogs.column_selection.summary",
                default="Selected features: {selected} (visible: {visible})",
                selected=len(checked),
                visible=len(visible),
            )
        )

    def get_selection(self):
        # Return raw column names (not normalized display text)
        try:
            tgt_raw = self.target.currentData()
            if tgt_raw is None:
                # fallback to text
                tgt_raw = self.target.currentText()
        except Exception:
            tgt_raw = self.target.currentText()
        return (
            tgt_raw,
            [c for c, cb in self.feats.items() if cb.isChecked() and cb.isEnabled()]
        )


    


class ModelSelectionDialog(QDialog):
    """On-demand model selection popup used by Train and View actions."""

    def __init__(self, model_names: list[str], selected_models: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.model_selection.title", default="Choose Models"))
        self.resize(560, 620)
        self.setMinimumSize(460, 420)
        self._selected_models = set(selected_models or [])

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        info = QLabel(tr("dialogs.model_selection.info", default="Select one or more models for training. You can reopen this popup any time."))
        info.setWordWrap(True)
        info.setObjectName("hintLabel")
        root.addWidget(info)

        hdr = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(tr("dialogs.model_selection.search", default="Search model..."))
        self.search_edit.setClearButtonEnabled(True)
        hdr.addWidget(self.search_edit, 1)
        btn_all = QPushButton(tr("dialogs.common.select_visible", default="Select Visible"))
        btn_none = QPushButton(tr("dialogs.common.clear_visible", default="Clear Visible"))
        hdr.addWidget(btn_all)
        hdr.addWidget(btn_none)
        root.addLayout(hdr)

        preset_grid = QGridLayout()
        preset_grid.setHorizontalSpacing(6)
        preset_grid.setVerticalSpacing(6)
        
        btn_fast = QPushButton(tr("dialogs.common.fast", default="Fast"))
        btn_balanced = QPushButton(tr("dialogs.common.balanced", default="Balanced"))
        btn_linear = QPushButton(tr("dialogs.common.linear", default="Linear"))
        btn_ensemble = QPushButton(tr("dialogs.common.ensemble", default="Ensemble"))
        btn_robust = QPushButton(tr("dialogs.common.robust", default="Robust"))
        btn_full = QPushButton(tr("dialogs.common.full", default="Full"))
        btn_clear = QPushButton(tr("dialogs.common.clear", default="Clear"))
        
        btn_fast.setToolTip(tr("dialogs.common.tip_fast", default="Quick baseline set for fastest iteration."))
        btn_balanced.setToolTip(tr("dialogs.common.tip_balanced", default="Balanced speed/quality starter set."))
        btn_linear.setToolTip(tr("dialogs.common.tip_linear", default="Linear-family models (interpretable and regularized)."))
        btn_ensemble.setToolTip(tr("dialogs.common.tip_ensemble", default="Tree ensemble models for non-linear patterns."))
        btn_robust.setToolTip(tr("dialogs.common.tip_robust", default="Strong mixed set for robust benchmark comparisons."))
        btn_full.setToolTip(tr("dialogs.common.tip_full", default="Enable all available models."))
        btn_clear.setToolTip(tr("dialogs.common.tip_clear", default="Disable all models and start from scratch."))
        
        preset_grid.addWidget(btn_fast, 0, 0)
        preset_grid.addWidget(btn_balanced, 0, 1)
        preset_grid.addWidget(btn_linear, 0, 2)
        preset_grid.addWidget(btn_ensemble, 1, 0)
        preset_grid.addWidget(btn_robust, 1, 1)
        preset_grid.addWidget(btn_full, 1, 2)
        preset_grid.addWidget(btn_clear, 2, 0, 1, 3)
        root.addLayout(preset_grid)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("hintLabel")
        root.addWidget(self.summary_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        self.model_checks: dict[str, QCheckBox] = {}
        for name in model_names:
            cb = QCheckBox(str(name))
            cb.setChecked(name in self._selected_models)
            cb.toggled.connect(self._update_ok)
            self.model_checks[name] = cb
            v.addWidget(cb)
        v.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self.btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        root.addWidget(self.btns)

        self.search_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self.search_shortcut.activated.connect(self.search_edit.setFocus)

        def apply_filter(text: str):
            t = text.strip().lower()
            for name, cb in self.model_checks.items():
                cb.setVisible((t in name.lower()) if t else True)
            self._update_ok()

        self.search_edit.textChanged.connect(apply_filter)

        def set_visible(state: bool):
            for cb in self.model_checks.values():
                if cb.isVisible() and cb.isEnabled():
                    cb.setChecked(state)

        btn_all.clicked.connect(lambda: set_visible(True))
        btn_none.clicked.connect(lambda: set_visible(False))

        # Model presets
        def apply_preset(names):
            for name, cb in self.model_checks.items():
                cb.setChecked(name in names)

        model_order = [
            "LinearRegression","RidgeCV","RandomForest","HistGB",
            "GradientBoostingRegressor","Lasso","ElasticNet",
            "SVR","KNeighborsRegressor","XGBoost"
        ]
        recommended = {"RidgeCV", "RandomForest", "HistGB"}
        preset_linear = {"LinearRegression", "RidgeCV", "Lasso", "ElasticNet"}
        preset_ensemble = {"RandomForest", "HistGB", "GradientBoostingRegressor", "XGBoost"}
        preset_robust = {"RidgeCV", "RandomForest", "HistGB", "XGBoost"}
        preset_full = set(model_order)

        btn_fast.clicked.connect(lambda: apply_preset({"LinearRegression", "RidgeCV", "HistGB"}))
        btn_balanced.clicked.connect(lambda: apply_preset(recommended))
        btn_linear.clicked.connect(lambda: apply_preset(preset_linear))
        btn_ensemble.clicked.connect(lambda: apply_preset(preset_ensemble))
        btn_robust.clicked.connect(lambda: apply_preset(preset_robust))
        btn_full.clicked.connect(lambda: apply_preset(preset_full))
        btn_clear.clicked.connect(lambda: set_visible(False))

        _restore_geometry(self, 'dialogs/ModelSelection/geometry')
        self.btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/ModelSelection/geometry'))
        self.btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/ModelSelection/geometry'))
        self._update_ok()
        self.search_edit.setFocus()

    def _update_ok(self):
        checked = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        visible = [name for name, cb in self.model_checks.items() if cb.isVisible()]
        self.summary_label.setText(
            tr(
                "dialogs.model_selection.summary",
                default="Selected: {selected} | Visible: {visible}",
                selected=len(checked),
                visible=len(visible),
            )
        )
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(len(checked) > 0)

    def get_selected_models(self) -> list[str]:
        return [name for name, cb in self.model_checks.items() if cb.isChecked()]


class PlotSelectionDialog(QDialog):
    """Dialog to customize which plots are enabled, grouped by pages.
    Uses same pages structure and QSettings keys as the main sidebar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.plot_selection.title", default="Customize Plots..."))
        self.resize(740, 700)
        self.setMinimumSize(600, 480)
        self.settings = QSettings()
        pages, tooltips = get_plot_pages()
        self.script_labels = set(get_optional_script_label_map().keys())
        self.recommended_script_labels = set(get_recommended_optional_script_labels())
        self.recommended_plot_labels = {
            "Residuals",
            "Predictions vs Actual",
            "Learning Curve",
            "Q-Q Plot",
            "Residual Distribution",
            "Feature Importance",
            "SHAP Summary",
        }

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)
        info_row = QHBoxLayout()
        info = QLabel(
            tr(
                "dialogs.plot_selection.info",
                default="Choose visual outputs and optional extra analyses to run automatically after training.",
            )
        )
        info.setWordWrap(True)
        info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        info_row.addWidget(info, 1)
        info_row.addStretch()
        # Quick search box to filter plot names
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText(tr("dialogs.plot_selection.search", default="Search plots or analyses..."))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(220)
        info_row.addWidget(self.search_edit)
        root.addLayout(info_row)
        self.search_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self.search_shortcut.activated.connect(self.search_edit.setFocus)

        # Live summary of selected plot count
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("hintLabel")
        root.addWidget(self.summary_label)

        # Scrollable area containing pages
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget(); v = QVBoxLayout(cont)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self.page_checks: dict[str, list[QCheckBox]] = {}
        self.page_headers: dict[str, QWidget] = {}

        # Global select/deselect row
        hdr = QHBoxLayout()
        btn_all = QPushButton(tr("dialogs.common.select_visible", default="Select Visible"))
        btn_none = QPushButton(tr("dialogs.common.clear_visible", default="Clear Visible"))
        hdr.addWidget(btn_all); hdr.addWidget(btn_none); hdr.addStretch()
        root.addLayout(hdr)

        def set_all_global(state: bool):
            for lst in self.page_checks.values():
                for c in lst:
                    if c.isVisible():
                        c.setChecked(state)
        btn_all.clicked.connect(lambda: set_all_global(True))
        btn_none.clicked.connect(lambda: set_all_global(False))

        preset_grid = QGridLayout()
        preset_grid.setSpacing(6)
        preset_grid.addWidget(QLabel(tr("dialogs.plot_selection.quick_presets", default="Quick presets:")), 0, 0, 2, 1, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        btn_visuals_only = QPushButton(tr("dialogs.plot_selection.visuals_only", default="Visuals Only"))
        btn_diagnostics = QPushButton(tr("dialogs.plot_selection.diagnostics", default="Diagnostics"))
        btn_explainability = QPushButton(tr("dialogs.plot_selection.explainability", default="Explainability"))
        btn_recommended = QPushButton(tr("dialogs.plot_selection.research_default", default="Research Default"))
        btn_everything = QPushButton(tr("dialogs.plot_selection.everything", default="Everything"))
        btn_clear = QPushButton(tr("dialogs.plot_selection.clear_all", default="Clear All"))
        
        btn_visuals_only.setToolTip(tr("dialogs.plot_selection.tip_visuals_only", default="Enable all plot visuals and disable extra analyses."))
        btn_diagnostics.setToolTip(tr("dialogs.plot_selection.tip_diagnostics", default="Keep only model-fit and diagnostics visuals enabled."))
        btn_explainability.setToolTip(tr("dialogs.plot_selection.tip_explainability", default="Keep only explainability visuals enabled (importance and SHAP)."))
        btn_recommended.setToolTip(tr("dialogs.plot_selection.tip_research_default", default="Uses widely reported regression diagnostics: residuals vs fitted, parity plot, Q-Q, residual distribution..."))
        btn_everything.setToolTip(tr("dialogs.plot_selection.tip_everything", default="Enable all visuals and all extra analyses."))
        btn_clear.setToolTip(tr("dialogs.plot_selection.tip_clear_all", default="Disable all visuals and extra analyses."))
        
        # Row 1
        preset_grid.addWidget(btn_visuals_only, 0, 1)
        preset_grid.addWidget(btn_diagnostics, 0, 2)
        preset_grid.addWidget(btn_explainability, 0, 3)
        # Row 2
        preset_grid.addWidget(btn_recommended, 1, 1)
        preset_grid.addWidget(btn_everything, 1, 2)
        preset_grid.addWidget(btn_clear, 1, 3)
        
        # Add stretch conceptually via columns
        preset_grid.setColumnStretch(4, 1)
        root.addLayout(preset_grid)

        def apply_preset(kind: str):
            diagnostic_plot_labels = {
                "Residuals",
                "Predictions vs Actual",
                "Learning Curve",
                "Q-Q Plot",
                "Residual Distribution",
                "Correlation Matrix",
            }
            explainability_plot_labels = {
                "Feature Importance",
                "Feature Importance Heatmap",
                "SHAP Summary",
                "SHAP Dependence",
            }
            for lst in self.page_checks.values():
                for cb in lst:
                    name = cb.text()
                    if kind == "visuals":
                        cb.setChecked(name not in self.script_labels)
                    elif kind == "diagnostics":
                        cb.setChecked(name in diagnostic_plot_labels)
                    elif kind == "explainability":
                        cb.setChecked(name in explainability_plot_labels)
                    elif kind == "recommended":
                        if name in self.script_labels:
                            cb.setChecked(name in self.recommended_script_labels)
                        else:
                            cb.setChecked(name in self.recommended_plot_labels)
                    elif kind == "clear":
                        cb.setChecked(False)
                    else:
                        cb.setChecked(True)

        btn_visuals_only.clicked.connect(lambda: apply_preset("visuals"))
        btn_diagnostics.clicked.connect(lambda: apply_preset("diagnostics"))
        btn_explainability.clicked.connect(lambda: apply_preset("explainability"))
        btn_recommended.clicked.connect(lambda: apply_preset("recommended"))
        btn_everything.clicked.connect(lambda: apply_preset("all"))
        btn_clear.clicked.connect(lambda: apply_preset("clear"))

        for title, items in pages.items():
            # Per-page header with select/deselect
            head = QHBoxLayout()
            header_widget = QWidget()
            header_widget.setObjectName("plotPageHeader")
            header_widget.setLayout(head)
            head.addWidget(QLabel(f"{title}"))
            page_all = QPushButton(tr("dialogs.column_selection.select_all", default="Select All"))
            page_none = QPushButton(tr("dialogs.column_selection.deselect_all", default="Deselect All"))
            head.addStretch(); head.addWidget(page_all); head.addWidget(page_none)
            v.addWidget(header_widget)
            self.page_headers[title] = header_widget

            local_checks: list[QCheckBox] = []
            for nm in items:
                chk = QCheckBox(nm)
                if nm in tooltips:
                    chk.setToolTip(tooltips[nm])
                v.addWidget(chk)
                local_checks.append(chk)
                # Update summary when any checkbox toggles
                chk.toggled.connect(lambda _=False: self._update_summary())
            # Initialize from settings; persist only when user presses OK.
            apply_settings_to_checks(title, local_checks, self.settings, persist_live=False)
            self.page_checks[title] = local_checks

            def set_all(state: bool, lst=local_checks):
                for c in lst:
                    if c.isVisible():
                        c.setChecked(state)
            page_all.clicked.connect(lambda _, lst=local_checks: set_all(True, lst))
            page_none.clicked.connect(lambda _, lst=local_checks: set_all(False, lst))

        v.addStretch()
        scroll.setWidget(cont)
        root.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
        # Wire search filter
        def apply_plot_filter(text: str):
            t = text.strip().lower()
            for title, checks in self.page_checks.items():
                any_visible = False
                for cb in checks:
                    visible = (t in cb.text().lower()) if t else True
                    cb.setVisible(visible)
                    any_visible = any_visible or visible
                header = self.page_headers.get(title)
                if header is not None:
                    header.setVisible(any_visible)
            self._update_summary()
        self.search_edit.textChanged.connect(apply_plot_filter)

        # Initial summary, geometry restore/persist
        self._update_summary()
        _restore_geometry(self, 'dialogs/PlotSelection/geometry')
        btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/PlotSelection/geometry'))
        btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/PlotSelection/geometry'))

        self.search_edit.setFocus()

    def accept(self):
        """Persist all current checkbox states explicitly on OK, then close.
        This lets Cancel close the dialog without mutating saved preferences.
        """
        try:
            # Iterate through the stored page_checks and write current states
            for title, checks in getattr(self, 'page_checks', {}).items():
                for chk in checks:
                    key = f"plots/{title}/{chk.text()}"
                    self.settings.setValue(key, bool(chk.isChecked()))
            # Flush to backing store
            try:
                self.settings.sync()
            except Exception:
                pass
        except Exception:
            # Do not block closing on settings errors
            pass
        # Save geometry on close
        _save_geometry(self, 'dialogs/PlotSelection/geometry')
        super().accept()

    def _update_summary(self) -> None:
        try:
            script_labels = set(get_optional_script_label_map().keys())
            total = sum(len(v) for v in self.page_checks.values())
            selected = sum(1 for lst in self.page_checks.values() for c in lst if c.isChecked())
            visible_total = sum(1 for lst in self.page_checks.values() for c in lst if c.isVisible())
            script_total = sum(1 for lst in self.page_checks.values() for c in lst if c.text() in script_labels)
            script_selected = sum(1 for lst in self.page_checks.values() for c in lst if c.isChecked() and c.text() in script_labels)
            plot_total = max(total - script_total, 0)
            plot_selected = max(selected - script_selected, 0)
            self.summary_label.setText(
                tr(
                    "dialogs.plot_selection.summary",
                    default="Plots: {plot_selected}/{plot_total} | Extra analyses: {script_selected}/{script_total} (visible: {visible_total})",
                    plot_selected=plot_selected,
                    plot_total=plot_total,
                    script_selected=script_selected,
                    script_total=script_total,
                    visible_total=visible_total,
                )
            )
        except Exception:
            self.summary_label.setText("")


class ShapSettingsDialog(QDialog):
    """Dedicated dialog to configure SHAP analysis behavior.
    Default behavior: analyze all features. User can optionally limit Top-N,
    apply a variance threshold, and specify always-include features.
    Persisted via QSettings under 'shap/*' keys.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.shap_settings.title", default="SHAP Settings"))
        self.resize(640, 340)
        self.setMinimumSize(560, 300)
        settings = QSettings()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)
        # Row with info label and small tooltip/clickable info
        title_row = QHBoxLayout()
        info = QLabel(tr("dialogs.shap_settings.info", default="Default: SHAP analyzes all features. Configure options below to customize."))
        info.setWordWrap(True)
        title_row.addWidget(info)
        title_row.addStretch()
        info_btn = QPushButton()
        info_btn.setObjectName("shapInfoButton")
        try:
            from PyQt6.QtWidgets import QStyle
            info_btn.setIcon(info_btn.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        except Exception:
            info_btn.setText("ⓘ")
        info_btn.setFlat(True)
        info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        info_btn.setToolTip(
            tr(
                "dialogs.shap_settings.info_tip",
                default="SHAP settings are saved and applied to future training runs automatically. Change them here anytime; they persist via QSettings.",
            )
        )
        def _show_info():
            QMessageBox.information(
                self,
                tr("dialogs.shap_settings.title", default="SHAP Settings"),
                tr(
                    "dialogs.shap_settings.info_detail",
                    default="These settings control how SHAP analyses are run:\n\n• All features vs Top-N: Limit plots to the top-N most important features, or include all.\n• Variance threshold: Optionally drop near-constant features to reduce noise.\n• Always include: Ensure certain raw feature names are always analysed.\n\nYour choices are persisted via QSettings and applied on the next training run.",
                ),
            )
        info_btn.clicked.connect(_show_info)
        title_row.addWidget(info_btn)
        root.addLayout(title_row)

        # Mode: All vs Top-N
        mode_row = QHBoxLayout()
        self.rb_all = QRadioButton(tr("dialogs.shap_settings.all_features", default="All features"))
        self.rb_topn = QRadioButton(tr("dialogs.shap_settings.top_n", default="Top-N"))
        self.topn_spin = QSpinBox(); self.topn_spin.setRange(1, 10000); self.topn_spin.setValue(20)
        self.topn_spin.setToolTip(tr("dialogs.shap_settings.top_n_tip", default="When enabled, limit SHAP analysis to the top-N most important features."))
        mode_row.addWidget(self.rb_all)
        mode_row.addSpacing(8)
        mode_row.addWidget(self.rb_topn)
        mode_row.addWidget(self.topn_spin)
        mode_row.addStretch()
        root.addLayout(mode_row)

        # Variance threshold (optional)
        var_row = QHBoxLayout()
        self.var_chk = QCheckBox(tr("dialogs.shap_settings.var_checkbox", default="Apply variance threshold"))
        self.var_spin = QDoubleSpinBox(); self.var_spin.setDecimals(10); self.var_spin.setRange(0.0, 1.0); self.var_spin.setSingleStep(1e-6); self.var_spin.setValue(1e-8)
        self.var_chk.setToolTip(tr("dialogs.shap_settings.var_checkbox_tip", default="Drop near-constant features before SHAP to reduce noise. Uncheck to disable."))
        self.var_spin.setToolTip(tr("dialogs.shap_settings.var_spin_tip", default="Treat features with variance below this threshold as near-constant."))
        var_row.addWidget(self.var_chk)
        var_row.addWidget(self.var_spin)
        var_row.addStretch()
        root.addLayout(var_row)

        # Always include
        inc_row = QHBoxLayout()
        inc_row.addWidget(QLabel(tr("dialogs.shap_settings.include_label", default="Always include (comma-separated raw names):")))
        self.include_edit = QLineEdit(); self.include_edit.setPlaceholderText(tr("dialogs.shap_settings.include_placeholder", default="gender, bullying_total ..."))
        self.include_edit.setToolTip(tr("dialogs.shap_settings.include_tip", default="Ensure these raw feature names are always included even if filtered."))
        inc_row.addWidget(self.include_edit)
        root.addLayout(inc_row)

        self.preview_label = QLabel("")
        self.preview_label.setObjectName("hintLabel")
        self.preview_label.setWordWrap(True)
        root.addWidget(self.preview_label)

        # Load from settings
        configured = str(settings.value('shap/configured', 'false')).lower() in ("true", "1", "yes")
        if not configured:
            # Defaults: All features, no var threshold, empty include
            self.rb_all.setChecked(True)
            self.var_chk.setChecked(False)
            self.var_spin.setEnabled(False)
        else:
            topn = settings.value('shap/top_n', None)
            if topn is None or str(topn) in ('', '-1'):
                self.rb_all.setChecked(True)
            else:
                self.rb_topn.setChecked(True)
                try:
                    self.topn_spin.setValue(int(topn))
                except Exception:
                    pass
            var_enabled = str(settings.value('shap/var_enabled', 'false')).lower() in ("true","1","yes")
            self.var_chk.setChecked(var_enabled)
            if var_enabled:
                try:
                    self.var_spin.setValue(float(settings.value('shap/var_thresh', 1e-8)))
                except Exception:
                    pass
            else:
                self.var_spin.setEnabled(False)
            self.include_edit.setText(str(settings.value('shap/always_include', '') or ''))

        self.var_chk.toggled.connect(self.var_spin.setEnabled)
        self.rb_all.toggled.connect(lambda _checked: self._sync_topn_controls())
        self.rb_topn.toggled.connect(lambda _checked: self._sync_topn_controls())
        self.var_chk.toggled.connect(lambda _checked: self._update_preview())
        self.topn_spin.valueChanged.connect(lambda _v: self._update_preview())
        self.var_spin.valueChanged.connect(lambda _v: self._update_preview())
        self.include_edit.textChanged.connect(lambda _t: self._update_preview())
        self._sync_topn_controls()
        self._update_preview()

        # Buttons: OK / Cancel / Restore Defaults
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._restore_defaults)
        root.addWidget(btns)

        # Restore/persist geometry
        _restore_geometry(self, 'dialogs/ShapSettings/geometry')
        btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/ShapSettings/geometry'))
        btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/ShapSettings/geometry'))

    def _sync_topn_controls(self):
        use_topn = self.rb_topn.isChecked()
        self.topn_spin.setEnabled(use_topn)
        self._update_preview()

    def _update_preview(self):
        mode_txt = (
            tr("dialogs.shap_settings.all_features", default="All features")
            if self.rb_all.isChecked()
            else tr("dialogs.shap_settings.top_n_value", default="Top-{value}", value=int(self.topn_spin.value()))
        )
        var_txt = (
            tr("dialogs.shap_settings.var_value", default="var >= {value}", value=f"{self.var_spin.value():.10f}")
            if self.var_chk.isChecked()
            else tr("dialogs.shap_settings.no_var_filter", default="no variance filter")
        )
        include = [x.strip() for x in self.include_edit.text().split(',') if x.strip()]
        if include:
            inc_txt = tr(
                "dialogs.shap_settings.include_preview",
                default="always include: {names}",
                names=", ".join(include[:3]) + ("..." if len(include) > 3 else ""),
            )
        else:
            inc_txt = tr("dialogs.shap_settings.include_none", default="always include: none")
        self.preview_label.setText(
            tr(
                "dialogs.shap_settings.current_profile",
                default="Current profile -> {mode}, {var}, {include}",
                mode=mode_txt,
                var=var_txt,
                include=inc_txt,
            )
        )

    def _on_accept(self):
        s = QSettings()
        # Mark configured when user saves
        s.setValue('shap/configured', True)
        if self.rb_all.isChecked():
            s.setValue('shap/top_n', -1)  # sentinel for All
        else:
            s.setValue('shap/top_n', int(self.topn_spin.value()))
        s.setValue('shap/var_enabled', bool(self.var_chk.isChecked()))
        if self.var_chk.isChecked():
            s.setValue('shap/var_thresh', float(self.var_spin.value()))
        s.setValue('shap/always_include', self.include_edit.text())
        try:
            s.sync()
        except Exception:
            pass
        # Save geometry and close
        _save_geometry(self, 'dialogs/ShapSettings/geometry')
        self.accept()

    def _restore_defaults(self):
        # UI defaults only; values are persisted on OK
        self.rb_all.setChecked(True)
        self.rb_topn.setChecked(False)
        self.topn_spin.setValue(20)
        self.var_chk.setChecked(False)
        self.var_spin.setValue(1e-8)
        self.var_spin.setEnabled(False)
        self.include_edit.clear()
        self._sync_topn_controls()
        self._update_preview()


class AboutDialog(QDialog):
    """General information about the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hakkında - Machine Learning Trainer")
        self.resize(680, 560)
        self.setMinimumSize(580, 460)
        _restore_geometry(self, "dialogs/About/geometry")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget()

        # --- TAB 1: UYGULAMA (APP INFO) ---
        tab_app = QWidget()
        v_app = QVBoxLayout(tab_app)
        v_app.setContentsMargins(20, 20, 20, 20)
        v_app.setSpacing(12)
        
        title = QLabel("Machine Learning Trainer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from PyQt6.QtGui import QFont
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setObjectName("titleLabel")
        
        subtitle = QLabel("Profesyonel Regresyon Modeli Eğitimi ve Değerlendirmesi")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("subtitleLabel")
        
        v_app.addWidget(title)
        v_app.addWidget(subtitle)
        v_app.addSpacing(16)
        
        desc = QLabel("Bu uygulama veri yükleme, özellik seçimi, arka planda paralel model eğitimi, \nve yayın kalitesinde açıklamalı (SHAP) çıktılar elde etmek için tasarlanmış \nprofesyonel bir makine öğrenmesi iş akışı sunar.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_app.addWidget(desc)
        v_app.addStretch()

        # --- TAB 2: GELIŞTIRICI (ABOUT ME) ---
        tab_dev = QWidget()
        v_dev = QVBoxLayout(tab_dev)
        v_dev.setContentsMargins(20, 20, 20, 20)
        v_dev.setSpacing(12)
        
        dev_title = QLabel("Geliştirici / Yazar")
        dev_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        dev_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_dev.addWidget(dev_title)
        
        dev_info = QLabel(
            "<div style='text-align: center;'><br><br>"
            "<span style='font-size:18px; font-weight:bold;'>Taha İlter Akar</span><br><br>"
            "<span style='font-size:14px;'>🎓 <i>FAU Erlangen-Nürnberg</i></span><br>"
            "<span style='font-size:14px;'>💼 Araştırma Görevlisi, <i>Fraunhofer IIS</i></span><br><br>"
            "<span style='font-size:14px;'>✉️ <a href='mailto:taha.ilter.akar@fau.de' style='color:#0B57D0; text-decoration:none;'>taha.ilter.akar@fau.de</a></span><br><br><br>"
            "<span style='font-size:13px; color:#555;'>Bu uygulama veri bilimi ve makine öğrenmesi süreçlerini otomatize edip <br>"
            "profesyonel bir araç seti sunmak üzere tasarlanmıştır.</span></div>"
        )
        dev_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dev_info.setOpenExternalLinks(True)
        v_dev.addWidget(dev_info)
        v_dev.addStretch()

        # --- TAB 3: LISANSLAR (LICENSES) ---
        tab_lic = QWidget()
        v_lic = QVBoxLayout(tab_lic)
        v_lic.setContentsMargins(12, 12, 12, 12)
        
        lic_txt = QTextEdit()
        lic_txt.setReadOnly(True)
        lic_content = """<h3>Açık Kaynak Lisansları ve Kullanılan Teknolojiler</h3>
            <p>Bu yazılım aşağıdaki açık kaynak kütüphane ve çatıları kullanmaktadır:</p>
            <ul>
            <li><b>Python 3</b> (PSF License)</li>
            <li><b>NumPy</b> (BSD 3-Clause)</li>
            <li><b>Pandas</b> (BSD 3-Clause)</li>
            <li><b>SciPy</b> (BSD 3-Clause)</li>
            <li><b>Scikit-Learn</b> (BSD 3-Clause)</li>
            <li><b>Matplotlib</b> (PSF License)</li>
            <li><b>Seaborn</b> (BSD 3-Clause)</li>
            <li><b>PyQt6</b> (GPL v3)</li>
            <li><b>SHAP</b> (MIT License)</li>
            <li><b>Statsmodels</b> (BSD 3-Clause)</li>
            <li><b>XGBoost</b> (Apache License 2.0)</li>
            <li><b>OpenPyXL</b> (MIT License)</li>
            </ul>
            <p>Tüm telif hakları ve ticari markalar ilgili hak sahiplerine aittir.</p>"""
        lic_txt.setHtml(lic_content)
        v_lic.addWidget(lic_txt)

        self.tabs.addTab(tab_app, "Uygulama")
        self.tabs.addTab(tab_dev, "Geliştirici")
        self.tabs.addTab(tab_lic, "Lisanslar")
        
        root.addWidget(self.tabs)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        root.addWidget(btns)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        btns.accepted.connect(lambda: _save_geometry(self, "dialogs/About/geometry"))
        btns.rejected.connect(lambda: _save_geometry(self, "dialogs/About/geometry"))

class PublicationExportDialog(QDialog):
    """Publication Studio with a guided and minimal publication export flow."""

    COL_ENABLED = 0
    COL_TYPE = 1
    COL_SOURCE = 2
    COL_LABEL = 3
    COL_FILENAME = 4

    VAL_COL_COLUMN = 0
    VAL_COL_SOURCE = 1
    VAL_COL_TARGET = 2
    VAL_COL_SCOPE = 3

    VAR_COL_SOURCE = 0
    VAR_COL_SUGGESTED = 1
    VAR_COL_OUTPUT = 2

    FMT_COL_COLUMN = 0
    FMT_COL_TYPE = 1
    FMT_COL_DECIMALS = 2

    def __init__(
        self,
        assets: list[dict],
        default_output_dir: str,
        parent=None,
        column_profile: dict | None = None,
        target_options: list[str] | None = None,
        default_target: str | None = None,
        selected_variables: list[str] | None = None,
        setup_mode: bool = False,
        initial_naming_rules: list[dict] | None = None,
        initial_value_rules: list[dict] | None = None,
        initial_format_rules: list[dict] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.publication_helper.title", default="Publication Studio"))
        self.resize(1120, 760)
        self.setMinimumSize(860, 560)

        self._setup_mode = bool(setup_mode)

        self._assets = list(assets or [])
        self._column_profile = dict(column_profile or {})
        self._target_options = [str(x) for x in (target_options or []) if str(x).strip()]
        self._default_target = str(default_target or "").strip()
        self._selected_variables = []
        seen_vars = set()
        for raw in (selected_variables or []):
            txt = str(raw).strip()
            if not txt:
                continue
            key = txt.lower()
            if key in seen_vars:
                continue
            seen_vars.add(key)
            self._selected_variables.append(txt)
        self._export_dir = ""
        self._export_plan: list[dict] = []
        self._value_rules: list[dict] = []
        self._naming_rules: list[dict] = []
        self._format_rules: list[dict] = []
        self._setup_payload: dict = {}
        self._initial_naming_rules = list(initial_naming_rules or [])
        self._initial_value_rules = list(initial_value_rules or [])
        self._initial_format_rules = list(initial_format_rules or [])
        self._updating_table = False
        self._updating_value_rules = False
        self._updating_naming_rules = False
        self._updating_format_rules = False
        self._variable_seed_target = ""
        self._initial_variable_outputs = self._build_initial_variable_output_map()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        info = QLabel(
            tr(
                "dialogs.publication_helper.info",
                default="Prepare manuscript-ready outputs in a guided flow: define study focus, review training variables, label values, then export.",
            )
        )
        info.setWordWrap(True)
        info.setObjectName("hintLabel")
        root.addWidget(info)

        if not self._setup_mode:
            self._build_output_folder_row(root, default_output_dir)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        self.variables_tab = QWidget()
        self.value_labels_tab = QWidget()
        self.tabs.addTab(self.variables_tab, tr("dialogs.publication_helper.step_variables", default="1. Variables"))
        self.tabs.addTab(self.value_labels_tab, tr("dialogs.publication_helper.step_value_labels", default="2. Value Labels"))

        self._build_variables_tab()
        self._build_value_labels_tab()

        if not self._setup_mode:
            self.assets_tab = QWidget()
            self.tabs.addTab(self.assets_tab, tr("dialogs.publication_helper.step_review", default="3. Package Review"))
            self._build_assets_tab(default_output_dir)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        if self._setup_mode:
            btns.button(QDialogButtonBox.StandardButton.Ok).setText(
                tr("dialogs.publication_helper.apply_studio_profile", default="Apply Studio Settings")
            )
        else:
            btns.button(QDialogButtonBox.StandardButton.Ok).setText(
                tr("dialogs.publication_helper.export_package", default="Export Package")
            )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        if hasattr(self, "btn_select_all"):
            self.btn_select_all.clicked.connect(lambda: self._set_all_rows_checked(True))
        if hasattr(self, "btn_clear_all"):
            self.btn_clear_all.clicked.connect(lambda: self._set_all_rows_checked(False))
        if hasattr(self, "table"):
            self.table.itemChanged.connect(lambda _item: self._update_summary())

        self.btn_value_add_rule.clicked.connect(self._add_value_rule_row)
        self.btn_value_remove_rule.clicked.connect(self._remove_selected_value_rule_row)
        self.value_rule_table.itemChanged.connect(lambda _item: self._refresh_value_rule_hint())
        self.value_column_list.currentItemChanged.connect(lambda _cur, _prev: self._on_value_column_selected())
        self.btn_value_add_quick.clicked.connect(self._on_add_quick_value_rule)
        if hasattr(self, "btn_format_add_quick"):
            self.btn_format_add_quick.clicked.connect(self._on_add_quick_format_rule)
        if hasattr(self, "btn_format_add_rule"):
            self.btn_format_add_rule.clicked.connect(self._add_format_rule_row)
        if hasattr(self, "btn_format_remove_rule"):
            self.btn_format_remove_rule.clicked.connect(self._remove_selected_format_rule_row)
        if hasattr(self, "format_rule_table"):
            self.format_rule_table.itemChanged.connect(lambda _item: self._refresh_format_rule_hint())

        self.btn_variable_apply_suggestions.clicked.connect(self._apply_variable_suggestions)
        self.btn_variable_reset.clicked.connect(self._reset_variable_outputs)
        self.variable_map_table.itemChanged.connect(self._on_variable_map_item_changed)

        if hasattr(self, "table"):
            self._populate_rows()
            self._auto_name_selected()

        self._seed_initial_value_rules()
        self._seed_initial_format_rules()
        self._update_summary()
        self._refresh_value_rule_hint()
        self._refresh_naming_rule_hint()
        self._refresh_format_rule_hint()

        _restore_geometry(self, 'dialogs/PublicationExport/geometry')
        btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/PublicationExport/geometry'))
        btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/PublicationExport/geometry'))

    def _available_profile_columns(self) -> list[str]:
        cols = [str(k) for k in self._column_profile.keys() if str(k).strip()]
        if not cols:
            cols = [str(x) for x in self._selected_variables if str(x).strip()]
        return sorted(cols, key=lambda x: x.lower())

    def _column_profile_entry(self, column_name: str) -> dict:
        return dict(self._column_profile.get(str(column_name), {}))

    def _profile_values_for_column(self, column_name: str) -> list[str]:
        entry = self._column_profile_entry(column_name)
        values = entry.get("sample_values", []) if isinstance(entry, dict) else []
        out = []
        seen = set()
        for v in values or []:
            txt = str(v).strip()
            if not txt:
                continue
            key = txt.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(txt)
        return out

    def _build_output_folder_row(self, parent_layout: QVBoxLayout, default_output_dir: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("dialogs.publication_helper.output_folder", default="Output folder:")))
        self.output_dir_edit = QLineEdit(default_output_dir or "")
        self.output_dir_edit.setClearButtonEnabled(True)
        row.addWidget(self.output_dir_edit, 1)
        browse_btn = QPushButton(tr("dialogs.publication_helper.browse", default="Browse..."))
        browse_btn.clicked.connect(self._browse_output_dir)
        row.addWidget(browse_btn)
        parent_layout.addLayout(row)

    def _training_variables(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        target = str(self._default_target or "").strip()
        if target:
            key = target.lower()
            if key not in seen:
                ordered.append(target)
                seen.add(key)

        for name in (self._selected_variables or self._available_profile_columns()):
            txt = str(name).strip()
            if not txt:
                continue
            key = txt.lower()
            if key in seen:
                continue
            ordered.append(txt)
            seen.add(key)
        return ordered

    def _suggested_label_for_variable(self, variable_name: str) -> str:
        mapping = self._load_social_science_mapping()
        renames = dict(mapping.get("column_renames", {})) if isinstance(mapping, dict) else {}

        target_norm = self._normalize_term(variable_name)
        if renames:
            for raw_src, raw_dst in renames.items():
                src = str(raw_src).strip()
                dst = str(raw_dst).strip()
                if not src or not dst:
                    continue
                if self._normalize_term(src) == target_norm:
                    return dst

        # Algorithmic fallback for high-quality English Q1 article column naming
        name = str(variable_name).strip()
        
        # Replace underscores and hyphens with spaces
        name = re.sub(r'[_\-]+', ' ', name)
        
        # Split CamelCase and PascalCase
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        
        # Title Capitalization strategy
        stop_words = {"and", "or", "but", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into", "like", "over"}
        words = name.split()
        if not words:
            return ""
        
        formatted_words = []
        for i, w in enumerate(words):
            if i > 0 and i < len(words) - 1 and w.lower() in stop_words:
                formatted_words.append(w.lower())
            else:
                formatted_words.append(w.capitalize())
                
        return " ".join(formatted_words)

    def _build_initial_variable_output_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rule in self._initial_naming_rules:
            if not isinstance(rule, dict):
                continue
            source = str(rule.get("source", "")).strip()
            target = str(rule.get("target", "")).strip()
            if not source or not target:
                continue
            out[self._normalize_term(source)] = target
        return out

    def _default_value_rule_scope(self) -> str:
        return "all" if self._setup_mode else "tables"

    def _seed_initial_value_rules(self):
        if not hasattr(self, "value_rule_table"):
            return
        if self.value_rule_table.rowCount() > 0:
            return

        valid_columns = {self._normalize_term(c) for c in self._available_profile_columns()}
        self._updating_value_rules = True
        try:
            for rule in self._initial_value_rules:
                if not isinstance(rule, dict):
                    continue
                column = str(rule.get("column", "")).strip()
                source = str(rule.get("source", "")).strip()
                target = str(rule.get("target", "")).strip()
                scope = str(rule.get("scope", self._default_value_rule_scope()) or self._default_value_rule_scope())
                if not source or not target:
                    continue
                if column and self._normalize_term(column) not in valid_columns:
                    continue
                for source_item in self._parse_value_rule_sources(source):
                    self._add_value_rule_row(column=column, source=source_item, target=target, scope=scope)
        finally:
            self._updating_value_rules = False

    def _seed_initial_format_rules(self):
        if not hasattr(self, "format_rule_table"):
            return
        if self.format_rule_table.rowCount() > 0:
            return

        valid_columns = {self._normalize_term(c) for c in self._available_profile_columns()}
        self._updating_format_rules = True
        try:
            for rule in self._initial_format_rules:
                if not isinstance(rule, dict):
                    continue
                column = str(rule.get("column", "")).strip()
                if not column or self._normalize_term(column) not in valid_columns:
                    continue
                fmt = str(rule.get("format", "auto") or "auto")
                try:
                    decimals = int(rule.get("decimals", 2))
                except Exception:
                    decimals = 2
                self._add_format_rule_row(column=column, fmt=fmt, decimals=decimals)
        finally:
            self._updating_format_rules = False

    def _build_variables_tab(self):
        layout = QVBoxLayout(self.variables_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        hint = QLabel(
            tr(
                "dialogs.publication_helper.variables_hint",
                default="Only dependent and independent variables used in training are shown here. Use recommendation labels if needed.",
            )
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)

        actions_row = QHBoxLayout()
        self.btn_variable_apply_suggestions = QPushButton(
            tr("dialogs.publication_helper.apply_recommended_labels", default="Apply Recommended Labels")
        )
        self.btn_variable_reset = QPushButton(
            tr("dialogs.publication_helper.reset_variable_labels", default="Reset To Original Names")
        )
        actions_row.addWidget(self.btn_variable_apply_suggestions)
        actions_row.addWidget(self.btn_variable_reset)
        actions_row.addStretch()
        layout.addLayout(actions_row)

        self.variable_map_table = QTableWidget(0, 3)
        self.variable_map_table.setAlternatingRowColors(True)
        self.variable_map_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.variable_map_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.variable_map_table.setHorizontalHeaderLabels(
            [
                tr("dialogs.publication_helper.variable_columns.source", default="Training Variable"),
                tr("dialogs.publication_helper.variable_columns.recommended", default="Recommended"),
                tr("dialogs.publication_helper.variable_columns.output", default="Output Label"),
            ]
        )
        self.variable_map_table.horizontalHeader().setSectionResizeMode(self.VAR_COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        self.variable_map_table.horizontalHeader().setSectionResizeMode(self.VAR_COL_SUGGESTED, QHeaderView.ResizeMode.Stretch)
        self.variable_map_table.horizontalHeader().setSectionResizeMode(self.VAR_COL_OUTPUT, QHeaderView.ResizeMode.Stretch)
        self.variable_map_table.verticalHeader().setVisible(False)
        layout.addWidget(self.variable_map_table, 1)

        self.variable_hint_label = QLabel("")
        self.variable_hint_label.setObjectName("hintLabel")
        self.variable_hint_label.setWordWrap(True)
        layout.addWidget(self.variable_hint_label)

        self._populate_variable_map_table()

    def _populate_variable_map_table(self):
        if not hasattr(self, "variable_map_table"):
            return

        variables = self._training_variables()
        existing_outputs = {}
        for row in range(self.variable_map_table.rowCount()):
            src_item = self.variable_map_table.item(row, self.VAR_COL_SOURCE)
            out_item = self.variable_map_table.item(row, self.VAR_COL_OUTPUT)
            if src_item is None or out_item is None:
                continue
            source = src_item.text().strip()
            output = out_item.text().strip()
            if source and output:
                existing_outputs[self._normalize_term(source)] = output

        self.variable_map_table.blockSignals(True)
        try:
            self.variable_map_table.setRowCount(0)
            for var_name in variables:
                row = self.variable_map_table.rowCount()
                self.variable_map_table.insertRow(row)

                src_item = QTableWidgetItem(var_name)
                src_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

                recommended = self._suggested_label_for_variable(var_name)
                rec_item = QTableWidgetItem(recommended)
                rec_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

                out_text = existing_outputs.get(
                    self._normalize_term(var_name),
                    self._initial_variable_outputs.get(self._normalize_term(var_name), recommended or var_name),
                )
                out_item = QTableWidgetItem(out_text)

                self.variable_map_table.setItem(row, self.VAR_COL_SOURCE, src_item)
                self.variable_map_table.setItem(row, self.VAR_COL_SUGGESTED, rec_item)
                self.variable_map_table.setItem(row, self.VAR_COL_OUTPUT, out_item)
        finally:
            self.variable_map_table.blockSignals(False)

        seed_target = self._default_target
        self._variable_seed_target = self._normalize_term(seed_target)
        self._refresh_naming_rule_hint()

    def _apply_variable_suggestions(self):
        if not hasattr(self, "variable_map_table"):
            return
        self.variable_map_table.blockSignals(True)
        try:
            for row in range(self.variable_map_table.rowCount()):
                src_item = self.variable_map_table.item(row, self.VAR_COL_SOURCE)
                rec_item = self.variable_map_table.item(row, self.VAR_COL_SUGGESTED)
                out_item = self.variable_map_table.item(row, self.VAR_COL_OUTPUT)
                if src_item is None or rec_item is None or out_item is None:
                    continue

                source = src_item.text().strip()
                recommended = rec_item.text().strip()
                output = out_item.text().strip()

                if recommended and (not output or output == source):
                    out_item.setText(recommended)
        finally:
            self.variable_map_table.blockSignals(False)

        self._refresh_naming_rule_hint()
        self._apply_naming_rules_to_labels(show_message=False)

    def _reset_variable_outputs(self):
        if not hasattr(self, "variable_map_table"):
            return
        self.variable_map_table.blockSignals(True)
        try:
            for row in range(self.variable_map_table.rowCount()):
                src_item = self.variable_map_table.item(row, self.VAR_COL_SOURCE)
                out_item = self.variable_map_table.item(row, self.VAR_COL_OUTPUT)
                if src_item is None or out_item is None:
                    continue
                out_item.setText(src_item.text().strip())
        finally:
            self.variable_map_table.blockSignals(False)

        self._refresh_naming_rule_hint()
        self._apply_naming_rules_to_labels(show_message=False)

    def _on_variable_map_item_changed(self, _item):
        self._refresh_naming_rule_hint()
        self._apply_naming_rules_to_labels(show_message=False)

    def _restore_asset_default_labels(self):
        if not hasattr(self, "table"):
            return
        self._updating_table = True
        try:
            for row, asset in enumerate(self._assets):
                label_item = self.table.item(row, self.COL_LABEL)
                if label_item is None:
                    continue
                default_label = str(asset.get("default_label") or asset.get("source_label") or "").strip()
                if default_label:
                    label_item.setText(default_label)
        finally:
            self._updating_table = False

    def _build_assets_tab(self, default_output_dir: str):
        layout = QVBoxLayout(self.assets_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        controls_row = QHBoxLayout()
        self.btn_select_all = QPushButton(tr("dialogs.publication_helper.select_all", default="Select All"))
        self.btn_clear_all = QPushButton(tr("dialogs.publication_helper.clear_all", default="Clear All"))
        controls_row.addWidget(self.btn_select_all)
        controls_row.addWidget(self.btn_clear_all)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self.table = QTableWidget(len(self._assets), 5)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setHorizontalHeaderLabels(
            [
                tr("dialogs.publication_helper.columns.include", default="Include"),
                tr("dialogs.publication_helper.columns.type", default="Type"),
                tr("dialogs.publication_helper.columns.source", default="Source"),
                tr("dialogs.publication_helper.columns.manuscript_label", default="Manuscript Label"),
                tr("dialogs.publication_helper.columns.output_name", default="Output File"),
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(self.COL_ENABLED, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_TYPE, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_LABEL, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(self.COL_FILENAME, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("hintLabel")
        layout.addWidget(self.summary_label)

    def _build_value_labels_tab(self):
        layout = QVBoxLayout(self.value_labels_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        hint = QLabel(
            tr(
                "dialogs.publication_helper.value_labels_hint_setup",
                default="Value labels convert coded values into readable text in training outputs (plots, figures, tables). Example: gender, 0 -> female, 1 -> male.",
            )
            if self._setup_mode
            else tr(
                "dialogs.publication_helper.value_labels_hint",
                default="Value labels convert coded values into readable text in exported tables. Example: gender, 0 -> female, 1 -> male.",
            )
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)

        axis_hint = QLabel(
            tr(
                "dialogs.publication_helper.axis_label_hint_setup",
                default="Axis and variable naming are managed in Step 1 (Variables). This step updates coded cell values used by training outputs.",
            )
            if self._setup_mode
            else tr(
                "dialogs.publication_helper.axis_label_hint",
                default="Axis and variable naming are managed in Step 1 (Variables). This step only changes cell values in table exports.",
            )
        )
        axis_hint.setWordWrap(True)
        axis_hint.setObjectName("hintLabel")
        layout.addWidget(axis_hint)

        explorer_card = QFrame()
        explorer_card.setObjectName("decisionCard")
        explorer_layout = QHBoxLayout(explorer_card)
        explorer_layout.setContentsMargins(10, 8, 10, 8)
        explorer_layout.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        left_col.addWidget(QLabel(tr("dialogs.publication_helper.variable_picker", default="Variable")))
        self.value_column_list = QListWidget()
        self.value_column_list.setAlternatingRowColors(True)
        self.value_column_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.value_column_list.setMinimumHeight(130)
        left_col.addWidget(self.value_column_list, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        right_col.addWidget(QLabel(tr("dialogs.publication_helper.value_label_editor", default="Value Label Editor")))
        right_col.addWidget(QLabel(tr("dialogs.publication_helper.raw_value", default="Raw value(s):")))
        self.quick_value_source_combo = QComboBox()
        self.quick_value_source_combo.setEditable(True)
        source_editor = self.quick_value_source_combo.lineEdit()
        if source_editor is not None:
            source_editor.setPlaceholderText(
                tr(
                    "dialogs.publication_helper.raw_value_placeholder",
                    default="Single value or comma-separated values, e.g. 1,2,3",
                )
            )
        right_col.addWidget(self.quick_value_source_combo)
        right_col.addWidget(QLabel(tr("dialogs.publication_helper.display_value", default="Display value:")))
        self.quick_value_target_edit = QLineEdit()
        self.quick_value_target_edit.setPlaceholderText(
            tr("dialogs.publication_helper.display_value_placeholder", default="Readable label")
        )
        right_col.addWidget(self.quick_value_target_edit)

        self.btn_value_add_quick = QPushButton(
            tr("dialogs.publication_helper.quick_add_value_rule", default="Add / Update Value Label")
        )
        quick_actions = QHBoxLayout()
        quick_actions.setSpacing(6)
        quick_actions.addWidget(self.btn_value_add_quick)
        right_col.addLayout(quick_actions)
        right_col.addStretch()

        self.value_column_hint = QLabel("")
        self.value_column_hint.setObjectName("hintLabel")
        self.value_column_hint.setWordWrap(True)
        right_col.addWidget(self.value_column_hint)

        left_wrap = QWidget()
        left_wrap.setLayout(left_col)
        right_wrap = QWidget()
        right_wrap.setLayout(right_col)
        explorer_layout.addWidget(left_wrap, 1)
        explorer_layout.addWidget(right_wrap, 1)

        layout.addWidget(explorer_card)

        actions_row = QHBoxLayout()
        self.btn_value_add_rule = QPushButton(tr("dialogs.publication_helper.value_add_rule", default="Add Empty Rule"))
        self.btn_value_remove_rule = QPushButton(
            tr("dialogs.publication_helper.value_remove_rule", default="Remove Selected Value Rule")
        )
        if not self._setup_mode:
            actions_row.addWidget(self.btn_value_add_rule)
        actions_row.addWidget(self.btn_value_remove_rule)
        actions_row.addStretch()
        layout.addLayout(actions_row)

        self.value_rule_table = QTableWidget(0, 4)
        self.value_rule_table.setAlternatingRowColors(True)
        self.value_rule_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.value_rule_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.value_rule_table.setHorizontalHeaderLabels(
            [
                tr("dialogs.publication_helper.value_columns.column", default="Column (optional)"),
                tr("dialogs.publication_helper.value_columns.source", default="Raw Value"),
                tr("dialogs.publication_helper.value_columns.target", default="Display Value"),
                tr("dialogs.publication_helper.value_columns.scope", default="Scope"),
            ]
        )
        self.value_rule_table.horizontalHeader().setSectionResizeMode(self.VAL_COL_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.value_rule_table.horizontalHeader().setSectionResizeMode(self.VAL_COL_SOURCE, QHeaderView.ResizeMode.Stretch)
        self.value_rule_table.horizontalHeader().setSectionResizeMode(self.VAL_COL_TARGET, QHeaderView.ResizeMode.Stretch)
        self.value_rule_table.horizontalHeader().setSectionResizeMode(self.VAL_COL_SCOPE, QHeaderView.ResizeMode.ResizeToContents)
        self.value_rule_table.verticalHeader().setVisible(False)
        if self._setup_mode:
            self.value_rule_table.setColumnHidden(self.VAL_COL_SCOPE, True)
        layout.addWidget(self.value_rule_table, 1)

        self.value_rule_hint_label = QLabel("")
        self.value_rule_hint_label.setObjectName("hintLabel")
        self.value_rule_hint_label.setWordWrap(True)
        layout.addWidget(self.value_rule_hint_label)

        self._populate_value_column_list()

    def _normalize_extension(self, ext: str, fallback: str) -> str:
        raw = str(ext or "").strip()
        if not raw:
            raw = fallback
        if not raw.startswith('.'):
            raw = f".{raw}"
        return raw.lower()

    def _slugify(self, text: str) -> str:
        cleaned = _qascii(str(text or "").strip()).lower()
        cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
        cleaned = cleaned.strip("_")
        return cleaned or "item"

    @staticmethod
    def _normalize_term(text: str) -> str:
        normalized = _qascii(str(text or "").strip()).lower()
        normalized = normalized.replace("ı", "i").replace("ğ", "g").replace("ü", "u")
        normalized = normalized.replace("ş", "s").replace("ö", "o").replace("ç", "c")
        return normalized

    @staticmethod
    def _social_science_mapping_path() -> Path:
        return Path(__file__).resolve().parents[2] / "scripts" / "q1_social_science_mappings.json"

    def _load_social_science_mapping(self) -> dict:
        path = self._social_science_mapping_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    @staticmethod
    def _scope_matches(rule_scope: str, target_scope: str) -> bool:
        return str(rule_scope or "all") in ("all", str(target_scope or "all"))

    @staticmethod
    def _column_matches(rule_column: str, column_name: str) -> bool:
        rc = str(rule_column or "").strip()
        if not rc:
            return True
        return PublicationExportDialog._normalize_term(rc) == PublicationExportDialog._normalize_term(column_name)

    def _apply_rules_to_text(self, text: str, rules: list[dict], target_scope: str) -> str:
        result = str(text or "")
        for rule in rules or []:
            scope = str(rule.get("scope", "all"))
            if not self._scope_matches(scope, target_scope):
                continue

            src = str(rule.get("source", "")).strip()
            dst = str(rule.get("target", "")).strip()
            if not src:
                continue

            # Treat underscores and slashes as token boundaries so renames also work for snake_case file names.
            token_pattern = rf"(?<![A-Za-z0-9]){re.escape(src)}(?![A-Za-z0-9])"
            if re.search(token_pattern, result, flags=re.IGNORECASE):
                result = re.sub(token_pattern, dst, result, flags=re.IGNORECASE)
            elif self._normalize_term(result) == self._normalize_term(src):
                result = dst
        return result

    def _sanitize_filename(self, filename: str, extension: str, fallback_stem: str) -> str:
        raw = os.path.basename(str(filename or "").strip())
        raw = re.sub(r'[<>:"/\\|?*]+', '_', raw)
        if not raw:
            raw = f"{fallback_stem}{extension}"

        stem, ext = os.path.splitext(raw)
        if not stem:
            stem = fallback_stem
        if not ext:
            ext = extension
        if ext.lower() != extension.lower():
            ext = extension
        return f"{stem}{ext}"

    def _asset_scope(self, asset_type: str) -> str:
        return "tables" if str(asset_type).lower() == "table" else "figures"

    def _asset_type_label(self, asset_type: str) -> str:
        if str(asset_type).lower() == "table":
            return tr("dialogs.publication_helper.type_table", default="Table")
        return tr("dialogs.publication_helper.type_figure", default="Figure")

    def _make_scope_combo(self, scope_value: str, on_change):
        combo = QComboBox()
        combo.addItem(tr("dialogs.publication_helper.scope_all", default="All"), "all")
        combo.addItem(tr("dialogs.publication_helper.scope_tables", default="Tables"), "tables")
        combo.addItem(tr("dialogs.publication_helper.scope_figures", default="Figures"), "figures")
        idx = combo.findData(scope_value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(on_change)
        return combo

    def _populate_rows(self):
        self._updating_table = True
        try:
            for row, asset in enumerate(self._assets):
                enabled_item = QTableWidgetItem()
                enabled_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled |
                    Qt.ItemFlag.ItemIsSelectable |
                    Qt.ItemFlag.ItemIsUserCheckable
                )
                enabled_item.setCheckState(Qt.CheckState.Checked)
                self.table.setItem(row, self.COL_ENABLED, enabled_item)

                asset_type = str(asset.get("asset_type", "figure"))
                source_label = str(asset.get("source_label") or asset.get("source_name") or "")
                default_label = str(asset.get("default_label") or source_label or "")
                default_filename = str(asset.get("default_filename") or "")

                type_item = QTableWidgetItem(self._asset_type_label(asset_type))
                type_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row, self.COL_TYPE, type_item)

                source_item = QTableWidgetItem(source_label)
                source_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row, self.COL_SOURCE, source_item)

                self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(default_label))
                self.table.setItem(row, self.COL_FILENAME, QTableWidgetItem(default_filename))
        finally:
            self._updating_table = False

    def _set_all_rows_checked(self, checked: bool):
        target_state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating_table = True
        try:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, self.COL_ENABLED)
                if item is not None:
                    item.setCheckState(target_state)
        finally:
            self._updating_table = False
        self._update_summary()

    def _row_checked(self, row: int) -> bool:
        item = self.table.item(row, self.COL_ENABLED)
        return bool(item is not None and item.checkState() == Qt.CheckState.Checked)

    def _browse_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or os.getcwd()
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            tr("dialogs.publication_helper.select_output_dir", default="Select Output Folder"),
            start_dir,
        )
        if selected_dir:
            self.output_dir_edit.setText(selected_dir)

    def _selected_profile_column(self) -> str:
        if not hasattr(self, "value_column_list"):
            return ""
        item = self.value_column_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "").strip()

    def _populate_value_column_list(self):
        if not hasattr(self, "value_column_list"):
            return
        self.value_column_list.clear()
        columns = self._available_profile_columns()
        if not columns:
            placeholder = QListWidgetItem(
                tr("dialogs.publication_helper.no_exportable_columns", default="No selected training variables found.")
            )
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.value_column_list.addItem(placeholder)
            self.value_column_hint.setText(
                tr(
                    "dialogs.publication_helper.no_exportable_columns_hint",
                    default="Select dependent and independent variables in training first, then reopen Publication Studio.",
                )
            )
            return

        for col in columns:
            entry = self._column_profile_entry(col)
            tables = ", ".join([str(t) for t in entry.get("tables", []) if str(t).strip()])
            values = self._profile_values_for_column(col)
            dtype = str(entry.get("dtype", "")).strip()
            item = QListWidgetItem(col)
            tooltip_parts = []
            if tables:
                tooltip_parts.append(f"tables: {tables}")
            if dtype:
                tooltip_parts.append(f"type: {dtype}")
            if values:
                tooltip_parts.append(
                    tr(
                        "dialogs.publication_helper.sample_values",
                        default="samples: {values}",
                        values=", ".join(values[:8]),
                    )
                )
            if tooltip_parts:
                item.setToolTip("\n".join(tooltip_parts))
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.value_column_list.addItem(item)

        if self.value_column_list.count() > 0:
            self.value_column_list.setCurrentRow(0)
        self._on_value_column_selected()

    def _on_value_column_selected(self):
        column = self._selected_profile_column()
        values = self._profile_values_for_column(column)
        entry = self._column_profile_entry(column)
        dtype = str(entry.get("dtype", "")).strip() if isinstance(entry, dict) else ""

        if hasattr(self, "quick_value_source_combo"):
            self.quick_value_source_combo.blockSignals(True)
            try:
                self.quick_value_source_combo.clear()
                for v in values:
                    self.quick_value_source_combo.addItem(v)
            finally:
                self.quick_value_source_combo.blockSignals(False)

        if hasattr(self, "quick_value_target_edit"):
            self.quick_value_target_edit.setText("")

        if hasattr(self, "value_column_hint"):
            if not column:
                self.value_column_hint.setText(
                    tr("dialogs.publication_helper.column_not_selected", default="Select an exportable column first.")
                )
            else:
                sample_text = ", ".join(values[:8]) if values else tr("common.none", default="none")
                self.value_column_hint.setText(
                    tr(
                        "dialogs.publication_helper.column_selected_hint",
                        default="Selected variable: {column} | Type: {dtype} | Sample values: {values}",
                        column=column,
                        dtype=dtype or "-",
                        values=sample_text,
                    )
                )
                
        # Every time a selection changes, automatically filter the rule table view
        self._filter_rule_table_by_current_column()

    def _parse_value_rule_sources(self, raw_sources: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for token in re.split(r"[,;\n]+", str(raw_sources or "")):
            txt = str(token).strip().strip("\"'")
            if not txt:
                continue
            key = self._normalize_term(txt)
            if key in seen:
                continue
            seen.add(key)
            out.append(txt)
        return out

    def _on_add_quick_value_rule(self):
        column = self._selected_profile_column()
        if not column:
            QMessageBox.information(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                tr("dialogs.publication_helper.column_not_selected", default="Select a training variable first."),
            )
            return

        raw_sources = ""
        if hasattr(self, "quick_value_source_combo"):
            raw_sources = self.quick_value_source_combo.currentText().strip()
        sources = self._parse_value_rule_sources(raw_sources)
        target = self.quick_value_target_edit.text().strip() if hasattr(self, "quick_value_target_edit") else ""

        if not sources or not target:
            QMessageBox.information(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                tr(
                    "dialogs.publication_helper.quick_value_required",
                    default="Provide raw value(s) and display value to add a rule. You can enter multiple raw values with commas.",
                ),
            )
            return

        added = 0
        updated = 0
        for source in sources:
            before = self.value_rule_table.rowCount()
            self._upsert_value_rule(column, source, target, self._default_value_rule_scope())
            if self.value_rule_table.rowCount() > before:
                added += 1
            else:
                updated += 1

        if hasattr(self, "quick_value_target_edit"):
            self.quick_value_target_edit.clear()
        if hasattr(self, "quick_value_source_combo"):
            self.quick_value_source_combo.setEditText("")

        if hasattr(self, "value_column_hint"):
            self.value_column_hint.setText(
                tr(
                    "dialogs.publication_helper.quick_value_result",
                    default="{added} mapping(s) added, {updated} updated for {column}.",
                    added=added,
                    updated=updated,
                    column=column,
                )
            )

    def _on_add_quick_format_rule(self):
        if not hasattr(self, "format_rule_table"):
            return
        column = self._selected_profile_column()
        if not column:
            QMessageBox.information(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                tr("dialogs.publication_helper.column_not_selected", default="Select a training variable first."),
            )
            return

        fmt = self.quick_format_combo.currentData() if hasattr(self, "quick_format_combo") else "auto"
        decimals = int(self.quick_format_decimals.value()) if hasattr(self, "quick_format_decimals") else 2

        # Replace existing format rule for the same column to keep the rule table concise.
        for row in range(self.format_rule_table.rowCount()):
            col_item = self.format_rule_table.item(row, self.FMT_COL_COLUMN)
            existing_col = col_item.text().strip() if col_item is not None else ""
            if self._normalize_term(existing_col) == self._normalize_term(column):
                self.format_rule_table.removeRow(row)
                break

        self._add_format_rule_row(column=column, fmt=str(fmt or "auto"), decimals=decimals)

    def _add_value_rule_row(self, column: str = "", source: str = "", target: str = "", scope: str | None = None):
        if not str(column).strip():
            column = self._selected_profile_column()
        scope_val = str(scope or self._default_value_rule_scope())
        row = self.value_rule_table.rowCount()
        self.value_rule_table.insertRow(row)
        col_item = QTableWidgetItem(str(column))
        if self._setup_mode:
            col_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.value_rule_table.setItem(row, self.VAL_COL_COLUMN, col_item)
        self.value_rule_table.setItem(row, self.VAL_COL_SOURCE, QTableWidgetItem(str(source)))
        self.value_rule_table.setItem(row, self.VAL_COL_TARGET, QTableWidgetItem(str(target)))
        self.value_rule_table.setCellWidget(
            row,
            self.VAL_COL_SCOPE,
            self._make_scope_combo(scope_val, lambda _i: self._refresh_value_rule_hint()),
        )
        self.value_rule_table.setCurrentCell(row, self.VAL_COL_SOURCE)
        self._refresh_value_rule_hint()

    def _remove_selected_value_rule_row(self):
        rows = sorted({idx.row() for idx in self.value_rule_table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            row = self.value_rule_table.currentRow()
            if row >= 0:
                rows = [row]
        for row in rows:
            self.value_rule_table.removeRow(row)
        if rows:
            self._refresh_value_rule_hint()

    def _upsert_value_rule(self, column: str, source: str, target: str, scope: str | None = None):
        src_norm = self._normalize_term(source)
        col_norm = self._normalize_term(column)
        scope_norm = str(scope or self._default_value_rule_scope())
        valid_columns = {self._normalize_term(c) for c in self._available_profile_columns()}

        if col_norm and col_norm not in valid_columns:
            return

        for row in range(self.value_rule_table.rowCount()):
            col_item = self.value_rule_table.item(row, self.VAL_COL_COLUMN)
            src_item = self.value_rule_table.item(row, self.VAL_COL_SOURCE)
            scope_widget = self.value_rule_table.cellWidget(row, self.VAL_COL_SCOPE)

            existing_col = col_item.text().strip() if col_item is not None else ""
            existing_src = src_item.text().strip() if src_item is not None else ""
            existing_scope = (
                scope_widget.currentData()
                if isinstance(scope_widget, QComboBox)
                else self._default_value_rule_scope()
            )

            if (
                self._normalize_term(existing_col) == col_norm
                and self._normalize_term(existing_src) == src_norm
                and str(existing_scope) == scope_norm
            ):
                tgt_item = self.value_rule_table.item(row, self.VAL_COL_TARGET)
                if tgt_item is None:
                    tgt_item = QTableWidgetItem(str(target))
                    self.value_rule_table.setItem(row, self.VAL_COL_TARGET, tgt_item)
                else:
                    tgt_item.setText(str(target))
                self._refresh_value_rule_hint()
                return

        self._add_value_rule_row(column=column, source=source, target=target, scope=scope_norm)

    def _collect_value_rules(self):
        rules = []
        seen_rule_keys: set[tuple[str, str, str, str]] = set()
        valid_columns = {self._normalize_term(c) for c in self._available_profile_columns()}
        for row in range(self.value_rule_table.rowCount()):
            col_item = self.value_rule_table.item(row, self.VAL_COL_COLUMN)
            src_item = self.value_rule_table.item(row, self.VAL_COL_SOURCE)
            tgt_item = self.value_rule_table.item(row, self.VAL_COL_TARGET)
            scope_widget = self.value_rule_table.cellWidget(row, self.VAL_COL_SCOPE)

            column = col_item.text().strip() if col_item is not None else ""
            source = src_item.text().strip() if src_item is not None else ""
            target = tgt_item.text().strip() if tgt_item is not None else ""
            scope = (
                "all"
                if self._setup_mode
                else (
                    scope_widget.currentData()
                    if isinstance(scope_widget, QComboBox)
                    else self._default_value_rule_scope()
                )
            )

            if not source or not target:
                continue
            if column and self._normalize_term(column) not in valid_columns:
                continue
            for source_item in self._parse_value_rule_sources(source):
                rule_key = (
                    self._normalize_term(column),
                    self._normalize_term(source_item),
                    self._normalize_term(target),
                    str(scope or self._default_value_rule_scope()),
                )
                if rule_key in seen_rule_keys:
                    continue
                seen_rule_keys.add(rule_key)
                rules.append(
                    {
                        "column": column,
                        "source": source_item,
                        "target": target,
                        "scope": str(scope or self._default_value_rule_scope()),
                    }
                )
        return rules

    def _refresh_value_rule_hint(self):
        if self._updating_value_rules:
            return
        rules = self._collect_value_rules()
        self.value_rule_hint_label.setText(
            tr(
                "dialogs.publication_helper.value_rules_summary_setup",
                default="{count} value-label rule(s) active. These are applied to training outputs (plots, figures, tables).",
                count=len(rules),
            )
            if self._setup_mode
            else tr(
                "dialogs.publication_helper.value_rules_summary",
                default="{count} value-label rule(s) active. These are applied to exported table values.",
                count=len(rules),
            )
        )
        # Visually refresh the list counts based on updated rules mapping
        self._sync_list_item_titles_with_rules(rules)
        # Apply filtering so that ONLY rules for the currently selected column are visible in the table.
        self._filter_rule_table_by_current_column()

    def _filter_rule_table_by_current_column(self):
        if not hasattr(self, "value_rule_table"):
            return
        column = self._selected_profile_column()
        target_norm = self._normalize_term(column)
        
        for row in range(self.value_rule_table.rowCount()):
            col_item = self.value_rule_table.item(row, self.VAL_COL_COLUMN)
            item_col = col_item.text().strip() if col_item is not None else ""
            if not target_norm or self._normalize_term(item_col) == target_norm:
                self.value_rule_table.setRowHidden(row, False)
            else:
                self.value_rule_table.setRowHidden(row, True)

    def _sync_list_item_titles_with_rules(self, current_rules):
        """Adds a visual mapped count to items in the list widget if they have rules"""
        if not hasattr(self, "value_column_list"):
            return
        
        column_rule_counts = {}
        for rule in current_rules:
            col_key = self._normalize_term(rule.get("column", ""))
            column_rule_counts[col_key] = column_rule_counts.get(col_key, 0) + 1

        self.value_column_list.blockSignals(True)
        try:
            for i in range(self.value_column_list.count()):
                item = self.value_column_list.item(i)
                if not item:
                    continue
                col_name = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if not col_name:
                    continue
                
                count = column_rule_counts.get(self._normalize_term(col_name), 0)
                if count > 0:
                    item.setText(f"{col_name} ✓ ({count})")
                else:
                    item.setText(col_name)
        finally:
            self.value_column_list.blockSignals(False)

    def _make_format_combo(self, format_value: str, on_change):
        combo = QComboBox()
        combo.addItem(tr("dialogs.publication_helper.format_auto", default="Auto"), "auto")
        combo.addItem(tr("dialogs.publication_helper.format_binary", default="Binary"), "binary")
        combo.addItem(tr("dialogs.publication_helper.format_percentage", default="Percentage"), "percentage")
        combo.addItem(tr("dialogs.publication_helper.format_decimal", default="Decimal"), "decimal")
        idx = combo.findData(str(format_value or "auto"))
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.currentIndexChanged.connect(on_change)
        return combo

    def _add_format_rule_row(self, column: str = "", fmt: str = "decimal", decimals: int = 2):
        if not hasattr(self, "format_rule_table"):
            return
        if not str(column).strip():
            column = self._selected_profile_column()
        row = self.format_rule_table.rowCount()
        self.format_rule_table.insertRow(row)
        self.format_rule_table.setItem(row, self.FMT_COL_COLUMN, QTableWidgetItem(str(column)))
        self.format_rule_table.setCellWidget(
            row,
            self.FMT_COL_TYPE,
            self._make_format_combo(str(fmt or "decimal"), lambda _i: self._refresh_format_rule_hint()),
        )
        self.format_rule_table.setItem(row, self.FMT_COL_DECIMALS, QTableWidgetItem(str(int(decimals))))
        self.format_rule_table.setCurrentCell(row, self.FMT_COL_COLUMN)
        self._refresh_format_rule_hint()

    def _remove_selected_format_rule_row(self):
        if not hasattr(self, "format_rule_table"):
            return
        row = self.format_rule_table.currentRow()
        if row >= 0:
            self.format_rule_table.removeRow(row)
            self._refresh_format_rule_hint()

    def _collect_format_rules(self):
        if not hasattr(self, "format_rule_table"):
            return []
        rules = []
        valid_columns = {self._normalize_term(c) for c in self._available_profile_columns()}
        for row in range(self.format_rule_table.rowCount()):
            col_item = self.format_rule_table.item(row, self.FMT_COL_COLUMN)
            dec_item = self.format_rule_table.item(row, self.FMT_COL_DECIMALS)
            fmt_widget = self.format_rule_table.cellWidget(row, self.FMT_COL_TYPE)

            column = col_item.text().strip() if col_item is not None else ""
            fmt = fmt_widget.currentData() if isinstance(fmt_widget, QComboBox) else "auto"
            decimals_raw = dec_item.text().strip() if dec_item is not None else "2"

            if not column:
                continue
            if self._normalize_term(column) not in valid_columns:
                continue
            try:
                decimals = max(0, min(int(decimals_raw), 8))
            except Exception:
                decimals = 2

            rules.append(
                {
                    "column": column,
                    "format": str(fmt or "auto"),
                    "decimals": int(decimals),
                    "scope": "tables",
                }
            )
        return rules

    def _refresh_format_rule_hint(self):
        if self._updating_format_rules:
            return
        if not hasattr(self, "format_rule_hint_label"):
            return
        rules = self._collect_format_rules()
        self.format_rule_hint_label.setText(
            tr(
                "dialogs.publication_helper.format_rules_summary",
                default="{count} display format rule(s) active. Applied to exported table columns.",
                count=len(rules),
            )
        )

    def _collect_naming_rules(self):
        rules = []
        if not hasattr(self, "variable_map_table"):
            return rules

        for row in range(self.variable_map_table.rowCount()):
            src_item = self.variable_map_table.item(row, self.VAR_COL_SOURCE)
            out_item = self.variable_map_table.item(row, self.VAR_COL_OUTPUT)
            if src_item is None or out_item is None:
                continue

            source = src_item.text().strip()
            target = out_item.text().strip()
            if not source or not target:
                continue
            if self._normalize_term(source) == self._normalize_term(target):
                continue
            rules.append({"source": source, "target": target, "scope": "all"})

        return rules

    def _apply_naming_rules_to_labels(self, show_message: bool = True):
        rules = self._collect_naming_rules()
        if not rules:
            self._restore_asset_default_labels()
            self._auto_name_selected()
            if show_message:
                QMessageBox.information(
                    self,
                    tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                    tr(
                        "dialogs.publication_helper.rules_empty",
                        default="Add at least one naming rule before applying to labels.",
                    ),
                )
            return

        self._updating_table = True
        try:
            for row, asset in enumerate(self._assets):
                if not self._row_checked(row):
                    continue
                scope = self._asset_scope(str(asset.get("asset_type", "figure")))
                label_item = self.table.item(row, self.COL_LABEL)
                if label_item is None:
                    continue
                mapped = self._apply_rules_to_text(label_item.text().strip(), rules, scope)
                label_item.setText(mapped)
        finally:
            self._updating_table = False

        self._auto_name_selected()
        if show_message:
            QMessageBox.information(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                tr("dialogs.publication_helper.rules_applied", default="Naming rules were applied to selected labels."),
            )

    def _refresh_naming_rule_hint(self):
        if self._updating_naming_rules:
            return
        rules = self._collect_naming_rules()
        if hasattr(self, "variable_hint_label"):
            self.variable_hint_label.setText(
                tr(
                    "dialogs.publication_helper.rules_summary",
                    default="{count} variable naming rule(s) active. These labels are applied to tables, figures, and captions.",
                    count=len(rules),
                )
            )

    def _auto_name_selected(self):
        naming_rules = self._collect_naming_rules()
        self._updating_table = True
        try:
            fig_index = 1
            table_index = 1
            for row, asset in enumerate(self._assets):
                if not self._row_checked(row):
                    continue

                asset_type = str(asset.get("asset_type", "figure")).lower()
                scope = self._asset_scope(asset_type)
                label_item = self.table.item(row, self.COL_LABEL)
                filename_item = self.table.item(row, self.COL_FILENAME)
                source_item = self.table.item(row, self.COL_SOURCE)

                label_text = label_item.text().strip() if label_item is not None else ""
                source_text = source_item.text().strip() if source_item is not None else ""
                mapped_label = self._apply_rules_to_text(label_text or source_text, naming_rules, scope)
                stem_text = self._slugify(mapped_label or source_text)

                if asset_type == "table":
                    ext = self._normalize_extension(str(asset.get("extension", ".csv")), ".csv")
                    auto_name = f"table_{table_index:02d}_{stem_text}{ext}"
                    table_index += 1
                else:
                    ext = self._normalize_extension(str(asset.get("extension", ".png")), ".png")
                    auto_name = f"figure_{fig_index:02d}_{stem_text}{ext}"
                    fig_index += 1

                if filename_item is not None:
                    filename_item.setText(auto_name)
        finally:
            self._updating_table = False
        self._update_summary()

    def _update_summary(self):
        if not hasattr(self, "table") or not hasattr(self, "summary_label"):
            return
        if self._updating_table:
            return
        selected = sum(1 for row in range(self.table.rowCount()) if self._row_checked(row))
        total = self.table.rowCount()
        selected_tables = 0
        selected_figures = 0
        for row, asset in enumerate(self._assets):
            if not self._row_checked(row):
                continue
            if str(asset.get("asset_type", "figure")).lower() == "table":
                selected_tables += 1
            else:
                selected_figures += 1
        self.summary_label.setText(
            tr(
                "dialogs.publication_helper.summary",
                default="Selected assets: {selected}/{total} (Tables: {tables}, Figures: {figures})",
                selected=selected,
                total=total,
                tables=selected_tables,
                figures=selected_figures,
            )
        )

    def _collect_export_plan(self):
        out_dir = self.output_dir_edit.text().strip()
        if not out_dir:
            return None, tr("dialogs.publication_helper.error_output_dir", default="Please choose an output folder.")

        value_rules = self._collect_value_rules()
        naming_rules = self._collect_naming_rules()
        format_rules = self._collect_format_rules()
        selected_items = []
        used_names = set()

        for row, asset in enumerate(self._assets):
            if not self._row_checked(row):
                continue

            asset_type = str(asset.get("asset_type", "figure")).lower()
            scope = self._asset_scope(asset_type)
            source_item = self.table.item(row, self.COL_SOURCE)
            label_item = self.table.item(row, self.COL_LABEL)
            filename_item = self.table.item(row, self.COL_FILENAME)

            source_text = source_item.text().strip() if source_item is not None else ""
            label_text = label_item.text().strip() if label_item is not None else ""
            file_text = filename_item.text().strip() if filename_item is not None else ""

            if not label_text:
                label_text = source_text or tr("dialogs.publication_helper.untitled", default="Untitled")

            label_text = self._apply_rules_to_text(label_text, naming_rules, scope)

            expected_ext = self._normalize_extension(
                str(asset.get("extension", ".csv" if asset_type == "table" else ".png")),
                ".csv" if asset_type == "table" else ".png",
            )
            fallback_stem = self._slugify(label_text)

            file_stem, file_ext = os.path.splitext(file_text)
            mapped_stem = self._apply_rules_to_text(file_stem or fallback_stem, naming_rules, scope)
            mapped_ext = file_ext or expected_ext
            candidate_name = f"{mapped_stem}{mapped_ext}"
            output_name = self._sanitize_filename(candidate_name, expected_ext, fallback_stem)

            lowered = output_name.lower()
            if lowered in used_names:
                return None, tr(
                    "dialogs.publication_helper.error_duplicate",
                    default="Duplicate output file names detected: {name}",
                    name=output_name,
                )
            used_names.add(lowered)

            row_payload = dict(asset)
            row_payload["manuscript_label"] = label_text
            row_payload["output_name"] = output_name
            selected_items.append(row_payload)

        if not selected_items:
            return None, tr("dialogs.publication_helper.error_no_selection", default="Select at least one asset to export.")

        return {
            "output_dir": out_dir,
            "items": selected_items,
            "value_rules": value_rules,
            "naming_rules": naming_rules,
            "format_rules": format_rules,
        }, None

    def get_export_payload(self):
        return (
            self._export_dir,
            list(self._export_plan),
            list(self._value_rules),
            list(self._naming_rules),
            list(self._format_rules),
        )

    def get_setup_payload(self) -> dict:
        return dict(self._setup_payload or {})

    def accept(self):
        if self._setup_mode:
            self._value_rules = list(self._collect_value_rules())
            self._naming_rules = list(self._collect_naming_rules())
            self._format_rules = list(self._collect_format_rules())
            self._setup_payload = {
                "value_rules": list(self._value_rules),
                "naming_rules": list(self._naming_rules),
                "format_rules": list(self._format_rules),
            }
            _save_geometry(self, 'dialogs/PublicationExport/geometry')
            super().accept()
            return

        payload, err = self._collect_export_plan()
        if err:
            QMessageBox.warning(
                self,
                tr("dialogs.publication_helper.validation_title", default="Publication Studio"),
                err,
            )
            return

        self._export_dir = str(payload.get("output_dir", ""))
        self._export_plan = list(payload.get("items", []))
        self._value_rules = list(payload.get("value_rules", []))
        self._naming_rules = list(payload.get("naming_rules", []))
        self._format_rules = list(payload.get("format_rules", []))
        _save_geometry(self, 'dialogs/PublicationExport/geometry')
        super().accept()


class CommandPaletteDialog(QDialog):
    """Searchable command launcher similar to modern desktop command palettes."""

    def __init__(self, commands, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.command_palette.title", default="Command Palette"))
        self.resize(760, 520)
        self.setMinimumSize(620, 420)
        self._commands = list(commands or [])

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("commandPaletteSearch")
        self.search_edit.setPlaceholderText(tr("dialogs.command_palette.search", default="Type a command or keyword..."))
        self.search_edit.setClearButtonEnabled(True)
        root.addWidget(self.search_edit)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("commandPaletteList")
        self.list_widget.setAlternatingRowColors(True)
        root.addWidget(self.list_widget, 1)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("hintLabel")
        root.addWidget(self.summary_label)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText(tr("dialogs.command_palette.run", default="Run Command"))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.search_edit.textChanged.connect(self._apply_filter)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list_widget.itemActivated.connect(lambda _item: self.accept())
        self.list_widget.currentRowChanged.connect(lambda _row: self._update_summary())

        self._search_shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        self._search_shortcut.activated.connect(self.search_edit.setFocus)

        self._apply_filter()
        self._update_summary()
        self.search_edit.setFocus()

        _restore_geometry(self, 'dialogs/CommandPalette/geometry')
        btns.accepted.connect(lambda: _save_geometry(self, 'dialogs/CommandPalette/geometry'))
        btns.rejected.connect(lambda: _save_geometry(self, 'dialogs/CommandPalette/geometry'))

    def _apply_filter(self):
        query = self.search_edit.text().strip().lower()
        self.list_widget.clear()

        for command in self._commands:
            text_blob = " ".join([
                str(command.get("title", "")),
                str(command.get("group", "")),
                str(command.get("description", "")),
                str(command.get("keywords", "")),
            ]).lower()
            if query and query not in text_blob:
                continue

            title = str(command.get("title", ""))
            group = str(command.get("group", tr("dialogs.command_palette.group_general", default="General")))
            description = str(command.get("description", ""))
            enabled = bool(command.get("enabled", True))
            status_suffix = "" if enabled else tr("dialogs.command_palette.unavailable", default=" [Unavailable]")

            item = QListWidgetItem(f"{group}: {title}{status_suffix}\n{description}")
            item.setData(Qt.ItemDataRole.UserRole, command)
            if not enabled:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        self._update_summary()

    def _update_summary(self):
        total = self.list_widget.count()
        current = self.list_widget.currentRow() + 1 if total > 0 else 0
        if total == 0:
            self.summary_label.setText(tr("dialogs.command_palette.none", default="No commands found for this search."))
        else:
            self.summary_label.setText(
                tr(
                    "dialogs.command_palette.summary",
                    default="{total} command(s) listed. Enter to run selected command ({current}/{total}).",
                    total=total,
                    current=current,
                )
            )

    def get_selected_command(self):
        item = self.list_widget.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data or not bool(data.get("enabled", True)):
            return None
        return data

    def accept(self):
        if self.get_selected_command() is None:
            return
        _save_geometry(self, 'dialogs/CommandPalette/geometry')
        super().accept()