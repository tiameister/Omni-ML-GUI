"""Modal dialog for configuring a single model's hyperparameters.

Opened from the ⚙️ gear button on each model card in "3. Models". Consumes
the schema registered in :mod:`models.hyperparameters` so the UI stays in
lockstep with the values that reach the estimator at training time.

Design goals
------------
* Human-readable labels with info tooltips explaining each parameter.
* Appropriate widgets per type (slider for ranges, dropdown for choices).
* "Restore Defaults" always visible.
* OK emits a ``dict`` of resolved values (pure data; no estimator instance)
  that the caller stores in :attr:`interface.logic.state.AppState.model_hyperparams`.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from models.hyperparameters import (
    decode_param_value,
    encode_param_value,
    get_default_hyperparams,
    get_param_schema,
)
from models.hyperparameter_presets import (
    CUSTOM_PRESET_ID,
    CUSTOM_PRESET_LABEL,
    get_presets,
    has_presets,
    match_preset,
    resolve_preset,
)
from utils.localization import tr


class _NoWheelFilter(QObject):
    """Swallow mouse-wheel events on spin boxes that live inside a QScrollArea.

    Without this, rolling the wheel over a spin box changes its value instead
    of scrolling the surrounding form, which is a classic Qt accessibility
    gotcha and a frequent source of accidental hyperparameter edits.
    """

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QAbstractSpinBox):
            if not obj.hasFocus():
                event.ignore()
                return True
        return False


# Pretty model labels for the dialog header (falls back to the raw key).
_MODEL_DISPLAY_NAMES = {
    "LinearRegression": "Linear Regression",
    "RidgeCV": "Ridge Regression (CV)",
    "RandomForest": "Random Forest",
    "HistGB": "Hist Gradient Boosting",
    "GradientBoostingRegressor": "Gradient Boosting",
    "Lasso": "Lasso (CV)",
    "ElasticNet": "Elastic Net (CV)",
    "SVR": "Support Vector Regression",
    "KNeighborsRegressor": "K-Nearest Neighbors",
    "XGBoost": "XGBoost",
}


class HyperparameterDialog(QDialog):
    """Settings modal for a single model's hyperparameters.

    Usage:
        dlg = HyperparameterDialog("RandomForest", current=state.model_hyperparams.get("RandomForest", {}))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            state.model_hyperparams["RandomForest"] = dlg.values()
    """

    def __init__(self, model_name: str,
                 current: dict[str, Any] | None = None,
                 parent=None):
        super().__init__(parent)
        self._model_name = str(model_name)
        self._schema = get_param_schema(self._model_name)
        self._defaults = get_default_hyperparams(self._model_name)
        # Per-param widget + a callable that reads the current raw value.
        self._readers: dict[str, callable] = {}
        self._widgets: dict[str, QWidget] = {}
        # Companion slider for widgets built with the slider+spin pairing;
        # tracked so preset application can sync both ends at once.
        self._sliders: dict[str, QSlider] = {}
        self._resolved: dict[str, Any] = {}
        # Owned by the dialog (same lifetime) so installEventFilter is safe.
        self._wheel_filter = _NoWheelFilter(self)
        # Preset dropdown handles. _suppress_preset_sync guards the combo
        # while the dialog pushes preset values into child widgets so the
        # combo doesn't immediately flip back to "Custom".
        self._preset_combo: QComboBox | None = None
        self._preset_description: QLabel | None = None
        self._preset_current_id: str = CUSTOM_PRESET_ID
        self._suppress_preset_sync: bool = False

        display_name = _MODEL_DISPLAY_NAMES.get(self._model_name, self._model_name)
        self.setWindowTitle(
            tr(
                "dialogs.hyperparameters.title",
                default="Configure: {model}",
                model=display_name,
            )
        )
        self.setMinimumSize(540, 360)
        self.resize(620, 500)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        header = QLabel(
            tr(
                "dialogs.hyperparameters.header",
                default="Adjust hyperparameters for <b>{model}</b>. "
                        "These values are passed directly to the estimator "
                        "at training time.",
                model=display_name,
            )
        )
        header.setWordWrap(True)
        root.addWidget(header)

        if not self._schema:
            info = QLabel(
                tr(
                    "dialogs.hyperparameters.no_schema",
                    default="This model has no user-configurable hyperparameters.",
                )
            )
            info.setWordWrap(True)
            root.addWidget(info)
        else:
            # Preset section (above the form). Only shown for models that
            # have at least one preset registered.
            if has_presets(self._model_name):
                root.addWidget(self._build_preset_section())

            # Scroll area keeps the dialog tidy even on small screens.
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            container = QWidget()
            form = QFormLayout(container)
            form.setContentsMargins(6, 6, 6, 6)
            form.setSpacing(10)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

            initial = dict(current or {})
            for spec in self._schema:
                label_widget = self._make_label(spec)
                field_widget = self._build_field(spec, initial)
                form.addRow(label_widget, field_widget)

            scroll.setWidget(container)
            root.addWidget(scroll, 1)

            # With every input widget built, connect its change signal to
            # _mark_custom so manual edits flip the preset dropdown to
            # "Custom (Manual Tuning)". The guard flag prevents preset
            # application (and the initial population) from triggering it.
            self._connect_change_listeners()
            self._sync_preset_from_values()

        btn_row = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        btn_row.accepted.connect(self._on_accept)
        btn_row.rejected.connect(self.reject)
        restore_btn = btn_row.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore_btn is not None:
            restore_btn.clicked.connect(self._restore_defaults)
            restore_btn.setToolTip(
                tr(
                    "dialogs.hyperparameters.restore_tooltip",
                    default="Reset all values to the recommended defaults.",
                )
            )
        root.addWidget(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def values(self) -> dict[str, Any]:
        """Return the resolved {param: value} map (populated on Accept)."""
        return dict(self._resolved)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _apply_spinbox_ux(self, spin: QAbstractSpinBox) -> None:
        """Common UX hardening for every spin box on this dialog."""
        # StrongFocus + wheel filter together prevent accidental value edits
        # from scrolling the surrounding form.
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spin.installEventFilter(self._wheel_filter)

    def _make_label(self, spec: dict[str, Any]) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        lbl = QLabel(str(spec.get("label", spec.get("name", ""))))
        font = QFont(lbl.font())
        lbl.setFont(font)
        lay.addWidget(lbl)

        tip_text = str(spec.get("tooltip", "")).strip()
        if tip_text:
            info = QToolButton()
            info.setAutoRaise(True)
            info.setCursor(Qt.CursorShape.WhatsThisCursor)
            try:
                info.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
                )
            except Exception:
                info.setText("ⓘ")
            info.setToolTip(tip_text)
            lay.addWidget(info)

        lay.addStretch(1)
        return wrap

    def _build_field(self, spec: dict[str, Any], initial: dict[str, Any]) -> QWidget:
        name = str(spec["name"])
        ptype = str(spec.get("type", ""))
        raw_default = encode_param_value(spec, spec.get("default"))
        if name in initial:
            raw_value = encode_param_value(spec, initial[name])
        else:
            raw_value = raw_default

        if ptype == "bool":
            return self._build_bool(spec, raw_value)
        if ptype == "choice":
            return self._build_choice(spec, raw_value)
        if ptype == "float":
            return self._build_float(spec, raw_value)
        if ptype in ("int", "int_or_none"):
            return self._build_int(spec, raw_value)

        # Unknown -> store value as-is via a disabled label so the user sees it.
        lbl = QLabel(str(raw_value))
        self._readers[name] = lambda v=raw_value: v
        return lbl

    def _build_bool(self, spec: dict[str, Any], raw_value: Any) -> QWidget:
        cb = QCheckBox()
        cb.setChecked(bool(raw_value))
        tip = str(spec.get("tooltip", "")).strip()
        if tip:
            cb.setToolTip(tip)
        name = str(spec["name"])
        self._widgets[name] = cb
        self._readers[name] = cb.isChecked
        return cb

    def _build_choice(self, spec: dict[str, Any], raw_value: Any) -> QWidget:
        combo = QComboBox()
        for label, value in spec.get("choices", []):
            combo.addItem(str(label), userData=value)
        # Select the item whose userData matches the current value.
        idx = 0
        for i in range(combo.count()):
            if combo.itemData(i) == raw_value:
                idx = i
                break
        combo.setCurrentIndex(idx)
        tip = str(spec.get("tooltip", "")).strip()
        if tip:
            combo.setToolTip(tip)
        name = str(spec["name"])
        self._widgets[name] = combo
        self._readers[name] = combo.currentData
        return combo

    def _build_float(self, spec: dict[str, Any], raw_value: Any) -> QWidget:
        spin = QDoubleSpinBox()
        spin.setRange(float(spec.get("min", -1e9)), float(spec.get("max", 1e9)))
        step = float(spec.get("step", 0.01))
        spin.setSingleStep(step)
        # Show enough precision to see the step size.
        if step < 1e-3:
            decimals = 6
        elif step < 1e-2:
            decimals = 4
        elif step < 1e-1:
            decimals = 3
        else:
            decimals = 2
        spin.setDecimals(decimals)
        try:
            spin.setValue(float(raw_value))
        except (TypeError, ValueError):
            spin.setValue(float(spec.get("default", 0.0)))
        tip = str(spec.get("tooltip", "")).strip()
        if tip:
            spin.setToolTip(tip)
        self._apply_spinbox_ux(spin)
        name = str(spec["name"])
        self._widgets[name] = spin
        self._readers[name] = spin.value
        return spin

    def _build_int(self, spec: dict[str, Any], raw_value: Any) -> QWidget:
        spin = QSpinBox()
        lo = int(spec.get("min", 0))
        hi = int(spec.get("max", 10_000))
        ptype = str(spec.get("type", ""))

        if ptype == "int_or_none":
            # The sentinel must equal the spinbox minimum so
            # setSpecialValueText triggers exactly on "None". Enforcing this
            # invariant here removes any ambiguity between schema values.
            sentinel = int(spec.get("none_sentinel", lo))
            lo = sentinel
            spin.setRange(lo, hi)
            spin.setSpecialValueText(
                tr("dialogs.hyperparameters.unlimited", default="Unlimited")
            )
        else:
            spin.setRange(lo, hi)

        spin.setSingleStep(int(spec.get("step", 1)))

        try:
            spin.setValue(int(raw_value))
        except (TypeError, ValueError):
            default_val = spec.get("default")
            if default_val is None and ptype == "int_or_none":
                spin.setValue(int(spec.get("none_sentinel", spin.minimum())))
            else:
                spin.setValue(int(default_val or 0))

        tip = str(spec.get("tooltip", "")).strip()
        if tip:
            spin.setToolTip(tip)
        self._apply_spinbox_ux(spin)

        container: QWidget = spin
        if bool(spec.get("slider", False)):
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(spin.minimum(), spin.maximum())
            slider.setSingleStep(spin.singleStep())
            slider.setPageStep(max(1, spin.singleStep() * 5))
            slider.setValue(spin.value())
            slider.setTracking(True)

            def _on_slider(v: int, s=spin):
                s.blockSignals(True)
                s.setValue(int(v))
                s.blockSignals(False)

            def _on_spin(v: int, s=slider):
                s.blockSignals(True)
                s.setValue(int(v))
                s.blockSignals(False)

            slider.valueChanged.connect(_on_slider)
            spin.valueChanged.connect(_on_spin)
            spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(slider, 1)
            row.addWidget(spin, 0)

        name = str(spec["name"])
        self._widgets[name] = spin
        self._readers[name] = spin.value
        if bool(spec.get("slider", False)):
            # 'slider' is defined in the slider+spin branch above; capture it
            # for later syncing from preset application.
            self._sliders[name] = slider  # type: ignore[name-defined]
        return container

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _build_preset_section(self) -> QWidget:
        """Build the "Configuration Preset" dropdown + description row."""
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        label = QLabel(
            tr(
                "dialogs.hyperparameters.preset_label",
                default="Configuration Preset:",
            )
        )
        label_font = QFont(label.font())
        label_font.setBold(True)
        label.setFont(label_font)

        combo = QComboBox()
        for preset in get_presets(self._model_name):
            combo.addItem(preset["label"], userData=preset["id"])
        combo.addItem(CUSTOM_PRESET_LABEL, userData=CUSTOM_PRESET_ID)
        combo.setToolTip(
            tr(
                "dialogs.hyperparameters.preset_tooltip",
                default="Pick a curated configuration. Manual edits below will "
                        "switch this back to 'Custom (Manual Tuning)'.",
            )
        )
        combo.currentIndexChanged.connect(self._on_preset_changed)
        self._preset_combo = combo

        row.addWidget(label, 0)
        row.addWidget(combo, 1)
        lay.addLayout(row)

        description = QLabel("")
        description.setWordWrap(True)
        description.setObjectName("hyperparameterPresetDescription")
        description.setStyleSheet("color: #5B6C7B; font-size: 11px;")
        self._preset_description = description
        lay.addWidget(description)

        return wrap

    def _connect_change_listeners(self) -> None:
        """Connect each input widget so manual edits flip combo to 'Custom'."""
        for name, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self._mark_custom)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._mark_custom)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._mark_custom)
            slider = self._sliders.get(name)
            if slider is not None:
                slider.valueChanged.connect(self._mark_custom)

    def _on_preset_changed(self, _idx: int) -> None:
        """Handle the user picking a new preset from the dropdown."""
        if self._suppress_preset_sync or self._preset_combo is None:
            return
        preset_id = str(self._preset_combo.currentData() or CUSTOM_PRESET_ID)
        self._preset_current_id = preset_id
        if preset_id == CUSTOM_PRESET_ID:
            self._update_preset_description(None)
            return
        resolved = resolve_preset(self._model_name, preset_id)
        if resolved is None:
            return
        self._apply_values(resolved)
        self._update_preset_description(preset_id)

    def _apply_values(self, values: dict[str, Any]) -> None:
        """Push ``values`` into every widget without flipping combo to 'Custom'.

        ``values`` is keyed by sklearn parameter name (decoded form).
        """
        self._suppress_preset_sync = True
        try:
            for spec in self._schema:
                name = str(spec["name"])
                if name not in values:
                    continue
                raw = encode_param_value(spec, values[name])
                self._set_widget_value(spec, raw)
        finally:
            self._suppress_preset_sync = False

    def _set_widget_value(self, spec: dict[str, Any], raw_value: Any) -> None:
        """Assign ``raw_value`` to the widget for ``spec`` (guarded setter)."""
        name = str(spec["name"])
        widget = self._widgets.get(name)
        if widget is None:
            return
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(raw_value))
        elif isinstance(widget, QComboBox):
            for i in range(widget.count()):
                if widget.itemData(i) == raw_value:
                    widget.setCurrentIndex(i)
                    break
        elif isinstance(widget, QDoubleSpinBox):
            try:
                widget.setValue(float(raw_value))
            except (TypeError, ValueError):
                widget.setValue(float(spec.get("default", 0.0)))
        elif isinstance(widget, QSpinBox):
            try:
                widget.setValue(int(raw_value))
            except (TypeError, ValueError):
                default_val = spec.get("default")
                if default_val is None and str(spec.get("type", "")) == "int_or_none":
                    widget.setValue(int(spec.get("none_sentinel", widget.minimum())))
                else:
                    widget.setValue(int(default_val or 0))
        # Keep the companion slider in sync with the spin value.
        slider = self._sliders.get(name)
        if slider is not None and isinstance(widget, QSpinBox):
            slider.setValue(widget.value())

    def _mark_custom(self, *_args) -> None:
        """Flip the preset combo to 'Custom' in response to a manual edit."""
        if self._suppress_preset_sync or self._preset_combo is None:
            return
        if self._preset_current_id == CUSTOM_PRESET_ID:
            return
        self._preset_current_id = CUSTOM_PRESET_ID
        self._preset_combo.blockSignals(True)
        try:
            for i in range(self._preset_combo.count()):
                if self._preset_combo.itemData(i) == CUSTOM_PRESET_ID:
                    self._preset_combo.setCurrentIndex(i)
                    break
        finally:
            self._preset_combo.blockSignals(False)
        self._update_preset_description(None)

    def _sync_preset_from_values(self) -> None:
        """On dialog open: pick the preset that matches current values, else Custom."""
        if self._preset_combo is None:
            return
        current_values = {
            str(spec["name"]): decode_param_value(spec, self._readers[str(spec["name"])]())
            for spec in self._schema
            if str(spec["name"]) in self._readers
        }
        matched = match_preset(self._model_name, current_values) or CUSTOM_PRESET_ID
        self._preset_current_id = matched
        self._preset_combo.blockSignals(True)
        try:
            for i in range(self._preset_combo.count()):
                if self._preset_combo.itemData(i) == matched:
                    self._preset_combo.setCurrentIndex(i)
                    break
        finally:
            self._preset_combo.blockSignals(False)
        self._update_preset_description(matched if matched != CUSTOM_PRESET_ID else None)

    def _update_preset_description(self, preset_id: str | None) -> None:
        if self._preset_description is None:
            return
        if not preset_id or preset_id == CUSTOM_PRESET_ID:
            self._preset_description.setText(
                tr(
                    "dialogs.hyperparameters.custom_description",
                    default="You have customized individual values.",
                )
            )
            return
        for preset in get_presets(self._model_name):
            if preset["id"] == preset_id:
                self._preset_description.setText(preset.get("description", ""))
                return
        self._preset_description.setText("")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_accept(self):
        resolved: dict[str, Any] = {}
        for spec in self._schema:
            name = str(spec["name"])
            reader = self._readers.get(name)
            raw = reader() if callable(reader) else spec.get("default")
            resolved[name] = decode_param_value(spec, raw)
        self._resolved = resolved
        self.accept()

    def _restore_defaults(self):
        """Re-populate every widget from the schema defaults.

        Routes through :meth:`_apply_values` so the combo is re-synced
        afterwards: if the defaults happen to match a registered preset,
        the dropdown picks it; otherwise it shows 'Custom'.
        """
        self._apply_values(dict(self._defaults))
        self._sync_preset_from_values()


__all__ = ["HyperparameterDialog"]
