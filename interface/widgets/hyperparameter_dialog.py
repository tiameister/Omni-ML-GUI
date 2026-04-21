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

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
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
from utils.localization import tr


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
        self._resolved: dict[str, Any] = {}

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
            info.clicked.connect(lambda _c=False, t=tip_text: info.setToolTip(t))
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
        name = str(spec["name"])
        self._widgets[name] = spin
        self._readers[name] = spin.value
        return spin

    def _build_int(self, spec: dict[str, Any], raw_value: Any) -> QWidget:
        spin = QSpinBox()
        lo = int(spec.get("min", 0))
        hi = int(spec.get("max", 10_000))
        spin.setRange(lo, hi)
        spin.setSingleStep(int(spec.get("step", 1)))

        if str(spec.get("type", "")) == "int_or_none":
            sentinel = int(spec.get("none_sentinel", lo))
            # Qt shows a custom string for the minimum; use that for "Unlimited".
            spin.setSpecialValueText(
                tr("dialogs.hyperparameters.unlimited", default="Unlimited")
            )
            # Ensure the sentinel lines up with the minimum so
            # setSpecialValueText triggers correctly.
            if sentinel != lo:
                spin.setRange(sentinel, hi)

        try:
            spin.setValue(int(raw_value))
        except (TypeError, ValueError):
            default_val = spec.get("default")
            if default_val is None and str(spec.get("type", "")) == "int_or_none":
                spin.setValue(int(spec.get("none_sentinel", spin.minimum())))
            else:
                spin.setValue(int(default_val or 0))

        tip = str(spec.get("tooltip", "")).strip()
        if tip:
            spin.setToolTip(tip)

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
        return container

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
        """Re-populate every widget from the schema defaults."""
        for spec in self._schema:
            name = str(spec["name"])
            widget = self._widgets.get(name)
            raw_default = encode_param_value(spec, spec.get("default"))
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(raw_default))
            elif isinstance(widget, QComboBox):
                for i in range(widget.count()):
                    if widget.itemData(i) == raw_default:
                        widget.setCurrentIndex(i)
                        break
            elif isinstance(widget, QDoubleSpinBox):
                try:
                    widget.setValue(float(raw_default))
                except (TypeError, ValueError):
                    pass
            elif isinstance(widget, QSpinBox):
                try:
                    widget.setValue(int(raw_default))
                except (TypeError, ValueError):
                    pass


__all__ = ["HyperparameterDialog"]
