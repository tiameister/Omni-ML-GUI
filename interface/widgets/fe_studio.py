"""
Feature Engineering Studio — professional configuration dialog for
Omni-ML-GUI's structural feature transforms.

Design principles
-----------------
* Every setting shows a plain-language description AND a risk badge so the
  user understands both *what* the transform does and *when* it is safe.
* An "Active Pipeline" banner updates live as settings change.
* High-impact options (Polynomial, Binning w/ data-leakage modes) carry
  explicit visual warnings.
* A "Reset to Defaults" button is always visible.
* Two mutually-exclusive exit paths:
    - Dynamic Pipeline (safe)  → transform runs inside CV folds
    - Generate Dataset          → static export with explicit leakage warning
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QFrame, QScrollArea, QWidget,
    QMessageBox, QApplication, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont

from interface.widgets.apple_helpers import create_apple_settings_row
from utils.localization import tr


# ---------------------------------------------------------------------------
# Risk-level colour palette
# ---------------------------------------------------------------------------
_RISK_COLORS = {
    "safe":    ("#34C759", "#E8F8ED"),   # green  — border, bg
    "cv":      ("#FF9500", "#FFF4E6"),   # amber  — fitted on CV fold (correct)
    "leakage": ("#FF3B30", "#FFF0EF"),   # red    — global fit / data leakage
    "heavy":   ("#5E5CE6", "#F0F0FF"),   # purple — expensive computation
}

_DEFAULT_CONFIG: dict = {
    "transform":         "yeo-johnson",
    "missing_indicators": True,
    "outliers":          "winsorize_1_99",
    "binning":           "none",
    "n_bins":            5,
    "poly_features":     False,
    "poly_degree":       2,
    "poly_max":          50,
    "interaction_only":  False,
}


# ---------------------------------------------------------------------------
# Helper: coloured chip label
# ---------------------------------------------------------------------------
def _make_chip(text: str, risk: str = "safe") -> QLabel:
    border_color, bg_color = _RISK_COLORS.get(risk, _RISK_COLORS["safe"])
    chip = QLabel(text)
    chip.setFixedHeight(24)
    chip.setContentsMargins(10, 0, 10, 0)
    chip.setStyleSheet(
        f"QLabel {{"
        f"  background: {bg_color};"
        f"  color: {border_color};"
        f"  border: 1px solid {border_color};"
        f"  border-radius: 12px;"
        f"  font-size: 11px;"
        f"  font-weight: 600;"
        f"}}"
    )
    chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return chip


def _make_section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size: 11px; font-weight: 700; color: #6E6E73;"
        "letter-spacing: 0.8px; padding-left: 4px;"
    )
    return lbl


def _make_risk_badge(risk: str, label: str) -> QLabel:
    border_color, bg_color = _RISK_COLORS.get(risk, _RISK_COLORS["safe"])
    badge = QLabel(label)
    badge.setFixedHeight(20)
    badge.setContentsMargins(8, 0, 8, 0)
    badge.setStyleSheet(
        f"QLabel {{"
        f"  background: {bg_color};"
        f"  color: {border_color};"
        f"  border: 1px solid {border_color};"
        f"  border-radius: 10px;"
        f"  font-size: 10px;"
        f"  font-weight: 700;"
        f"}}"
    )
    badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return badge


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class FeatureEngineeringStudioDialog(QDialog):
    """
    Feature Engineering Studio.

    Accepts the current FE config dict, lets the user edit every parameter,
    then returns the new config via ``get_config()``.

    Attributes
    ----------
    export_requested : bool
        True when the user chose "Generate Dataset" instead of "Apply Pipeline".
    """

    def __init__(self, current_config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            tr("dialogs.fe_studio.title", default="Feature Engineering Studio")
        )
        self.setMinimumSize(660, 740)
        self.resize(700, 800)
        self.current_config = {**_DEFAULT_CONFIG, **current_config}
        self.export_requested = False

        self._setup_ui()
        self._apply_config()
        self._apply_styles()
        self._refresh_pipeline_summary()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("studioHeader")
        hdr_lay = QVBoxLayout(header)
        hdr_lay.setContentsMargins(28, 24, 28, 20)
        hdr_lay.setSpacing(4)

        title_lbl = QLabel(
            tr("dialogs.fe_studio.header", default="Feature Engineering Studio")
        )
        title_lbl.setObjectName("studioTitle")

        subtitle_lbl = QLabel(
            tr(
                "dialogs.fe_studio.subtitle",
                default=(
                    "Configure automated transformations that run safely inside each "
                    "cross-validation fold, preventing data leakage."
                ),
            )
        )
        subtitle_lbl.setObjectName("studioSubtitle")
        subtitle_lbl.setWordWrap(True)

        hdr_lay.addWidget(title_lbl)
        hdr_lay.addWidget(subtitle_lbl)
        root.addWidget(header)

        # ── Active Pipeline Banner ─────────────────────────────────────
        banner_frame = QFrame()
        banner_frame.setObjectName("pipelineBanner")
        self._banner_lay = QVBoxLayout(banner_frame)
        self._banner_lay.setContentsMargins(28, 14, 28, 14)
        self._banner_lay.setSpacing(8)

        banner_title = QLabel("ACTIVE PIPELINE")
        banner_title.setStyleSheet(
            "font-size: 10px; font-weight: 700; color: #8E8E93; letter-spacing: 0.8px;"
        )
        self._banner_lay.addWidget(banner_title)

        self._chips_container = QWidget()
        self._chips_layout = QHBoxLayout(self._chips_container)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(8)
        self._banner_lay.addWidget(self._chips_container)
        root.addWidget(banner_frame)

        # ── Scrollable Settings Area ───────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("studioScroll")

        content = QWidget()
        content.setObjectName("studioContent")
        self._lay = QVBoxLayout(content)
        self._lay.setContentsMargins(28, 20, 28, 28)
        self._lay.setSpacing(0)

        self._build_distribution_section()
        self._lay.addSpacing(20)
        self._build_robustness_section()
        self._lay.addSpacing(20)
        self._build_shaping_section()
        self._lay.addSpacing(20)
        self._build_expansion_section()
        self._lay.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ── Footer ────────────────────────────────────────────────────
        self._build_footer(root)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_label(self, text: str, risk_key: str | None = None) -> QFrame:
        """Returns a section-header row with optional risk badge."""
        row = QFrame()
        row.setObjectName("sectionLabelRow")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(8)
        lay.addWidget(_make_section_header(text))
        if risk_key:
            risk_labels = {
                "safe":    "CV-SAFE",
                "cv":      "CV-FITTED",
                "leakage": "LEAKAGE RISK",
                "heavy":   "MEMORY-HEAVY",
            }
            lay.addWidget(_make_risk_badge(risk_key, risk_labels.get(risk_key, risk_key)))
        lay.addStretch(1)
        return row

    def _build_distribution_section(self):
        self._lay.addWidget(self._section_label("DISTRIBUTION & IMPUTATION", "cv"))

        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        # Power Transform
        self.transform_combo = QComboBox()
        self.transform_combo.setObjectName("appleCombo")
        self.transform_combo.addItem("None — keep raw distribution", "none")
        self.transform_combo.addItem("Yeo-Johnson  (handles zeros & negatives)", "yeo-johnson")
        self.transform_combo.addItem("Log(1+x)  (simple, interpretable)", "log1p")
        self.transform_combo.setMinimumWidth(260)
        self.transform_combo.currentIndexChanged.connect(self._refresh_pipeline_summary)

        row_transform, _, _ = create_apple_settings_row(
            self.transform_combo,
            title_text="Power Transformation",
            subtitle_text=(
                "Reshapes heavily skewed columns toward a normal distribution — "
                "improves linear model performance and SHAP interpretability."
            ),
            show_bottom_line=True,
        )
        card_lay.addWidget(row_transform)

        # Missing Indicators
        self.missing_toggle = QCheckBox()
        self.missing_toggle.setObjectName("toggleSwitch")
        self.missing_toggle.toggled.connect(self._refresh_pipeline_summary)

        row_miss, _, _ = create_apple_settings_row(
            self.missing_toggle,
            title_text="Missing Value Indicators",
            subtitle_text=(
                "Adds a binary 0/1 column for every feature that contains nulls, "
                "encoding the missingness pattern as a learnable signal."
            ),
            show_bottom_line=False,
        )
        card_lay.addWidget(row_miss)
        self._lay.addWidget(card)

    def _build_robustness_section(self):
        self._lay.addWidget(self._section_label("ROBUSTNESS", "cv"))

        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        self.outlier_combo = QComboBox()
        self.outlier_combo.setObjectName("appleCombo")
        self.outlier_combo.addItem("None — keep all extreme values", "none")
        self.outlier_combo.addItem("Winsorize  1st – 99th percentile  (mild)", "winsorize_1_99")
        self.outlier_combo.addItem("Winsorize  5th – 95th percentile  (aggressive)", "winsorize_5_95")
        self.outlier_combo.setMinimumWidth(260)
        self.outlier_combo.currentIndexChanged.connect(self._refresh_pipeline_summary)

        row_out, _, _ = create_apple_settings_row(
            self.outlier_combo,
            title_text="Outlier Clipping (Winsorization)",
            subtitle_text=(
                "Caps extreme values at a percentile boundary computed on the training "
                "fold. Protects non-robust models (Linear, SVR) from noise spikes "
                "without removing any rows."
            ),
            show_bottom_line=False,
        )
        card_lay.addWidget(row_out)
        self._lay.addWidget(card)

    def _build_shaping_section(self):
        self._lay.addWidget(self._section_label("MACRO SHAPING", "leakage"))

        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        self.bin_combo = QComboBox()
        self.bin_combo.setObjectName("appleCombo")
        self.bin_combo.addItem("None — keep continuous", "none")
        self.bin_combo.addItem("Uniform  (equal-width buckets)", "uniform")
        self.bin_combo.addItem("Quantile  (equal-frequency buckets)", "quantile")
        self.bin_combo.addItem("K-Means  (cluster-boundary buckets)", "kmeans")
        self.bin_combo.setMinimumWidth(260)
        self.bin_combo.currentIndexChanged.connect(self._on_bin_changed)
        self.bin_combo.currentIndexChanged.connect(self._refresh_pipeline_summary)

        row_bin, _, _ = create_apple_settings_row(
            self.bin_combo,
            title_text="Discretization (Binning)",
            subtitle_text=(
                "Converts continuous features into discrete ordinal buckets. "
                "Useful for tree models to capture threshold effects. "
                "Bin boundaries are fitted on the training fold only."
            ),
            show_bottom_line=True,
        )
        card_lay.addWidget(row_bin)

        self.bin_spin = QSpinBox()
        self.bin_spin.setObjectName("appleSpinBox")
        self.bin_spin.setRange(2, 20)
        self.bin_spin.setMinimumWidth(70)
        self.bin_spin.valueChanged.connect(self._refresh_pipeline_summary)

        self._row_n_bins, _, _ = create_apple_settings_row(
            self.bin_spin,
            title_text="Number of Bins",
            subtitle_text="How many discrete segments per continuous feature.",
            show_bottom_line=False,
        )
        card_lay.addWidget(self._row_n_bins)
        self._lay.addWidget(card)

    def _build_expansion_section(self):
        self._lay.addWidget(self._section_label("DIMENSIONALITY EXPANSION", "heavy"))

        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        # Polynomial toggle
        self.poly_toggle = QCheckBox()
        self.poly_toggle.setObjectName("toggleSwitch")
        self.poly_toggle.toggled.connect(self._on_poly_toggled)
        self.poly_toggle.toggled.connect(self._refresh_pipeline_summary)

        row_poly, _, _ = create_apple_settings_row(
            self.poly_toggle,
            title_text="Polynomial / Interaction Features",
            subtitle_text=(
                "Generates cross-product and power terms (A·B, A², …) so that linear "
                "models can capture non-linear relationships. Increases feature count "
                "exponentially — use with small feature sets only."
            ),
            show_bottom_line=True,
        )
        card_lay.addWidget(row_poly)

        # Degree
        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setObjectName("appleSpinBox")
        self.poly_degree_spin.setRange(2, 4)
        self.poly_degree_spin.setMinimumWidth(70)
        self.poly_degree_spin.valueChanged.connect(self._refresh_poly_hint)
        self.poly_degree_spin.valueChanged.connect(self._refresh_pipeline_summary)

        self._row_degree, _, _ = create_apple_settings_row(
            self.poly_degree_spin,
            title_text="Polynomial Degree",
            subtitle_text="Maximum term degree. Degree 2: A², A·B. Degree 3: A³, A²·B, A·B·C.",
            show_bottom_line=True,
        )
        card_lay.addWidget(self._row_degree)

        # Interaction-only toggle
        self.interaction_only_toggle = QCheckBox()
        self.interaction_only_toggle.setObjectName("toggleSwitch")
        self.interaction_only_toggle.toggled.connect(self._refresh_poly_hint)
        self.interaction_only_toggle.toggled.connect(self._refresh_pipeline_summary)

        self._row_interaction, _, _ = create_apple_settings_row(
            self.interaction_only_toggle,
            title_text="Interactions Only  (no power terms)",
            subtitle_text=(
                "Suppresses A², A³ … terms and produces only cross-products A·B, A·B·C. "
                "Recommended for ordinal / psychometric scales where squaring is "
                "semantically meaningless."
            ),
            show_bottom_line=True,
        )
        card_lay.addWidget(self._row_interaction)

        # Max features
        self.poly_max_spin = QSpinBox()
        self.poly_max_spin.setObjectName("appleSpinBox")
        self.poly_max_spin.setRange(5, 500)
        self.poly_max_spin.setSingleStep(5)
        self.poly_max_spin.setMinimumWidth(70)
        self.poly_max_spin.valueChanged.connect(self._refresh_poly_hint)

        self._row_poly_max, _, _ = create_apple_settings_row(
            self.poly_max_spin,
            title_text="Max Input Features",
            subtitle_text=(
                "Caps the number of continuous features fed into the expander "
                "(selected by variance). Prevents RAM exhaustion."
            ),
            show_bottom_line=True,
        )
        card_lay.addWidget(self._row_poly_max)

        # Poly feature-count hint label
        self._poly_hint_lbl = QLabel()
        self._poly_hint_lbl.setObjectName("polyHintLabel")
        self._poly_hint_lbl.setWordWrap(True)
        self._poly_hint_lbl.setContentsMargins(16, 8, 16, 10)
        card_lay.addWidget(self._poly_hint_lbl)

        self._lay.addWidget(card)

    def _build_footer(self, root: QVBoxLayout):
        footer = QFrame()
        footer.setObjectName("studioFooter")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(28, 14, 28, 14)
        foot_lay.setSpacing(10)

        btn_reset = QPushButton(tr("common.reset_defaults", default="Reset Defaults"))
        btn_reset.setObjectName("ghostButton")
        btn_reset.setToolTip("Restore all settings to their original safe defaults.")
        btn_reset.clicked.connect(self._on_reset)

        self.btn_cancel = QPushButton(tr("common.cancel", default="Cancel"))
        self.btn_cancel.setObjectName("actionButton")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_export = QPushButton(
            tr("dialogs.fe_studio.export", default="⚠  Generate CSV")
        )
        self.btn_export.setObjectName("warningButton")
        self.btn_export.setToolTip(
            "Export a static CSV transformed on the full dataset.\n"
            "Warning: distribution-based transforms fitted globally introduce data leakage."
        )
        self.btn_export.clicked.connect(self._on_export_clicked)

        self.btn_apply = QPushButton(
            tr("dialogs.fe_studio.apply_pipeline", default="✓  Apply to Pipeline")
        )
        self.btn_apply.setObjectName("accentButton")
        self.btn_apply.setMinimumWidth(170)
        self.btn_apply.setToolTip(
            "Schedule these transforms to run inside each CV fold — no data leakage."
        )
        self.btn_apply.clicked.connect(self.accept)

        foot_lay.addWidget(btn_reset)
        foot_lay.addStretch(1)
        foot_lay.addWidget(self.btn_cancel)
        foot_lay.addWidget(self.btn_export)
        foot_lay.addWidget(self.btn_apply)
        root.addWidget(footer)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def _refresh_pipeline_summary(self):
        """Rebuild the active-pipeline chip bar based on current widget state."""
        # Clear existing chips
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        active: list[tuple[str, str]] = []

        transform = self.transform_combo.currentData()
        if transform == "yeo-johnson":
            active.append(("Yeo-Johnson", "cv"))
        elif transform == "log1p":
            active.append(("Log(1+x)", "cv"))

        if self.missing_toggle.isChecked():
            active.append(("Missing Indicators", "safe"))

        outlier = self.outlier_combo.currentData()
        if outlier == "winsorize_1_99":
            active.append(("Winsorize 1-99", "cv"))
        elif outlier == "winsorize_5_95":
            active.append(("Winsorize 5-95", "cv"))

        binning = self.bin_combo.currentData()
        if binning == "uniform":
            active.append(("Uniform Bins", "cv"))
        elif binning == "quantile":
            active.append(("Quantile Bins", "leakage"))
        elif binning == "kmeans":
            active.append(("K-Means Bins", "leakage"))

        if self.poly_toggle.isChecked():
            label = "Interactions" if self.interaction_only_toggle.isChecked() else f"Poly^{self.poly_degree_spin.value()}"
            active.append((label, "heavy"))

        if not active:
            none_lbl = QLabel("No transforms active  —  raw features passed through")
            none_lbl.setStyleSheet("color: #8E8E93; font-size: 12px; font-style: italic;")
            self._chips_layout.addWidget(none_lbl)
        else:
            for text, risk in active:
                self._chips_layout.addWidget(_make_chip(text, risk))

        self._chips_layout.addStretch(1)

    def _refresh_poly_hint(self):
        """Show estimated output feature count based on degree and max_features."""
        if not self.poly_toggle.isChecked():
            self._poly_hint_lbl.setText("")
            return
        n = self.poly_max_spin.value()
        d = self.poly_degree_spin.value()
        interaction_only = self.interaction_only_toggle.isChecked()
        try:
            from math import comb
            if interaction_only:
                # sum of C(n, k) for k in 2..d
                count = sum(comb(n, k) for k in range(2, d + 1))
            else:
                # C(n+d, d) - 1  (includes powers but not bias)
                count = comb(n + d, d) - 1
            if count > 5_000:
                colour = "#FF3B30"
                warn = "  ⚠ Very large — may cause memory issues."
            elif count > 1_000:
                colour = "#FF9500"
                warn = "  Consider reducing max features or degree."
            else:
                colour = "#34C759"
                warn = ""
            self._poly_hint_lbl.setText(
                f"Estimated new features from {n} inputs: "
                f"<b style='color:{colour}'>{count:,}</b>{warn}"
            )
        except Exception:
            self._poly_hint_lbl.setText("")

    def _on_poly_toggled(self, checked: bool):
        for w in (
            self._row_degree,
            self._row_interaction,
            self._row_poly_max,
            self._poly_hint_lbl,
        ):
            w.setEnabled(checked)
            w.setStyleSheet("" if checked else "opacity: 0.4;")
        if checked:
            self._refresh_poly_hint()
        else:
            self._poly_hint_lbl.setText("")

    def _on_bin_changed(self):
        enabled = self.bin_combo.currentData() != "none"
        self._row_n_bins.setEnabled(enabled)
        self._row_n_bins.setStyleSheet("" if enabled else "opacity: 0.4;")

    # ------------------------------------------------------------------
    # Config round-trip
    # ------------------------------------------------------------------

    def _apply_config(self):
        c = self.current_config

        idx = self.transform_combo.findData(c.get("transform", "yeo-johnson"))
        if idx >= 0:
            self.transform_combo.setCurrentIndex(idx)

        self.missing_toggle.setChecked(bool(c.get("missing_indicators", True)))

        idx = self.outlier_combo.findData(c.get("outliers", "winsorize_1_99"))
        if idx >= 0:
            self.outlier_combo.setCurrentIndex(idx)

        idx = self.bin_combo.findData(c.get("binning", "none"))
        if idx >= 0:
            self.bin_combo.setCurrentIndex(idx)
        self.bin_spin.setValue(int(c.get("n_bins", 5)))

        poly_on = bool(c.get("poly_features", False))
        self.poly_toggle.setChecked(poly_on)
        self.poly_degree_spin.setValue(int(c.get("poly_degree", 2)))
        self.interaction_only_toggle.setChecked(bool(c.get("interaction_only", False)))
        self.poly_max_spin.setValue(int(c.get("poly_max", 50)))

        # Trigger dependent states
        self._on_poly_toggled(poly_on)
        self._on_bin_changed()

    def get_config(self) -> dict:
        return {
            "transform":          self.transform_combo.currentData(),
            "missing_indicators": self.missing_toggle.isChecked(),
            "outliers":           self.outlier_combo.currentData(),
            "binning":            self.bin_combo.currentData(),
            "n_bins":             self.bin_spin.value(),
            "poly_features":      self.poly_toggle.isChecked(),
            "poly_degree":        self.poly_degree_spin.value(),
            "interaction_only":   self.interaction_only_toggle.isChecked(),
            "poly_max":           self.poly_max_spin.value(),
        }

    def _on_reset(self):
        self.current_config = dict(_DEFAULT_CONFIG)
        self._apply_config()
        self._refresh_pipeline_summary()

    # ------------------------------------------------------------------
    # Export with leakage warning
    # ------------------------------------------------------------------

    def _on_export_clicked(self):
        c = self.get_config()
        leaks: list[str] = []
        if c["outliers"] != "none":
            leaks.append(
                f"• Outlier Clipping ({c['outliers']}) computes global percentile "
                "bounds on the full dataset."
            )
        if c["transform"] != "none":
            leaks.append(
                f"• Power Transform ({c['transform']}) fits distribution parameters "
                "on the full dataset."
            )
        if c["binning"] in ("quantile", "kmeans"):
            leaks.append(
                f"• Discretization ({c['binning']}) calculates bin boundaries across "
                "the full dataset."
            )

        warn_msg = (
            "⚠  Data Leakage Risk Detected\n\n"
            "Generating a static CSV fits the following transforms on the ENTIRE "
            "dataset — including rows that would be held out in cross-validation:\n\n"
            + "\n".join(leaks or ["• None detected for current settings."])
            + "\n\n"
            "The resulting file must NOT be used as-is for CV-based evaluation "
            "without understanding that validation rows have seen training statistics.\n\n"
            "For scientifically rigorous evaluation use  ✓ Apply to Pipeline  instead, "
            "which applies every transform inside each fold.\n\n"
            "Generate the static dataset anyway?"
        )
        reply = QMessageBox.warning(
            self,
            "Data Leakage Warning",
            warn_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.export_requested = True
        self.accept()

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    def _apply_styles(self):
        self.setStyleSheet("""
            /* ── Dialog ──────────────────────────────────────────── */
            QDialog {
                background: #F2F2F7;
            }

            /* ── Header ──────────────────────────────────────────── */
            QFrame#studioHeader {
                background: #FFFFFF;
                border-bottom: 1px solid #E5E5EA;
            }
            QLabel#studioTitle {
                font-size: 20px;
                font-weight: 700;
                color: #1C1C1E;
                letter-spacing: -0.3px;
            }
            QLabel#studioSubtitle {
                font-size: 12px;
                color: #8E8E93;
                line-height: 1.4;
            }

            /* ── Pipeline Banner ─────────────────────────────────── */
            QFrame#pipelineBanner {
                background: #FFFFFF;
                border-bottom: 1px solid #E5E5EA;
            }

            /* ── Section label row ───────────────────────────────── */
            QFrame#sectionLabelRow {
                background: transparent;
            }

            /* ── Cards ───────────────────────────────────────────── */
            QFrame#appleCard {
                background: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E5E5EA;
            }
            QFrame#appleRowContainer {
                background: transparent;
            }
            QFrame#appleRowContent {
                background: transparent;
            }
            QLabel#appleRowTitle {
                font-size: 13px;
                font-weight: 600;
                color: #1C1C1E;
            }
            QLabel#appleRowSubtitle {
                font-size: 11px;
                color: #8E8E93;
                line-height: 1.3;
            }

            /* ── Poly hint ───────────────────────────────────────── */
            QLabel#polyHintLabel {
                font-size: 12px;
                color: #3C3C43;
                background: #F9F9FB;
                border-top: 1px solid #E5E5EA;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }

            /* ── Controls ────────────────────────────────────────── */
            QComboBox#appleCombo {
                background-color: #F2F2F7;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                padding: 6px 12px;
                color: #1C1C1E;
                font-size: 13px;
                font-weight: 500;
                min-width: 200px;
            }
            QComboBox#appleCombo::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox#appleCombo:hover {
                border-color: #007AFF;
            }
            QSpinBox#appleSpinBox {
                background-color: #F2F2F7;
                border: 1px solid #D1D1D6;
                border-radius: 8px;
                padding: 6px 10px;
                color: #1C1C1E;
                font-size: 13px;
                font-weight: 500;
            }
            QSpinBox#appleSpinBox:hover {
                border-color: #007AFF;
            }
            QCheckBox#toggleSwitch {
                spacing: 0px;
            }
            QCheckBox#toggleSwitch::indicator {
                width: 44px;
                height: 26px;
                border-radius: 13px;
                background: #D1D1D6;
                border: none;
            }
            QCheckBox#toggleSwitch::indicator:checked {
                background: #34C759;
            }

            /* ── Scroll area ─────────────────────────────────────── */
            QScrollArea#studioScroll, QWidget#studioContent {
                background: #F2F2F7;
                border: none;
            }

            /* ── Footer ──────────────────────────────────────────── */
            QFrame#studioFooter {
                background: #FFFFFF;
                border-top: 1px solid #E5E5EA;
            }
            QPushButton#ghostButton {
                background: transparent;
                border: none;
                color: #8E8E93;
                font-size: 13px;
                font-weight: 500;
                padding: 8px 4px;
            }
            QPushButton#ghostButton:hover {
                color: #007AFF;
            }
            QPushButton#actionButton {
                background: #E5E5EA;
                color: #1C1C1E;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#actionButton:hover {
                background: #D1D1D6;
            }
            QPushButton#warningButton {
                background: transparent;
                border: 1px solid #FF9500;
                color: #FF9500;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#warningButton:hover {
                background: rgba(255, 149, 0, 0.10);
            }
            QPushButton#accentButton {
                background: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#accentButton:hover {
                background: #0071E3;
            }
            QPushButton#accentButton:pressed {
                background: #0062C8;
            }
        """)
