import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QSpinBox, QCheckBox, QFrame, QScrollArea, QWidget, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt
from interface.widgets.apple_helpers import create_apple_settings_row
from utils.localization import tr

class FeatureEngineeringStudioDialog(QDialog):
    def __init__(self, current_config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dialogs.fe_studio.title", default="Feature Engineering Studio"))
        self.setMinimumSize(600, 700)
        self.current_config = current_config.copy()
        self.exported_path = None
        self.export_requested = False
        
        self._setup_ui()
        self._apply_config()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setObjectName("studioHeader")
        header_lay = QVBoxLayout(header)
        header_lay.setContentsMargins(24, 24, 24, 16)
        title = QLabel(tr("dialogs.fe_studio.header", default="Feature Engineering Studio"))
        title.setStyleSheet("font-size: 20pt; font-weight: 700; color: #1C1C1E; letter-spacing: -0.5px;")
        subtitle = QLabel(tr("dialogs.fe_studio.subtitle", default="Configure data expansions and structural transformations. Modern ML pipelines manage these automatically to prevent data leakage."))
        subtitle.setStyleSheet("font-size: 12pt; color: #8E8E93;")
        subtitle.setWordWrap(True)
        header_lay.addWidget(title)
        header_lay.addWidget(subtitle)
        main_layout.addWidget(header)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: #F2F2F7;")
        
        container = QWidget()
        container.setStyleSheet("background: #F2F2F7;")
        self.lay = QVBoxLayout(container)
        self.lay.setContentsMargins(24, 16, 24, 24)
        self.lay.setSpacing(24)
        
        self._build_distribution_card()
        self._build_outlier_card()
        self._build_discretization_card()
        self._build_expansion_card()
        
        self.lay.addStretch(1)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Footer / Buttons
        footer = QFrame()
        footer.setStyleSheet("background: #FFFFFF; border-top: 1px solid #E5E5EA;")
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(24, 16, 24, 16)
        
        self.btn_export = QPushButton(tr("dialogs.fe_studio.export", default="Generate Dataset"))
        self.btn_export.setObjectName("outlineButton")
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_export.setToolTip("Create a physical CSV from these rules and switch to it immediately. Careful: Data Leakage risk.")

        self.btn_cancel = QPushButton(tr("common.cancel", default="Cancel"))
        self.btn_cancel.setObjectName("actionButton")
        self.btn_cancel.setMinimumWidth(100)
        self.btn_cancel.clicked.connect(self.reject)
        
        self.btn_apply = QPushButton(tr("dialogs.fe_studio.apply_pipeline", default="Dynamic Pipeline (Safe)"))
        self.btn_apply.setObjectName("accentButton")
        self.btn_apply.setMinimumWidth(160)
        self.btn_apply.clicked.connect(self.accept)
        self.btn_apply.setToolTip("Schedule these rules to run safely inside the cross-validation folds.")
        
        footer_lay.addStretch(1)
        footer_lay.addWidget(self.btn_cancel)
        footer_lay.addWidget(self.btn_export)
        footer_lay.addWidget(self.btn_apply)
        main_layout.addWidget(footer)

    def _build_distribution_card(self):
        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        
        self.transform_combo = QComboBox()
        self.transform_combo.addItem("None", "none")
        self.transform_combo.addItem("Yeo-Johnson (Handles 0 & Neg)", "yeo-johnson")
        self.transform_combo.setMinimumWidth(220)
        self.transform_combo.setObjectName("appleCombo")
        
        row1, _, _ = create_apple_settings_row(
            self.transform_combo,
            title_text="Power Transformation",
            subtitle_text="Forces extreme skewed values toward a Gaussian (normal) distribution.",
            show_bottom_line=True
        )
        
        self.missing_toggle = QCheckBox()
        self.missing_toggle.setObjectName("toggleSwitch")
        row2, _, _ = create_apple_settings_row(
            self.missing_toggle,
            title_text="Missing Indicators",
            subtitle_text="Creates binary flags (0/1) tracing which rows had originally missing records.",
            show_bottom_line=False
        )
        
        card_lay.addWidget(row1)
        card_lay.addWidget(row2)
        
        header = QLabel("DISTRIBUTION & IMPUTATION")
        header.setObjectName("cardHeaderLabel")
        self.lay.addWidget(header)
        self.lay.addWidget(card)
        self.lay.setSpacing(8) # tighten spacing between label and card

    def _build_outlier_card(self):
        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        
        self.outlier_combo = QComboBox()
        self.outlier_combo.addItem("None", "none")
        self.outlier_combo.addItem("Winsorize (1st - 99th Percentile)", "winsorize_1_99")
        self.outlier_combo.addItem("Winsorize (5th - 95th Percentile)", "winsorize_5_95")
        self.outlier_combo.setMinimumWidth(220)
        self.outlier_combo.setObjectName("appleCombo")
        
        row1, _, _ = create_apple_settings_row(
            self.outlier_combo,
            title_text="Outlier Clipping (Winsorization)",
            subtitle_text="Caps extreme anomalies at specific bounds to protect non-robust models from noise.",
            show_bottom_line=False
        )
        card_lay.addWidget(row1)
        
        header = QLabel("ROBUSTNESS")
        header.setObjectName("cardHeaderLabel")
        self.lay.addSpacing(16)
        self.lay.addWidget(header)
        self.lay.addWidget(card)

    def _build_discretization_card(self):
        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        
        self.bin_combo = QComboBox()
        self.bin_combo.addItem("None", "none")
        self.bin_combo.addItem("Quantile (Equal Frequency)", "quantile")
        self.bin_combo.addItem("Uniform (Equal Width)", "uniform")
        self.bin_combo.addItem("K-Means (Clustering)", "kmeans")
        self.bin_combo.setMinimumWidth(220)
        self.bin_combo.setObjectName("appleCombo")
        
        self.bin_combo.currentIndexChanged.connect(self._toggle_bin_states)
        
        row1, _, _ = create_apple_settings_row(
            self.bin_combo,
            title_text="Discretization (Binning)",
            subtitle_text="Modern capability to slice continuous features into discrete categorical buckets.",
            show_bottom_line=True
        )
        card_lay.addWidget(row1)
        
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(2, 20)
        self.bin_spin.setObjectName("appleSpinBox")
        self.bin_spin.setMinimumWidth(80)
        
        self.row_bin, _, _ = create_apple_settings_row(
            self.bin_spin,
            title_text="Number of Bins",
            subtitle_text="How many discrete segments to divide each continuous feature into.",
            show_bottom_line=False
        )
        card_lay.addWidget(self.row_bin)
        
        header = QLabel("MACRO SHAPING")
        header.setObjectName("cardHeaderLabel")
        self.lay.addSpacing(16)
        self.lay.addWidget(header)
        self.lay.addWidget(card)

    def _build_expansion_card(self):
        card = QFrame()
        card.setObjectName("appleCard")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)
        
        self.poly_toggle = QCheckBox()
        self.poly_toggle.setObjectName("toggleSwitch")
        self.poly_toggle.toggled.connect(self._toggle_poly_states)
        
        row1, _, _ = create_apple_settings_row(
            self.poly_toggle,
            title_text="Polynomial Features",
            subtitle_text="Forces models to learn interaction terms (e.g. A*B, A^2) by providing them purely.",
            show_bottom_line=True
        )
        card_lay.addWidget(row1)
        
        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setRange(2, 4)
        self.poly_degree_spin.setObjectName("appleSpinBox")
        self.poly_degree_spin.setMinimumWidth(80)
        
        self.row2, _, _ = create_apple_settings_row(
            self.poly_degree_spin,
            title_text="Polynomial Degree",
            subtitle_text="Maximum degree of interactions. Extremely heavy on memory when high.",
            show_bottom_line=True
        )
        card_lay.addWidget(self.row2)
        
        self.poly_max_spin = QSpinBox()
        self.poly_max_spin.setRange(10, 500)
        self.poly_max_spin.setSingleStep(10)
        self.poly_max_spin.setObjectName("appleSpinBox")
        self.poly_max_spin.setMinimumWidth(80)
        
        self.row3, _, _ = create_apple_settings_row(
            self.poly_max_spin,
            title_text="Max Features Retained",
            subtitle_text="Keeps only the top variables based on highest variance to prevent RAM crashes.",
            show_bottom_line=False
        )
        card_lay.addWidget(self.row3)
        
        header = QLabel("DIMENSIONALITY EXPANSION")
        header.setObjectName("cardHeaderLabel")
        self.lay.addSpacing(16)
        self.lay.addWidget(header)
        self.lay.addWidget(card)

    def _toggle_poly_states(self, checked):
        self.poly_degree_spin.setEnabled(checked)
        self.poly_max_spin.setEnabled(checked)
        op = 1.0 if checked else 0.4
        for w in [self.row2, self.row3]:
            w.setStyleSheet(f"QFrame {{ opacity: {op}; }}")

    def _toggle_bin_states(self, index):
        checked = (self.bin_combo.currentData() != "none")
        self.bin_spin.setEnabled(checked)
        op = 1.0 if checked else 0.4
        self.row_bin.setStyleSheet(f"QFrame {{ opacity: {op}; }}")

    def _apply_config(self):
        c = self.current_config
        
        idx = self.transform_combo.findData(c.get("transform", "yeo-johnson"))
        if idx >= 0: self.transform_combo.setCurrentIndex(idx)
        self.missing_toggle.setChecked(c.get("missing_indicators", True))
        idx = self.outlier_combo.findData(c.get("outliers", "winsorize_1_99"))
        if idx >= 0: self.outlier_combo.setCurrentIndex(idx)
        
        idx = self.bin_combo.findData(c.get("binning", "none"))
        if idx >= 0: self.bin_combo.setCurrentIndex(idx)
        self.bin_spin.setValue(c.get("n_bins", 5))
        
        poly_on = c.get("poly_features", False)
        self.poly_toggle.setChecked(poly_on)
        self.poly_degree_spin.setValue(c.get("poly_degree", 2))
        self.poly_max_spin.setValue(c.get("poly_max", 50))
        
        self._toggle_poly_states(poly_on)
        self._toggle_bin_states(0)

    def get_config(self) -> dict:
        return {
            "transform": self.transform_combo.currentData(),
            "missing_indicators": self.missing_toggle.isChecked(),
            "outliers": self.outlier_combo.currentData(),
            "binning": self.bin_combo.currentData(),
            "n_bins": self.bin_spin.value(),
            "poly_features": self.poly_toggle.isChecked(),
            "poly_degree": self.poly_degree_spin.value(),
            "poly_max": self.poly_max_spin.value(),
        }

    def _on_export_clicked(self):
        # We check leakages: Outliers, Yeo-Johnson, K-Means binning, Quantile binning.
        c = self.get_config()
        leaks = []
        if c["outliers"] != "none":
            leaks.append(f"• Outlier Clipping ({c['outliers']}) computes global min/max percentiles.")
        if c["transform"] != "none":
            leaks.append(f"• Power Transform ({c['transform']}) fits distribution constants globally.")
        if c["binning"] in ["quantile", "kmeans"]:
            leaks.append(f"• Discretization ({c['binning']}) calculates boundaries across the full dataset.")
        
        if leaks:
            warn_msg = (
                "Data Leakage Risk Detected!\n\n"
                "You are attempting to generate a static CSV bypassing the cross-validation boundaries.\n\n"
                "The following selected operations analyze the statistical distribution of your ENTIRE dataset before feeding it to models:\n"
                + "\n".join(leaks) +
                "\n\nIn rigorous/modern ML (e.g. Kaggle/Publications), this causes validation data to 'leak' into the training process, artificially inflating your scores. To prevent this, cancel and choose 'Dynamic Pipeline'.\n\n"
                "Do you still want to generate the manipulated static dataset anyway?"
            )
            # Actually, QMessageBox is fine
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.warning(self, "Data Leakage Warning", warn_msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.export_requested = True
        self.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog { background: #F2F2F7; }
            QFrame#studioHeader { background: #FFFFFF; border-bottom: 1px solid #E5E5EA; }
            QLabel#cardHeaderLabel {
                font-size: 11px;
                color: #6E6E73;
                padding-left: 14px;
                text-transform: uppercase;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QComboBox#appleCombo {
                background-color: #E5E5EA; border: none; border-radius: 6px; padding: 6px 12px; color: #1C1C1E; font-weight: 400; font-size: 13px;
            }
            QSpinBox#appleSpinBox {
                background-color: #F2F2F7; border: 1px solid #D1D1D6; border-radius: 6px; padding: 6px 10px; color: #1C1C1E; font-weight: 500; font-size: 13px;
                qproperty-alignment: 'AlignHCenter';
            }
            QPushButton#outlineButton {
                background-color: transparent; border: 1px solid #007AFF; color: #007AFF; border-radius: 8px; padding: 8px 14px; font-weight: 600;
            }
            QPushButton#outlineButton:hover {
                background-color: rgba(0, 122, 255, 0.1);
            }
            QPushButton#actionButton {
                background-color: #E5E5EA; color: #1C1C1E; border-radius: 8px; padding: 8px 14px; font-weight: 600;
            }
            QPushButton#accentButton {
                background-color: #007AFF; color: #FFFFFF; border-radius: 8px; padding: 8px 14px; font-weight: 600;
            }
        """)
