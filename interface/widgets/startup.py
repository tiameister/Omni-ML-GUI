import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame, QMessageBox, QCheckBox,
    QComboBox
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QSettings
from utils.localization import i18n, tr
from utils.logger import get_logger

LOGGER = get_logger(__name__)

class StartupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("startupDialog")
        self.resize(620, 420)
        self.setMinimumSize(500, 340)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog#startupDialog {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 1,
                    stop: 0 #F8FBFF,
                    stop: 1 #EDF4FB
                );
                border: 1px solid #CFDDEC;
                border-radius: 14px;
            }
            QLabel#startupTitle {
                color: #0F2E46;
                font-weight: 700;
            }
            QLabel#startupSubtitle {
                color: #4C6279;
            }
            QLabel#startupQuickStart {
                color: #2C506E;
                background: #EAF4FF;
                border: 1px solid #C8DDED;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QCheckBox {
                color: #44596D;
            }
            """
        )
        self.settings = QSettings()
        self._listener = self._apply_translations
        i18n.add_listener(self._listener)
        self.destroyed.connect(lambda _obj=None: i18n.remove_listener(self._listener))

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        # Resolve project root and logo path
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        logo_path = os.path.join(proj_root, "images", "fau.png")
        pix = QPixmap(logo_path)
        logo = QLabel(self)
        if not pix.isNull():
            pass # no logo)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        # Title / Name / Institution
        self.title_label = QLabel(self)
        self.title_label.setObjectName("startupTitle")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setWeight(QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(self)
        self.subtitle_label.setObjectName("startupSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("color:#5B6778;")
        layout.addWidget(self.subtitle_label)

        # Separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Welcome message and quick actions row
        self.welcome_label = QLabel()
        self.welcome_label.setObjectName("startupQuickStart")
        self.welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.welcome_label.setStyleSheet("color:#475569;")
        layout.addWidget(self.welcome_label)

        lang_row = QHBoxLayout()
        lang_row.addStretch()
        self.language_label = QLabel(self)
        self.language_combo = QComboBox(self)
        self.language_combo.setMinimumWidth(150)
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        lang_row.addWidget(self.language_label)
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch()
        layout.addLayout(lang_row)

        actions = QHBoxLayout()
        self.btn_start = QPushButton()
        self.btn_start.setObjectName("startButton")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.setDefault(True)
        self.btn_about = QPushButton()
        self.btn_guide = QPushButton()
        actions.addStretch()
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_about)
        actions.addWidget(self.btn_guide)
        actions.addStretch()
        layout.addLayout(actions)

        self.show_on_start = QCheckBox()
        self.show_on_start.setChecked(self._get_show_on_startup())
        layout.addWidget(self.show_on_start, 0, Qt.AlignmentFlag.AlignCenter)

        # Wire actions
        def _start():
            self._persist_preferences()
            self.accept()
        self.btn_start.clicked.connect(_start)

        def _about():
            try:
                from interface.widgets.dialogs import AboutDialog
                dlg = AboutDialog(self)
                dlg.exec()
            except Exception:
                LOGGER.exception("About dialog failed")
        self.btn_about.clicked.connect(_about)

        def _guide():
            try:
                from PyQt6.QtCore import QUrl
                from PyQt6.QtGui import QDesktopServices
                proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
                pdf_path = os.path.join(proj_root, 'info.pdf')
                if os.path.exists(pdf_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
                else:
                    QMessageBox.information(
                        self,
                        tr("startup.guide_title", default="User Guide"),
                        tr("startup.guide_missing", default="No local guide file was found."),
                    )
            except Exception:
                LOGGER.exception("Guide open failed")
        self.btn_guide.clicked.connect(_guide)

        if not os.path.exists(os.path.join(proj_root, 'info.pdf')):
            self.btn_guide.setEnabled(False)
            self.btn_guide.setToolTip(tr("startup.guide_not_found_tip", default="Local guide not found."))

        # Move version label to bottom
        version_text = "1.0.0"
        try:
            import config as _cfg
            version_text = str(getattr(_cfg, 'VERSION', version_text))
        except Exception:
            LOGGER.exception("Failed to read VERSION from config")
        self.version_label = QLabel(f"Version {version_text}", self)
        version_font = QFont()
        version_font.setPointSize(8)
        self.version_label.setFont(version_font)
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.version_label.setStyleSheet("color: gray;")
        layout.addWidget(self.version_label)

        self._restore_saved_language()
        self._apply_translations()

    def _restore_saved_language(self):
        raw = str(self.settings.value("ui/language", "")).strip().lower()
        if raw in i18n.get_supported_languages() and raw != i18n.get_language():
            i18n.set_language(raw)

    def _on_language_selected(self, *_args):
        code = self.language_combo.currentData()
        if not code:
            return
        code = str(code)
        if code != i18n.get_language():
            i18n.set_language(code)
        self.settings.setValue("ui/language", code)

    def _apply_translations(self):
        self.setWindowTitle(tr("startup.window_title", default="Welcome"))
        self.title_label.setText(tr("app.title", default="Machine Learning Trainer"))
        self.subtitle_label.setText(
            tr(
                "startup.subtitle",
                default="Build, train, and explain regression models with a guided workflow.",
            )
        )
        self.welcome_label.setText(
            tr("startup.quick_start", default="Quick start: Load dataset -> Select variables -> Choose models -> Start training")
        )
        self.language_label.setText(tr("menu.language", default="Language"))

        current = i18n.get_language()
        self.language_combo.blockSignals(True)
        try:
            self.language_combo.clear()
            self.language_combo.addItem(tr("menu.lang_en", default="English"), "en")
            self.language_combo.addItem(tr("menu.lang_tr", default="Turkish"), "tr")
            idx = self.language_combo.findData(current)
            self.language_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self.language_combo.blockSignals(False)

        self.btn_start.setText(tr("startup.start", default="Start"))
        self.btn_about.setText(tr("help.about", default="About"))
        self.btn_guide.setText(tr("help.open_guide", default="Open Guide"))
        self.show_on_start.setText(
            tr("startup.show_on_startup", default="Show this welcome screen on startup")
        )
        self.show_on_start.setToolTip(
            tr("startup.show_on_startup_tip", default="Disable to open the main window directly next time.")
        )

    def _get_show_on_startup(self) -> bool:
        raw = str(self.settings.value("ui/show_startup", "true")).strip().lower()
        return raw in ("1", "true", "yes", "on")

    def _persist_preferences(self) -> None:
        try:
            self.settings.setValue("ui/show_startup", bool(self.show_on_start.isChecked()))
            self.settings.sync()
        except Exception:
            LOGGER.exception("Failed to persist startup preferences")

    def reject(self):
        self._persist_preferences()
        super().reject()

