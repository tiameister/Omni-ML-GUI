import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QSettings

LOGGER = logging.getLogger(__name__)

class ThemeManager:
    """Enterprise-grade Theme Manager for Dark/Light mode integration."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance.current_theme = "system"
        return cls._instance

    def _is_system_dark_mode(self, app: QApplication) -> bool:
        """Kullanıcının işletim sisteminin karanlık modda olup olmadığını algılar."""
        try:
            # Check window background lightness
            palette = app.palette()
            bg_color = palette.color(QPalette.ColorRole.Window)
            return bg_color.lightnessF() < 0.5
        except Exception as e:
            LOGGER.warning(f"Sistem tema algılaması başarısız: {e}")
            return False

    def load_theme(self, app: QApplication, theme_name: str):
        """Temayı (QSS) bellekten veya diskten yükleyip uygular."""
        self.current_theme = theme_name
        settings = QSettings()
        settings.setValue("ui/theme", theme_name)
        
        is_dark = False
        if theme_name == "dark":
            is_dark = True
        elif theme_name == "system":
            is_dark = self._is_system_dark_mode(app)
            
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # 1. Native Fusion Palette ayarları (özellikle QMessageBox ve menüler MacOS ile tam uyum sağlasın diye)
        app.setStyle("Fusion")
        
        # 2. QSS yükleme
        qss_filename = "dark_style.qss" if is_dark else "style.qss"
        qss_path = os.path.join(base_dir, "style", qss_filename)
        
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                qss_text = f.read()
                
                # Dinamik token haritası kullanımı
                if is_dark:
                    token_map = {
                        "{{SURFACE_BG_TOP}}": "#1E1E1E",
                        "{{SURFACE_BG_MID}}": "#202020",
                        "{{SURFACE_BG_END}}": "#252526",
                        "{{BORDER_SOFT}}": "#3A3A3D",
                        "{{BORDER_PANEL}}": "#3A3A3D",
                        "{{COLOR_PRIMARY}}": "#0A84FF",
                        "{{COLOR_ACCENT}}": "#5E5CE6",
                        "{{TEXT_TITLE}}": "#FFFFFF",
                        "{{TEXT_SUBTITLE}}": "#EAEB5E",
                        "{{TEXT_ACCENT}}": "#0A84FF",
                        "{{RADIUS_PANEL}}": "14",
                        "{{RADIUS_CARD}}": "10",
                        "{{RADIUS_WORKFLOW}}": "12",
                    }
                else:
                    token_map = {
                        "{{SURFACE_BG_TOP}}": "#EEF3F8",
                        "{{SURFACE_BG_MID}}": "#E9F0F7",
                        "{{SURFACE_BG_END}}": "#DDE8F2",
                        "{{BORDER_SOFT}}": "#C8D8EA",
                        "{{BORDER_PANEL}}": "#D4DFEC",
                        "{{COLOR_PRIMARY}}": "#0068B3",
                        "{{COLOR_ACCENT}}": "#3D8ED1",
                        "{{TEXT_TITLE}}": "#102333",
                        "{{TEXT_SUBTITLE}}": "#496279",
                        "{{TEXT_ACCENT}}": "#1F4D73",
                        "{{RADIUS_PANEL}}": "14",
                        "{{RADIUS_CARD}}": "10",
                        "{{RADIUS_WORKFLOW}}": "12",
                    }
                    
                for token, value in token_map.items():
                    qss_text = qss_text.replace(token, value)
                    
                app.setStyleSheet(qss_text)
                
        except Exception as e:
            LOGGER.error(f"Tema {theme_name} yüklenirken hata: {e}")

theme_manager = ThemeManager()
