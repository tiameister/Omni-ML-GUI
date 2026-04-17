import json
import logging
import locale
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalizationManager:
    """
    Enterprise-grade i18n yöneticisi (Singleton).
    Tıpkı VS Code, IntelliJ gibi yazılımların çalıştığı standartlarda;
    - Metinleri koddan ayırıp dışarıdan (JSON) okur (Lazy Loading).
    - Namespace mantığını destekler ("ui.buttons.ok" gibi . (nokta) notasyonu).
    - OS Sistem dilini algılama özelliği mevcuttur.
    - İngilizceyi her zaman çekirdek fallback (geri dönüş) dili olarak tutar.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalizationManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.current_language = 'en'
        self.locales_dir = Path(__file__).parent.parent / "locales"
        
        self.translations = {}
        self.fallback_translations = {} 
        self._callbacks = []
        
        # Sistemi ayağa kaldırırken varsayılan İngilizceyi Fallback olarak yükle
        self._load_fallback()
        
        # Kullanıcının varsayılan işletim sistemi dilini tespit et
        self._detect_system_language()

    def _load_fallback(self):
        """Öncelikli olarak sistemi kurtaracak olan temel(İngilizce) dili yükler."""
        fallback_path = self.locales_dir / "en.json"
        if fallback_path.exists():
            try:
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    self.fallback_translations = json.load(f)
            except Exception as e:
                logger.error(f"Fallback dili yüklenemedi: {e}")

    def _detect_system_language(self):
        """İşletim sisteminin dil kodunu okuyup uygun olanı yükler (Örn: 'tr_TR' -> 'tr')"""
        try:
            sys_lang, _ = locale.getdefaultlocale()
            if sys_lang:
                lang_prefix = sys_lang.split('_')[0].lower() # 'tr_TR' -> 'tr'
                if (self.locales_dir / f"{lang_prefix}.json").exists():
                    self.set_language(lang_prefix)
                    return
        except Exception as e:
            logger.warning(f"Sistem dili tespit edilemedi: {e}")
            
        # Eğer özel bir tespit yapılamazsa İngilizce'ye geç
        self.set_language('en')

    def set_language(self, lang_code: str) -> bool:
        """Sadece çağırılan dili okuyarak (Lazy Load) belleği gereksiz yere şişirmeyi engeller."""
        file_path = self.locales_dir / f"{lang_code}.json"
        
        if not file_path.exists():
            logger.error(f"Dil desteklenmiyor veya dosya eksik: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            
            self.current_language = lang_code
            logger.info(f"Dil '{lang_code}' olarak değiştirildi.")
            self._notify_listeners()
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Dil dosyasında sözdizimi(syntax) hatası: {file_path} - {e}")
            return False

    def get_language(self) -> str:
        return self.current_language

    def get_supported_languages(self) -> list[str]:
        """Return available language codes from the locales directory."""
        try:
            codes = [p.stem for p in self.locales_dir.glob("*.json") if p.is_file()]
            return sorted(set(codes))
        except Exception:
            return ["en"]

    def tr(self, key_path: str, default: str | None = None, **kwargs) -> str:
        """
        Nokta notasyonlu ('ui.buttons.ok' gibi) gelen anahtarı okur.
        1- Seçili dilde (örn. TR) arar.
        2- Bulamazsa, Fallback'te (EN) arar.
        3- Orada da yoksa geliştiricinin fark etmesi için anahtarın kendisini raw "[key_path]" olarak döndürür.
        """
        text = self._get_nested_value(key_path, self.translations)
        
        # Seçili dilde bulunamadıysa fallback(İngilizce) dosyasına bak
        if text is None:
            text = self._get_nested_value(key_path, self.fallback_translations)
            if text is None:
                logger.warning(f"Eksik çeviri anahtarı: '{key_path}'")
                if default is not None:
                    text = default
                else:
                    return f"[{key_path}]"
        
        # Eğer içerikte '{param}' gibi dinamik değerler varsa, yerleştir
        if kwargs and isinstance(text, str):
            try:
                return text.format(**kwargs)
            except KeyError as e:
                logger.error(f"'{key_path}' için format argümanı eksik: {e}")
                return text
                
        return text

    def _get_nested_value(self, key_path: str, data: dict):
        """'ui.buttons.ok' gibi içiçe geçmiş JSON anahtarlarını ayrıştırır."""
        keys = key_path.split('.')
        val = data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val

    def add_listener(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_listener(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_listeners(self):
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Dil güncelleme tetikleyicisinde hata: {e}")

# Uygulama genelinde modülü başlatan singleton instance
i18n = LocalizationManager()
tr = i18n.tr
