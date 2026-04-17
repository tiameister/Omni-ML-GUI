import json
import logging
import locale
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

class LocalizationManager:
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
        
        self._load_fallback()
        self._detect_system_language()

    def _load_fallback(self):
        fallback_path = self.locales_dir / "en.json"
        if fallback_path.exists():
            try:
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    self.fallback_translations = _flatten_dict(json.load(f))
            except Exception as e:
                logger.error(f"Fallback dili yüklenemedi: {e}", exc_info=True)

    def _detect_system_language(self):
        try:
            sys_lang, _ = locale.getdefaultlocale()
            if sys_lang:
                lang_prefix = str(sys_lang).split('_')[0].lower()
                if (self.locales_dir / f"{lang_prefix}.json").exists():
                    self.set_language(lang_prefix)
                    return
        except Exception as e:
            logger.warning(f"Sistem dili tespit edilemedi: {e}", exc_info=True)
            
        self.set_language('en')

    def set_language(self, lang_code: str) -> bool:
        if self.current_language == lang_code and self.translations:
            return True
            
        file_path = self.locales_dir / f"{lang_code}.json"
        if not file_path.exists():
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.translations = _flatten_dict(json.load(f))
            
            self.current_language = lang_code
            logger.info(f"Dil '{lang_code}' olarak değiştirildi.")
            self._notify_listeners()
            return True
        except Exception as e:
            logger.error(f"Dil dosyasi yuklenemedi: {file_path}", exc_info=True)
            return False

    def get_language(self) -> str:
        return self.current_language

    def get_supported_languages(self) -> list[str]:
        try:
            codes = [p.stem for p in self.locales_dir.glob("*.json") if p.is_file()]
            return sorted(set(codes))
        except Exception:
            return ["en"]

    def tr(self, key_path: str, default: Optional[str] = None, **kwargs) -> str:
        text = self.translations.get(key_path)
        
        if text is None:
            text = self.fallback_translations.get(key_path)
            if text is None:
                if default is not None:
                    text = default
                else:
                    return f"[{key_path}]"
        
        if kwargs:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
                
        return text

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
                logger.error(f"Dil güncelleme tetikleyicisinde hata: {e}", exc_info=True)

i18n = LocalizationManager()
tr = i18n.tr