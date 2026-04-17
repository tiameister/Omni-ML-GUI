import os
import pandas as pd
from typing import Tuple
import re
from pathlib import Path
from exceptions import DataLoadError
from utils.logger import get_logger
from utils.text import normalize_text


LOGGER = get_logger(__name__)


EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".xlsb"}
SUPPORTED_DATASET_EXTENSIONS = tuple(sorted({".csv", *EXCEL_EXTENSIONS}))


def _coerce_numeric_like_object_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Convert object columns to numeric only when values are predominantly numeric-like.

    This avoids corrupting true categorical text columns while still handling
    comma decimals (e.g., "1,25").
    """
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].astype("string")
        non_empty = s.notna() & (s.str.strip() != "")
        if int(non_empty.sum()) == 0:
            continue

        candidate = s.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(candidate, errors="coerce")
        ratio = float(numeric[non_empty].notna().mean())
        if ratio >= threshold:
            df[col] = numeric
    return df


def _drop_high_cardinality_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_candidates = [c for c in df.columns if normalize_text(c) == "province"]
    if drop_candidates:
        LOGGER.info("Dropping high-cardinality geography columns: %s", drop_candidates)
    return df.drop(columns=drop_candidates, errors="ignore")


def _read_excel_safely(path: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = pd.read_excel(path, sheet_name=0)
        df = _coerce_numeric_like_object_columns(df)
        df = _drop_high_cardinality_geo_columns(df)
        LOGGER.info("Loaded Excel dataset '%s' shape=%s", path, tuple(df.shape))
        return df, "excel"
    except FileNotFoundError as exc:
        raise DataLoadError(f"Dataset not found: {path}") from exc
    except ImportError as exc:
        raise DataLoadError(
            "Excel support requires optional dependencies. Install 'openpyxl' for .xlsx/.xlsm, "
            "'xlrd' for legacy .xls, and 'pyxlsb' for .xlsb files."
        ) from exc
    except Exception as exc:
        raise DataLoadError(f"Could not parse Excel dataset '{path}': {exc}") from exc


def _read_csv_with_sniffing(path: str) -> Tuple[pd.DataFrame, str]:
    attempts: list[str] = []
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8", "utf-8-sig", "cp1254", "latin-1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] < 2:
                    attempts.append(f"sep={sep} enc={enc} -> too few columns ({df.shape[1]})")
                    continue

                df = _coerce_numeric_like_object_columns(df)
                df = _drop_high_cardinality_geo_columns(df)
                LOGGER.info("Loaded CSV '%s' with sep='%s' encoding='%s' shape=%s", path, sep, enc, tuple(df.shape))
                return df, sep
            except FileNotFoundError as exc:
                raise DataLoadError(f"Dataset not found: {path}") from exc
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                attempts.append(f"sep={sep} enc={enc} -> {type(exc).__name__}: {exc}")
                continue

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df = _coerce_numeric_like_object_columns(df)
        df = _drop_high_cardinality_geo_columns(df)
        LOGGER.warning("Loaded CSV '%s' using fallback auto separator; prior attempts failed", path)
        return df, "auto"
    except Exception as exc:
        details = " | ".join(attempts[-8:]) if attempts else "no parser attempts recorded"
        raise DataLoadError(f"Could not parse CSV '{path}'. Attempts: {details}") from exc


def read_dataset_safely(path: str) -> Tuple[pd.DataFrame, str]:
    """Load CSV or Excel datasets with robust parsing and normalization."""
    if not path:
        raise DataLoadError("Dataset path is empty")

    suffix = Path(path).suffix.lower()
    if suffix in EXCEL_EXTENSIONS:
        return _read_excel_safely(path)
    return _read_csv_with_sniffing(path)


def read_csv_safely(path: str) -> Tuple[pd.DataFrame, str]:
    """Backward-compatible alias; now supports Excel datasets too."""
    return read_dataset_safely(path)

def detect_cols(df: pd.DataFrame):
    nm = {c: normalize_text(c) for c in df.columns}

    sensitive_norms = {"irk", "irki", "etnik", "etnisite", "ethnicity", "race"}
    drop_norms = {"il","ilce","il_ilce","ililce","il_ilce_adi","il_ilcesi"} | sensitive_norms
    drop_cols = [c for c, n in nm.items() if n in drop_norms]
    drop_cols += [c for c, n in nm.items() if re.match(r"faal", n)]
    drop_cols += [c for c, n in nm.items() if n == "id"]
    drop_cols += [c for c, n in nm.items() if re.match(r"(irk|etnik|etnisite|ethnicity|race)(_|$)", n)]

    # Detect target (happiness) by Turkish or English column names or environment variable
    target_env = os.environ.get("TARGET_COL")
    if target_env and target_env in df.columns:
        target = target_env
    else:
        target = next((c for c, n in nm.items() if n in {"mutluluk","mutluluktoplam","mutluluk_toplam",
                                                          "happiness","happinessscore","happiness_score"}), None)
        
    if target is None:
        # Instead of crashing, just return the first numeric column as a fallback or None if we really want to be dynamic.
        # But to not break old scripts, we can log a warning and let the user select it via GUI.
        # However, for completely unknown datasets in CLI, we could just say target is None.
        LOGGER.warning("Hedef sütunomatik bulunamadı: (mutluluk vb.). GUI üzerinden seçmeniz gerekebilir.")
        target = None
    
    # Detect bully (total bullying score) by Turkish or English column names
    bully = next((c for c, n in nm.items() if n in {
        "zorbalik", "zorbaliktoplam", "zorbalik_toplam",
        "totalbullyingscore","total_bullying_score","bullying_score"
    }), None)

    bully_subs = [c for c, n in nm.items() if n in {"fizikselz","sozelz","duygusalz","dijitalz"}]

    m_items = [c for c, n in nm.items() if re.fullmatch(r"m[0-9]+", n)]
    z_items = [c for c, n in nm.items() if re.fullmatch(r"z[0-9]+", n)]
    bully_subs = [c for c, n in nm.items() if n in {"fizikselz", "fizkselz", "sozelz", "duygusalz", "dijitalz"}]

    LOGGER.debug(
        "Detected target=%s bully=%s m_items=%d z_items=%d bully_subs=%d drop_cols=%d",
        target,
        bully,
        len(m_items),
        len(z_items),
        len(bully_subs),
        len(drop_cols),
    )

    return target, bully, m_items, z_items, bully_subs, drop_cols
