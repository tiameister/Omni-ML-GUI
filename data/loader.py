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
    Optimized: Sample-based strict detection (O(1) sampling) to avoid 
    O(N) full-string iterations over purely categorical columns.
    """
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) == 0:
        return df

    for col in object_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        
        # Optimize: Test a small sample first (100 rows)
        sample = col_data.head(100).astype(str).str.strip()
        sample_candidate = sample.str.replace(",", ".", regex=False)
        sample_num = pd.to_numeric(sample_candidate, errors="coerce")
        
        # If the sample has less than 50% numeric, it's overwhelmingly a text column, skip full conversion
        if sample_num.notna().mean() < 0.5:
            continue
            
        # Full conversion explicitly over C backend if pass
        s_full = col_data.astype(str).str.strip()
        # Drop empties from denominator
        s_full = s_full[s_full != ""]
        if s_full.empty:
            continue
            
        candidate = s_full.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(candidate, errors="coerce")
        
        ratio = numeric.notna().mean()
        if ratio >= threshold:
            # Map back to original dataframe size
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Shift ALL remaining completely text-based categorical object columns to the PyArrow C++ Backend (if available)
    try:
        import pyarrow
        # Converting remaining Object strings to PyArrow String arrays drops ~70% of memory requirements
        # and vastly speeds up downstream string hashing
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("string[pyarrow]")
    except ImportError:
        pass
            
    return df



def _optimize_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    '''Lossless downcasting of numeric columns to save memory and speed up cache hits.'''
    for col in df.select_dtypes(include=["float"]):
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["integer"]):
        df[col] = pd.to_numeric(df[col], downcast="integer")
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
        df = _optimize_numeric_types(df)
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
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="c" if len(sep)==1 else "python", low_memory=False)
                if df.shape[1] < 2:
                    attempts.append(f"sep={sep} enc={enc} -> too few columns ({df.shape[1]})")
                    continue

                df = _coerce_numeric_like_object_columns(df)
                df = _optimize_numeric_types(df)
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
        df = _optimize_numeric_types(df)
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
    # Compile regexes once
    re_faal = re.compile(r"faal")
    re_sensitive = re.compile(r"(irk|etnik|etnisite|ethnicity|race)(_|$)")
    
    # Collect drop columns efficiently with O(N) single-pass lookup
    drop_cols = []
    for c, n in nm.items():
        if n in drop_norms or n == "id" or re_faal.match(n) or re_sensitive.match(n):
            drop_cols.append(c)

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
