from typing import List, Tuple

import pandas as pd
from utils.text import normalize_text as _norm


# Desired groups (names as they appear conceptually). We will resolve them to actual columns present.
NUMERIC_COLS = [
    'Age',
    'TV Time (Daily Hours)',
    'Mobile Phone (Daily Hours)',
    'Total Bullying Score',
    'Household Size',
    'Number of Siblings',
]

ORDINAL_COLS = [
    'Household Income',
    "Mother's Education",  # ascii apostrophe variant
    "Mother’s Education",  # curly apostrophe variant
    "Father's Education",
    "Father’s Education",
    'Birth Order',
    'Grade/Class',
    'Teacher Intervention',
    'Reporting Bullying to Family',
    "Parents' Marital Status (Together/Separated)",
    'Parents’ Marital Status (Together/Separated)',
]

BINARY_COLS = [
    'Gender',
    'Extracurricular Activity Participation',
]


def resolve_column_groups(df_columns: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Resolve (numeric, ordinal, binary, other_categorical) by matching intended names to actual
    DataFrame columns via normalized comparison. Columns not found fall back to 'other_categorical'
    if they are in df but not in numeric/ordinal/binary.
    """
    cols = list(df_columns)
    norm_map = {c: _norm(c) for c in cols}
    inv_map = {}
    for c, n in norm_map.items():
        inv_map.setdefault(n, []).append(c)

    def _resolve(desired: List[str]) -> List[str]:
        out = []
        for d in desired:
            nd = _norm(d)
            if nd in inv_map:
                # pick the first matching actual column
                out.append(inv_map[nd][0])
        return out

    num = _resolve(NUMERIC_COLS)
    ord_ = _resolve(ORDINAL_COLS)
    bin_ = _resolve(BINARY_COLS)
    taken = set(num) | set(ord_) | set(bin_)
    other = [c for c in cols if c not in taken]
    return num, ord_, bin_, other
