from __future__ import annotations

EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".xlsb"}
SUPPORTED_DATASET_EXTENSIONS = tuple(sorted({".csv", *EXCEL_EXTENSIONS}))
