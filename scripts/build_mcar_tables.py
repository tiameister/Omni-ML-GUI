"""
Build Table R1.2 (Little's MCAR summary) for both raw and cleaned datasets.
Outputs:
- exports/manuscript_exports/table_R1_2_mcar.csv
- exports/manuscript_exports/table_R1_2_mcar.md (optional, simple markdown)
"""
from __future__ import annotations

import os
import csv
from typing import Dict, Any

from utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS = os.path.join(ROOT, 'exports', 'manuscript_exports')
os.makedirs(EXPORTS, exist_ok=True)

from data.loader import read_csv_safely
from scripts.missingness_report import little_mcar_test


def summarize_missingness(df) -> Dict[str, Any]:
    nrows = int(df.shape[0])
    total_cells = int(df.size)
    total_missing = int(df.isna().sum().sum())
    total_missing_pct = round(100.0 * total_missing / total_cells, 3) if total_cells else 0.0
    import numpy as np
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows_num_nan = int((~df[numeric_cols].notna().all(axis=1)).sum()) if numeric_cols else 0
    return {
        'rows': nrows,
        'total_cells': total_cells,
        'total_missing_pct': total_missing_pct,
        'rows_with_any_numeric_missingness': rows_num_nan,
    }


def parse_mcar_line(txt: str):
    # Expected formats:
    # "Little's MCAR: chi2=..., dof=..., p=..." or
    # "Little's MCAR (approx.): chi2=..., dof=..., p=..." or failure message
    if 'chi2=' in txt and 'dof=' in txt and 'p=' in txt:
        try:
            parts = {}
            for token in txt.replace(',', ' ').split():
                if token.startswith('chi2='):
                    parts['chi2'] = float(token.split('=')[1])
                elif token.startswith('dof='):
                    parts['df'] = int(float(token.split('=')[1]))
                elif token.startswith('p='):
                    parts['p_value'] = float(token.split('=')[1])
            return parts
        except Exception as e:
            LOGGER.exception("Failed parsing MCAR stats line")
    return {'chi2': 'N/A', 'df': 'N/A', 'p_value': 'N/A'}


def main():
    rows = []
    # Raw schema
    raw_path = os.path.join(ROOT, 'dataset', 'data.csv')
    if os.path.exists(raw_path):
        df_raw, _ = read_csv_safely(raw_path)
        mcar_txt = little_mcar_test(df_raw.copy()).strip()
        miss = summarize_missingness(df_raw)
        stats = parse_mcar_line(mcar_txt)
        rows.append({
            'dataset_scope': 'raw_schema',
            'chi2': stats['chi2'],
            'df': stats['df'],
            'p_value': stats['p_value'],
            'rows': miss['rows'],
            'total_cells': miss['total_cells'],
            'total_missing_pct': miss['total_missing_pct'],
            'notes': 'Raw dataset (dataset/data.csv)'
        })
    # Analytic cleaned
    cleaned_path = os.path.join(ROOT, 'dataset', 'data_cleaned.csv')
    if os.path.exists(cleaned_path):
        df_cln, _ = read_csv_safely(cleaned_path)
        mcar_txt = little_mcar_test(df_cln.copy()).strip()
        miss = summarize_missingness(df_cln)
        stats = parse_mcar_line(mcar_txt)
        # If no missingness, MCAR is effectively not applicable; keep stats as N/A
        if miss['total_missing_pct'] == 0.0:
            stats = {'chi2': 'N/A', 'df': 'N/A', 'p_value': 'N/A'}
        rows.append({
            'dataset_scope': 'analytic_cleaned',
            'chi2': stats['chi2'],
            'df': stats['df'],
            'p_value': stats['p_value'],
            'rows': miss['rows'],
            'total_cells': miss['total_cells'],
            'total_missing_pct': miss['total_missing_pct'],
            'notes': 'Cleaned dataset (dataset/data_cleaned.csv)'
        })

    # Write CSV
    out_csv = os.path.join(EXPORTS, 'table_R1_2_mcar.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset_scope','chi2','df','p_value','rows','total_cells','total_missing_pct','notes'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Optional Markdown convenience
    out_md = os.path.join(EXPORTS, 'table_R1_2_mcar.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('Table R1.2. Little\'s MCAR test on raw schema vs analytic (cleaned) dataset\n\n')
        f.write('| Dataset scope | chi² | df | p-value | Rows | Total cells | Total missing (%) | Notes |\n')
        f.write('|---|---:|---:|:--:|---:|---:|---:|---|\n')
        for r in rows:
            chi2 = r['chi2'] if isinstance(r['chi2'], str) else f"{r['chi2']:.3f}"
            dfv = r['df'] if isinstance(r['df'], str) else str(r['df'])
            pv  = r['p_value'] if isinstance(r['p_value'], str) else f"{r['p_value']:.6f}".rstrip('0').rstrip('.')
            f.write(f"| {r['dataset_scope']} | {chi2} | {dfv} | {pv} | {r['rows']} | {r['total_cells']:,} | {r['total_missing_pct']} | {r['notes']} |\n")

    print('[OK] Wrote:', out_csv)
    print('[OK] Wrote:', out_md)


if __name__ == '__main__':
    main()
