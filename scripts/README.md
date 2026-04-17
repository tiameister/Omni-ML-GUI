# Scripts Catalog and Run Guide

The scripts folder now has a searchable catalog for day-to-day operations.

## 1) Discover what each script does

From project root:

```powershell
python scripts/catalog.py
```

Filter by category:

```powershell
python scripts/catalog.py --category manuscript
python scripts/catalog.py --category xai
python scripts/catalog.py --category smoke
```

Show expected outputs:

```powershell
python scripts/catalog.py --show-outputs
```

Machine-readable output:

```powershell
python scripts/catalog.py --json
```

Validate that catalog entries point to existing files:

```powershell
python scripts/catalog.py --validate
```

## 2) Recommended run order (manuscript workflow)

1. `python -X faulthandler main.py`
2. `python -X faulthandler scripts/build_model_benchmark_artifacts.py`
3. `python -X faulthandler scripts/build_psychometrics.py`
4. `python -X faulthandler scripts/build_supplements_named.py`
5. `python -X faulthandler scripts/collect_manuscript_exports.py`

Validation protocol comparison helper:

```powershell
python scripts/run_validation_compare.py --protocols kfold,repeated,nested --selected-models RandomForest
```

Dry-run mode (no training, only planned runs):

```powershell
python scripts/run_validation_compare.py --dry-run
```

## 3) Category intent

- `manuscript`: final tables/figures and export bundle scripts
- `benchmark`: model comparison and baseline analyses
- `diagnostics`: cleaning, missingness, calibration, and CV diagnostics
- `xai`: SHAP and rank stability analyses
- `smoke`: fast pipeline checks
- `stats`: statistical significance tooling
- `maintenance`: environment and debug helpers

## 4) Naming caveats

- `build_supplements.py` is a generic supplements builder.
- `build_supplements_named.py` targets fixed manuscript S-table/S-figure naming.
- `make_rank_stability_heatmap.py` currently produces a bar-style stability visualization despite its name.

## 5) Q1 social-science auto-translation helper

Use the helper to convert Turkish manuscript columns/labels into publication-friendly English:

```powershell
python scripts/auto_translate_q1_terms.py --input dataset/my_data.xlsx --output output/my_data_q1.xlsx
```

Default behavior automatically loads social-science mappings from:

- `scripts/q1_social_science_mappings.json`

This JSON is the main edit file for adding new column/value mappings (for example, bullying, school belonging, school happiness, region labels, parental education, income labels).
Publication Studio preset buttons reuse the same mapping file, so updates here are reflected in both script and UI workflows.

You can also provide an extra mapping file to extend or override defaults:

```powershell
python scripts/auto_translate_q1_terms.py --input dataset/my_data.xlsx --mapping-file scripts/my_extra_mappings.json
```
