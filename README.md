# Machine Learning Application — Happiness Score Prediction

This application trains multiple ML models (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, HistGB, SVR, KNN, XGBoost) with 5-fold cross-validation to predict Happiness Score. It produces publication-ready figures and reports including SHAP explainability, permutation importance, regression statistics, and more.

It can be used from the command line (batch runs) or via a simple GUI for interactive workflows.

## Highlights

- End-to-end pipeline: data loading → preprocessing → model training → evaluation → explainability
- 5-fold CV metrics (R², RMSE) with bar charts and Excel exports
- Permutation importance (CSV/Excel + bar plots)
- PDP (Partial Dependence) for top features
- SHAP explainability: summary + dependence plots
- Publication-ready SHAP visuals with human-readable categorical labels and consistent scales (saved under `./figures/`)
- Regression tables and coefficient plots (with standardized Betas and 95% CI)

## Requirements

- Python 3.10+ (tested with 3.11)
- Windows (PowerShell examples below), macOS/Linux also work with minor command changes

Recommended Python packages (install if not already available):

- numpy, pandas, scikit-learn, seaborn, matplotlib
- shap, statsmodels, openpyxl
- xgboost (optional; required for XGBoost model)

Quick install from manifest:

```powershell
pip install -r requirements.txt
```

## Quickstart (CLI)

1) Place your dataset at `dataset/data.csv` (default path). The pipeline auto-detects the target and feature columns; you can customize `config/__init__.py` and `config/columns.py` if needed.

2) From the project root, run:

- Windows PowerShell

```powershell
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install common dependencies
pip install numpy pandas scikit-learn seaborn matplotlib shap statsmodels openpyxl xgboost

# Run the training pipeline
python -X faulthandler main.py
```

3) Outputs

- `./output/<ModelName>_output/` directory with:
  - evaluation: CV metrics, regression stats, curves
  - explainability: SHAP results, PDPs, feature importance
  - diagnostics/fit subfolders for other plots/artifacts
- Publication-ready SHAP plots under:
  - `./figures/shap_summary_beeswarm.png`
  - `./figures/shap_dependence/`

## Quickstart (GUI)

Run the GUI to load a CSV, select target/features, choose models/plots, and run training interactively.

```powershell
python -X faulthandler run_gui.py
```

Inside the GUI:
- Load your dataset (CSV)
- Select variables (target and features)
- Configure SHAP settings (optional)
- Train models and view outputs. The app saves the same artifacts under `./output/` and will display SHAP summary if available.
- Use drag-and-drop to load CSV files directly into the main window.
- Use keyboard shortcuts for speed: `Ctrl+O` (load dataset), `Ctrl+L` (select variables), `Ctrl+Enter` (start training).
- Resize panel widths using the splitter; layout and panel sizes are persisted between sessions.

## SHAP Visualization (Publication-ready)

Kanonik SHAP akışı evaluation/plots + utils/plotting_helpers üzerindedir:

- `evaluation/plots/pdp_shap.py`
  - `generate_shap_summary(...)`: SHAP beeswarm ve bar özetleri (insan-okur adlarla)
  - `generate_shap_dependence(...)`: ham birimler ve düzenli kategorik etiketlerle bağımlılık grafikleri
  - `explain_with_shap(...)`: yukarıdakilerin hızlı toplayıcısı
- `utils/plotting_helpers.py`
  - Kategorik etiket eşleme, sıralı eksenler, yayın-uyumlu stiller
  - SHAP bağımlılık çizimi için ortak yardımcılar ve tutarlı y-limit/klipleme mantığı

Ana akış (main.py) bu modülleri kullanır ve sonuçları `output/.../explainability/` altına kaydeder.

## Preprocessing configuration

Preprocessing is defined in `features/preprocess.py`:
- Numeric columns: median imputation + StandardScaler
- Ordinal columns: most-frequent imputation + OrdinalEncoder (no scaling)
- Binary columns: passthrough (no scaling)
- Other categoricals: most-frequent imputation + OneHotEncoder

Column groups are resolved based on your dataset using `config/columns.py`:
- Edit `NUMERIC_COLS`, `ORDINAL_COLS`, `BINARY_COLS` to match your schema
- The resolver will match these names to the actual columns in your CSV (robust to minor naming differences)

## Configuration

Edit `config/__init__.py` for global settings:
- `DATASET_PATH`: input CSV path
- `DO_SHAP`: enable/disable SHAP explainability
- `SHAP_TOP_N`, `SHAP_VAR_THRESH`: SHAP selection behavior
- `OUTPUT_DIR`, `SAVE_PDF`: output structure and formats

Edit `config/columns.py` for preprocessing groups.
Customize category labels and ordering via `utils/plotting_helpers.py`.

## Reproducibility

- Random seed is controlled via `config.RSTATE`
- Cross-validation folds and other stochastic steps use this seed when possible

## Troubleshooting

- Missing packages: install `shap`, `statsmodels`, `openpyxl`, `xgboost` if needed
- SHAP on non-linear models can be slow; consider sampling or setting `SHAP_TOP_N`
- If feature names appear unexpected in plots, update `config/columns.py` and label mappings in `utils/plotting_helpers.py`

## Project structure (selected)

```
config/__init__.py
config/columns.py                # column grouping for preprocessing
features/preprocess.py           # ColumnTransformer (num/ord/bin/cat)
models/train.py                  # training + CV
evaluation/                      # metrics, plots (explainability/export)
evaluation/README.md             # evaluation module responsibilities
scripts/catalog.py               # script purpose/category index
scripts/README.md                # script run guide and workflow
scripts/target_analysis.py       # target descriptive analysis helper
utils/plotting_helpers.py        # publication-ready SHAP helpers
main.py                          # CLI entrypoint
run_gui.py                       # GUI entrypoint
output/                          # model outputs (per run)
figures/                         # publication-ready SHAP figures
```

## Operational catalogs

- Script map and run-order guidance: `scripts/README.md`
- Evaluation module responsibility map: `evaluation/README.md`

## License

Proprietary/internal (update as appropriate).
