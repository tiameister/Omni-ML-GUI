# Machine Learning Application — Advanced Regression & Evaluation Pipeline

This application provides an end-to-end, publication-ready machine learning pipeline for regression tasks. It trains multiple ML models with advanced cross-validation strategies, automates feature engineering, and produces extensive diagnostics, including permutation importance, SHAP explainability, partial dependence plots (PDPs), and regression statistics.

It supports fully automated batch execution via the command line Interface (CLI) and an interactive, step-by-step Graphical User Interface (GUI).

## Highlights

- **Comprehensive End-to-End Pipeline:** Data loading -> Preprocessing -> Feature Engineering -> Model Training/CV -> Evaluation -> XAI (Explainability).
- **Supported Models:** Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, HistGB, SVR, KNN, XGBoost.
- **Cross-Validation Modes:** Standard K-Fold, Repeated K-Fold, and Nested CV.
- **Robust Preprocessing:** Handles numeric, ordinal, binary, and categorical grouped data with appropriate imputation and scaling techniques.
- **Automated Feature Engineering:** Optional inclusion of Polynomial Features and Missing-Value Indicators directly from the GUI.
- **Publication-Ready Figures:** Automatically styled outputs for CV splits, diagnostic plots, permutation importance, and SHAP results mapped with human-readable categorical labels.
- **Interactive GUI:** Built with PyQt6, providing a local "Publication Studio" for configuring variables before execution, tracking parallel job execution, and evaluating multi-model results.

## Output Directory Structure

Outputs are generated under `output/runs/{RUN_ID}/` with the following stage-based structure:

```text
output/runs/job1_.../
├── 0_Feature_Selection/
│   └── ui_feature_selection_meta.json
├── 1_Overall_Evaluation/
│   ├── metrics.xlsx (CV Metrics for all models)
│   ├── metrics_R2_cv.png (Bar charts)
│   └── permutation_importance_*.png
├── 2_Model_Diagnostics/
│   ├── HistGB/
│   │   ├── actual_vs_predicted.png
│   │   ├── learning_curve.png
│   │   ├── qq_plot.png
│   │   ├── regression_stats.xlsx
│   │   └── residuals_plot.png
│   └── RandomForest/
├── 3_Manuscript_Figures/
│   ├── HistGB/
│   │   ├── *_feature_importance.png
│   │   ├── *_shap_summary.png
│   │   └── *_shap_dependence.png
│   └── RandomForest/
└── Run_Log_and_Warnings.md
```

## Requirements

- Python 3.10+ (tested with 3.11)
- OS: Windows, macOS, or Linux.

Recommended Python packages:
```powershell
pip install numpy pandas scikit-learn seaborn matplotlib shap statsmodels openpyxl PyQt6
pip install xgboost  # Optional, but required if you want to use the XGBoost model
```

Or quickly install via `requirements.txt`:
```powershell
pip install -r requirements.txt
```

## Quickstart (GUI)

Run the GUI to interactively load your dataset, select target/features, apply feature engineering, choose models, and review the results.

```powershell
python run_gui.py
```

**GUI Workflow:**
1. **Load Dataset:** Drag-and-drop or select your CSV/Excel file.
2. **Select Variables:** Pick the target and feature columns. Adjust categorical definitions.
3. **Train Models:** Check the models to train.
4. **Evaluate Results:** The embedded result viewer lets you explore correlation matrices, SHAP outputs, CV metrics, and regression stats.

*Pro-tip:* Use shortcuts like `Ctrl+O` for dataset loading, `Ctrl+L` for variable selection, and `Ctrl+Enter` to queue training.

## Quickstart (CLI)

1) Place your dataset. You can configure data schemas in `config/columns.py` and execution rules in `config/__init__.py`.
2) Trigger batch jobs from root via:

```powershell
python main.py
```

All trained models and their artifacts are dropped into the timestamped specific run folder under `output/runs/`.

## Preprocessing Configuration

Preprocessing is defined dynamically via mappings in `config/columns.py` and logic in `features/preprocess.py`:
- **Numeric columns**: Median imputation + StandardScaler
- **Ordinal columns**: Most-frequent imputation + OrdinalEncoder
- **Binary columns**: Most-frequent imputation + Passthrough
- **Categorical (Nominal) columns**: Most-frequent imputation + OneHotEncoder

## Reproducibility and Configuration

- **Reproducibility**: Random seed (`config.RSTATE`) is distributed to cross-validation folds, sub-shuffles, and stochastic model initializations. 
- **Configuration Paths**:
  - `config/__init__.py` handles primary operational rules (`DO_SHAP`, dataset path, `CV_FOLDS`).
  - `utils/plotting_helpers.py` handles cosmetic transformations (e.g., matching encoded variables like `Gender=1` back to `Male/Female` in matplotlib).

## License
Proprietary / Internal.
