# TIA Machine Learning Studio
**Enterprise-Grade Automated Machine Learning & XAI Desktop Application**

TIA Machine Learning Studio is an end-to-end, publication-ready machine learning desktop platform optimized for speed, memory efficiency, and rigorous academic standards. Built with PyQt6 and powered by an aggressively optimized Scikit-Learn/C++ backend, the platform automates data ingestion, robust feature engineering, nested cross-validation, and Explainable AI (XAI) extraction without requiring programming knowledge.

## Features & Architectural Optimizations

This pipeline is engineered to industrial standards, prioritizing mathematical integrity, zero memory leaks, and C-level execution speed:

- **PyArrow Memory Backend:** String caching is intelligently mapped to PyArrow arrays. Combined with lossless numeric downcasting, the RAM footprint during dataframe loading is drastically reduced.
- **O(1) Data Coercion:** Full dataset string iterations are replaced with intelligent heuristic sampling. Mismatched delimiters and categorical texts are handled directly with minimal compute penalty.
- **Thread Explosion Protection:** Nested Cross-Validation defaults to inner and outer thread limits matched with job dispatchers, strictly avoiding RAM exhaustion on multi-core workstations.
- **Academic XAI Integrity:** 
  - Prevents target leakage via constant empty value imputation rather than the heavily biased most-frequent mode. 
  - Early Stopping is natively configured for structural Tree algorithms to deterministically prevent overfitting.
  - Generates natively exact analytical SHAP values via directly optimized TreeExplainer backends instead of stochastic samplers.
- **Asynchronous UI (QRunnable):** Heavy I/O blocking during dataset ingestion is pushed entirely to PyQt background workers, keeping the UI fully responsive under heavy data loads.
- **Reproducibility Serializer:** All runs emit an `experiment_metadata.json` capturing strict hardware info, Python version, cv-strategies, hyperparameters, and test performances for publication validation.

## Installation & Setup

1. **Clone the repository and create a virtual environment:**
```bash
git clone <repository_url>
cd MachineLearning
python -m venv venv
```

2. **Activate the virtual environment:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**
Ensure you have the required optimization libraries (including PyArrow) installed.
```bash
pip install -r requirements.txt
```

## User Guide (Graphical Interface)

The application provides a no-code visual interface to guide you through the entire machine learning process.

**1. Launch the Application:**
```bash
python run_gui.py
```

**2. Load Dataset:** 
Use the "Load Dataset" button to import your CSV or Excel file. The application asynchronously reads and optimizes the memory usage of your dataset in the background.

**3. Configure Variables:**
Navigate to the "Variable Selection" section. Select the target variable you want to predict, and choose the variables you want the model to learn from (Features). The system automatically detects data types, but you can explicitly define categorical or numeric column rules.

**4. Select Models & Settings:**
Choose from a list of linear models, tree-based models, and distance-based algorithms. You can set up the cross-validation strategy (e.g., 5-Fold) to evaluate how well the models generalize.

**5. Execute Training:**
Click "Run Training". The system will train the selected models, perform cross-validation, and generate Partial Dependence Plots (PDP) and SHAP values for interpretability. 

**6. Publication Studio & Export:**
Once finished, all performance tables, evaluation plots, and feature importance matrices will be visible inside the GUI. You can review and export these directly via the built-in Publication Studio for manuscript writing.

## Supported Algorithms
- **Linear Models:** Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Ensembles:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost
- **Distance/Margin:** SVR (Support Vector Machines), KNN

## Output Artifacts Structure

Outputs are rigorously version-controlled and saved locally:
```text
output/runs/test_run_.../
├── experiment_metadata.json
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

## Batch Execution (Command Line)
For advanced users who prefer headless servers or batch operations, the exact same pipeline can be executed via command line. Configure `config/columns.py` and `config/__init__.py`, then run:
```bash
python main.py
```

---

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

This project is licensed under the MIT License - see the LICENSE file for details. Contributions, bug reports, and feature requests are always welcome!
# TIA Machine Learning Studio
**Enterprise-Grade Automated Machine Learning & XAI Desktop Application**

TIA Machine Learning Studio is an end-to-end, publication-ready machine learning desktop platform optimized for speed, memory efficiency, and rigorous academic standards. Built with PyQt6 and powered by an aggressively optimized Scikit-Learn/C++ backend, the platform automates data ingestion, robust feature engineering, nested cross-validation, and Explainable AI (XAI) extraction without requiring programming knowledge.

## Features & Architectural Optimizations

This pipeline is engineered to industrial standards, prioritizing mathematical integrity, zero memory leaks, and C-level execution speed:

- **PyArrow Memory Backend:** String caching is intelligently mapped to PyArrow arrays. Combined with lossless numeric downcasting, the RAM footprint during dataframe loading is drastically reduced.
- **O(1) Data Coercion:** Full dataset string iterations are replaced with intelligent heuristic sampling. Mismatched delimiters and categorical texts are handled directly with minimal compute penalty.
- **Thread Explosion Protection:** Nested Cross-Validation defaults to inner and outer thread limits matched with job dispatchers, strictly avoiding RAM exhaustion on multi-core workstations.
- **Academic XAI Integrity:** 
  - Prevents target leakage via constant empty value imputation rather than the heavily biased most-frequent mode. 
  - Early Stopping is natively configured for structural Tree algorithms to deterministically prevent overfitting.
  - Generates natively exact analytical SHAP values via directly optimized TreeExplainer backends instead of stochastic samplers.
- **Asynchronous UI (QRunnable):** Heavy I/O blocking during dataset ingestion is pushed entirely to PyQt background workers, keeping the UI fully responsive under heavy data loads.
- **Reproducibility Serializer:** All runs emit an `experiment_metadata.json` capturing strict hardware info, Python version, cv-strategies, hyperparameters, and test performances for publication validation.

## Installation & Setup

1. **Clone the repository and create a virtual environment:**
```bash
git clone <repository_url>
cd MachineLearning
python -m venv venv
<<<<<<< HEAD
```

2. **Activate the virtual environment:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**
Ensure you have the required optimization libraries (including PyArrow) installed.
```bash
pip install -r requirements.txt
```

## User Guide (Graphical Interface)

The application provides a no-code visual interface to guide you through the entire machine learning process.

**1. Launch the Application:**
```bash
python run_gui.py
```

**2. Load Dataset:** 
Use the "Load Dataset" button to import your CSV or Excel file. The application asynchronously reads and optimizes the memory usage of your dataset in the background.

**3. Configure Variables:**
Navigate to the "Variable Selection" section. Select the target variable you want to predict, and choose the variables you want the model to learn from (Features). The system automatically detects data types, but you can explicitly define categorical or numeric column rules.

**4. Select Models & Settings:**
Choose from a list of linear models, tree-based models, and distance-based algorithms. You can set up the cross-validation strategy (e.g., 5-Fold) to evaluate how well the models generalize.

**5. Execute Training:**
Click "Run Training". The system will train the selected models, perform cross-validation, and generate Partial Dependence Plots (PDP) and SHAP values for interpretability. 

**6. Publication Studio & Export:**
Once finished, all performance tables, evaluation plots, and feature importance matrices will be visible inside the GUI. You can review and export these directly via the built-in Publication Studio for manuscript writing.

## Supported Algorithms
- **Linear Models:** Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Ensembles:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost
- **Distance/Margin:** SVR (Support Vector Machines), KNN

## Output Artifacts Structure

Outputs are rigorously version-controlled and saved locally:
```text
output/runs/test_run_.../
├── experiment_metadata.json
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

=======
```

2. **Activate the virtual environment:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**
Ensure you have the required optimization libraries (including PyArrow) installed.
```bash
pip install -r requirements.txt
```

## User Guide (Graphical Interface)

The application provides a no-code visual interface to guide you through the entire machine learning process.

**1. Launch the Application:**
```bash
python run_gui.py
```

**2. Load Dataset:** 
Use the "Load Dataset" button to import your CSV or Excel file. The application asynchronously reads and optimizes the memory usage of your dataset in the background.

**3. Configure Variables:**
Navigate to the "Variable Selection" section. Select the target variable you want to predict, and choose the variables you want the model to learn from (Features). The system automatically detects data types, but you can explicitly define categorical or numeric column rules.

**4. Select Models & Settings:**
Choose from a list of linear models, tree-based models, and distance-based algorithms. You can set up the cross-validation strategy (e.g., 5-Fold) to evaluate how well the models generalize.

**5. Execute Training:**
Click "Run Training". The system will train the selected models, perform cross-validation, and generate Partial Dependence Plots (PDP) and SHAP values for interpretability. 

**6. Publication Studio & Export:**
Once finished, all performance tables, evaluation plots, and feature importance matrices will be visible inside the GUI. You can review and export these directly via the built-in Publication Studio for manuscript writing.

## Supported Algorithms
- **Linear Models:** Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Ensembles:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost
- **Distance/Margin:** SVR (Support Vector Machines), KNN

## Output Artifacts Structure

Outputs are rigorously version-controlled and saved locally:
```text
output/runs/test_run_.../
├── experiment_metadata.json
├── 0_Feature_Selection/
├── 1_Overall_Evaluation/
│   ├── metrics.xlsx (CV Metrics for all models)
│   ├── R2_cv_bar.png 
│   └── permutation_importance_*.png
├── 2_Model_Diagnostics/
│   └── HistGB/
│       ├── regression_stats.xlsx
│       ├── learning_curve.png
│       └── residuals_plot.png
├── 3_Manuscript_Figures/
│   └── HistGB/
│       ├── histgb_feature_importance.png
│       └── histgb_shap_summary.png
└── Run_Log_and_Warnings.md
```

>>>>>>> updated_logic
## Batch Execution (Command Line)
For advanced users who prefer headless servers or batch operations, the exact same pipeline can be executed via command line. Configure `config/columns.py` and `config/__init__.py`, then run:
```bash
python main.py
```

<<<<<<< HEAD
---

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

=======
>>>>>>> updated_logic
## License

This project is licensed under the MIT License - see the LICENSE file for details. Contributions, bug reports, and feature requests are always welcome!
# TIA Machine Learning Studio
**Enterprise-Grade Automated Machine Learning & XAI Desktop Application**

TIA Machine Learning Studio is an end-to-end, publication-ready machine learning desktop platform optimized for speed, memory efficiency, and rigorous academic standards. Built with PyQt6 and powered by an aggressively optimized Scikit-Learn/C++ backend, the platform automates data ingestion, robust feature engineering, nested cross-validation, and Explainable AI (XAI) extraction without requiring programming knowledge.

## Features & Architectural Optimizations

This pipeline is engineered to industrial standards, prioritizing mathematical integrity, zero memory leaks, and C-level execution speed:

- **PyArrow Memory Backend:** String caching is intelligently mapped to PyArrow arrays. Combined with lossless numeric downcasting, the RAM footprint during dataframe loading is drastically reduced.
- **O(1) Data Coercion:** Full dataset string iterations are replaced with intelligent heuristic sampling. Mismatched delimiters and categorical texts are handled directly with minimal compute penalty.
- **Thread Explosion Protection:** Nested Cross-Validation defaults to inner and outer thread limits matched with job dispatchers, strictly avoiding RAM exhaustion on multi-core workstations.
- **Academic XAI Integrity:** 
  - Prevents target leakage via constant empty value imputation rather than the heavily biased most-frequent mode. 
  - Early Stopping is natively configured for structural Tree algorithms to deterministically prevent overfitting.
  - Generates natively exact analytical SHAP values via directly optimized TreeExplainer backends instead of stochastic samplers.
- **Asynchronous UI (QRunnable):** Heavy I/O blocking during dataset ingestion is pushed entirely to PyQt background workers, keeping the UI fully responsive under heavy data loads.
- **Reproducibility Serializer:** All runs emit an `experiment_metadata.json` capturing strict hardware info, Python version, cv-strategies, hyperparameters, and test performances for publication validation.

## Installation & Setup

1. **Clone the repository and create a virtual environment:**
```bash
git clone <repository_url>
cd MachineLearning
python -m venv venv
```

2. **Activate the virtual environment:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

3. **Install dependencies:**
Ensure you have the required optimization libraries (including PyArrow) installed.
```bash
pip install -r requirements.txt
```

## User Guide (Graphical Interface)

The application provides a no-code visual interface to guide you through the entire machine learning process.

**1. Launch the Application:**
```bash
python run_gui.py
```

**2. Load Dataset:** 
Use the "Load Dataset" button to import your CSV or Excel file. The application asynchronously reads and optimizes the memory usage of your dataset in the background.

**3. Configure Variables:**
Navigate to the "Variable Selection" section. Select the target variable you want to predict, and choose the variables you want the model to learn from (Features). The system automatically detects data types, but you can explicitly define categorical or numeric column rules.

**4. Select Models & Settings:**
Choose from a list of linear models, tree-based models, and distance-based algorithms. You can set up the cross-validation strategy (e.g., 5-Fold) to evaluate how well the models generalize.

**5. Execute Training:**
Click "Run Training". The system will train the selected models, perform cross-validation, and generate Partial Dependence Plots (PDP) and SHAP values for interpretability. 

**6. Publication Studio & Export:**
Once finished, all performance tables, evaluation plots, and feature importance matrices will be visible inside the GUI. You can review and export these directly via the built-in Publication Studio for manuscript writing.

## Supported Algorithms
- **Linear Models:** Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Ensembles:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost
- **Distance/Margin:** SVR (Support Vector Machines), KNN

## Output Artifacts Structure

Outputs are rigorously version-controlled and saved locally:
```text
output/runs/test_run_.../
├── experiment_metadata.json
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

## Batch Execution (Command Line)
For advanced users who prefer headless servers or batch operations, the exact same pipeline can be executed via command line. Configure `config/columns.py` and `config/__init__.py`, then run:
```bash
python main.py
```

---

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

This project is licensed under the MIT License - see the LICENSE file for details. Contributions, bug reports, and feature requests are always welcome!