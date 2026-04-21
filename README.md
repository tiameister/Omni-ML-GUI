
# Omni-ML-GUI (Omni-Machine Learning Studio)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19649882-blue)](https://doi.org/10.5281/zenodo.19649882)

**A No-Code Machine Learning & XAI Workbench for Researchers and Students**

---

```bash
git clone https://github.com/tiameister/Omni-ML-GUI.git
cd Omni-ML-GUI
python -m venv venv
pip install -r requirements.txt
```


Omni-ML-GUI is an accessible, end-to-end desktop application designed to help students, researchers, and data enthusiasts run rigorous machine learning experiments without needing to write complex code. Built with PySide6 and powered by Scikit-Learn, this tool automates data processing, model training, cross-validation, and Explainable AI (XAI) extraction, so you can focus on your research instead of debugging pipelines.


## Why use this tool?

I built this platform to bring academic rigor to a simple user interface. It is optimized to run smoothly on personal computers while outputting results that are ready for publication:

- **Memory-Friendly & Fast:** Uses a PyArrow backend to efficiently handle dataset loading and formatting. It prevents RAM exhaustion and system freezes, even when running heavy nested cross-validation on standard workstations.
- **Academic-Grade Explainability (XAI):** Ensures robust analysis by avoiding target leakage during missing value imputation. It generates exact analytical SHAP values and Partial Dependence Plots (PDP) to help you explain your models transparently.
- **Responsive No-Code Interface:** Heavy data processing runs in the background, keeping the user interface smooth and responsive while your models train.
- **Reproducible Results:** Every experiment automatically saves an `experiment_metadata.json` file. This logs your hardware details, Python version, hyperparameter settings, and cross-validation strategies, making it easy to validate and reproduce your findings for papers or thesis work.

## 🌟 Key Feature: Publication Studio

Writing a thesis or a paper? **Publication Studio** is a built-in module designed to eliminate the tedious process of re-formatting results.

Accessed during the "Pre-training" phase, it allows you to:
- **Rename Variables & Labels:** Instantly map raw column names (e.g., `feat_01_val`) to publication-ready names (e.g., `Standardized Income ($)`).
- **Consistent Visual Branding:** Apply custom labels and figure titles globally.
- **Auto-Formatting:** All exported tables (Excel) and figures (PNG/PDF) will automatically use your custom labels, ensuring consistency throughout your entire manuscript without manual editing.


## Installation & Setup

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone https://github.com/tiameister/Omni-ML-GUI.git
   cd Omni-ML-GUI
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## How to Use (Graphical Interface)

The application provides a visual step-by-step workflow:

1. **Launch the App:** Run `python run_gui.py` in your terminal.
2. **Load Dataset:** Import your CSV or Excel files. The app optimizes the dataset for memory usage automatically.
3. **Configure Variables:** Select your target and features.
4. **Publication Studio (Optional):** Open the Studio dialog to customize how variable names and categories will appear in your final charts and tables.
5. **Select Models & Settings:** Choose from various algorithms and cross-validation strategies.
6. **Run Training:** The app will handle the training, validation, and generation of XAI metrics (like SHAP and PDP).
7. **Export for Publication:** Review performance tables, learning curves, and feature importance matrices directly in the GUI. You can easily export these figures for your thesis, manuscript, or presentation.


## Supported Algorithms

- **Linear Models:** Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based Ensembles:** Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost
- **Distance/Margin:** SVR (Support Vector Machines), KNN


## Output Artifacts

Outputs are organized neatly in your local directory, ready to be attached to your research notes:

```
output/runs/test_run_.../
├── experiment_metadata.json
├── feature_selection/
├── metrics/
│   ├── metrics.xlsx (CV Metrics for all models)
│   ├── R2_cv_bar.png 
│   └── permutation_importance_*.png
├── diagnostics/
│   └── HistGB/
│       ├── regression_stats.xlsx
│       ├── learning_curve.png
│       └── residuals_plot.png
├── figures/
│   └── HistGB/
│       ├── histgb_feature_importance.png
│       └── histgb_shap_summary.png
└── Run_Log_and_Warnings.md
```



## Batch Execution (For Advanced Users)

If you prefer working without a GUI or want to run batch operations on a server, you can execute the exact same pipeline via the command line. Just configure `config/columns.py` and `config/__init__.py`, then run:

```bash
python main.py
```



## Citation
If you use this tool in your research, please cite it as follows:

**APA:**  
Akar, T. I. (2026). Omni-ML-GUI (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.19649882

**BibTeX:**
```bibtex
@software{Akar_Omni_ML_GUI_2026,
	author = {Akar, Taha Ilter},
	title = {{Omni-ML-GUI: A Comprehensive GUI for Machine Learning and XAI Research}},
	month = {4},
	year = {2026},
	publisher = {Zenodo},
	version = {1.0.1},
	doi = {10.5281/zenodo.19649882},
	url = {https://github.com/tiameister/Omni-ML-GUI}
}
```


## License

This project is licensed under the MIT License - see the LICENSE file for details. Contributions, bug reports, and feature requests are always welcome!

---

## About the Developer

This tool is a solo project developed as part of my ongoing Master's research at FAU. It is in active development, and feedback is highly appreciated.
