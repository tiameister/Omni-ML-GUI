# Evaluation Module Guide

This folder contains reusable evaluation and explainability helpers used by both CLI and GUI runs.

## Core modules

- `metrics.py`
  - Exports model metrics and CV split artifacts.
  - Includes feature-name extraction and permutation importance dump helpers.
- `shap_analysis.py`
  - High-level SHAP orchestration wrapper.
  - Produces summary/dependence artifacts and optional humanized previews.
- `explain.py`
  - Convenience re-export surface for plotting functions.

## Plot modules (`evaluation/plots`)

- `pdp_shap.py`: PDP and SHAP visual outputs (summary + dependence)
- `residuals.py`: residual scatter, residual distribution, Q-Q diagnostics
- `curves.py`: learning curves and prediction-vs-actual plots
- `regression_stats.py`: statsmodels-based coefficient and p-value reporting
- `correlation.py`: correlation matrix helper
- `feature_importance.py`: feature importance heatmap/bar helpers

## Output conventions

These helpers are expected to write under model-scoped output directories, for example:

- `output/<ModelName>_output/evaluation/*`
- `output/<ModelName>_output/explainability/*`
- `output/<ModelName>_output/diagnostics/*`

## Import guidance

Prefer importing directly from stable modules used by the training pipeline:

```python
from evaluation.metrics import save_model_metrics, dump_permutation
from evaluation.shap_analysis import run_shap_analysis
from evaluation.plots.curves import plot_learning_curve
```

Use `evaluation.explain` only as a convenience import layer when you explicitly need mixed plotting exports.
