import matplotlib
matplotlib.use('Agg')
"""
Aggregator module for evaluation plotting and SHAP utilities.
Sets matplotlib to non-GUI backend and re-exports key plotting functions.
"""
# Thin aggregator re-exporting plotting and analysis utilities from evaluation.plots
from .plots import (
    generate_pdp,
    explain_with_shap,
    # expose fine-grained SHAP functions as well
    generate_shap_summary,
    generate_shap_dependence,
    generate_regression_stats,
    plot_residuals,
    plot_residual_distribution,
    plot_qq,
    plot_correlation_matrix,
    plot_learning_curve,
    plot_predictions_vs_actual,
    plot_feature_importance_heatmap,
)

__all__ = [
    'generate_pdp', 'explain_with_shap', 'generate_shap_summary', 'generate_shap_dependence',
    'generate_regression_stats',
    'plot_residuals', 'plot_residual_distribution', 'plot_qq',
    'plot_correlation_matrix',
    'plot_learning_curve', 'plot_predictions_vs_actual',
    'plot_feature_importance_heatmap'
]

