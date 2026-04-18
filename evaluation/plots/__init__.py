import functools
import matplotlib.pyplot as plt

def ensure_closed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            plt.close('all')
    return wrapper

# Plot modules package for modular plotting utilities
from .pdp_shap import generate_pdp as _g_pdp, explain_with_shap as _e_shap, generate_shap_summary as _g_shap_sum, generate_shap_dependence as _g_shap_dep
from .regression_stats import generate_regression_stats as _g_reg_stats
from .residuals import plot_residuals as _p_res, plot_residual_distribution as _p_res_dist, plot_qq as _p_qq
from .correlation import plot_correlation_matrix as _p_corr
from .curves import plot_learning_curve as _p_lc, plot_predictions_vs_actual as _p_p_v_a
from .feature_importance import plot_feature_importance_heatmap as _p_fih

generate_pdp = ensure_closed(_g_pdp)
explain_with_shap = ensure_closed(_e_shap)
generate_shap_summary = ensure_closed(_g_shap_sum)
generate_shap_dependence = ensure_closed(_g_shap_dep)
generate_regression_stats = ensure_closed(_g_reg_stats)
plot_residuals = ensure_closed(_p_res)
plot_residual_distribution = ensure_closed(_p_res_dist)
plot_qq = ensure_closed(_p_qq)
plot_correlation_matrix = ensure_closed(_p_corr)
plot_learning_curve = ensure_closed(_p_lc)
plot_predictions_vs_actual = ensure_closed(_p_p_v_a)
plot_feature_importance_heatmap = ensure_closed(_p_fih)

__all__ = [
    'generate_pdp', 'explain_with_shap', 'generate_shap_summary', 'generate_shap_dependence',
    'generate_regression_stats',
    'plot_residuals', 'plot_residual_distribution', 'plot_qq',
    'plot_correlation_matrix',
    'plot_learning_curve', 'plot_predictions_vs_actual',
    'plot_feature_importance_heatmap'
]
