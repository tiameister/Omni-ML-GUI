import os
import re
from utils.paths import DIAGNOSTICS_DIR, EVALUATION_DIR, FEATURE_SELECTION_DIR, MANUSCRIPT_DIR

def write_manuscript_guide():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.environ.get("MLTRAINER_RUN_ROOT")

    if target_dir and os.path.isdir(target_dir):
        output_location = target_dir
    else:
        # Fallback to output/runs
        target_dir = os.path.join(root, "output")
        runs_dir = os.path.join(target_dir, "runs")
        output_location = target_dir
        if os.path.isdir(runs_dir):
            subdirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
            if subdirs:
                output_location = os.path.join(runs_dir, subdirs[0])
    
    manuscript_dir = os.path.join(output_location, MANUSCRIPT_DIR)
    os.makedirs(manuscript_dir, exist_ok=True)
    guide_path = os.path.join(manuscript_dir, "MANUSCRIPT_REPORT_GUIDE.txt")
    
    # Collect available files
    all_files = []
    # Only walk within output_location
    for r, d, f in os.walk(output_location):
        for file in f:
            if not file.startswith(".") and file.endswith((".png", ".pdf", ".csv", ".xlsx", ".txt")):
                all_files.append(os.path.join(r, file))
                
    def check_file(pattern):
        for f in all_files:
            if re.search(pattern, os.path.basename(f), re.IGNORECASE):
                return True
        return False

    content = []
    content.append("="*80)
    content.append(" MACHINE LEARNING MANUSCRIPT ARTIFACT GUIDE (FOR Q1 ACADEMIC JOURNALS) ")
    content.append("="*80)
    content.append("\nThis guide surveys the files generated in your output directory and maps ")
    content.append("them to standard sections of an academic Machine Learning manuscript.\n")
    
    # --- 1. METHODS SECTION ---
    content.append("1. MATERIALS & METHODS SECTION")
    content.append("-" * 35)
    content.append("These generated files should be used to describe the data profiling and the analytical pipeline:\n")
    if check_file(r"mcar"):
        content.append("- Table/Description: Missing Data Analysis (MCAR check). Justifies data imputation strategies using the 'mcar' outputs.")
    if check_file(r"target"):
        content.append("- Descriptive Stats: Target distribution descriptives. Usually located in 'baseline' or 'target' output folders.")
    if check_file(r"metrics_?.*\.xlsx|cv_splits"):
        content.append("- Validation Strategy: Detail your k-fold, nested, or repeated validation using the metric sheets. State the CV hyperparameter setup.")
    content.append("\n")

    # --- 2. RESULTS SECTION ---
    content.append("2. MAIN RESULTS SECTION")
    content.append("-" * 35)
    content.append("Place the most impactful, high-level findings here. Keep it concise:\n")
    if check_file(r"metrics_?.*\.xlsx"):
        content.append("- Primary Table (Model Performance): Average R2, RMSE, and MAE across models. Create a comparison table from 'metrics.xlsx'.")
    if check_file(r"shap_summary_beeswarm"):
        content.append("- Primary Figure (Global Feature Importance): Insert the '...shap_summary_beeswarm.png'. This is extremely popular in Q1 papers to explain overall feature impact.")
    if check_file(r"shap_dependence"):
        content.append("- Additional Figures (Feature Behaviors): SHAP dependence plots for the top 2-3 most important features showing non-linear thresholds.")
    if check_file(r"predictions_vs_actual.*png"):
        content.append("- Primary Plot (Actual vs Predicted): Use the 'predictions_vs_actual' scatter plot to show model fit visually.")
    content.append("\n")

    # --- 3. SUPPLEMENTARY MATERIALS ---
    content.append("3. SUPPLEMENTARY MATERIALS (APPENDIX)")
    content.append("-" * 35)
    content.append("Place detailed diagnostic and stability checks here to prove robustness to reviewers:\n")
    if check_file(r"learning_curve"):
        content.append("- Figure S1 (Learning Curves): Insert '..._learning_curve.png' to prove the model is not overfitting and data size is adequate.")
    if check_file(r"residuals|qq_plot|residual_distribution"):
        content.append("- Figure S2 (Diagnostics): Insert '..._residuals.png' and '..._qq_plot.png'. Reviewers often ask for homoskedasticity and normality checks.")
    if check_file(r"calibration"):
        content.append("- Figure S3 (Calibration/Reliability): Insert 'calibration_curves.png' to demonstrate prediction confidence.")
    if check_file(r"stability|consistency"):
        content.append("- Table S1 (XAI Rank Stability): If you generated the rank stability heatmaps/tables, add them to prove top features are independent of the model chosen.")
    if check_file(r"stats|test"):
        content.append("- Table S2 (Statistical Tests): The 'corrected_t_tests.csv' or 'fdr_adjusted_pvalues' proves the performance gap between models is statistically significant.")
    content.append("\n")
    
    # --- WRAP UP ---
    content.append("="*80)
    content.append("QUICK TIPS FOR WRITING:")
    content.append(
        f"- Folder Structure: Look in '{EVALUATION_DIR}/' for model comparisons and '{FEATURE_SELECTION_DIR}/' for selection outputs. "
        f"Models are isolated inside '{DIAGNOSTICS_DIR}/' and '{MANUSCRIPT_DIR}/' with subfolders, so running 10+ models won't clutter the root directory."
    )
    content.append("- Do not overload the main results with 10 identical graphics. Pick the best model for the SHAP/Predicted plots and relegate the rest to the supplements.")
    content.append("- Explicitly mention 'SHAP (SHapley Additive exPlanations) values were utilized for model explainability' in your methods.")
    content.append("- Use 'tight' bounding box geometries for figures directly in your Word/LaTeX manuscript to prevent cropped axis labels.")

    with open(guide_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    
    print(f"Manuscript Guide generated at: {guide_path}")


if __name__ == "__main__":
    write_manuscript_guide()