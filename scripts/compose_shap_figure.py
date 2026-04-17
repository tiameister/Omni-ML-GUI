"""
Compose SHAP summary + dependence plots into a single figure.

Searches for SHAP exports under output/*_output*/explainability and
assembles the following panels:
 - SHAP summary bar
 - SHAP summary beeswarm
 - Dependence plots for: Total Bullying Score, Mobile Phone (Daily Hours),
   Teacher Intervention, Reading Books (Frequency), TV Time (Daily Hours)

Outputs:
 - new_plot/shap_summary_dependence_BestModel.png
 - new_plot/shap_summary_dependence_BestModel.pdf (optional if matplotlib supports)
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.image import imread


def find_rf_shap_dir(base: str, run_root: str | None = None) -> str | None:
    candidates = []
    if run_root and os.path.isdir(run_root):
        candidates.extend(glob.glob(os.path.join(run_root, "models", "*", "3_Manuscript_Figures")))
    candidates.extend(glob.glob(os.path.join(base, "output", "*_output*", "3_Manuscript_Figures")))
    candidates = sorted(set(candidates), key=lambda p: (len(p), p), reverse=True)
    for p in candidates:
        # Check presence of core images
        bar_candidates = glob.glob(os.path.join(p, "*_shap_summary_bar.png"))
        bees_candidates = glob.glob(os.path.join(p, "*_shap_summary_beeswarm.png"))
        if bar_candidates and bees_candidates:
            return p
    return None


def expect_files(shap_dir: str) -> Dict[str, str]:
    bar_candidates = sorted(
        glob.glob(os.path.join(shap_dir, "*_shap_summary_bar.png")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not bar_candidates:
        return {}
    bar_path = bar_candidates[0]
    bar_name = os.path.basename(bar_path)
    # Extract prefix and model name before '_shap_summary_bar.png'
    token = "_shap_summary_bar.png"
    prefix_and_model = bar_name[:-len(token)] if bar_name.endswith(token) else ""

    # Try mapping logical names via glob search because we might not know exact feature names or they might be different
    files = {
        "summary_bar": bar_path,
        "summary_bees": os.path.join(shap_dir, f"{prefix_and_model}_shap_summary_beeswarm.png"),
    }
    
    # Best effort feature dependencies
    dep_candidates = glob.glob(os.path.join(shap_dir, f"{prefix_and_model}_shap_dependence_*.png"))
    for idx, dep_path in enumerate(dep_candidates[:5]):
        files[f"dep::Feature_{idx+1}"] = dep_path
        
    return files


def compose_figure(files: Dict[str, str], out_png: str, out_pdf: str | None = None, out_svg: str | None = None) -> None:
    missing: List[str] = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing panels, will skip these:", ", ".join(missing))

    # Layout: 2 rows x 4 cols
    # Row 1: summary bar, summary bees, dep1, dep2
    # Row 2: dep3, dep4, dep5, empty
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    panels = [
        ("summary_bar", "SHAP summary — bar"),
        ("summary_bees", "SHAP summary — beeswarm"),
        ("dep::Total Bullying Score", "Dependence — Total Bullying Score"),
        ("dep::Mobile Phone (Daily Hours)", "Dependence — Mobile Phone (daily hours)"),
        ("dep::Teacher Intervention", "Dependence — Teacher Intervention"),
        ("dep::Reading Books (Frequency)", "Dependence — Reading Books (frequency)"),
        ("dep::TV Time (Daily Hours)", "Dependence — TV Time (daily hours)"),
    ]

    for ax in axes:
        ax.axis("off")

    for idx, (key, title) in enumerate(panels):
        if idx >= len(axes):
            break
        ax = axes[idx]
        path = files.get(key)
        if not path or not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing: {title}", ha="center", va="center")
            continue
        img = imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle("Figure R2.3 — SHAP summary + dependence (Best Model)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[OK] Saved: {out_pdf}")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"[OK] Saved: {out_svg}")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    run_root = str(os.environ.get("MLTRAINER_RUN_ROOT", "") or "").strip()
    analysis_root = str(os.environ.get("MLTRAINER_ANALYSIS_ROOT", "") or "").strip()

    shap_dir = find_rf_shap_dir(repo_root, run_root=run_root)
    if shap_dir is None:
        raise SystemExit("Could not locate SHAP exports under output/**/explainability")
    files = expect_files(shap_dir)
    out_dir = os.path.join(analysis_root, "figures") if analysis_root else os.path.join(repo_root, "new_plot")
    out_png = os.path.join(out_dir, "shap_summary_dependence_BestModel.png")
    out_pdf = os.path.join(out_dir, "shap_summary_dependence_BestModel.pdf")
    out_svg = os.path.join(out_dir, "shap_summary_dependence_BestModel.svg")
    compose_figure(files, out_png, out_pdf, out_svg)


if __name__ == "__main__":
    main()
