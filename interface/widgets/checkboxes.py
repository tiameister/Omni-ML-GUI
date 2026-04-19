from config import DO_SHAP
from functools import lru_cache
import os
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
    QToolBox,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QGridLayout,
    QFrame,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap


LEGACY_PAGE_TITLES = {
    "Model Fit": "📊 Model Fit",
    "Diagnostics": "🔍 Diagnostics",
    "Explainability": "📈 Explainability",
}

_SCRIPT_CATEGORY_ORDER = [
    "insights",
    "validation",
    "xai",
    "reporting",
]

_SCRIPT_CATEGORY_TITLES = {
    "insights": "Extra Insights",
    "validation": "Validation and Robustness",
    "xai": "Advanced Explainability",
    "reporting": "Reporting and Export",
}

_OPTIONAL_ANALYSIS_OPTIONS = [
    {
        "label": "Missing Data Health Check",
        "category": "insights",
        "filename": "missingness_report.py",
        "purpose": "Build a missingness summary with MCAR-oriented checks.",
        "outputs": "run/analysis/mcar/*",
        "recommended": True,
    },
    {
        "label": "Residual Diagnostics Report",
        "category": "insights",
        "filename": "diagnostics.py",
        "purpose": "Create residual diagnostics and heteroskedasticity outputs.",
        "outputs": "run/analysis/diagnostics/*",
        "recommended": True,
    },
    {
        "label": "Calibration and Cumulative Curves",
        "category": "insights",
        "filename": "calibration_and_cumulative.py",
        "purpose": "Generate calibration and cumulative importance artifacts.",
        "outputs": "run/analysis/calibration/*, run/analysis/cumulative/*, run/analysis/figures/*",
        "recommended": True,
    },
    {
        "label": "Target Profile Snapshot",
        "category": "insights",
        "filename": "target_analysis.py",
        "purpose": "Create descriptive summaries and plots for the target variable.",
        "outputs": "output/target/*",
        "recommended": False,
    },
    {
        "label": "Validation Protocol Comparison",
        "category": "validation",
        "filename": "run_validation_compare.py",
        "purpose": "Run protocol comparison across holdout, k-fold, repeated and nested validation.",
        "outputs": "validation_compare/*",
        "recommended": False,
    },
    {
        "label": "Validation Comparison Summary Table",
        "category": "validation",
        "filename": "summarize_validation_compare.py",
        "purpose": "Summarize validation protocol metrics into publication-ready tables.",
        "outputs": "validation_compare/tables/*",
        "recommended": False,
    },
    {
        "label": "Statistical Significance Test Pack",
        "category": "validation",
        "filename": "stats_tests.py",
        "purpose": "Run corrected resampled tests with multiple-comparison control.",
        "outputs": "output/stats_tests/*",
        "recommended": False,
    },
    {
        "label": "Baseline Confidence Interval Pack",
        "category": "validation",
        "filename": "baseline_ci.py",
        "purpose": "Compute baseline confidence intervals for context benchmarks.",
        "outputs": "output/baseline/*",
        "recommended": False,
    },
    {
        "label": "SHAP Composite Figure",
        "category": "xai",
        "filename": "compose_shap_figure.py",
        "purpose": "Compose SHAP summary and dependence outputs into a single figure.",
        "outputs": "run/analysis/figures/shap_summary_dependence_BestModel.*",
        "recommended": True,
    },
    {
        "label": "XAI Consistency Analysis",
        "category": "xai",
        "filename": "xai_consistency.py",
        "purpose": "Build cross-model explainability consistency artifacts.",
        "outputs": "output/xai_consistency/*",
        "recommended": False,
    },
    {
        "label": "Rank Stability Plot",
        "category": "xai",
        "filename": "make_rank_stability_plot.py",
        "purpose": "Generate rank-distribution stability visualizations.",
        "outputs": "output/xai_consistency/*",
        "recommended": False,
    },
    {
        "label": "Rank Stability Heatmap",
        "category": "xai",
        "filename": "make_rank_stability_heatmap.py",
        "purpose": "Generate top-N inclusion probability stability visualizations.",
        "outputs": "output/xai_consistency/*",
        "recommended": False,
    },
    {
        "label": "Model Benchmark Artifact Pack",
        "category": "reporting",
        "filename": "build_model_benchmark_artifacts.py",
        "purpose": "Produce benchmark figures and confidence-interval metric tables.",
        "outputs": "exports/manuscript_exports/figure_models_r2_distribution.*, table_model_metrics_ci.*",
        "recommended": False,
    },
    {
        "label": "Manuscript Publication Guide",
        "category": "reporting",
        "filename": "generate_manuscript_guide.py",
        "purpose": "Provides a text-based guide mapping generated ML artifacts to Q1 manuscript sections.",
        "outputs": "run/README_manuscript_guide.txt",
        "recommended": True,
    },
    {
        "label": "MCAR Manuscript Table",
        "category": "reporting",
        "filename": "build_mcar_tables.py",
        "purpose": "Build the manuscript-grade MCAR summary table.",
        "outputs": "exports/manuscript_exports/table_R1_2_mcar.*",
        "recommended": False,
    },
    {
        "label": "Psychometrics Artifact Pack",
        "category": "reporting",
        "filename": "build_psychometrics.py",
        "purpose": "Create reliability and factor-analysis artifacts.",
        "outputs": "supplements/tables/psychometrics_*, supplements/figures/psychometrics_*",
        "recommended": False,
    },
]


def _scripts_root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "scripts"))


@lru_cache(maxsize=1)
def _optional_script_catalog_data():
    """Build optional user-facing extra-analysis pages.
    Returns (pages, tooltips, label_to_filename).
    """
    pages: dict[str, list[str]] = {}
    tooltips: dict[str, str] = {}
    label_to_filename: dict[str, str] = {}

    scripts_root = _scripts_root_dir()
    order = {cat: idx for idx, cat in enumerate(_SCRIPT_CATEGORY_ORDER)}
    options = sorted(
        _OPTIONAL_ANALYSIS_OPTIONS,
        key=lambda x: (order.get(str(x.get("category", "")), 999), str(x.get("label", ""))),
    )

    for option in options:
        category = str(option.get("category", ""))
        label = str(option.get("label", "")).strip()
        filename = str(option.get("filename", "")).strip()
        if not label or not filename:
            continue
        if not os.path.exists(os.path.join(scripts_root, filename)):
            continue
        page_title = _SCRIPT_CATEGORY_TITLES.get(category, "Extra Insights")
        pages.setdefault(page_title, []).append(label)
        purpose = str(option.get("purpose", "Extra analysis task."))
        outputs = str(option.get("outputs", "output/*"))
        tooltips[label] = f"{purpose}\nExpected outputs: {outputs}"
        label_to_filename[label] = filename

    return pages, tooltips, label_to_filename


def get_optional_script_pages():
    pages, tooltips, _ = _optional_script_catalog_data()
    return {k: list(v) for k, v in pages.items()}, dict(tooltips)


def get_optional_script_label_map() -> dict[str, str]:
    _, _, label_to_filename = _optional_script_catalog_data()
    return dict(label_to_filename)


def get_recommended_optional_script_labels() -> set[str]:
    available = set(get_optional_script_label_map().keys())
    return {
        str(opt.get("label", "")).strip()
        for opt in _OPTIONAL_ANALYSIS_OPTIONS
        if bool(opt.get("recommended")) and str(opt.get("label", "")).strip() in available
    }


def is_optional_script_option(name: str) -> bool:
    _, _, label_to_filename = _optional_script_catalog_data()
    return name in label_to_filename

def create_model_checkboxes():
    """
    Returns a dict of {model_name: QCheckBox} and the containing QGroupBox.
    """
    group = QGroupBox("Models")
    layout = QVBoxLayout(group)
    # Row 1: search (full width)
    header_row = QHBoxLayout()
    filter_edit = QLineEdit()
    filter_edit.setPlaceholderText("Filter models...")
    filter_edit.setClearButtonEnabled(True)

    # Add a leading magnifier icon inside the input.
    try:
        def _text_icon(text: str, *, size: int = 14, color: str = "#5B6C7B") -> QIcon:
            dpr = 1.0
            try:
                dpr = float(filter_edit.devicePixelRatioF())
            except Exception:
                pass
            pm = QPixmap(int(size * dpr), int(size * dpr))
            pm.fill(Qt.GlobalColor.transparent)
            pm.setDevicePixelRatio(dpr)

            p = QPainter(pm)
            try:
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            except Exception:
                pass
            f = QFont(filter_edit.font())
            f.setPointSize(max(9, int(size * 0.9)))
            p.setFont(f)
            p.setPen(QColor(color))
            p.drawText(pm.rect(), int(Qt.AlignmentFlag.AlignCenter), text)
            p.end()
            return QIcon(pm)

        act = QAction(_text_icon("🔍"), "", filter_edit)
        filter_edit.addAction(act, QLineEdit.ActionPosition.LeadingPosition)
    except Exception:
        pass

    header_row.addWidget(filter_edit)
    layout.addLayout(header_row)

    # Row 2: quick presets
    preset_grid = QGridLayout()
    preset_grid.setHorizontalSpacing(6)
    preset_grid.setVerticalSpacing(6)
    btn_fast = QPushButton("Fast")
    btn_balanced = QPushButton("Balanced")
    btn_linear = QPushButton("Linear")
    btn_ensemble = QPushButton("Ensemble")
    btn_robust = QPushButton("Robust")
    btn_full = QPushButton("Full")
    btn_clear = QPushButton("Clear")

    chip_buttons = [btn_fast, btn_balanced, btn_linear, btn_ensemble, btn_robust, btn_full]
    for b in chip_buttons:
        b.setProperty("chip", True)
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        try:
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        except Exception:
            pass

    btn_clear.setProperty("chip", True)
    btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
    try:
        btn_clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    except Exception:
        pass

    preset_group = QButtonGroup(group)
    try:
        preset_group.setExclusive(True)
    except Exception:
        pass
    for b in chip_buttons:
        preset_group.addButton(b)
    btn_fast.setMinimumWidth(62)
    btn_balanced.setMinimumWidth(80)
    btn_linear.setMinimumWidth(72)
    btn_ensemble.setMinimumWidth(84)
    btn_robust.setMinimumWidth(72)
    btn_full.setMinimumWidth(62)
    btn_clear.setMinimumWidth(62)
    preset_grid.addWidget(btn_fast, 0, 0)
    preset_grid.addWidget(btn_balanced, 0, 1)
    preset_grid.addWidget(btn_linear, 0, 2)
    preset_grid.addWidget(btn_ensemble, 1, 0)
    preset_grid.addWidget(btn_robust, 1, 1)
    preset_grid.addWidget(btn_full, 1, 2)
    preset_grid.addWidget(btn_clear, 2, 0, 1, 3)
    layout.addLayout(preset_grid)
    # Margins and spacing
    m = layout.contentsMargins()
    layout.setContentsMargins(max(8, m.left()), 12, max(8, m.right()), max(8, m.bottom()))
    layout.setSpacing(5)
    checks = {}
    tooltips = {
        "LinearRegression": "Ordinary Least Squares linear regression.",
        "RidgeCV": "Ridge regression with built-in cross-validation to select alpha.",
        "RandomForest": "Ensemble of decision trees; robust to non-linearities and outliers.",
        "HistGB": "Histogram-based Gradient Boosting (fast, efficient boosting).",
        "GradientBoostingRegressor": "Gradient boosting regression trees (scikit-learn).",
        "Lasso": "L1-regularized linear regression; performs feature selection.",
        "ElasticNet": "Combination of L1 and L2 regularization; balances Lasso and Ridge.",
        "SVR": "Support Vector Regression with kernel trick for non-linear relationships.",
        "KNeighborsRegressor": "Non-parametric regression based on nearest neighbors.",
        "XGBoost": "Extreme Gradient Boosting (requires xgboost package)."
    }
    model_order = [
        "LinearRegression","RidgeCV","RandomForest","HistGB",
        "GradientBoostingRegressor","Lasso","ElasticNet",
        "SVR","KNeighborsRegressor","XGBoost"
    ]
    recommended = {"RidgeCV", "RandomForest", "HistGB"}

    # Card grid (BigTech-style inline picker)
    model_cards: dict[str, QFrame] = {}
    meta = {
        "LinearRegression": {"title": "Linear Regression", "tag": "Fast", "desc": "Baseline linear model"},
        "RidgeCV": {"title": "Ridge (CV)", "tag": "Recommended", "desc": "Regularized linear model with CV"},
        "RandomForest": {"title": "Random Forest", "tag": "Robust", "desc": "Non-linear ensemble (trees)"},
        "HistGB": {"title": "Hist Gradient Boost", "tag": "Fast", "desc": "Efficient boosting (strong default)"},
        "GradientBoostingRegressor": {"title": "Gradient Boosting", "tag": "Accurate", "desc": "Classic boosting regressor"},
        "Lasso": {"title": "Lasso", "tag": "Sparse", "desc": "L1 regularization (feature selection)"},
        "ElasticNet": {"title": "Elastic Net", "tag": "Balanced", "desc": "L1/L2 mix regularization"},
        "SVR": {"title": "SVR", "tag": "Flexible", "desc": "Kernel regression (can be slower)"},
        "KNeighborsRegressor": {"title": "KNN Regressor", "tag": "Simple", "desc": "Nearest-neighbors regression"},
        "XGBoost": {"title": "XGBoost", "tag": "Boosted", "desc": "High-performance gradient boosting"},
    }

    grid = QGridLayout()
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)

    cols = 2
    for idx, name in enumerate(model_order):
        info = meta.get(name, {"title": name, "tag": "", "desc": ""})

        card = QFrame()
        card.setObjectName("modelCard")
        card.setProperty("selected", name in recommended)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.setCursor(Qt.CursorShape.PointingHandCursor)

        v = QVBoxLayout(card)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        title = QLabel(str(info.get("title", name)))
        title.setObjectName("modelCardTitle")
        title.setToolTip(tooltips.get(name, name))

        tag = QLabel(str(info.get("tag", "")).strip())
        tag.setObjectName("modelCardTag")
        tag.setVisible(bool(tag.text()))

        toggle = QPushButton("Select")
        toggle.setCheckable(True)
        toggle.setChecked(name in recommended)
        toggle.setObjectName("modelToggle")
        toggle.setToolTip(tooltips.get(name, name))

        def _sync_toggle_text(state: bool, btn=toggle, fr=card):
            btn.setText("Selected" if state else "Select")
            fr.setProperty("selected", bool(state))
            fr.style().unpolish(fr)
            fr.style().polish(fr)

        toggle.toggled.connect(_sync_toggle_text)
        _sync_toggle_text(toggle.isChecked())

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_row.addWidget(title)
        title_row.addWidget(tag)
        title_row.addStretch(1)

        top.addLayout(title_row, 1)
        top.addWidget(toggle, 0)
        v.addLayout(top)

        desc = QLabel(str(info.get("desc", "")).strip())
        desc.setObjectName("modelCardDesc")
        desc.setWordWrap(True)
        desc.setVisible(bool(desc.text()))
        v.addWidget(desc)

        def _card_click(event, btn=toggle):
            try:
                btn.toggle()
            except Exception:
                pass
            try:
                event.accept()
            except Exception:
                pass

        card.mousePressEvent = _card_click

        model_cards[name] = card
        checks[name] = toggle

        r = idx // cols
        c = idx % cols
        grid.addWidget(card, r, c)

    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    layout.addLayout(grid)

    # Wire bulk presets
    def set_all(state: bool):
        for cb in checks.values():
            cb.setChecked(state)

    def apply_filter(text: str):
        t = (text or '').strip().lower()
        for name, fr in model_cards.items():
            fr.setVisible((t in name.lower()) if t else True)

    # Debounce filter updates to keep typing responsive on large UIs.
    _filter_timer = QTimer(group)
    _filter_timer.setSingleShot(True)
    _filter_timer.setInterval(120)

    def _apply_filter_now():
        apply_filter(filter_edit.text())

    _filter_timer.timeout.connect(_apply_filter_now)
    filter_edit.textChanged.connect(lambda _t: _filter_timer.start())
    _apply_filter_now()
    group._filter_timer = _filter_timer

    def apply_preset(names: set[str]):
        for name, cb in checks.items():
            cb.setChecked(name in names)

    def _clear_preset_checks():
        try:
            preset_group.setExclusive(False)
        except Exception:
            pass
        for b in chip_buttons:
            try:
                b.setChecked(False)
            except Exception:
                pass
        try:
            preset_group.setExclusive(True)
        except Exception:
            pass

    btn_fast.setToolTip("Quick baseline set for fastest iteration.")
    btn_balanced.setToolTip("Balanced speed/quality starter set.")
    btn_linear.setToolTip("Linear-family models (interpretable and regularized).")
    btn_ensemble.setToolTip("Tree ensemble models for non-linear patterns.")
    btn_robust.setToolTip("Strong mixed set for robust benchmark comparisons.")
    btn_full.setToolTip("Enable all available models.")
    btn_clear.setToolTip("Disable all models and start from scratch.")

    preset_linear = {"LinearRegression", "RidgeCV", "Lasso", "ElasticNet"}
    preset_ensemble = {"RandomForest", "HistGB", "GradientBoostingRegressor", "XGBoost"}
    preset_robust = {"RidgeCV", "RandomForest", "HistGB", "XGBoost"}
    preset_full = set(model_order)

    btn_fast.clicked.connect(lambda: apply_preset({"LinearRegression", "RidgeCV", "HistGB"}))
    btn_balanced.clicked.connect(lambda: apply_preset(recommended))
    btn_linear.clicked.connect(lambda: apply_preset(preset_linear))
    btn_ensemble.clicked.connect(lambda: apply_preset(preset_ensemble))
    btn_robust.clicked.connect(lambda: apply_preset(preset_robust))
    btn_full.clicked.connect(lambda: apply_preset(preset_full))
    btn_clear.clicked.connect(lambda: (set_all(False), _clear_preset_checks()))

    # expose a few handles for inline UX
    group.filter_edit = filter_edit
    group.preset_buttons = {
        "fast": btn_fast,
        "balanced": btn_balanced,
        "linear": btn_linear,
        "ensemble": btn_ensemble,
        "robust": btn_robust,
        "full": btn_full,
        "clear": btn_clear,
    }

    return checks, group

def get_plot_pages():
    """Return the structured pages dict and tooltips mapping for plots.
    Keeping this separate lets other dialogs reuse the same structure.
    """
    pages = {
        "Model Fit": [
            "Residuals",
            "Predictions vs Actual",
            "Learning Curve",
        ],
        "Diagnostics": [
            "Q-Q Plot",
            "Residual Distribution",
            "Correlation Matrix",
        ],
        "Explainability": [
            "Feature Importance",
            "Feature Importance Heatmap",
        ],
    }
    if DO_SHAP:
        pages["Explainability"] += ["SHAP Summary", "SHAP Dependence"]

    script_pages, script_tooltips = get_optional_script_pages()
    for page_title, items in script_pages.items():
        if items:
            pages[page_title] = items

    tooltips = {
        "Residuals": "Scatter of residuals vs predictions to inspect bias/variance patterns.",
        "Residual Distribution": "Histogram/KDE of residuals to assess normality and spread.",
        "Q-Q Plot": "Quantile-Quantile plot comparing residuals to a normal distribution.",
        "Correlation Matrix": "Heatmap of feature correlations; helps detect multicollinearity.",
        "Feature Importance": "Model-derived importance scores indicating influential features.",
        "Predictions vs Actual": "Parity plot to compare predicted vs true target values.",
        "Learning Curve": "Train/test score vs training size to assess bias/variance.",
        "Feature Importance Heatmap": "Heatmap visualization of feature importances across runs/models.",
        "SHAP Summary": "SHAP beeswarm and bar summary of feature contributions.",
        "SHAP Dependence": "SHAP dependence plots for top features.",
    }
    tooltips.update(script_tooltips)
    return pages, tooltips


def apply_settings_to_checks(
    title: str,
    checks: list[QCheckBox],
    settings: QSettings,
    persist_live: bool = True,
):
    """Initialize checkboxes from QSettings and optionally persist toggles live."""
    script_labels = set(get_optional_script_label_map().keys())
    for chk in checks:
        key = f"plots/{title}/{chk.text()}"
        val = settings.value(key, None)
        if val is None and title in LEGACY_PAGE_TITLES:
            legacy_key = f"plots/{LEGACY_PAGE_TITLES[title]}/{chk.text()}"
            val = settings.value(legacy_key, None)
        if val is None:
            # Keep regular plots enabled by default, but extra analyses disabled.
            chk.setChecked(chk.text() not in script_labels)
        else:
            chk.setChecked(str(val).lower() in ("true", "1", "yes"))
        if persist_live:
            chk.toggled.connect(lambda state, k=key, s=settings: s.setValue(k, bool(state)))


def create_plot_checkboxes():
    """
    Returns a dict of {plot_name: QCheckBox} and a QToolBox with 4 collapsible pages.
    Includes SHAP plots if DO_SHAP=True in config.py.
    Persists selections via QSettings.
    """
    settings = QSettings()
    pages, tooltips = get_plot_pages()

    toolbox = QToolBox()
    checks: dict[str, QCheckBox] = {}

    def make_page(title: str, items: list[str]) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Deselect All")
        btn_row.addWidget(btn_all); btn_row.addWidget(btn_none); btn_row.addStretch()
        vbox.addLayout(btn_row)

        # Checkboxes
        local_checks = []
        for nm in items:
            chk = QCheckBox(nm)
            chk.setToolTip(tooltips.get(nm, nm))
            vbox.addWidget(chk)
            checks[nm] = chk
            local_checks.append(chk)

        # Apply settings and persistence
        apply_settings_to_checks(title, local_checks, settings)

        # Wire select all/none
        def set_all(state: bool):
            for c in local_checks:
                c.setChecked(state)
        btn_all.clicked.connect(lambda: set_all(True))
        btn_none.clicked.connect(lambda: set_all(False))

        vbox.addStretch()
        return page

    for title, items in pages.items():
        toolbox.addItem(make_page(title, items), title)

    return checks, toolbox