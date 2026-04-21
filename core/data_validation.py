"""Strict pre-flight validation for the training input.

The training pipeline already cleans up several issues internally
(NaN target rows, ridiculously high-cardinality categoricals, etc.),
but anything detected *after* the worker thread is launched produces
the worst possible UX: a frozen progress bar and a cryptic error log.
This module exists to fail fast in the GUI thread instead.

Two outcomes are possible:

* ``DataValidationReport.is_blocking`` is True
    The GUI must abort the training launch and display the report to
    the user (typically via ``ErrorDialog`` or a critical
    ``QMessageBox``).
* otherwise the report may still carry warnings the GUI can surface;
  training can proceed (typically after a confirm dialog).

Design notes
------------
* No Qt imports here. The validator must remain importable from CLI
  scripts and unit tests.
* All checks are O(n_features) or cheaper; they're cheap enough to
  run synchronously on click.
* Memory thresholds are deliberately conservative — they exist to
  protect a typical 16 GB workstation from an OOM hang during a long
  cross-validated training run, not to be tight bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from exceptions import DataValidationError


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Estimated working-set above which we *warn* the user before training.
MEMORY_WARN_BYTES = 1_500_000_000  # ~1.5 GB

#: Estimated working-set above which we *block* training to avoid OOM.
#: CV multiplies the working set across folds and joblib workers, so the
#: real peak is several times the raw matrix size.
MEMORY_BLOCK_BYTES = 6_000_000_000  # ~6 GB

#: Minimum non-NaN target rows required to attempt training at all.
MIN_ROWS_FOR_TRAINING = 20


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class DataValidationReport:
    """Structured result of :func:`validate_training_input`.

    Attributes
    ----------
    errors:
        Hard failures. If non-empty, training must not start.
    warnings:
        Non-fatal issues. The GUI typically asks the user to confirm
        before proceeding.
    info:
        Purely informational notes (kept separate so the dialog can
        de-emphasise them visually).
    estimated_bytes:
        Estimated in-memory size of the selected (target + features)
        matrix, computed via ``DataFrame.memory_usage(deep=True)``.
    memory_block / memory_warn:
        Convenience flags so callers can react specifically to memory
        pressure (e.g. show a different icon).
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    estimated_bytes: int = 0
    memory_block: bool = False
    memory_warn: bool = False

    @property
    def is_blocking(self) -> bool:
        return bool(self.errors) or self.memory_block

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def render(self) -> str:
        """Return a single human-readable string suitable for a modal."""
        sections: list[str] = []
        if self.errors:
            sections.append("Errors:\n• " + "\n• ".join(self.errors))
        if self.warnings:
            sections.append("Warnings:\n• " + "\n• ".join(self.warnings))
        if self.info:
            sections.append("Notes:\n• " + "\n• ".join(self.info))
        return "\n\n".join(sections) if sections else "All checks passed."

    def raise_if_blocking(self) -> None:
        """Convenience for non-GUI callers."""
        if self.is_blocking:
            raise DataValidationError(self.render())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def validate_training_input(
    df: pd.DataFrame | None,
    target: str | None,
    features: Sequence[str] | None,
) -> DataValidationReport:
    """Run a strict sanity pass on ``(df, target, features)``.

    The function is intentionally cheap — purely structural and dtype
    checks — so it can run synchronously in the GUI thread before
    spawning the training worker.
    """
    report = DataValidationReport()

    # ---- structural preconditions ----
    if df is None or not isinstance(df, pd.DataFrame):
        report.errors.append("No dataset is loaded.")
        return report

    if df.empty:
        report.errors.append("The loaded dataset is empty.")
        return report

    if target is None or not str(target).strip():
        report.errors.append("No target variable has been selected.")
        return report

    if not features:
        report.errors.append("No feature variables have been selected.")
        return report

    if target in features:
        # Soft notice — training_runner removes it automatically, but the
        # user should know.
        report.warnings.append(
            f"Target '{target}' was found in the feature list and will "
            "be removed automatically before training."
        )

    if target not in df.columns:
        report.errors.append(f"Target column '{target}' is not present in the dataset.")

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        shown = ", ".join(repr(c) for c in missing_cols[:10])
        suffix = " …" if len(missing_cols) > 10 else ""
        report.errors.append(f"Selected feature(s) not present in the dataset: {shown}{suffix}")

    if report.errors:
        return report

    feats = [c for c in features if c != target]

    # ---- target validation ----
    target_series = df[target]
    if not pd.api.types.is_numeric_dtype(target_series):
        report.errors.append(
            f"Target '{target}' must be numeric for regression. "
            f"Detected dtype: {target_series.dtype}."
        )
    else:
        non_na = int(target_series.notna().sum())
        if non_na < MIN_ROWS_FOR_TRAINING:
            report.errors.append(
                f"Target '{target}' has only {non_na} usable (non-missing) row(s); "
                f"at least {MIN_ROWS_FOR_TRAINING} are required to train."
            )
        if non_na > 0:
            arr = pd.to_numeric(target_series, errors="coerce").to_numpy()
            finite_mask = np.isfinite(arr) | np.isnan(arr)
            if not finite_mask.all():
                report.errors.append(
                    f"Target '{target}' contains infinite values. "
                    "Please clean ±inf entries before training."
                )

    # ---- per-feature validation ----
    for col in feats:
        s = df[col]
        if s.isna().all():
            report.errors.append(f"Feature '{col}' is entirely missing (all NaN).")
            continue

        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy()
            finite_mask = np.isfinite(arr) | np.isnan(arr)
            if not finite_mask.all():
                report.errors.append(
                    f"Feature '{col}' contains infinite values; "
                    "replace ±inf entries before training."
                )
            continue

        sample = s.dropna()
        if len(sample) == 0:
            continue
        try:
            sample.astype(str)
        except Exception as exc:  # extremely defensive; pandas almost always succeeds
            report.errors.append(
                f"Feature '{col}' could not be coerced to string ({exc}); "
                "the dataset appears corrupted."
            )
            continue

        n_unique = int(sample.nunique())
        uniq_ratio = n_unique / max(len(sample), 1)
        if n_unique > 50 and uniq_ratio > 0.15:
            # Mirrors the pipeline's own auto-drop heuristic so the user
            # is informed up-front instead of seeing a surprise log line.
            report.warnings.append(
                f"Feature '{col}' has {n_unique} distinct categorical values "
                "and will be dropped automatically to prevent dimensionality explosion."
            )

    # ---- memory estimate (selected matrix only) ----
    try:
        usage = df[[target, *feats]].memory_usage(deep=True, index=False)
        estimated = int(usage.sum())
    except Exception:
        estimated = int(df.memory_usage(deep=True, index=False).sum())
    report.estimated_bytes = estimated

    if estimated >= MEMORY_BLOCK_BYTES:
        report.memory_block = True
        report.errors.append(
            f"Selected matrix is ~{estimated / 1e9:.1f} GB which exceeds the safe "
            "training threshold. Reduce the feature set or sample the dataset before continuing."
        )
    elif estimated >= MEMORY_WARN_BYTES:
        report.memory_warn = True
        report.warnings.append(
            f"Selected matrix is ~{estimated / 1e9:.2f} GB. Training may be slow "
            "and could exhaust memory on smaller machines."
        )

    return report


__all__ = [
    "DataValidationError",
    "DataValidationReport",
    "MEMORY_BLOCK_BYTES",
    "MEMORY_WARN_BYTES",
    "MIN_ROWS_FOR_TRAINING",
    "validate_training_input",
]
