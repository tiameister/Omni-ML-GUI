import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


def _get_bundle_dir() -> Path:
    """
    Return the directory containing read-only bundled resources.

    In a PyInstaller one-file build sys._MEIPASS points to the temporary
    extraction directory where data files (locales, QSS, …) are unpacked.
    In a normal source install it is the project root (parent of utils/).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def _get_writable_root() -> Path:
    """
    Return the directory that should contain writable output (runs, exports…).

    In a PyInstaller build we write next to the executable so the user can
    find the results without digging into a temp directory.
    In a source install it is the project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Project root for resource resolution (read-only assets)."""
    return _get_bundle_dir()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dataset_candidates() -> list[Path]:
    env_path = os.environ.get("DATASET_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    root = get_project_root()
    candidates.extend([
        root / "dataset" / "data_cleaned.csv",
        root / "dataset" / "data.csv",
    ])
    return candidates


def resolve_dataset_path(strict: bool = False) -> Path:
    for candidate in get_dataset_candidates():
        if candidate.exists():
            return candidate

    if strict:
        raise FileNotFoundError("No dataset file found. Checked DATASET_PATH and dataset/data_cleaned.csv, dataset/data.csv")

    # Return primary default for caller-side fallback messaging.
    return get_project_root() / "dataset" / "data.csv"


def get_output_root(output_dir: str = "output", run_tag: str | None = None) -> Path:
    # Use the writable root so that frozen builds write next to the .exe,
    # not inside PyInstaller's temporary extraction directory.
    root = _get_writable_root() / output_dir
    if run_tag:
        root = root / run_tag
    return ensure_directory(root)


def get_versioned_output_folder(base_name: str, output_dir: str = "output", run_tag: str | None = None) -> Path:
    root = get_output_root(output_dir=output_dir, run_tag=run_tag)
    folder = root / f"{base_name}_output"
    if not folder.exists():
        return folder

    idx = 1
    while True:
        candidate = root / f"{base_name}_output_v{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def safe_folder_name(value: str, fallback: str = "item") -> str:
    """Return a filesystem-friendly folder token (keeps alnum, '_' and '-')."""
    value = str(value or "").strip()
    if not value:
        return fallback
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def make_run_id(prefix: str | None = None) -> str:
    """Generate a human-readable, unique run id."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    base = f"{ts}_{rand}"
    if prefix:
        prefix = safe_folder_name(prefix, fallback="run")
        return f"{prefix}_{base}"
    return base


def get_runs_root(output_dir: str = "output", run_tag: str | None = None) -> Path:
    """Root folder that contains per-run output folders."""
    return ensure_directory(get_output_root(output_dir=output_dir, run_tag=run_tag) / "runs")


def get_run_root(run_id: str, output_dir: str = "output", run_tag: str | None = None) -> Path:
    """Return the per-run root folder (created if missing)."""
    run_id = safe_folder_name(run_id, fallback="run")
    return ensure_directory(get_runs_root(output_dir=output_dir, run_tag=run_tag) / run_id)


def get_transient_runs_root() -> Path:
    """Ephemeral root used when user disables persistent disk output."""
    return ensure_directory(Path(tempfile.gettempdir()) / "mltrainer" / "runs")


def get_transient_run_root(run_id: str) -> Path:
    run_id = safe_folder_name(run_id, fallback="run")
    return ensure_directory(get_transient_runs_root() / run_id)


def get_run_subdir(run_root: Path, name: str) -> Path:
    return ensure_directory(Path(run_root) / safe_folder_name(name, fallback="dir"))


def get_run_model_root(run_root: Path) -> Path:
    return ensure_directory(Path(run_root) / "models")


def get_run_model_dir(run_root: Path, model_name: str) -> Path:
    return ensure_directory(get_run_model_root(run_root) / safe_folder_name(model_name, fallback="model"))


def get_supplements_root(
    *,
    run_root: str | Path | None = None,
    output_dir: str = "output",
    run_tag: str | None = None,
) -> Path:
    """
    Resolve canonical supplements root.

    Priority:
    1) Explicit run_root -> <run_root>/analysis/supplements
    2) Env MLTRAINER_SUPPLEMENTS_ROOT
    3) <output_root>/supplements
    """
    if run_root is not None:
        return ensure_directory(Path(run_root) / "analysis" / "supplements")

    env_value = str(os.environ.get("MLTRAINER_SUPPLEMENTS_ROOT", "")).strip()
    if env_value:
        return ensure_directory(Path(env_value))

    return ensure_directory(get_output_root(output_dir=output_dir, run_tag=run_tag) / "supplements")

# Common predefined output folder names to avoid hardcoding across scripts
EVALUATION_DIR = "1_Overall_Evaluation"
FEATURE_IMP_DIR = "2_Feature_Importance"
MANUSCRIPT_DIR = "3_Manuscript_Figures"
MODELS_DIR = "models"
EXPLAINABILITY_DIR = "explainability"

def get_latest_output_matches(base_path: str, sub_dir: str, file_pattern: str) -> list[str]:
    """Helper to find files in dynamic output folders (e.g. '*_output*')."""
    import glob
    import os
    pattern = os.path.join(base_path, "*_output*", sub_dir, file_pattern)
    return sorted(glob.glob(pattern))


def ensure_outdir(p="output"):
    os.makedirs(p, exist_ok=True)
    return p
