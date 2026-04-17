import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    # utils/paths.py -> project root is parent of utils.
    return Path(__file__).resolve().parents[1]


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
    root = get_project_root() / output_dir
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
    # Instead of creating /models/model_name inside get_run_model_dir, we just return the run_root itself.
    # The individual scripts will append ModelName where appropriate.
    return ensure_directory(run_root)
