"""
Build a Windows installer (Setup.exe) for Omni-ML-GUI using Inno Setup.

Expected workflow on Windows:
  1) python scripts/build_windows_exe.py --mode fast --clean-output
  2) python scripts/build_windows_installer.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_build_meta(root: Path) -> dict[str, str]:
    meta_path = root / "build_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing build metadata file: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh) or {}

    required = ("app_name", "exe_name", "version", "publisher", "url", "copyright")
    missing = [k for k in required if not str(meta.get(k, "")).strip()]
    if missing:
        raise ValueError(f"build_meta.json is missing required keys: {', '.join(missing)}")
    return {k: str(v).strip() for k, v in meta.items()}


def _find_iscc() -> str | None:
    # Most common Inno Setup command names/locations on Windows.
    candidates = [
        "iscc",
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Omni-ML-GUI Windows installer.")
    parser.add_argument(
        "--skip-exe-check",
        action="store_true",
        help="Skip check for dist/OmniMLGUI/OmniMLGUI.exe (not recommended).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    meta = _load_build_meta(root)
    iss_script = root / "installer" / "OmniMLGUI.iss"
    dist_exe = root / "dist" / meta["exe_name"] / f"{meta['exe_name']}.exe"

    if platform.system().lower() != "windows":
        print(
            "Warning: installer builds should be run on Windows where Inno Setup is installed.",
            file=sys.stderr,
        )

    if not iss_script.exists():
        raise FileNotFoundError(f"Inno Setup script not found: {iss_script}")

    if not args.skip_exe_check and not dist_exe.exists():
        raise FileNotFoundError(
            f"Expected EXE not found at dist/{meta['exe_name']}/{meta['exe_name']}.exe. "
            "Run: python scripts/build_windows_exe.py --mode fast --clean-output"
        )

    iscc = _find_iscc()
    if not iscc:
        raise RuntimeError(
            "Inno Setup compiler (ISCC.exe) not found.\n"
            "Install Inno Setup 6: https://jrsoftware.org/isinfo.php"
        )

    cmd = [
        iscc,
        f"/DAppName={meta['app_name']}",
        f"/DAppVersion={meta['version']}",
        f"/DAppPublisher={meta['publisher']}",
        f"/DAppURL={meta['url']}",
        f"/DAppExeName={meta['exe_name']}.exe",
        f"/DAppCopyright={meta['copyright']}",
        str(iss_script),
    ]
    _run(cmd, cwd=root)
    print("\nInstaller build complete.")
    setup_file = f"{meta['exe_name']}-Setup.exe"
    print(f"Output: {root / 'dist' / setup_file}")


if __name__ == "__main__":
    main()
