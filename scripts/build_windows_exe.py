"""
Build a Windows executable for Omni-ML-GUI using PyInstaller.

Usage examples (run on Windows):
  python scripts/build_windows_exe.py --mode fast
  python scripts/build_windows_exe.py --mode single
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_fast(root: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(root / "omni_ml_gui_fast.spec"),
        ],
        cwd=root,
    )


def _build_single_file(root: Path, use_upx: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "OmniMLGUI",
        "--optimize",
        "2",
        "--strip",
        "--hidden-import",
        "shiboken6",
        "--hidden-import",
        "PySide6.QtSvg",
        "--hidden-import",
        "PySide6.QtOpenGLWidgets",
        "--collect-submodules",
        "sklearn",
        "--add-data",
        f"{root / 'locales'};locales",
        "--add-data",
        f"{root / 'interface' / 'style'};interface/style",
        "--exclude-module",
        "tkinter",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "jupyter",
        "--exclude-module",
        "notebook",
        str(root / "run_gui.py"),
    ]

    mapping_json = root / "scripts" / "q1_social_science_mappings.json"
    if mapping_json.exists():
        cmd.extend(["--add-data", f"{mapping_json};scripts"])

    icon_path = root / "assets" / "app.ico"
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])

    version_file = root / "installer" / "windows_version_info.txt"
    if version_file.exists():
        cmd.extend(["--version-file", str(version_file)])

    if not use_upx:
        cmd.append("--noupx")

    _run(cmd, cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight Windows executable for Omni-ML-GUI."
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "single"),
        default="fast",
        help="fast=onedir fastest startup, single=onefile smaller distribution unit",
    )
    parser.add_argument(
        "--upx",
        action="store_true",
        help="Use UPX compression when available (smaller exe, possibly slower startup).",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete old build/ and dist/ folders before build.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if platform.system().lower() != "windows":
        print(
            "Warning: .exe builds are only produced on Windows. "
            "Run this command on a Windows machine.",
            file=sys.stderr,
        )

    if args.clean_output:
        for folder_name in ("build", "dist"):
            folder = root / folder_name
            if folder.exists():
                shutil.rmtree(folder)
                print(f"Deleted {folder}")

    if args.mode == "fast":
        _build_fast(root)
    else:
        _build_single_file(root, use_upx=args.upx)

    print("\nBuild complete.")
    if args.mode == "fast":
        print(f"Output: {root / 'dist' / 'OmniMLGUI' / 'OmniMLGUI.exe'}")
    else:
        print(f"Output: {root / 'dist' / 'OmniMLGUI.exe'}")


if __name__ == "__main__":
    main()
