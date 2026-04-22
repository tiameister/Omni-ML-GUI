"""
Build a Windows installer (Setup.exe) for Omni-ML-GUI using Inno Setup.

Expected workflow on Windows:
  1) python scripts/build_windows_exe.py --mode fast --clean-output
  2) python scripts/build_windows_installer.py
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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
    iss_script = root / "installer" / "OmniMLGUI.iss"
    dist_exe = root / "dist" / "OmniMLGUI" / "OmniMLGUI.exe"

    if platform.system().lower() != "windows":
        print(
            "Warning: installer builds should be run on Windows where Inno Setup is installed.",
            file=sys.stderr,
        )

    if not iss_script.exists():
        raise FileNotFoundError(f"Inno Setup script not found: {iss_script}")

    if not args.skip_exe_check and not dist_exe.exists():
        raise FileNotFoundError(
            "Expected EXE not found at dist/OmniMLGUI/OmniMLGUI.exe. "
            "Run: python scripts/build_windows_exe.py --mode fast --clean-output"
        )

    iscc = _find_iscc()
    if not iscc:
        raise RuntimeError(
            "Inno Setup compiler (ISCC.exe) not found.\n"
            "Install Inno Setup 6: https://jrsoftware.org/isinfo.php"
        )

    _run([iscc, str(iss_script)], cwd=root)
    print("\nInstaller build complete.")
    print(f"Output: {root / 'dist' / 'OmniMLGUI-Setup.exe'}")


if __name__ == "__main__":
    main()
