# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

project_root = Path(SPECPATH)
icon_path = project_root / "assets" / "app.ico"
version_file = project_root / "installer" / "windows_version_info.txt"

datas = [
    (str(project_root / "locales"), "locales"),
    (str(project_root / "interface" / "style"), "interface/style"),
]

mapping_json = project_root / "scripts" / "q1_social_science_mappings.json"
if mapping_json.exists():
    datas.append((str(mapping_json), "scripts"))

hiddenimports = [
    "shiboken6",
    "PySide6.QtSvg",
    "PySide6.QtOpenGLWidgets",
] + collect_submodules("sklearn")

excludes = [
    "tkinter",
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "pydoc",
    "setuptools",
]

a = Analysis(
    ["run_gui.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="OmniMLGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
    version=str(version_file) if version_file.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name="OmniMLGUI",
)
