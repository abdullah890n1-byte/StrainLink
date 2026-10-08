# PyInstaller recipe for the stand-alone StrainLink app (macOS and Windows).
# Build with packaging/build_mac.py or packaging\build_windows.bat, not directly.

import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from strainlink import __version__

STATIC = os.path.join(ROOT, "src", "strainlink", "app", "static")
ICONS = os.path.join(ROOT, "build", "icons")
BUNDLE_ID = "com.strainlink.app"
# Why macOS should let StrainLink read these folders (it asks the user once).
FOLDER_REASON = "StrainLink reads the photos, videos and logger files of your tests from this folder, and saves its results next to them."

analysis = Analysis(
    [os.path.join(SPECPATH, "launch.py")],
    pathex=[os.path.join(ROOT, "src")],
    datas=[(STATIC, os.path.join("strainlink", "app", "static"))],
    hiddenimports=[],
    # Not used by the app: leaving them out keeps the download smaller. The charts need only
    # plotly's JavaScript file (in the static folder), not the plotly Python package.
    excludes=["tkinter", "pytest", "IPython", "plotly", "PyQt5", "PyQt6", "PySide2", "PySide6", "notebook"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [],
    exclude_binaries=True,
    name="StrainLink",
    console=False,                      # a windowed app: no terminal behind it
    icon=os.path.join(ICONS, "StrainLink.icns" if sys.platform == "darwin" else "StrainLink.ico"),
)
collected = COLLECT(exe, analysis.binaries, analysis.datas, name="StrainLink")
if sys.platform == "darwin":
    app = BUNDLE(
        collected,
        name="StrainLink.app",
        icon=os.path.join(ICONS, "StrainLink.icns"),
        bundle_identifier=BUNDLE_ID,
        version=__version__,
        info_plist={
            "CFBundleDisplayName": "StrainLink",
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSDocumentsFolderUsageDescription": FOLDER_REASON,
            "NSDesktopFolderUsageDescription": FOLDER_REASON,
            "NSDownloadsFolderUsageDescription": FOLDER_REASON,
            "NSRemovableVolumesUsageDescription": FOLDER_REASON,
            "NSNetworkVolumesUsageDescription": FOLDER_REASON,
        },
    )
