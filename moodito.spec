# PyInstaller spec for Moodito — builds a standalone macOS menu bar .app.
# Build with:  pyinstaller moodito.spec
import os

from PyInstaller.utils.hooks import collect_all

# Version comes from the release tag (set by CI as MOODITO_VERSION, e.g. "1.0.0");
# falls back to a sensible default for local builds.
VERSION = os.environ.get("MOODITO_VERSION", "1.5.1")

# Bundle MediaPipe's native libraries and data files (.tflite, .binarypb, etc.).
mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=mp_binaries,
    datas=mp_datas + [("assets/moodito.png", "assets"), ("assets/bmc_qr.png", "assets")],
    hiddenimports=mp_hiddenimports + ["AVFoundation", "objc"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Moodito",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Moodito",
)

app = BUNDLE(
    coll,
    name="Moodito.app",
    icon="assets/moodito.icns",
    bundle_identifier="com.moodito.app",
    info_plist={
        # Menu-bar-only app: no Dock icon, no main window.
        "LSUIElement": True,
        # Required so macOS can prompt for and grant camera access.
        "NSCameraUsageDescription": "Moodito uses the camera to detect your facial expression.",
        "CFBundleName": "Moodito",
        "CFBundleDisplayName": "Moodito",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
    },
)
