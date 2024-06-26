# -*- mode: python ; coding: utf-8 -*-
import os
block_cipher = None
a = Analysis(
    ['home.py'],
    pathex=['.'],
    binaries=[],
    datas=[("utils","utils"),("utils/model/default.yaml", 'ultralytics/cfg')],
    hiddenimports=['ultralytics', 'opencv-python','torch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Illegal Vehicle Smart Surveillance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.abspath('utils/img/icon.ico') 
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Illegal Vehicle Smart Surveillance',
)
