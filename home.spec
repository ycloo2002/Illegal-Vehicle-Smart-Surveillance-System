# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
a = Analysis(
    ['home.py'],
    pathex=['.'],
    binaries=[],
    datas=[("utils","utils"),("utils\img\icon.png","."),("utils/model/default.yaml", 'ultralytics/cfg')],
    hiddenimports=['ultralytics','numpy', 'opencv-python','torch'],
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='utils\img\icon.png'
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