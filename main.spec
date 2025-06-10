# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('C:\\Users\\LEE\\anaconda3\\envs\\ocr\\Lib\\site-packages\\opensmile\\core\\bin\\win\\SMILEapi.dll', 'opensmile/core/bin/win'),
    ('dtrb', 'dtrb'),
    ('yolov5', 'yolov5'),
    ('best_full_kobert_epoch2', 'best_full_kobert_epoch2'),
    ('epoch12_final.pth', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
