# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for whisper2text.

Collects pywhispercpp (with CUDA native libs), evdev, webrtcvad, pyaudio,
and all project submodules. UPX is disabled because it corrupts .so files.
"""

import os
import glob
from PyInstaller.utils.hooks import collect_all, collect_submodules

# ── Data files ──────────────────────────────────────────────────────────
datas = [
    ('icon.png', '.'),
    ('icon_recording.png', '.'),
]

# ── Binary collection ───────────────────────────────────────────────────
binaries = []

# pywhispercpp + its .libs/ directory (CUDA, ggml, whisper shared libs)
pwcpp_datas, pwcpp_bins, pwcpp_hiddens = collect_all('pywhispercpp')
datas += pwcpp_datas
binaries += pwcpp_bins

# Also explicitly grab pywhispercpp.libs/* in case collect_all misses them
import pywhispercpp as _pwcpp
_pwcpp_root = os.path.dirname(_pwcpp.__file__)
_pwcpp_libs = os.path.join(os.path.dirname(_pwcpp_root), 'pywhispercpp.libs')
if os.path.isdir(_pwcpp_libs):
    for so_file in glob.glob(os.path.join(_pwcpp_libs, '*')):
        binaries.append((so_file, 'pywhispercpp.libs'))

# evdev native extensions
evdev_datas, evdev_bins, evdev_hiddens = collect_all('evdev')
datas += evdev_datas
binaries += evdev_bins

# webrtcvad native extension
webrtcvad_datas, webrtcvad_bins, webrtcvad_hiddens = collect_all('webrtcvad')
datas += webrtcvad_datas
binaries += webrtcvad_bins

# pyaudio native extension
pyaudio_datas, pyaudio_bins, pyaudio_hiddens = collect_all('pyaudio')
datas += pyaudio_datas
binaries += pyaudio_bins

# ── Hidden imports ──────────────────────────────────────────────────────
hiddenimports = (
    pwcpp_hiddens
    + evdev_hiddens
    + webrtcvad_hiddens
    + pyaudio_hiddens
    + collect_submodules('config')
    + collect_submodules('engine')
    + collect_submodules('audio')
    + collect_submodules('ui')
    + ['pyperclip']
)

# ── Analysis ────────────────────────────────────────────────────────────
a = Analysis(
    ['whisper2text.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/rthook_cuda.py'],
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
    name='whisper2text',
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
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='whisper2text',
)
