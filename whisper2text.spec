# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for whisper2text.

Collects pywhispercpp (with CUDA native libs), evdev, pyaudio,
onnxruntime (with CUDA execution provider libs), onnx_asr, and
huggingface_hub, plus all project submodules. UPX is disabled
because it corrupts .so files.
"""

import os
import glob
from PyInstaller.utils.hooks import collect_all, collect_submodules

# ── Data files ──────────────────────────────────────────────────────────
datas = [
    ('icon.png', '.'),
    ('icon_recording.png', '.'),
    ('icon_tray.png', '.'),
    ('icon_recording_tray.png', '.'),
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

# pyaudio native extension
pyaudio_datas, pyaudio_bins, pyaudio_hiddens = collect_all('pyaudio')
datas += pyaudio_datas
binaries += pyaudio_bins

# onnxruntime (with CUDAExecutionProvider libs + bundled CUDA 12 DLLs)
ort_datas, ort_bins, ort_hiddens = collect_all('onnxruntime')
datas += ort_datas
binaries += ort_bins

# onnx_asr (Parakeet wrapper) — pulls its ONNX schema + resources
onnxasr_datas, onnxasr_bins, onnxasr_hiddens = collect_all('onnx_asr')
datas += onnxasr_datas
binaries += onnxasr_bins

# huggingface_hub (used by ModelManager to snapshot Parakeet repos)
hf_datas, hf_bins, hf_hiddens = collect_all('huggingface_hub')
datas += hf_datas
binaries += hf_bins

# hf_xet (optional — accelerates HuggingFace Xet-backed downloads).
# Non-fatal if absent; collect_all just returns empty tuples.
try:
    xet_datas, xet_bins, xet_hiddens = collect_all('hf_xet')
    datas += xet_datas
    binaries += xet_bins
except Exception:
    xet_hiddens = []

# ── Hidden imports ──────────────────────────────────────────────────────
hiddenimports = (
    pwcpp_hiddens
    + evdev_hiddens
    + pyaudio_hiddens
    + ort_hiddens
    + onnxasr_hiddens
    + hf_hiddens
    + xet_hiddens
    + collect_submodules('config')
    + collect_submodules('engine')
    + collect_submodules('audio')
    + collect_submodules('ui')
    + ['pyperclip', 'pkg_resources', 'setuptools']
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
