"""Centralized path resolution for whisper2text.

Handles both normal source runs and PyInstaller frozen bundles
(where resources are extracted to sys._MEIPASS).
"""

import os
import sys

# Application directory: sys._MEIPASS when frozen, project root otherwise
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    APP_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# User data directory (persists across installs/upgrades)
USER_DATA_DIR = os.path.join(os.path.expanduser('~'), '.whisper2text')

# Icon paths (bundled with the app)
ICON_NORMAL = os.path.join(APP_DIR, 'icon.png')
ICON_RECORDING = os.path.join(APP_DIR, 'icon_recording.png')
ICON_TRAY_NORMAL = os.path.join(APP_DIR, 'icon_tray.png')
ICON_TRAY_RECORDING = os.path.join(APP_DIR, 'icon_recording_tray.png')

# Dash icon — mutable file that .desktop points to, overwritten at runtime
INSTALL_DIR = os.path.join(os.path.expanduser('~'), '.local', 'share', 'whisperLocal')
ICON_DASH = os.path.join(INSTALL_DIR, 'whisper2text.png')

# Default models directory (in user data, not in app bundle)
DEFAULT_MODELS_DIR = os.path.join(USER_DATA_DIR, 'models')
