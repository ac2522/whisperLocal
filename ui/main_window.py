"""Main application window integrating all components.

Replaces the old monolithic whisper2text.py with a cleanly separated
MainWindow widget that delegates to the engine, audio, config and UI
sub-modules.
"""

import atexit
import logging
import os
import platform
import signal
import sys
import threading
import time

import select
import subprocess

import evdev
import evdev.ecodes as ec
import pyperclip

from PyQt5.QtWidgets import (
    QApplication,
    QAction,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSignal

import shutil

from config.paths import (
    ICON_NORMAL, ICON_RECORDING,
    ICON_TRAY_NORMAL, ICON_TRAY_RECORDING,
    ICON_DASH, DEFAULT_MODELS_DIR,
)
from config.settings import SettingsManager
from engine.model_manager import ModelManager
from engine.whisper_engine import WhisperEngine
from audio.recorder import Recorder
from audio.device_manager import DeviceManager
from ui.error_panel import ErrorPanel
from ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# evdev keycode ↔ character mappings for hotkey detection
# ---------------------------------------------------------------------------
_CHAR_TO_EVDEV = {
    'a': ec.KEY_A, 'b': ec.KEY_B, 'c': ec.KEY_C, 'd': ec.KEY_D,
    'e': ec.KEY_E, 'f': ec.KEY_F, 'g': ec.KEY_G, 'h': ec.KEY_H,
    'i': ec.KEY_I, 'j': ec.KEY_J, 'k': ec.KEY_K, 'l': ec.KEY_L,
    'm': ec.KEY_M, 'n': ec.KEY_N, 'o': ec.KEY_O, 'p': ec.KEY_P,
    'q': ec.KEY_Q, 'r': ec.KEY_R, 's': ec.KEY_S, 't': ec.KEY_T,
    'u': ec.KEY_U, 'v': ec.KEY_V, 'w': ec.KEY_W, 'x': ec.KEY_X,
    'y': ec.KEY_Y, 'z': ec.KEY_Z,
    '1': ec.KEY_1, '2': ec.KEY_2, '3': ec.KEY_3, '4': ec.KEY_4,
    '5': ec.KEY_5, '6': ec.KEY_6, '7': ec.KEY_7, '8': ec.KEY_8,
    '9': ec.KEY_9, '0': ec.KEY_0,
    '`': ec.KEY_GRAVE, '-': ec.KEY_MINUS, '=': ec.KEY_EQUAL,
    '[': ec.KEY_LEFTBRACE, ']': ec.KEY_RIGHTBRACE,
    '\\': ec.KEY_BACKSLASH, ';': ec.KEY_SEMICOLON,
    "'": ec.KEY_APOSTROPHE, ',': ec.KEY_COMMA,
    '.': ec.KEY_DOT, '/': ec.KEY_SLASH,
    'space': ec.KEY_SPACE, 'enter': ec.KEY_ENTER, 'tab': ec.KEY_TAB,
    'backspace': ec.KEY_BACKSPACE, 'delete': ec.KEY_DELETE,
    'esc': ec.KEY_ESC,
    'f1': ec.KEY_F1, 'f2': ec.KEY_F2, 'f3': ec.KEY_F3,
    'f4': ec.KEY_F4, 'f5': ec.KEY_F5, 'f6': ec.KEY_F6,
    'f7': ec.KEY_F7, 'f8': ec.KEY_F8, 'f9': ec.KEY_F9,
    'f10': ec.KEY_F10, 'f11': ec.KEY_F11, 'f12': ec.KEY_F12,
}

_EVDEV_MODIFIERS = {
    ec.KEY_LEFTCTRL: 'ctrl', ec.KEY_RIGHTCTRL: 'ctrl',
    ec.KEY_LEFTALT: 'alt', ec.KEY_RIGHTALT: 'alt',
    ec.KEY_LEFTSHIFT: 'shift', ec.KEY_RIGHTSHIFT: 'shift',
    ec.KEY_LEFTMETA: 'super', ec.KEY_RIGHTMETA: 'super',
}

_EVDEV_TO_CHAR = {v: k for k, v in _CHAR_TO_EVDEV.items()}


def _find_keyboard_device():
    """Find the primary keyboard input device."""
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        caps = dev.capabilities()
        if 1 in caps and ec.KEY_A in caps[1]:
            return dev.path
    return None


class MainWindow(QWidget):
    """Main application window for Speech to Text.

    Integrates SettingsManager, ModelManager, DeviceManager, WhisperEngine,
    Recorder, ErrorPanel, and SettingsDialog into a single cohesive window.
    """

    # Signals for cross-thread communication
    hotkey_signal = pyqtSignal()
    transcript_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    recording_stopped_signal = pyqtSignal()
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Speech to Text")
        self.setWindowIcon(QIcon(ICON_NORMAL))
        self.resize(400, 600)

        self.is_quitting = False
        self.is_transcribing = False
        self._keyboard_dev_path = _find_keyboard_device()
        self.last_clicked_button = None
        self.max_transcripts = 10
        self._hotkey_device = None
        self._hotkey_stop_event = threading.Event()
        self._hotkey_thread = None

        # --- Core managers ---
        self.settings = SettingsManager()
        self.model_manager = ModelManager(DEFAULT_MODELS_DIR)
        self.device_manager = DeviceManager()

        # --- Load WhisperEngine from saved model setting ---
        self.engine = self._load_initial_engine()

        # --- Create Recorder with saved device ---
        device_index = self.settings.get('audio_device_index')
        self.recorder = Recorder(device_index=device_index)

        # --- Load transcript history ---
        self.transcripts = self.settings.get('transcripts', [])

        # --- Connect signals ---
        self.hotkey_signal.connect(self._toggle_recording)
        self.transcript_signal.connect(self._on_transcript)
        self.error_signal.connect(self._on_error)
        self.recording_stopped_signal.connect(self._on_recording_stopped)
        self.update_status_signal.connect(self._on_update_status)

        # --- Build the UI ---
        self._init_ui()

        # --- System tray ---
        self._setup_tray()

        # --- Hotkey listener ---
        self._hotkey_thread = threading.Thread(target=self._start_hotkey_listener, daemon=True)
        self._hotkey_thread.start()

        # --- Cleanup handlers ---
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        sys.excepthook = self._handle_exception

    # ------------------------------------------------------------------
    # Engine initialisation
    # ------------------------------------------------------------------

    def _load_initial_engine(self):
        """Load the WhisperEngine from the saved model setting.

        Handles the old model_size format (e.g. 'base' -> 'ggml-base.bin').
        Falls back to any available model if the saved one is not found.
        Returns None if no models are available.
        """
        self._apply_compute_backend()
        model_size = self.settings.get('model_size')

        # Handle old format: if model_size doesn't end in '.bin', convert it
        if model_size and not model_size.endswith('.bin'):
            model_size = f'ggml-{model_size}.bin'
            self.settings.set('model_size', model_size)
            self.settings.save()

        # Try to load the saved model
        if model_size:
            try:
                model_path = self.model_manager.get_model_path(model_size)
                return WhisperEngine(model_path)
            except FileNotFoundError:
                logger.warning("Saved model '%s' not found, trying fallback", model_size)
            except Exception:
                logger.error("Failed to load saved model '%s'", model_size, exc_info=True)

        # Fallback: try any available model
        downloaded = self.model_manager.list_downloaded()
        if downloaded:
            first = downloaded[0]
            logger.info("Falling back to model '%s'", first['name'])
            self.settings.set('model_size', first['name'])
            self.settings.save()
            try:
                return WhisperEngine(first['path'])
            except Exception:
                logger.error("Failed to load fallback model '%s'", first['name'], exc_info=True)

        logger.warning("No models available -- engine is None")
        return None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        """Build the main window layout."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # --- Top bar: spacer + settings gear ---
        top_bar = QHBoxLayout()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(spacer)

        settings_button = QPushButton()
        settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        settings_button.setIconSize(QSize(40, 40))
        settings_button.setFixedSize(40, 40)
        settings_button.setFlat(True)
        settings_button.setToolTip("Settings")
        settings_button.clicked.connect(self._open_settings)
        top_bar.addWidget(settings_button)

        main_layout.addLayout(top_bar)

        # --- Scroll area for transcript buttons ---
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.transcript_widget = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_widget)
        self.transcript_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.transcript_widget)

        main_layout.addWidget(self.scroll_area)

        # Populate saved transcripts
        recent = self.transcripts[-self.max_transcripts:]
        for text in recent:
            self._create_transcript_button(text)

        # --- Record button ---
        self.record_button = QPushButton('Record', self)
        self.record_button.clicked.connect(self._toggle_recording)
        main_layout.addWidget(self.record_button)

        # --- Status label ---
        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(8)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)
        self._update_status_label()

        # --- Error panel ---
        self.error_panel = ErrorPanel(self)
        main_layout.addWidget(self.error_panel)

    def _apply_compute_backend(self):
        """Set environment variables to control GPU usage based on settings."""
        backend = self.settings.get('compute_backend', 'cpu')
        if backend == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    def _detect_compute_backend(self):
        """Detect the active compute backend."""
        backend = self.settings.get('compute_backend', 'cpu')
        if backend == 'vulkan':
            return "Vulkan"
        elif backend == 'cuda':
            return "CUDA"
        return "CPU"

    def _update_status_label(self):
        """Refresh the status label with current model, compute, and mic info."""
        parts = []

        # Model name
        model_name = self.settings.get('model_size', 'None')
        parts.append(f"Model: {model_name}")

        # GPU/CPU
        parts.append(self._detect_compute_backend())

        # Mic name
        device_name = self.settings.get('audio_device_name')
        if device_name:
            parts.append(f"Mic: {device_name}")
        else:
            parts.append("Mic: System Default")

        self.status_label.setText(" | ".join(parts))

    # ------------------------------------------------------------------
    # Transcript buttons
    # ------------------------------------------------------------------

    def _get_button_style(self, selected=False):
        """Return the stylesheet string for transcript buttons."""
        base = """
        QPushButton {
            border: 1px solid blue;
            border-radius: 5px;
            text-align: left;
            padding: 5px;
            background-color: white;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        """
        if selected:
            return base + "QPushButton { background-color: rgba(0, 0, 255, 50); }"
        return base

    def _create_transcript_button(self, text):
        """Create and add a transcript button to the layout."""
        label = text[:50] + '...' if len(text) > 50 else text
        button = QPushButton(label)
        button.setToolTip(text)
        button.setStyleSheet(self._get_button_style())
        button.clicked.connect(lambda checked, b=button, t=text: self._on_transcript_click(b, t))
        self.transcript_layout.addWidget(button)
        self._highlight_button(button)

    def _highlight_button(self, button):
        """Highlight the given button and un-highlight the previous one."""
        if self.last_clicked_button:
            self.last_clicked_button.setStyleSheet(self._get_button_style(selected=False))
        button.setStyleSheet(self._get_button_style(selected=True))
        self.last_clicked_button = button

    def _on_transcript_click(self, button, text):
        """Handle a transcript button click: copy to clipboard, optionally paste."""
        try:
            pyperclip.copy(text)
            auto_paste = self.settings.get('auto_paste', False)
            if auto_paste:
                self._paste_text(text)
            self._highlight_button(button)
        except Exception as e:
            logger.error("Failed to copy/paste text: %s", e, exc_info=True)

    def _on_transcript(self, text):
        """Slot for transcript_signal: add a new transcript to the UI."""
        if not text.strip():
            return

        self.transcripts.append(text)
        self._create_transcript_button(text)

        # Enforce max transcripts limit
        while self.transcript_layout.count() > self.max_transcripts:
            item = self.transcript_layout.takeAt(0)
            widget = item.widget()
            if widget:
                if widget == self.last_clicked_button:
                    self.last_clicked_button = None
                widget.deleteLater()

        # Copy to clipboard
        try:
            pyperclip.copy(text)
            auto_paste = self.settings.get('auto_paste', False)
            if auto_paste:
                self._paste_text(text)
        except Exception as e:
            logger.error("Failed to copy/paste text: %s", e, exc_info=True)

        # Save transcripts
        self.settings.set('transcripts', self.transcripts)
        self.settings.save()

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------

    def _setup_tray(self):
        """Set up the system tray icon and its context menu."""
        self.tray_icon = QSystemTrayIcon(self)

        if os.path.exists(ICON_TRAY_NORMAL):
            self.tray_icon.setIcon(QIcon(ICON_TRAY_NORMAL))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.tray_icon.setToolTip("Speech to Text")

        # Menu
        self.tray_menu = QMenu()

        self.show_action = QAction("Show", self)
        self.show_action.triggered.connect(self.show)

        self.hide_action = QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide)

        self.record_action = QAction("Start Recording", self)
        self.record_action.triggered.connect(self._toggle_recording)

        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self._open_settings)

        self.quit_action = QAction("Quit", self)
        self.quit_action.triggered.connect(self._quit)

        self.tray_menu.addAction(self.show_action)
        self.tray_menu.addAction(self.hide_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.record_action)
        self.tray_menu.addAction(self.settings_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self._tray_activated)
        self.tray_icon.show()

    def _tray_activated(self, reason):
        """Toggle window visibility on tray icon double-click."""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def _update_tray_icon(self):
        """Update tray, window, and dash icon based on recording/transcribing state."""
        is_active = self.recorder.is_recording or self.is_transcribing
        if is_active:
            tray_path = ICON_TRAY_RECORDING
            window_path = ICON_RECORDING
            fallback = self.style().standardIcon(QStyle.SP_MediaStop)
        else:
            tray_path = ICON_TRAY_NORMAL
            window_path = ICON_NORMAL
            fallback = self.style().standardIcon(QStyle.SP_MediaPlay)

        self.tray_icon.setIcon(QIcon(tray_path) if os.path.exists(tray_path) else fallback)
        self.setWindowIcon(QIcon(window_path) if os.path.exists(window_path) else fallback)

        # Overwrite the dash icon file so GNOME picks up the change
        try:
            if os.path.exists(window_path) and os.path.exists(ICON_DASH):
                shutil.copy2(window_path, ICON_DASH)
        except Exception:
            pass

    def _update_record_action_text(self):
        """Update the tray record action text and tooltip."""
        if self.recorder.is_recording:
            self.record_action.setText("Stop Recording")
            self.tray_icon.setToolTip("Speech to Text (Recording...)")
        else:
            self.record_action.setText("Start Recording")
            self.tray_icon.setToolTip("Speech to Text")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _toggle_recording(self):
        """Start or stop recording based on the current state."""
        # If currently recording, stop it (works for both button and silence modes)
        if self.recorder.is_recording:
            self.recorder.stop()
            return

        # If transcribing, ignore (can't start a new recording yet)
        if self.is_transcribing:
            return

        if self.engine is None or not self.engine.is_loaded():
            self.error_signal.emit("No model loaded. Please download or select a model in Settings.")
            return

        recording_mode = self.settings.get('recording_mode', 'silence')

        self.record_button.setText('Stop Recording')
        self.record_button.setEnabled(True)  # Keep enabled so user can click to stop
        self.record_action.setText("Stop Recording")
        self.tray_icon.setToolTip("Speech to Text (Recording...)")
        # Set red icons immediately (before recording thread starts)
        if os.path.exists(ICON_TRAY_RECORDING):
            self.tray_icon.setIcon(QIcon(ICON_TRAY_RECORDING))
        self.setWindowIcon(QIcon(ICON_RECORDING) if os.path.exists(ICON_RECORDING)
                           else self.style().standardIcon(QStyle.SP_MediaStop))
        # Overwrite dash icon to red
        try:
            if os.path.exists(ICON_RECORDING) and os.path.exists(ICON_DASH):
                shutil.copy2(ICON_RECORDING, ICON_DASH)
        except Exception:
            pass

        if recording_mode == 'button':
            threading.Thread(target=self._record_button, daemon=True).start()
        else:
            threading.Thread(target=self._record_silence, daemon=True).start()

    def _record_button(self):
        """Button-mode recording thread: record until stop() is called."""
        try:
            audio_data = self.recorder.record_button_mode()
            self._transcribe_and_emit(audio_data)
        except Exception as e:
            logger.error("Error in button recording mode: %s", e, exc_info=True)
            self.error_signal.emit(str(e))
            self.recording_stopped_signal.emit()

    def _record_silence(self):
        """Silence-mode recording thread: record until silence is detected."""
        try:
            vad = self.settings.get('vad_aggressiveness', 1)
            break_length = self.settings.get('break_length', 5)
            audio_data = self.recorder.record_silence_mode(vad, break_length)
            self._transcribe_and_emit(audio_data)
        except Exception as e:
            logger.error("Error in silence recording mode: %s", e, exc_info=True)
            self.error_signal.emit(str(e))
            self.recording_stopped_signal.emit()

    def _transcribe_and_emit(self, audio_data):
        """Transcribe audio and emit the result. Disables record button during transcription."""
        self.is_transcribing = True
        self.recording_stopped_signal.emit()  # Update button to "Transcribing..."
        try:
            text = self.engine.transcribe(audio_data)
            if text and text.strip():
                self.transcript_signal.emit(text)
        except Exception as e:
            logger.error("Error during transcription: %s", e, exc_info=True)
            self.error_signal.emit(str(e))
        finally:
            self.is_transcribing = False
            self.recording_stopped_signal.emit()

    def _on_recording_stopped(self):
        """Slot: reset UI after recording/transcription finishes."""
        if self.is_transcribing:
            self.record_button.setText('Transcribing...')
            self.record_button.setEnabled(False)
        else:
            self.record_button.setText('Record')
            self.record_button.setEnabled(True)
        self._update_record_action_text()
        self._update_tray_icon()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _open_settings(self):
        """Open the settings dialog. On accept, apply changes."""
        old_model = self.settings.get('model_size')
        old_device = self.settings.get('audio_device_index')
        old_backend = self.settings.get('compute_backend')

        dialog = SettingsDialog(self.settings, self.model_manager, self.device_manager, parent=self)
        if dialog.exec_() == SettingsDialog.Accepted:
            new_model = self.settings.get('model_size')
            new_device = self.settings.get('audio_device_index')
            new_backend = self.settings.get('compute_backend')

            # Reload engine if model or compute backend changed
            if new_model != old_model or new_backend != old_backend:
                self._apply_compute_backend()
                threading.Thread(target=self._reload_engine, daemon=True).start()

            # Recreate recorder if device changed
            if new_device != old_device:
                self.recorder.cleanup()
                self.recorder = Recorder(device_index=new_device)

            # Reload hotkey binding
            self._reload_hotkey()

            self._update_status_label()

    def _reload_engine(self):
        """Reload or create the WhisperEngine in a background thread."""
        model_name = self.settings.get('model_size')
        if not model_name:
            return

        try:
            model_path = self.model_manager.get_model_path(model_name)
        except FileNotFoundError:
            self.error_signal.emit(f"Model '{model_name}' not found.")
            return

        try:
            if self.engine is not None and self.engine.is_loaded():
                self.update_status_signal.emit("Reloading model...")
                self.engine.reload(model_path)
            else:
                self.update_status_signal.emit("Loading model...")
                self.engine = WhisperEngine(model_path)
            self.update_status_signal.emit("")
            logger.info("Engine reloaded with model '%s'", model_name)
        except Exception as e:
            logger.error("Failed to reload engine: %s", e, exc_info=True)
            self.error_signal.emit(f"Failed to load model: {e}")

    # ------------------------------------------------------------------
    # Hotkey listener (configurable)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hotkey(hotkey_str: str):
        """Parse a hotkey string like 'Ctrl+Alt+Shift+L' into (modifiers_set, key_char)."""
        parts = [p.strip() for p in hotkey_str.split('+')]
        modifiers = set()
        key_char = ''
        for p in parts:
            low = p.lower()
            if low in ('ctrl', 'control'):
                modifiers.add('ctrl')
            elif low in ('alt',):
                modifiers.add('alt')
            elif low in ('shift',):
                modifiers.add('shift')
            elif low in ('super', 'meta', 'win'):
                modifiers.add('super')
            else:
                key_char = low
        return modifiers, key_char

    def _start_hotkey_listener(self):
        """Start a global hotkey listener using evdev (works on Wayland)."""
        stop_event = self._hotkey_stop_event

        hotkey_str = self.settings.get('hotkey', 'Ctrl+Alt+Shift+L')
        self._hotkey_modifiers, self._hotkey_char = self._parse_hotkey(hotkey_str)

        target_keycode = _CHAR_TO_EVDEV.get(self._hotkey_char)
        if target_keycode is None:
            logger.error("Unknown hotkey character: %s", self._hotkey_char)
            return

        dev_path = self._keyboard_dev_path
        if not dev_path:
            logger.error("No keyboard device found for hotkey listener")
            return

        logger.info("Hotkey listener starting (evdev %s), target=%s+%s (keycode=%d)",
                     dev_path, self._hotkey_modifiers, self._hotkey_char, target_keycode)

        active_mods = set()

        try:
            dev = evdev.InputDevice(dev_path)
        except Exception as e:
            logger.error("Failed to open keyboard device %s: %s", dev_path, e)
            return

        self._hotkey_device = dev

        while not self.is_quitting and not stop_event.is_set():
            try:
                r, _, _ = select.select([dev], [], [], 0.5)
                if stop_event.is_set():
                    break
                if not r:
                    continue
                for event in dev.read():
                    if event.type != 1:  # EV_KEY
                        continue
                    code = event.code
                    value = event.value  # 1=press, 0=release, 2=repeat

                    # Track modifiers
                    mod_name = _EVDEV_MODIFIERS.get(code)
                    if mod_name:
                        if value >= 1:
                            active_mods.add(mod_name)
                        else:
                            active_mods.discard(mod_name)
                        continue

                    # Check for hotkey on press (not repeat)
                    if value == 1 and code == target_keycode and active_mods == self._hotkey_modifiers:
                        logger.info("Hotkey triggered!")
                        self.hotkey_signal.emit()
            except Exception as e:
                if not self.is_quitting and not stop_event.is_set():
                    logger.error("Error in hotkey listener: %s", e, exc_info=True)
                break

        # Clean up the device when this thread exits
        try:
            dev.close()
        except Exception:
            pass
        logger.info("Hotkey listener stopped")

    def _reload_hotkey(self):
        """Re-read the hotkey setting and restart the listener thread."""
        hotkey_str = self.settings.get('hotkey', 'Ctrl+Alt+Shift+L')
        logger.info("Hotkey updated to %s", hotkey_str)

        # Signal the old listener thread to stop
        self._hotkey_stop_event.set()
        if self._hotkey_thread and self._hotkey_thread.is_alive():
            self._hotkey_thread.join(timeout=2)

        # Reset the stop event and start a new listener
        self._hotkey_stop_event = threading.Event()
        self._hotkey_thread = threading.Thread(target=self._start_hotkey_listener, daemon=True)
        self._hotkey_thread.start()

    # ------------------------------------------------------------------
    # Auto-paste
    # ------------------------------------------------------------------

    def _paste_text(self, text):
        """Paste text into the focused window via clipboard + Ctrl+V.

        Text is already on the clipboard (pyperclip.copy is called before this).
        We simulate Ctrl+V using ydotool, which is instant regardless of
        text length.  ydotool injects keystrokes at the kernel level via
        /dev/uinput, which works on both Wayland and X11.
        """
        try:
            time.sleep(0.05)
            subprocess.run(
                ['ydotool', 'key', 'ctrl+v'],
                timeout=5, check=False,
            )
        except Exception as e:
            logger.error("Failed to paste text: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # Error / status signal slots
    # ------------------------------------------------------------------

    def _on_error(self, message):
        """Slot for error_signal: log and display the error."""
        logger.error("Application error: %s", message)

    def _on_update_status(self, message):
        """Slot for update_status_signal: update status label temporarily."""
        if message:
            self.status_label.setText(message)
        else:
            self._update_status_label()

    # ------------------------------------------------------------------
    # Exception hook
    # ------------------------------------------------------------------

    def _handle_exception(self, exc_type, exc_value, exc_tb):
        """Handle uncaught exceptions by logging them."""
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    # ------------------------------------------------------------------
    # Cleanup and shutdown
    # ------------------------------------------------------------------

    def _cleanup(self):
        """Release all resources."""
        if self.engine is not None:
            try:
                self.engine.unload()
            except Exception:
                pass
        if self.recorder is not None:
            try:
                self.recorder.cleanup()
            except Exception:
                pass
        if self.device_manager is not None:
            try:
                self.device_manager.cleanup()
            except Exception:
                pass

    def _quit(self):
        """Gracefully shut down the application."""
        self.is_quitting = True
        self._hotkey_stop_event.set()
        self.tray_icon.hide()
        self._cleanup()
        if self._hotkey_thread and self._hotkey_thread.is_alive():
            self._hotkey_thread.join(timeout=2)
        self.close()
        QApplication.instance().quit()

    def closeEvent(self, event):
        """Handle the window close event."""
        self.is_quitting = True
        self._hotkey_stop_event.set()
        self.tray_icon.hide()
        self._cleanup()
        if self._hotkey_thread and self._hotkey_thread.is_alive():
            self._hotkey_thread.join(timeout=2)
        event.accept()

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM by triggering quit."""
        self._quit()


