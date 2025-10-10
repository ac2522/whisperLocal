#!/usr/bin/env python3

import os
os.environ['ALSA_DEBUG'] = '0'  # Suppress ALSA warnings

import sys
import logging
import threading
import pyaudio
import wave
import whisper
import pyperclip
import webrtcvad
import numpy as np
import json
import platform
import time
import atexit
import signal
from pathlib import Path

from pynput import keyboard
from pynput.keyboard import Controller, Key

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox,
    QScrollArea, QHBoxLayout, QAction, QMenuBar, QMenu, QMessageBox,
    QComboBox, QDialog, QSpinBox, QSizePolicy, QStyle, QSystemTrayIcon
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QMetaObject, Q_ARG

if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    ICON_BASE = sys._MEIPASS
else:
    # Running in a normal Python environment
    ICON_BASE = os.path.dirname(os.path.abspath(__file__))

ICON_NORMAL = os.path.join(ICON_BASE, 'icon.png')
ICON_RECORDING = os.path.join(ICON_BASE, 'icon_recording.png')


SETTINGS_DIR = os.path.join(Path.home(), '.whisper2text')
if not os.path.exists(SETTINGS_DIR):
    os.makedirs(SETTINGS_DIR)

SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')

LOG_FILE = os.path.join(SETTINGS_DIR, 'app_errors.log')
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                   format='%(asctime)s:%(levelname)s:%(message)s')


class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    add_transcript = pyqtSignal(str)
    show_error = pyqtSignal(str)
    recording_stopped = pyqtSignal()
    update_recording_state = pyqtSignal(bool)
    update_button_text = pyqtSignal(str)
    update_button_enabled = pyqtSignal(bool)

class SettingsManager:
    """Manage application settings with thread safety."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SettingsManager, cls).__new__(cls)
                cls._instance._settings = {}
                cls._instance._load_settings()
            return cls._instance

    def _load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    self._settings = json.load(f)
                logging.info("Settings loaded from settings.json")
            except Exception as e:
                logging.error(f"Failed to load settings.json: {str(e)}", exc_info=True)
                self._settings = {}
        else:
            self._settings = {}

    def get(self, key, default=None):
        with self._lock:
            return self._settings.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._settings[key] = value

    def save(self):
        with self._lock:
            try:
                with open(SETTINGS_FILE, 'w') as f:
                    json.dump(self._settings, f, indent=4)
                logging.info("Settings saved to settings.json")
            except Exception as e:
                logging.error(f"Failed to save settings.json: {str(e)}", exc_info=True)

class AudioManager:
    """Manage PyAudio resources safely."""
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        atexit.register(self.cleanup)

    def open_stream(self, *args, **kwargs):
        self.stream = self.pa.open(*args, **kwargs)
        return self.stream

    def cleanup(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logging.error("Error closing stream:", exc_info=True)
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception as e:
                logging.error("Error terminating PyAudio:", exc_info=True)

class SpeechToTextApp(QWidget):
    hotkey_signal = pyqtSignal()

    def __init__(self, recording_mode='silence', break_length=5):
        super().__init__()
        self.setWindowTitle("Speech to Text")
        self.setWindowIcon(QIcon(ICON_NORMAL))
        self.resize(400, 600)
        self.is_quitting = False

        self.keyboard_controller = Controller()

        # Initialize default settings first
        self.settings_manager = SettingsManager()
        self.vad_aggressiveness = self.settings_manager.get('vad_aggressiveness', 1)
        self.model_size = self.settings_manager.get('model_size', 'base')
        self.padding_duration_ms = self.settings_manager.get('padding_duration_ms', 1000)
        self.recording_mode = self.settings_manager.get('recording_mode', recording_mode)
        self.break_length = self.settings_manager.get('break_length', break_length)
        self.transcripts = self.settings_manager.get('transcripts', [])
        self.auto_paste = self.settings_manager.get('auto_paste', False)

        self.max_transcripts = 10
        self.last_clicked_button = None

        # Initialize signals
        self.signals = WorkerSignals()
        self.signals.add_transcript.connect(self.add_transcript)
        self.signals.show_error.connect(self.show_error)
        self.signals.recording_stopped.connect(self.on_recording_stopped)
        self.signals.update_recording_state.connect(self.update_recording_state)

        self.signals.update_button_text.connect(self._update_button_text)
        self.signals.update_button_enabled.connect(self._update_button_enabled)

        # Initialize UI
        self.init_ui()

        # Initialize VAD and model
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.model = whisper.load_model(self.model_size)

        # Initialize AudioManager
        self.audio_manager = AudioManager()

        # Initialize recording state
        self.is_recording = False
        self.recording_lock = threading.Lock()

        # Error handling
        sys.excepthook = self.handle_exception

        # Initialize system tray
        self.setup_tray_icon()

        # Start hotkey listener
        self.hotkey_signal.connect(self.start_recording)
        threading.Thread(target=self.start_hotkey_listener, daemon=True).start()

    def _update_button_text(self, text):
        """Slot to update button text safely from any thread."""
        self.record_button.setText(text)

    def _update_button_enabled(self, enabled):
        """Slot to update button enabled state safely from any thread."""
        self.record_button.setEnabled(enabled)


    def setup_tray_icon(self):
        """Set up the system tray icon and its menu."""
        try:
            # Create the tray icon
            self.tray_icon = QSystemTrayIcon(self)

            if os.path.exists(ICON_NORMAL):
                self.tray_icon.setIcon(QIcon(ICON_NORMAL))
            else:
                logging.warning(f"Icon not found at {ICON_NORMAL}, using system icon")
                self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        except Exception as e:
            logging.error(f"Failed to set up system tray: {e}")
            QMessageBox.warning(self, "Warning", 
                              "Could not create system tray icon. The application will run without it.")
        
        # Create the tray menu
        self.tray_menu = QMenu()

        self.tray_icon.setToolTip("Speech to Text")

        # Add menu actions
        self.show_action = QAction("Show", self)
        self.show_action.triggered.connect(self.show)
        
        self.hide_action = QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide)
        
        self.record_action = QAction("Start Recording", self)
        self.record_action.triggered.connect(self.start_recording)
        
        self.settings_action = QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.settings_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        
        self.quit_action = QAction("Quit", self)
        self.quit_action.triggered.connect(self.quit_application)
        
        # Add actions to menu
        self.tray_menu.addAction(self.show_action)
        self.tray_menu.addAction(self.hide_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.record_action)
        self.tray_menu.addAction(self.settings_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.quit_action)
        
        # Set the menu for tray icon
        self.tray_icon.setContextMenu(self.tray_menu)
        
        # Connect double-click action
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
        # Show the tray icon
        self.tray_icon.show()
        
        # Update record action text based on recording state
        self.update_record_action_text()

    def update_record_action_text(self):
        """Update the recording action text based on current state."""
        if self.is_recording:
            self.record_action.setText("Stop Recording")
            self.tray_icon.setToolTip("Speech to Text (Recording...)")
        else:
            self.record_action.setText("Start Recording")
            self.tray_icon.setToolTip("Speech to Text")

    def tray_icon_activated(self, reason):
        """Handle tray icon activation (clicks)."""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def quit_application(self):
        """Properly close the application."""
        self.is_quitting = True  # Set a flag to indicate we're quitting
        self.tray_icon.hide()
        try:
            self.hotkey_listener.stop()
        except AttributeError:
            pass
        self.audio_manager.cleanup()
        self.close()
        QApplication.instance().quit()


    def init_ui(self):
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        top_bar = QHBoxLayout()

        # Menu bar
        menu_bar = QMenuBar(self)
        self.layout.setMenuBar(menu_bar)

        # Create a spacer menu to push the settings menu to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(spacer)

        # Settings menu with icon
        settings_button = QPushButton()
        settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        settings_button.setFlat(True)  # Makes the button look more like a menu item
        settings_button.clicked.connect(self.open_settings)
        settings_button.setToolTip('Preferences')
        top_bar.addWidget(settings_button)

        self.layout.addLayout(top_bar)

        # Scroll area for transcripts
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.transcript_widget = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_widget)
        self.transcript_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.transcript_widget)
        self.layout.addWidget(self.scroll_area)

        # Record button
        self.record_button = QPushButton('Record', self)
        self.record_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.record_button)

        # Adjust scroll area size policy
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialize transcript buttons
        recent_transcripts = self.transcripts[-self.max_transcripts:]
        for text in recent_transcripts:
            self.create_transcript_button(text)

        

    def open_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Settings')

        layout = QVBoxLayout()

        # VAD Aggressiveness
        vad_label = QLabel('VAD Aggressiveness (0-3):')
        vad_input = QSpinBox()
        vad_input.setRange(0, 3)
        vad_input.setValue(self.vad_aggressiveness)

        # Model Size
        model_label = QLabel('Model Size:')
        model_input = QComboBox()
        model_input.addItems(['tiny', 'base', 'small', 'medium', 'large'])
        model_input.setCurrentText(self.model_size)

        # Padding Duration
        padding_label = QLabel('Padding Duration (ms):')
        padding_input = QSpinBox()
        padding_input.setRange(100, 5000)
        padding_input.setValue(self.padding_duration_ms)

        # Recording Mode
        recording_mode_label = QLabel('Recording Mode:')
        recording_mode_input = QComboBox()
        recording_mode_input.addItems(['silence', 'button'])
        recording_mode_input.setCurrentText(self.recording_mode)

        # Break Length
        break_length_label = QLabel('Silence Duration to Stop Recording (seconds):')
        break_length_input = QSpinBox()
        break_length_input.setRange(1, 30)
        break_length_input.setValue(self.break_length)

        auto_paste_checkbox = QCheckBox('Automatically paste after copying')
        auto_paste_checkbox.setChecked(self.auto_paste)

        # Save button
        save_button = QPushButton('Save')
        save_button.clicked.connect(lambda: self.save_settings_dialog(
            vad_input.value(),
            model_input.currentText(),
            padding_input.value(),
            recording_mode_input.currentText(),
            break_length_input.value(),
            auto_paste_checkbox.isChecked(),
            dialog
        ))

        # Add widgets to layout
        layout.addWidget(vad_label)
        layout.addWidget(vad_input)
        layout.addWidget(model_label)
        layout.addWidget(model_input)
        layout.addWidget(padding_label)
        layout.addWidget(padding_input)
        layout.addWidget(recording_mode_label)
        layout.addWidget(recording_mode_input)
        layout.addWidget(break_length_label)
        layout.addWidget(break_length_input)
        layout.addWidget(auto_paste_checkbox)
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_settings_dialog(self, vad_value, model_size, padding_duration, recording_mode, break_length, auto_paste, dialog):
        with threading.Lock():
            self.vad_aggressiveness = vad_value
            self.model_size = model_size
            self.padding_duration_ms = padding_duration
            self.recording_mode = recording_mode
            self.break_length = break_length
            self.auto_paste = auto_paste

        # Reinitialize VAD and model
        try:
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            # Load model in a separate thread to prevent UI freeze
            threading.Thread(target=self.load_model_thread, args=(self.model_size,), daemon=True).start()
            self.settings_manager.set('vad_aggressiveness', self.vad_aggressiveness)
            self.settings_manager.set('model_size', self.model_size)
            self.settings_manager.set('padding_duration_ms', self.padding_duration_ms)
            self.settings_manager.set('recording_mode', self.recording_mode)
            self.settings_manager.set('break_length', self.break_length)
            self.settings_manager.set('auto_paste', self.auto_paste)
            self.settings_manager.save()
            dialog.accept()
            QMessageBox.information(self, 'Settings', 'Settings saved successfully.')
        except Exception as e:
            logging.error("Failed to reinitialize VAD or model:", exc_info=True)
            QMessageBox.critical(self, 'Error', f"Failed to save settings: {e}")

    def load_model_thread(self, model_size):
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            logging.error("Failed to load model in background thread:", exc_info=True)
            self.signals.show_error.emit(f"Failed to load model: {e}")

    def create_transcript_button(self, text):
        """Create a transcript button."""
        button = QPushButton(text[:50] + '...' if len(text) > 50 else text)
        button.setToolTip(text)
        button.setStyleSheet(self.get_button_style())
        button.clicked.connect(lambda: self.on_transcript_click(button, text))
        self.transcript_layout.addWidget(button)
        self.highlight_button(button)

    def add_transcript(self, text):
        if not text.strip():
            return

        # Insert at the top
        self.transcripts.append(text)
        self.create_transcript_button(text)

        # Keep only the last 10 transcripts
        while self.transcript_layout.count() > self.max_transcripts:
            first_button = self.transcript_layout.takeAt(0).widget()
            if first_button:
                # If the button we're deleting was the last clicked, reset it
                if first_button == self.last_clicked_button:
                    self.last_clicked_button = None
                first_button.deleteLater()

        # Copy to clipboard and paste if enabled
        try:
            pyperclip.copy(text)
            if self.auto_paste:
                self.paste_text()
        except Exception as e:
            logging.error("Failed to copy/paste text:", exc_info=True)
            QMessageBox.critical(self, 'Error', f"Failed to copy/paste text: {e}")

        # Update settings.json
        self.settings_manager.set('transcripts', self.transcripts)
        self.settings_manager.save()


    def on_transcript_click(self, button, text):
        """Handle transcript button click."""
        try:
            pyperclip.copy(text)
            if self.auto_paste:
                self.paste_text()
            self.highlight_button(button)
        except Exception as e:
            logging.error("Failed to copy/paste text:", exc_info=True)
            QMessageBox.critical(self, 'Error', f"Failed to copy/paste text: {e}")

    def paste_text(self):
        """Simulate pressing the paste keyboard shortcut."""
        try:
            # Give a slight delay to ensure clipboard is ready
            time.sleep(0.1)
            if platform.system() == 'Darwin':  # macOS
                with self.keyboard_controller.pressed(Key.cmd):
                    self.keyboard_controller.press('v')
                    self.keyboard_controller.release('v')
            else:  # Windows and Linux
                with self.keyboard_controller.pressed(Key.ctrl):
                    self.keyboard_controller.press('v')
                    self.keyboard_controller.release('v')
        except Exception as e:
            logging.error("Failed to paste text:", exc_info=True)
            QMessageBox.critical(self, 'Error', f"Failed to paste text: {e}")

    def highlight_button(self, button):
        """Highlight the clicked button."""
        if self.last_clicked_button:
            self.last_clicked_button.setStyleSheet(self.get_button_style(selected=False))
        button.setStyleSheet(self.get_button_style(selected=True))
        self.last_clicked_button = button

    def get_button_style(self, selected=False):
        """Return the stylesheet for buttons."""
        base_style = """
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
            return base_style + "QPushButton { background-color: rgba(0, 0, 255, 50); }"
        else:
            return base_style

    def start_recording(self):
        """Start or stop recording based on the current mode."""
        with self.recording_lock:
            if self.is_recording:
                # Stop recording
                self.is_recording = False
                self.update_record_action_text()
                self.update_tray_icon()
            else:
                # Start recording
                self.is_recording = True
                self.record_button.setText('Stop Recording')
                self.update_record_action_text()
                self.update_tray_icon()

                if self.recording_mode == 'button':
                    threading.Thread(target=self.record_and_transcribe_button_mode, daemon=True).start()
                else:
                    # 'silence' mode
                    self.record_button.setEnabled(False)
                    threading.Thread(target=self.record_and_transcribe_silence_mode, daemon=True).start()


    def update_tray_icon(self):
        """Update the tray and window icons based on recording state."""
        if self.is_recording:
            icon_path = ICON_RECORDING if os.path.exists(ICON_RECORDING) else None
            default_icon = self.style().standardIcon(QStyle.SP_MediaStop)
        else:
            icon_path = ICON_NORMAL if os.path.exists(ICON_NORMAL) else None
            default_icon = self.style().standardIcon(QStyle.SP_MediaPlay)

        if icon_path:
            icon = QIcon(icon_path)
        else:
            icon = default_icon

        # Update tray icon
        self.tray_icon.setIcon(icon)
        # This line updates the icon in the main window, which also updates
        # the icon shown in the Ubuntu dash/dock for most desktop environments.
        self.setWindowIcon(icon)



    def record_and_transcribe_button_mode(self):
        """Handle recording in 'button' mode."""
        try:
            audio_data = self.record_audio_button_mode()
            if audio_data:  # Check if we have audio data
                text = self.transcribe_audio(audio_data)
                if text and text.strip():  # Check if we have text
                    self.signals.add_transcript.emit(text)
        except Exception as e:
            logging.error("Error in button mode:", exc_info=True)
            self.signals.show_error.emit(str(e))
        finally:
            # Always emit recording stopped signal
            self.signals.recording_stopped.emit()


    def record_and_transcribe_silence_mode(self):
        """Handle recording in 'silence' mode."""
        try:
            audio_data = self.record_audio_silence_mode()
            if audio_data:  # Check if we have audio data
                text = self.transcribe_audio(audio_data)
                if text and text.strip():  # Check if we have text
                    self.signals.add_transcript.emit(text)
        except Exception as e:
            logging.error("Error in silence mode:", exc_info=True)
            self.signals.show_error.emit(str(e))
        finally:
            # Always emit recording stopped signal
            self.signals.recording_stopped.emit()


    def record_audio_button_mode(self):
        """Record audio until the recording is stopped."""
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_SIZE = int(RATE * 30 / 1000)  # 30 ms

        stream = self.audio_manager.open_stream(format=FORMAT,
                                                channels=CHANNELS,
                                                rate=RATE,
                                                input=True,
                                                frames_per_buffer=CHUNK_SIZE)

        frames = []

        try:
            while True:
                with self.recording_lock:
                    if not self.is_recording:
                        break
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()

        audio_data = b''.join(frames)
        return audio_data

    def record_audio_silence_mode(self):
        """Record audio until a specified duration of silence is detected."""
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_DURATION_MS = 30
        CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
        NUM_SILENT_CHUNKS = int(self.break_length * 1000 / CHUNK_DURATION_MS)

        stream = self.audio_manager.open_stream(format=FORMAT,
                                                channels=CHANNELS,
                                                rate=RATE,
                                                input=True,
                                                frames_per_buffer=CHUNK_SIZE)

        frames = []
        silence_chunks = 0
        triggered = False

        try:
            while True:
                with self.recording_lock:
                    if not self.is_recording:
                        break
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, RATE)

                if is_speech:
                    if not triggered:
                        triggered = True
                    frames.append(data)
                    silence_chunks = 0
                else:
                    if triggered:
                        silence_chunks += 1
                        frames.append(data)
                        if silence_chunks > NUM_SILENT_CHUNKS:
                            break
        finally:
            stream.stop_stream()
            stream.close()

        audio_data = b''.join(frames)
        return audio_data

    def transcribe_audio(self, audio_data):
        """Transcribe the recorded audio using Whisper."""
        temp_filename = 'temp_audio.wav'
        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio_manager.pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            result = self.model.transcribe(temp_filename)
            transcription = result['text'].strip()
            return transcription
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def handle_exception(self, exctype, value, traceback_obj):
        """Handle uncaught exceptions."""
        logging.error("Uncaught exception:", exc_info=(exctype, value, traceback_obj))
        QMessageBox.critical(self, 'Error', f"An unexpected error occurred: {value}")

    def start_hotkey_listener(self):
        """Start a global hotkey listener using pynput."""
        try:
            def on_press(key):
                try:

                    # Update key states
                    if key in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
                        self.ctrl_pressed = True
                    elif key in [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r]:
                        self.alt_pressed = True
                    elif key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                        self.shift_pressed = True
                    elif hasattr(key, 'char') and key.char is not None:
                        if key.char.lower() == 'l' and self.ctrl_pressed and self.alt_pressed and self.shift_pressed:
                            self.hotkey_signal.emit()
                except Exception as e:
                    logging.error(f"Error in on_press: {e}", exc_info=True)

            def on_release(key):
                try:

                    if key in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
                        self.ctrl_pressed = False
                    elif key in [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r]:
                        self.alt_pressed = False
                    elif key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
                        self.shift_pressed = False
                except Exception as e:
                    logging.error(f"Error in on_release: {e}", exc_info=True)

            # Initialize key state
            self.ctrl_pressed = False
            self.alt_pressed = False
            self.shift_pressed = False

            # Start listener
            self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.keyboard_listener.start()

        except Exception as e:
            logging.error("Hotkey listener error:", exc_info=True)
            QMessageBox.warning(self, 'Warning', 
                                'Could not start global hotkey listener. Please use the GUI buttons.')



    def show_error(self, message):
        """Display error messages."""
        QMessageBox.critical(self, 'Error', message)

    def closeEvent(self, event):
        """Properly close the application when the window is closed."""
        self.is_quitting = True  # Indicate that the application is quitting
        self.tray_icon.hide()
        self.audio_manager.cleanup()
        try:
            self.keyboard_listener.stop()
        except AttributeError:
            pass
        event.accept()


    def on_recording_stopped(self):
        """Slot to handle recording stopped signal."""
        try:
            with self.recording_lock:
                if hasattr(self, 'is_recording'):
                    self.is_recording = False
            
            if hasattr(self, 'record_button'):
                self.record_button.setText('Record')
                self.record_button.setEnabled(True)

            if hasattr(self, 'record_action'):
                self.update_record_action_text()

            if hasattr(self, 'tray_icon'):
                self.update_tray_icon()

        except Exception as e:
            logging.error(f"Error in on_recording_stopped: {e}", exc_info=True)


    def update_recording_state(self, is_recording):
        """Slot to update recording state in the UI."""
        try:
            self.is_recording = is_recording
            if is_recording:
                self.signals.update_button_text.emit('Stop Recording')
            else:
                self.signals.update_button_text.emit('Record')
            
            self.signals.update_button_enabled.emit(True)
            self.update_record_action_text()
            self.update_tray_icon()
        except Exception as e:
            logging.error(f"Error in update_recording_state: {e}", exc_info=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Speech to Text Application')
    parser.add_argument('--recording_mode', choices=['button', 'silence'], default='silence', help='Recording mode: "button" or "silence"')
    parser.add_argument('--break_length', type=int, default=5, help='Silence duration in seconds to stop recording in "silence" mode')

    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_NORMAL))
    window = SpeechToTextApp(recording_mode=args.recording_mode, break_length=args.break_length)
    window.show()

    # Handle Ctrl-C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
