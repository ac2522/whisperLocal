"""Settings dialog for the whisper2text application.

Provides a PyQt5 dialog that allows users to configure the Whisper model,
audio input device, recording behaviour, and other application settings.
"""

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class DownloadThread(QThread):
    """Background thread that downloads a Whisper model.

    Signals
    -------
    progress(int)
        Emitted with the download percentage (0-100).
    finished_ok()
        Emitted when the download completes successfully.
    error(str)
        Emitted with an error message when the download fails.
    """

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_manager, model_name, parent=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._model_name = model_name

    def run(self):
        try:
            def _on_progress(percent, _downloaded, _total):
                self.progress.emit(int(percent))

            self._model_manager.download_model(
                self._model_name, progress_callback=_on_progress
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class SettingsDialog(QDialog):
    """Application settings dialog.

    Parameters
    ----------
    settings_manager:
        A :class:`config.settings.SettingsManager` instance.
    model_manager:
        A :class:`engine.model_manager.ModelManager` instance.
    device_manager:
        A :class:`audio.device_manager.DeviceManager` instance.
    parent:
        Optional parent widget.
    """

    def __init__(self, settings_manager, model_manager, device_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self._settings = settings_manager
        self._model_manager = model_manager
        self._device_manager = device_manager

        self._download_thread = None
        self._progress_dialog = None

        layout = QVBoxLayout(self)

        # --- Model group ---
        layout.addWidget(self._build_model_group())

        # --- Compute group ---
        layout.addWidget(self._build_compute_group())

        # --- Audio group ---
        layout.addWidget(self._build_audio_group())

        # --- Recording group ---
        layout.addWidget(self._build_recording_group())

        # --- Hotkey group ---
        layout.addWidget(self._build_hotkey_group())

        # --- Save button ---
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _build_model_group(self):
        group = QGroupBox("Model")
        vbox = QVBoxLayout()

        # Whisper Model dropdown (downloaded models)
        vbox.addWidget(QLabel("Whisper Model"))
        self._model_combo = QComboBox()
        vbox.addWidget(self._model_combo)
        self._refresh_model_list()

        # Download Model dropdown + Download button
        vbox.addWidget(QLabel("Download Model"))
        dl_row = QHBoxLayout()
        self._download_combo = QComboBox()
        for m in self._model_manager.list_available():
            label = f"{m['name']} ({m['size_mb']} MB)"
            self._download_combo.addItem(label, m["name"])
        dl_row.addWidget(self._download_combo)

        download_btn = QPushButton("Download")
        download_btn.clicked.connect(self._download_model)
        dl_row.addWidget(download_btn)
        vbox.addLayout(dl_row)

        # Delete selected model button
        delete_btn = QPushButton("Delete Selected Model")
        delete_btn.clicked.connect(self._delete_model)
        vbox.addWidget(delete_btn)

        group.setLayout(vbox)
        return group

    def _build_audio_group(self):
        group = QGroupBox("Audio")
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel("Input Device"))
        self._device_combo = QComboBox()

        # "System Default" entry with data=None
        self._device_combo.addItem("System Default", None)

        saved_index = self._settings.get("audio_device_index")
        select_idx = 0  # default to "System Default"

        for dev in self._device_manager.list_input_devices():
            self._device_combo.addItem(dev["name"], dev["index"])
            if dev["index"] == saved_index:
                select_idx = self._device_combo.count() - 1

        self._device_combo.setCurrentIndex(select_idx)
        vbox.addWidget(self._device_combo)

        group.setLayout(vbox)
        return group

    def _build_recording_group(self):
        group = QGroupBox("Recording")
        vbox = QVBoxLayout()

        # VAD Aggressiveness
        vbox.addWidget(QLabel("VAD Aggressiveness"))
        self._vad_spin = QSpinBox()
        self._vad_spin.setRange(0, 3)
        self._vad_spin.setValue(self._settings.get("vad_aggressiveness", 1))
        vbox.addWidget(self._vad_spin)

        # Recording Mode
        vbox.addWidget(QLabel("Recording Mode"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["silence", "button"])
        current_mode = self._settings.get("recording_mode", "silence")
        idx = self._mode_combo.findText(current_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        vbox.addWidget(self._mode_combo)

        # Silence Duration (break_length)
        vbox.addWidget(QLabel("Silence Duration"))
        self._silence_spin = QSpinBox()
        self._silence_spin.setRange(1, 30)
        self._silence_spin.setValue(self._settings.get("break_length", 5))
        vbox.addWidget(self._silence_spin)

        # Padding Duration
        vbox.addWidget(QLabel("Padding Duration"))
        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(100, 5000)
        self._padding_spin.setValue(
            self._settings.get("padding_duration_ms", 1000)
        )
        vbox.addWidget(self._padding_spin)

        # Auto-paste checkbox
        self._autopaste_cb = QCheckBox("Auto-paste")
        self._autopaste_cb.setChecked(bool(self._settings.get("auto_paste", False)))
        vbox.addWidget(self._autopaste_cb)

        group.setLayout(vbox)
        return group

    def _build_compute_group(self):
        group = QGroupBox("Compute")
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel("Backend"))
        self._compute_combo = QComboBox()
        self._compute_combo.addItem("CPU", "cpu")

        # Detect available GPU backends from pywhispercpp shared libs
        try:
            import importlib.util, os
            spec = importlib.util.find_spec('_pywhispercpp')
            if spec and spec.origin:
                lib_dir = os.path.dirname(spec.origin)
                for f in os.listdir(lib_dir):
                    if 'vulkan' in f.lower():
                        self._compute_combo.addItem("Vulkan GPU", "vulkan")
                    if 'cuda' in f.lower():
                        self._compute_combo.addItem("CUDA GPU", "cuda")
        except Exception:
            pass

        # Select saved backend
        saved = self._settings.get("compute_backend", "vulkan")
        idx = self._compute_combo.findData(saved)
        if idx >= 0:
            self._compute_combo.setCurrentIndex(idx)

        vbox.addWidget(self._compute_combo)

        group.setLayout(vbox)
        return group

    def _build_hotkey_group(self):
        group = QGroupBox("Hotkey")
        vbox = QVBoxLayout()

        current = self._settings.get("hotkey", "Ctrl+Alt+Shift+L")
        vbox.addWidget(QLabel("Press 'Capture' to set a new hotkey"))

        row = QHBoxLayout()
        self._hotkey_display = QLineEdit(current)
        self._hotkey_display.setReadOnly(True)
        self._hotkey_display.setAlignment(Qt.AlignCenter)
        self._hotkey_display.setStyleSheet(
            "QLineEdit { background-color: white; color: black; border: 2px solid blue;"
            " border-radius: 5px; padding: 4px; font-weight: bold; }"
        )
        row.addWidget(self._hotkey_display)

        capture_btn = QPushButton("Capture")
        capture_btn.clicked.connect(self._capture_hotkey)
        row.addWidget(capture_btn)

        vbox.addLayout(row)
        group.setLayout(vbox)
        return group

    def _capture_hotkey(self):
        """Open a small dialog that captures a hotkey combo."""
        dialog = _HotkeyCaptureDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.captured_hotkey:
            self._hotkey_display.setText(dialog.captured_hotkey)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _refresh_model_list(self):
        """Repopulate the Whisper Model dropdown with downloaded models."""
        self._model_combo.clear()
        current_model = self._settings.get("model_size")
        select_idx = 0

        for i, m in enumerate(self._model_manager.list_downloaded()):
            label = f"{m['name']} ({m['size_mb']} MB)"
            self._model_combo.addItem(label, m["name"])
            if m["name"] == current_model:
                select_idx = i

        if self._model_combo.count() > 0:
            self._model_combo.setCurrentIndex(select_idx)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_model(self):
        """Start downloading the model selected in the download combo."""
        model_name = self._download_combo.currentData()
        if model_name is None:
            return

        if self._model_manager.is_downloaded(model_name):
            QMessageBox.information(
                self, "Already Downloaded", f"{model_name} is already downloaded."
            )
            return

        self._progress_dialog = QProgressDialog(
            f"Downloading {model_name}...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Downloading")
        self._progress_dialog.setMinimumDuration(0)

        self._download_thread = DownloadThread(
            self._model_manager, model_name, parent=self
        )
        self._download_thread.progress.connect(self._progress_dialog.setValue)
        self._download_thread.finished_ok.connect(self._on_download_done)
        self._download_thread.error.connect(self._on_download_error)

        self._progress_dialog.canceled.connect(self._download_thread.terminate)

        self._download_thread.start()

    def _on_download_done(self):
        """Handle a successful download."""
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

        self._refresh_model_list()

        QMessageBox.information(self, "Download Complete", "Model downloaded successfully.")

    def _on_download_error(self, message):
        """Handle a failed download."""
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

        QMessageBox.critical(self, "Download Error", f"Failed to download model:\n{message}")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_model(self):
        """Delete the currently selected downloaded model after confirmation."""
        model_name = self._model_combo.currentData()
        if model_name is None:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {model_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._model_manager.delete_model(model_name)
            self._refresh_model_list()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        """Persist all widget values to the settings manager and close."""
        # Model
        model_name = self._model_combo.currentData()
        if model_name is not None:
            self._settings.set("model_size", model_name)

        # Compute backend
        self._settings.set("compute_backend", self._compute_combo.currentData())

        # Audio device
        device_index = self._device_combo.currentData()
        self._settings.set("audio_device_index", device_index)
        device_name = self._device_combo.currentText()
        self._settings.set(
            "audio_device_name",
            None if device_index is None else device_name,
        )

        # Recording
        self._settings.set("vad_aggressiveness", self._vad_spin.value())
        self._settings.set("recording_mode", self._mode_combo.currentText())
        self._settings.set("break_length", self._silence_spin.value())
        self._settings.set("padding_duration_ms", self._padding_spin.value())
        self._settings.set("auto_paste", self._autopaste_cb.isChecked())

        # Hotkey
        self._settings.set("hotkey", self._hotkey_display.text())

        self._settings.save()
        self.accept()


class _HotkeyCaptureDialog(QDialog):
    """Dialog that captures a keyboard shortcut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Hotkey")
        self.setMinimumSize(300, 150)
        self.captured_hotkey = ""
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; }"
            " QPushButton { color: white; background-color: #444;"
            "   border: 1px solid #666; border-radius: 4px; padding: 4px 12px; }"
            " QPushButton:hover { background-color: #555; }"
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        self._label = QLabel("Press your desired key combination:")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self._label)

        self._display = QLabel("(waiting for keypress)")
        self._display.setStyleSheet(
            "QLabel { background-color: white; color: #222; border: 2px solid #4a90d9;"
            " border-radius: 8px; padding: 8px; font-weight: bold; }"
        )
        self._display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._display)

        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()
        self.grabKeyboard()

    def closeEvent(self, event):
        self.releaseKeyboard()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        parts = []
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            parts.append("Ctrl")
        if mods & Qt.AltModifier:
            parts.append("Alt")
        if mods & Qt.ShiftModifier:
            parts.append("Shift")
        if mods & Qt.MetaModifier:
            parts.append("Super")

        key = event.key()
        if key in (Qt.Key_Control, Qt.Key_Alt, Qt.Key_Shift, Qt.Key_Meta,
                   Qt.Key_AltGr, Qt.Key_Super_L, Qt.Key_Super_R):
            self._display.setText("+".join(parts) + "+...")
            return

        # Resolve the key name from Qt key code (always reliable, unlike event.text())
        special_keys = {
            Qt.Key_Space: "Space", Qt.Key_Return: "Enter",
            Qt.Key_Tab: "Tab", Qt.Key_Backspace: "Backspace",
            Qt.Key_Delete: "Delete", Qt.Key_Escape: "Esc",
            Qt.Key_F1: "F1", Qt.Key_F2: "F2", Qt.Key_F3: "F3",
            Qt.Key_F4: "F4", Qt.Key_F5: "F5", Qt.Key_F6: "F6",
            Qt.Key_F7: "F7", Qt.Key_F8: "F8", Qt.Key_F9: "F9",
            Qt.Key_F10: "F10", Qt.Key_F11: "F11", Qt.Key_F12: "F12",
            Qt.Key_Home: "Home", Qt.Key_End: "End",
            Qt.Key_PageUp: "PageUp", Qt.Key_PageDown: "PageDown",
            Qt.Key_Insert: "Insert",
            Qt.Key_Left: "Left", Qt.Key_Right: "Right",
            Qt.Key_Up: "Up", Qt.Key_Down: "Down",
            Qt.Key_QuoteLeft: "`", Qt.Key_Minus: "-", Qt.Key_Equal: "=",
            Qt.Key_BracketLeft: "[", Qt.Key_BracketRight: "]",
            Qt.Key_Backslash: "\\", Qt.Key_Semicolon: ";",
            Qt.Key_Apostrophe: "'", Qt.Key_Comma: ",",
            Qt.Key_Period: ".", Qt.Key_Slash: "/",
        }

        key_text = special_keys.get(key, "")
        if not key_text and Qt.Key_A <= key <= Qt.Key_Z:
            key_text = chr(key)
        elif not key_text and Qt.Key_0 <= key <= Qt.Key_9:
            key_text = chr(key)

        if key_text:
            parts.append(key_text)
            self.captured_hotkey = "+".join(parts)
            self._display.setText(self.captured_hotkey)
