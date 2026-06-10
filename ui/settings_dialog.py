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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import api_keys


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

        # Start smaller and let users resize — without tabs this dialog grew
        # taller than many displays.
        self.resize(520, 560)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Tab 1: Model & Compute ---
        tab_model = QWidget()
        tab_model_layout = QVBoxLayout(tab_model)
        tab_model_layout.addWidget(self._build_model_group())
        tab_model_layout.addWidget(self._build_compute_group())
        tab_model_layout.addWidget(self._build_deepgram_group())
        tab_model_layout.addStretch()
        tabs.addTab(tab_model, "Model && Compute")

        # --- Tab 2: Audio & Recording ---
        tab_audio = QWidget()
        tab_audio_layout = QVBoxLayout(tab_audio)
        tab_audio_layout.addWidget(self._build_audio_group())
        tab_audio_layout.addWidget(self._build_recording_group())
        tab_audio_layout.addStretch()
        tabs.addTab(tab_audio, "Audio && Recording")

        # --- Tab 3: Vocabulary ---
        tab_vocab = QWidget()
        tab_vocab_layout = QVBoxLayout(tab_vocab)
        tab_vocab_layout.addWidget(self._build_vocabulary_group())
        tab_vocab_layout.addStretch()
        tabs.addTab(tab_vocab, "Vocabulary")

        # --- Tab 4: Hotkey ---
        tab_hotkey = QWidget()
        tab_hotkey_layout = QVBoxLayout(tab_hotkey)
        tab_hotkey_layout.addWidget(self._build_hotkey_group())
        tab_hotkey_layout.addStretch()
        tabs.addTab(tab_hotkey, "Hotkey")

        # --- Save button (outside tabs — always visible) ---
        # _save reads values from every widget on every tab regardless of
        # which tab is currently visible, because the widgets are held on
        # self._* and persist across tab switches.
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_model_label(model: dict) -> str:
        """Format dropdown label per engine type."""
        t = model.get("type")
        if t == "cloud":
            return f"[Cloud] {model['name']}"
        engine_tag = "Parakeet" if t == "parakeet" else "Whisper"
        return f"[{engine_tag}] {model['name']} ({model['size_mb']} MB)"

    def _build_model_group(self):
        group = QGroupBox("Model")
        vbox = QVBoxLayout()

        # Transcription Model dropdown (downloaded models, both engines)
        vbox.addWidget(QLabel("Transcription Model"))
        self._model_combo = QComboBox()
        vbox.addWidget(self._model_combo)
        self._refresh_model_list()

        # Download Model dropdown + Download button (both engines)
        vbox.addWidget(QLabel("Download Model"))
        dl_row = QHBoxLayout()
        self._download_combo = QComboBox()
        for m in self._model_manager.list_available():
            if m.get("type") == "cloud":
                continue  # Cloud entries don't need downloading.
            self._download_combo.addItem(self._format_model_label(m), m["name"])
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
        # QComboBox subclassed so that opening the dropdown re-queries
        # the OS for newly-plugged-in audio devices. PyAudio caches its
        # device list at init time, so we also ask the DeviceManager to
        # restart PyAudio before re-enumerating.
        dialog = self

        class _RefreshingComboBox(QComboBox):
            def showPopup(self_inner):  # noqa: N805 — Qt convention
                dialog._refresh_device_list()
                super().showPopup()

        self._device_combo = _RefreshingComboBox()
        self._populate_device_combo()
        vbox.addWidget(self._device_combo)

        group.setLayout(vbox)
        return group

    def _populate_device_combo(self) -> None:
        """Fill the device combo with PipeWire sources (friendly names).

        Item data carries the PipeWire ``node.name`` (stable across
        hotplug), or None for "follow the system default".  A saved mic
        that's currently unplugged still appears — marked "(not
        connected)" — so opening Settings never silently discards it.
        """
        self._device_combo.clear()

        default = self._device_manager.get_default_source()
        if default is not None:
            label = f"System Default (currently: {default['description']})"
        else:
            label = "System Default"
        self._device_combo.addItem(label, None)

        saved_node = self._settings.get("audio_device_node")
        saved_label = self._settings.get("audio_device_label")
        select_idx = 0  # default to "System Default"

        for src in self._device_manager.list_sources():
            self._device_combo.addItem(src["description"], src["node_name"])
            if src["node_name"] == saved_node:
                select_idx = self._device_combo.count() - 1

        if saved_node is not None and select_idx == 0:
            self._device_combo.addItem(
                f"{saved_label or saved_node} (not connected)", saved_node
            )
            select_idx = self._device_combo.count() - 1

        self._device_combo.setCurrentIndex(select_idx)

    def _refresh_device_list(self) -> None:
        """Refill the combo from a fresh pw-dump. Called from showPopup."""
        self._device_manager.refresh()
        self._populate_device_combo()

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
        self._compute_combo.addItem("Auto (Recommended)", "auto")
        self._compute_combo.addItem("CPU", "cpu")

        # Detect available GPU backends from pywhispercpp shared libs.
        # Use explicit lib-name prefixes to avoid false positives
        # (e.g. libicudata.so contains "cuda" as a substring).
        backends_seen: set[str] = set()
        _CUDA_LIB_PREFIXES = ("libcuda", "libcudart", "libcublas", "libggml-cuda")
        _VULKAN_LIB_PREFIXES = ("libvulkan", "libggml-vulkan")
        try:
            import importlib.util, os
            spec = importlib.util.find_spec('_pywhispercpp')
            if spec and spec.origin:
                lib_dir = os.path.dirname(spec.origin)
                for f in os.listdir(lib_dir):
                    low = f.lower()
                    if any(low.startswith(p) for p in _VULKAN_LIB_PREFIXES):
                        if "vulkan" not in backends_seen:
                            backends_seen.add("vulkan")
                            self._compute_combo.addItem("Vulkan GPU", "vulkan")
                    if any(low.startswith(p) for p in _CUDA_LIB_PREFIXES):
                        if "cuda" not in backends_seen:
                            backends_seen.add("cuda")
                            self._compute_combo.addItem("CUDA GPU", "cuda")
        except Exception:
            pass

        # Select saved backend
        saved = self._settings.get("compute_backend", "auto")
        idx = self._compute_combo.findData(saved)
        if idx >= 0:
            self._compute_combo.setCurrentIndex(idx)

        vbox.addWidget(self._compute_combo)

        group.setLayout(vbox)
        return group

    def _build_deepgram_group(self):
        group = QGroupBox("Deepgram API Key")
        vbox = QVBoxLayout()

        self._deepgram_status_label = QLabel()
        vbox.addWidget(self._deepgram_status_label)

        row = QHBoxLayout()
        self._deepgram_key_edit = QLineEdit()
        self._deepgram_key_edit.setEchoMode(QLineEdit.Password)
        self._deepgram_key_edit.setPlaceholderText("Paste API key, then press Save")
        row.addWidget(self._deepgram_key_edit)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_deepgram_key)
        row.addWidget(save_btn)
        vbox.addLayout(row)

        clear_btn = QPushButton("Clear stored key")
        clear_btn.clicked.connect(self._clear_deepgram_key)
        vbox.addWidget(clear_btn)

        tip = QLabel("Tip: env var DEEPGRAM_API_KEY overrides the stored key.")
        tip.setStyleSheet("color: gray;")
        vbox.addWidget(tip)

        group.setLayout(vbox)
        self._refresh_deepgram_status()
        return group

    def _refresh_deepgram_status(self) -> None:
        source = api_keys.get_key_source()
        if source == "env":
            self._deepgram_status_label.setText(
                "Status: ● Configured (via DEEPGRAM_API_KEY env var)"
            )
        elif source == "keyring":
            self._deepgram_status_label.setText(
                "Status: ● Configured (in OS keyring)"
            )
        else:
            self._deepgram_status_label.setText("Status: ○ Not set")

    def _save_deepgram_key(self) -> None:
        value = self._deepgram_key_edit.text().strip()
        if not value:
            QMessageBox.warning(
                self, "Empty key", "Enter a key before pressing Save."
            )
            return
        try:
            api_keys.set_deepgram_key(value)
        except Exception as exc:
            QMessageBox.critical(
                self, "Keyring error", f"Failed to save key:\n{exc}"
            )
            return
        self._deepgram_key_edit.clear()
        self._refresh_deepgram_status()
        QMessageBox.information(self, "Saved", "Deepgram API key saved.")

    def _clear_deepgram_key(self) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Remove the stored Deepgram API key?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            api_keys.clear_deepgram_key()
        except Exception as exc:
            QMessageBox.critical(
                self, "Keyring error", f"Failed to clear key:\n{exc}"
            )
            return
        self._refresh_deepgram_status()

    def _build_hotkey_group(self):
        group = QGroupBox("Hotkey")
        vbox = QVBoxLayout()

        current = self._settings.get("hotkey", "Ctrl+Alt+Shift+L")
        vbox.addWidget(QLabel("Hotkey (e.g. Ctrl+Alt+S, Ctrl+Shift+Z)"))

        self._hotkey_display = QLineEdit(current)
        self._hotkey_display.setAlignment(Qt.AlignCenter)
        self._hotkey_display.setStyleSheet(
            "QLineEdit { background-color: white; color: black; border: 2px solid blue;"
            " border-radius: 5px; padding: 4px; font-weight: bold; }"
        )
        vbox.addWidget(self._hotkey_display)
        group.setLayout(vbox)
        return group

    def _build_vocabulary_group(self):
        group = QGroupBox("Custom Vocabulary")
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel(
            "One word or short phrase per line. Used to bias Whisper's decoder "
            "(real prompt) and to fuzzy-fix Parakeet's output (post-processing)."
        ))

        self._vocab_edit = QTextEdit()
        self._vocab_edit.setPlaceholderText(
            "Avrillo\nconveyancing\nSDLT"
        )
        existing = self._settings.get("custom_vocabulary") or []
        self._vocab_edit.setPlainText("\n".join(existing))
        self._vocab_edit.setFixedHeight(120)
        vbox.addWidget(self._vocab_edit)

        group.setLayout(vbox)
        return group

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _refresh_model_list(self):
        """Repopulate the Transcription Model dropdown with downloaded models."""
        self._model_combo.clear()
        current_model = self._settings.get("model_size")
        select_idx = 0

        for i, m in enumerate(self._model_manager.list_downloaded()):
            self._model_combo.addItem(self._format_model_label(m), m["name"])
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

        # Cloud entries have nothing to delete on disk.
        from engine.model_manager import _MODELS_BY_NAME
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            QMessageBox.information(
                self,
                "Cloud model",
                "Cloud models are not stored locally and cannot be deleted.",
            )
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
            from engine.model_manager import _MODELS_BY_NAME
            known = _MODELS_BY_NAME.get(model_name)
            if (
                known
                and known.get("type") == "cloud"
                and not api_keys.has_deepgram_key()
            ):
                QMessageBox.warning(
                    self,
                    "Deepgram API key not set",
                    "You selected Deepgram Nova-3 but no API key is configured. "
                    "Set DEEPGRAM_API_KEY or save a key in this dialog before "
                    "recording, otherwise transcription will fail.",
                )

        # Compute backend
        self._settings.set("compute_backend", self._compute_combo.currentData())

        # Audio device
        device_node = self._device_combo.currentData()
        self._settings.set("audio_device_node", device_node)
        device_label = self._device_combo.currentText()
        if device_label.endswith(" (not connected)"):
            device_label = device_label[: -len(" (not connected)")]
        self._settings.set(
            "audio_device_label",
            None if device_node is None else device_label,
        )

        # Recording
        self._settings.set("vad_aggressiveness", self._vad_spin.value())
        self._settings.set("recording_mode", self._mode_combo.currentText())
        self._settings.set("break_length", self._silence_spin.value())
        self._settings.set("padding_duration_ms", self._padding_spin.value())
        self._settings.set("auto_paste", self._autopaste_cb.isChecked())

        # Hotkey
        self._settings.set("hotkey", self._hotkey_display.text())

        # Custom vocabulary — split on newlines, strip, drop empties
        vocab_raw = self._vocab_edit.toPlainText().splitlines()
        vocab_list = [line.strip() for line in vocab_raw if line.strip()]
        self._settings.set("custom_vocabulary", vocab_list)

        self._settings.save()
        self.accept()

