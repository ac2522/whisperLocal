"""Settings manager for the whisper2text application.

Provides thread-safe access to application settings with automatic
persistence to a JSON file.
"""

import json
import logging
import os
import threading


logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    'model_size': 'base',
    'vad_aggressiveness': 1,
    'padding_duration_ms': 1000,
    'recording_mode': 'silence',
    'break_length': 5,
    'auto_paste': False,
    'transcripts': [],
    'audio_device_index': None,
    'audio_device_name': None,
}


def _migrate_model_size(model_size):
    """Migrate old model_size format to new format.

    Old format stored the model size as a bare name like 'base'.
    New format uses the full filename like 'ggml-base.bin'.
    """
    if model_size is None:
        return model_size
    if not isinstance(model_size, str):
        return model_size
    if not model_size.endswith('.bin'):
        return f'ggml-{model_size}.bin'
    return model_size


class SettingsManager:
    """Manage application settings with thread safety.

    Settings are stored as a JSON file in the settings directory.
    All access is protected by a threading lock to ensure thread safety.

    Args:
        settings_dir: Path to the directory for storing settings.
            Defaults to ``~/.whisper2text``.
    """

    def __init__(self, settings_dir=None):
        if settings_dir is None:
            settings_dir = os.path.join(os.path.expanduser('~'), '.whisper2text')

        self._settings_dir = settings_dir
        self._settings_file = os.path.join(settings_dir, 'settings.json')
        self._lock = threading.Lock()
        self._settings = dict(DEFAULT_SETTINGS)

        # Create settings directory if it doesn't exist
        os.makedirs(self._settings_dir, exist_ok=True)

        self._load_settings()

    def _load_settings(self):
        """Load settings from the JSON file on disk.

        Merges loaded values on top of defaults so that any new default
        keys are present even when loading an older settings file.
        Applies migrations for backward compatibility.
        """
        if os.path.exists(self._settings_file):
            try:
                with open(self._settings_file, 'r') as f:
                    data = json.load(f)
                # Merge loaded data on top of defaults
                self._settings.update(data)
                logger.info("Settings loaded from %s", self._settings_file)
            except Exception:
                logger.error(
                    "Failed to load %s", self._settings_file, exc_info=True
                )
                # Keep defaults on failure
        # Apply migrations
        self._migrate()

    def _migrate(self):
        """Apply any necessary migrations to loaded settings."""
        model_size = self._settings.get('model_size')
        if model_size is not None:
            migrated = _migrate_model_size(model_size)
            if migrated != model_size:
                logger.info(
                    "Migrated model_size from '%s' to '%s'",
                    model_size,
                    migrated,
                )
                self._settings['model_size'] = migrated

    def get(self, key, default=None):
        """Return the value for *key*, or *default* if not present.

        Args:
            key: The setting name.
            default: Value to return when *key* is absent.

        Returns:
            The setting value or *default*.
        """
        with self._lock:
            return self._settings.get(key, default)

    def set(self, key, value):
        """Set *key* to *value*.

        Args:
            key: The setting name.
            value: The new value.
        """
        with self._lock:
            self._settings[key] = value

    def get_all(self):
        """Return a shallow copy of all current settings.

        Returns:
            dict: A copy of the internal settings dictionary.
        """
        with self._lock:
            return dict(self._settings)

    def save(self):
        """Persist current settings to the JSON file on disk."""
        with self._lock:
            try:
                with open(self._settings_file, 'w') as f:
                    json.dump(self._settings, f, indent=4)
                logger.info("Settings saved to %s", self._settings_file)
            except Exception:
                logger.error(
                    "Failed to save %s", self._settings_file, exc_info=True
                )
