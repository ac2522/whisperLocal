"""Tests for config.settings.SettingsManager."""

import json
import os
import tempfile

import pytest

from config.settings import DEFAULT_SETTINGS, SettingsManager, _migrate_model_size


@pytest.fixture
def tmp_settings_dir(tmp_path):
    """Provide a temporary directory for settings."""
    return str(tmp_path / "settings")


class TestDefaultValues:
    """SettingsManager should expose sensible defaults."""

    def test_defaults_are_set(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('model_size') == 'ggml-base.bin'
        assert sm.get('vad_aggressiveness') == 1
        assert sm.get('padding_duration_ms') == 1000
        assert sm.get('recording_mode') == 'silence'
        assert sm.get('break_length') == 5
        assert sm.get('auto_paste') is False
        assert sm.get('transcripts') == []
        assert sm.get('audio_device_node') is None
        assert sm.get('audio_device_label') is None

    def test_creates_settings_dir(self, tmp_settings_dir):
        assert not os.path.exists(tmp_settings_dir)
        SettingsManager(settings_dir=tmp_settings_dir)
        assert os.path.isdir(tmp_settings_dir)


class TestGetAndSet:
    """get() and set() should read and write individual keys."""

    def test_set_and_get(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        sm.set('model_size', 'ggml-large.bin')
        assert sm.get('model_size') == 'ggml-large.bin'

    def test_set_new_key(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        sm.set('custom_key', 42)
        assert sm.get('custom_key') == 42

    def test_get_unknown_key_returns_none(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('nonexistent_key') is None

    def test_get_unknown_key_returns_provided_default(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('nonexistent_key', 'fallback') == 'fallback'


class TestSaveAndLoad:
    """Settings should survive a save/load round-trip."""

    def test_round_trip(self, tmp_settings_dir):
        sm1 = SettingsManager(settings_dir=tmp_settings_dir)
        sm1.set('recording_mode', 'manual')
        sm1.set('break_length', 10)
        sm1.set('auto_paste', True)
        sm1.save()

        # Create a new manager pointed at the same directory
        sm2 = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm2.get('recording_mode') == 'manual'
        assert sm2.get('break_length') == 10
        assert sm2.get('auto_paste') is True

    def test_save_creates_json_file(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        sm.save()
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        assert os.path.isfile(settings_file)
        with open(settings_file, 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_load_merges_with_defaults(self, tmp_settings_dir):
        """Loading an older file that is missing new keys should still
        have the new keys with their defaults."""
        os.makedirs(tmp_settings_dir, exist_ok=True)
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        # Write a minimal old-style file
        with open(settings_file, 'w') as f:
            json.dump({'model_size': 'ggml-tiny.bin', 'auto_paste': True}, f)

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        # The loaded values should be honoured
        assert sm.get('model_size') == 'ggml-tiny.bin'
        assert sm.get('auto_paste') is True
        # Missing keys should fall back to defaults
        assert sm.get('vad_aggressiveness') == 1
        assert sm.get('padding_duration_ms') == 1000
        assert sm.get('audio_device_node') is None


class TestGetAll:
    """get_all() should return a copy of all settings."""

    def test_get_all_returns_dict(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        all_settings = sm.get_all()
        assert isinstance(all_settings, dict)

    def test_get_all_contains_defaults(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        all_settings = sm.get_all()
        for key in DEFAULT_SETTINGS:
            assert key in all_settings

    def test_get_all_returns_copy(self, tmp_settings_dir):
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        all_settings = sm.get_all()
        all_settings['model_size'] = 'MUTATED'
        # The internal state must not be affected
        assert sm.get('model_size') != 'MUTATED'


class TestBackwardCompatibility:
    """The manager should correctly handle settings files written by the
    old single-file application."""

    def test_loads_old_format_settings(self, tmp_settings_dir):
        """Simulate the settings.json format used by whisper2text.py."""
        os.makedirs(tmp_settings_dir, exist_ok=True)
        old_settings = {
            "transcripts": ["hello world"],
            "vad_aggressiveness": 2,
            "model_size": "base",
            "padding_duration_ms": 500,
            "recording_mode": "manual",
            "break_length": 3,
            "auto_paste": True,
        }
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump(old_settings, f, indent=4)

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('transcripts') == ["hello world"]
        assert sm.get('vad_aggressiveness') == 2
        assert sm.get('padding_duration_ms') == 500
        assert sm.get('recording_mode') == 'manual'
        assert sm.get('break_length') == 3
        assert sm.get('auto_paste') is True

    def test_old_format_gets_new_default_keys(self, tmp_settings_dir):
        """Old settings files should gain any newly introduced keys."""
        os.makedirs(tmp_settings_dir, exist_ok=True)
        old_settings = {
            "model_size": "base",
            "vad_aggressiveness": 1,
        }
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump(old_settings, f)

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        # New keys should be present
        assert sm.get('audio_device_node') is None
        assert sm.get('audio_device_label') is None


class TestModelSizeMigration:
    """The old app stored model_size as a bare name (e.g. 'base').
    The new format is 'ggml-base.bin'. Migration should be automatic."""

    def test_migrate_bare_name(self):
        assert _migrate_model_size('base') == 'ggml-base.bin'

    def test_migrate_tiny(self):
        assert _migrate_model_size('tiny') == 'ggml-tiny.bin'

    def test_migrate_large(self):
        assert _migrate_model_size('large') == 'ggml-large.bin'

    def test_already_migrated(self):
        assert _migrate_model_size('ggml-base.bin') == 'ggml-base.bin'

    def test_migrate_none(self):
        assert _migrate_model_size(None) is None

    def test_migration_on_load(self, tmp_settings_dir):
        """Loading old-format model_size should auto-migrate."""
        os.makedirs(tmp_settings_dir, exist_ok=True)
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump({"model_size": "base"}, f)

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('model_size') == 'ggml-base.bin'

    def test_migration_on_default(self, tmp_settings_dir):
        """Even the default model_size should be migrated."""
        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('model_size') == 'ggml-base.bin'

    def test_already_migrated_not_double_migrated(self, tmp_settings_dir):
        """A model_size that already ends in .bin should not be changed."""
        os.makedirs(tmp_settings_dir, exist_ok=True)
        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump({"model_size": "ggml-medium.bin"}, f)

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('model_size') == 'ggml-medium.bin'

    # ── Non-whisper model names must NOT be migrated ──────────────────
    # The earlier migration was over-broad: any value not ending in
    # ``.bin`` was rewritten to ``ggml-<name>.bin``. That mangled
    # Parakeet directory names and cloud sentinel entries, so on every
    # relaunch the saved model would be replaced by a fallback.

    def test_parakeet_dir_name_unchanged(self):
        assert _migrate_model_size('parakeet-tdt-0.6b-v3-int8') \
            == 'parakeet-tdt-0.6b-v3-int8'

    def test_parakeet_v2_dir_name_unchanged(self):
        assert _migrate_model_size('parakeet-tdt-0.6b-v2-int8') \
            == 'parakeet-tdt-0.6b-v2-int8'

    def test_partial_download_unchanged(self):
        # In-flight download artifacts must round-trip without mangling
        # so the eventual rename to the real name still matches.
        assert _migrate_model_size('parakeet-tdt-0.6b-v2-int8.partial') \
            == 'parakeet-tdt-0.6b-v2-int8.partial'

    def test_cloud_sentinel_name_unchanged(self):
        assert _migrate_model_size('deepgram-nova-3') == 'deepgram-nova-3'

    def test_unknown_string_unchanged(self):
        # Strings that look nothing like the legacy short names should
        # pass through; the engine factory will raise a clear error.
        assert _migrate_model_size('something-else-entirely') \
            == 'something-else-entirely'

    def test_parakeet_survives_save_and_reload(self, tmp_settings_dir):
        """End-to-end: pick Parakeet, save, reload — must still be Parakeet."""
        sm1 = SettingsManager(settings_dir=tmp_settings_dir)
        sm1.set('model_size', 'parakeet-tdt-0.6b-v3-int8')
        sm1.save()

        sm2 = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm2.get('model_size') == 'parakeet-tdt-0.6b-v3-int8'

    def test_cloud_survives_save_and_reload(self, tmp_settings_dir):
        """End-to-end: pick Deepgram, save, reload — must still be Deepgram."""
        sm1 = SettingsManager(settings_dir=tmp_settings_dir)
        sm1.set('model_size', 'deepgram-nova-3')
        sm1.save()

        sm2 = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm2.get('model_size') == 'deepgram-nova-3'


class TestAudioDeviceMigration:
    """PyAudio device indices were replaced by PipeWire node names.

    The legacy keys ``audio_device_index`` / ``audio_device_name`` are
    dropped on load (an old index cannot be resolved to a node name) and
    the new ``audio_device_node`` / ``audio_device_label`` keys take over.
    """

    def _write(self, settings_dir, data):
        os.makedirs(settings_dir, exist_ok=True)
        settings_file = os.path.join(settings_dir, 'settings.json')
        with open(settings_file, 'w') as f:
            json.dump(data, f)

    def test_new_keys_in_defaults(self):
        assert DEFAULT_SETTINGS['audio_device_node'] is None
        assert DEFAULT_SETTINGS['audio_device_label'] is None

    def test_legacy_keys_removed_from_defaults(self):
        assert 'audio_device_index' not in DEFAULT_SETTINGS
        assert 'audio_device_name' not in DEFAULT_SETTINGS

    def test_legacy_keys_removed_on_load(self, tmp_settings_dir):
        self._write(tmp_settings_dir, {
            "audio_device_index": 13,
            "audio_device_name": "foo",
            "model_size": "ggml-base.bin",
            "auto_paste": True,
            "break_length": 7,
        })

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        all_settings = sm.get_all()

        # Legacy keys are gone — NOT mapped to the new keys.
        assert 'audio_device_index' not in all_settings
        assert 'audio_device_name' not in all_settings
        # New keys are present at their defaults.
        assert 'audio_device_node' in all_settings
        assert 'audio_device_label' in all_settings
        assert all_settings['audio_device_node'] is None
        assert all_settings['audio_device_label'] is None

    def test_other_values_preserved_through_migration(self, tmp_settings_dir):
        self._write(tmp_settings_dir, {
            "audio_device_index": 13,
            "audio_device_name": "foo",
            "model_size": "ggml-base.bin",
            "auto_paste": True,
            "break_length": 7,
            "transcripts": ["hello world"],
            "vad_aggressiveness": 2,
        })

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('model_size') == 'ggml-base.bin'
        assert sm.get('auto_paste') is True
        assert sm.get('break_length') == 7
        assert sm.get('transcripts') == ["hello world"]
        assert sm.get('vad_aggressiveness') == 2

    def test_saved_node_and_label_round_trip(self, tmp_settings_dir):
        self._write(tmp_settings_dir, {
            "audio_device_node": "alsa_input.usb-foo.analog-stereo",
            "audio_device_label": "Foo Mic",
        })

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        assert sm.get('audio_device_node') == "alsa_input.usb-foo.analog-stereo"
        assert sm.get('audio_device_label') == "Foo Mic"

    def test_legacy_keys_do_not_survive_save(self, tmp_settings_dir):
        """After load+save, the file on disk has no legacy keys either."""
        self._write(tmp_settings_dir, {
            "audio_device_index": 13,
            "audio_device_name": "foo",
        })

        sm = SettingsManager(settings_dir=tmp_settings_dir)
        sm.save()

        settings_file = os.path.join(tmp_settings_dir, 'settings.json')
        with open(settings_file, 'r') as f:
            on_disk = json.load(f)
        assert 'audio_device_index' not in on_disk
        assert 'audio_device_name' not in on_disk
        assert on_disk['audio_device_node'] is None
        assert on_disk['audio_device_label'] is None
