"""Tests for audio.recorder.Recorder (PyAudio is mocked)."""

import struct
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audio.recorder import (
    WHISPER_RATE, CHUNK_DURATION_MS,
    Recorder, _validate_device, _pick_sample_rate,
)

# Use the whisper target rate for mock chunks
_CHUNK_SAMPLES = int(WHISPER_RATE * CHUNK_DURATION_MS / 1000)


def _make_fake_chunk(value: int = 1000, n_samples: int = _CHUNK_SAMPLES) -> bytes:
    """Return *n_samples* of 16-bit PCM audio packed as bytes."""
    return struct.pack(f"<{n_samples}h", *([value] * n_samples))


@pytest.fixture
def mock_pyaudio():
    """Patch ``pyaudio.PyAudio`` so no real audio hardware is needed."""
    fake_stream = MagicMock()
    fake_stream.read.return_value = _make_fake_chunk()

    fake_pa_instance = MagicMock()
    fake_pa_instance.open.return_value = fake_stream
    fake_pa_instance.is_format_supported.return_value = True
    fake_pa_instance.get_device_info_by_index.return_value = {
        "name": "Test Mic",
        "maxInputChannels": 2,
        "defaultSampleRate": 44100.0,
    }

    with patch("audio.recorder.pyaudio.PyAudio", return_value=fake_pa_instance):
        yield fake_pa_instance, fake_stream


# -----------------------------------------------------------------------
# _validate_device tests
# -----------------------------------------------------------------------

class TestValidateDevice:
    """Tests for device index validation."""

    def test_none_returns_none(self):
        pa = MagicMock()
        assert _validate_device(pa, None) is None

    def test_valid_device_returns_index(self):
        pa = MagicMock()
        pa.get_device_info_by_index.return_value = {
            "name": "Mic", "maxInputChannels": 1,
        }
        assert _validate_device(pa, 5) == 5

    def test_device_with_no_input_channels_returns_none(self):
        pa = MagicMock()
        pa.get_device_info_by_index.return_value = {
            "name": "Speakers", "maxInputChannels": 0,
        }
        assert _validate_device(pa, 3) is None

    def test_nonexistent_device_returns_none(self):
        pa = MagicMock()
        pa.get_device_info_by_index.side_effect = IOError("Invalid device")
        assert _validate_device(pa, 99) is None

    def test_oserror_returns_none(self):
        pa = MagicMock()
        pa.get_device_info_by_index.side_effect = OSError("No such device")
        assert _validate_device(pa, 11) is None


# -----------------------------------------------------------------------
# _pick_sample_rate tests
# -----------------------------------------------------------------------

class TestPickSampleRate:
    """Tests for sample rate selection."""

    def test_none_device_returns_whisper_rate(self):
        pa = MagicMock()
        assert _pick_sample_rate(pa, None) == WHISPER_RATE

    def test_supported_device_returns_whisper_rate(self):
        pa = MagicMock()
        pa.is_format_supported.return_value = True
        assert _pick_sample_rate(pa, 0) == WHISPER_RATE

    def test_unsupported_rate_returns_native_rate(self):
        pa = MagicMock()
        pa.is_format_supported.side_effect = ValueError("unsupported")
        pa.get_device_info_by_index.return_value = {"defaultSampleRate": 44100.0}
        assert _pick_sample_rate(pa, 0) == 44100


# -----------------------------------------------------------------------
# Recorder construction
# -----------------------------------------------------------------------

class TestRecorderInit:
    """Tests for Recorder initialization."""

    def test_invalid_device_falls_back_to_default(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        pa_instance.get_device_info_by_index.side_effect = IOError("bad device")
        recorder = Recorder(device_index=99)
        assert recorder._device_index is None
        recorder.cleanup()

    def test_output_only_device_falls_back_to_default(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        pa_instance.get_device_info_by_index.return_value = {
            "name": "Speakers", "maxInputChannels": 0,
        }
        recorder = Recorder(device_index=3)
        assert recorder._device_index is None
        recorder.cleanup()

    def test_valid_device_is_kept(self, mock_pyaudio):
        recorder = Recorder(device_index=5)
        assert recorder._device_index == 5
        recorder.cleanup()

    def test_none_device_stays_none(self, mock_pyaudio):
        recorder = Recorder(device_index=None)
        assert recorder._device_index is None
        recorder.cleanup()


# -----------------------------------------------------------------------
# Recording - button mode
# -----------------------------------------------------------------------

class TestRecorderButtonMode:
    """record_button_mode should return float32 audio when stopped."""

    def test_returns_float32_audio(self, mock_pyaudio):
        _pa_instance, _stream = mock_pyaudio
        recorder = Recorder()

        def _stop_soon():
            time.sleep(0.05)
            recorder.stop()

        stopper = threading.Thread(target=_stop_soon)
        stopper.start()

        audio = recorder.record_button_mode()
        stopper.join()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert not recorder.is_recording
        recorder.cleanup()

    def test_stream_open_failure_resets_recording_flag(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        pa_instance.open.side_effect = OSError("[Errno -9998] Invalid number of channels")

        recorder = Recorder()
        with pytest.raises(OSError):
            recorder.record_button_mode()

        assert not recorder.is_recording
        recorder.cleanup()

    def test_stream_read_failure_resets_recording_flag(self, mock_pyaudio):
        pa_instance, stream = mock_pyaudio
        stream.read.side_effect = OSError("Stream read error")

        recorder = Recorder()
        with pytest.raises(OSError):
            recorder.record_button_mode()

        assert not recorder.is_recording
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        recorder.cleanup()

    def test_stream_cleanup_failure_does_not_raise(self, mock_pyaudio):
        _pa_instance, stream = mock_pyaudio
        stream.stop_stream.side_effect = OSError("cleanup failed")

        recorder = Recorder()

        def _stop_soon():
            time.sleep(0.05)
            recorder.stop()

        stopper = threading.Thread(target=_stop_soon)
        stopper.start()
        audio = recorder.record_button_mode()
        stopper.join()

        assert isinstance(audio, np.ndarray)
        assert not recorder.is_recording
        recorder.cleanup()


# -----------------------------------------------------------------------
# Recording - silence mode
# -----------------------------------------------------------------------

class TestRecorderSilenceMode:
    """record_silence_mode tests."""

    def test_stream_open_failure_resets_recording_flag(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        pa_instance.open.side_effect = OSError("[Errno -9993] Illegal combination")

        recorder = Recorder()
        with pytest.raises(OSError):
            recorder.record_silence_mode()

        assert not recorder.is_recording
        recorder.cleanup()

    def test_stops_on_external_stop(self, mock_pyaudio):
        _pa_instance, _stream = mock_pyaudio
        recorder = Recorder()

        def _stop_soon():
            time.sleep(0.05)
            recorder.stop()

        stopper = threading.Thread(target=_stop_soon)
        stopper.start()

        with patch("audio.recorder.webrtcvad.Vad") as mock_vad_cls:
            mock_vad = MagicMock()
            mock_vad.is_speech.return_value = False
            mock_vad_cls.return_value = mock_vad
            audio = recorder.record_silence_mode()

        stopper.join()

        assert isinstance(audio, np.ndarray)
        assert not recorder.is_recording
        recorder.cleanup()


# -----------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------

class TestToFloat32:
    """_to_float32 should produce normalised float32 output."""

    def test_converts_to_float32(self, mock_pyaudio):
        recorder = Recorder()
        raw = struct.pack("<4h", -32768, -16384, 0, 32767)
        result = recorder._to_float32(raw, WHISPER_RATE)

        assert result.dtype == np.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        np.testing.assert_almost_equal(result[2], 0.0)
        recorder.cleanup()


# -----------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------

class TestCleanup:
    """cleanup() should be safe to call multiple times."""

    def test_cleanup_twice(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        recorder = Recorder()
        recorder.cleanup()
        recorder.cleanup()  # must not raise
        pa_instance.terminate.assert_called_once()

    def test_cleanup_resets_recording_flag(self, mock_pyaudio):
        _pa_instance, _ = mock_pyaudio
        recorder = Recorder()
        with recorder._lock:
            recorder._recording = True
        recorder.cleanup()
        assert not recorder.is_recording
