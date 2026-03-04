"""Tests for audio.recorder.Recorder (PyAudio is mocked)."""

import struct
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audio.recorder import WHISPER_RATE, CHUNK_DURATION_MS, Recorder

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
    # Make is_format_supported succeed so _pick_sample_rate returns WHISPER_RATE
    fake_pa_instance.is_format_supported.return_value = True

    with patch("audio.recorder.pyaudio.PyAudio", return_value=fake_pa_instance):
        yield fake_pa_instance, fake_stream


class TestRecorderButtonMode:
    """record_button_mode should return float32 audio when stopped."""

    def test_recorder_button_mode_returns_audio(self, mock_pyaudio):
        """Stop recording from a background thread after a short delay and
        verify the returned array is a float32 numpy array."""
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

        recorder.cleanup()


class TestToFloat32:
    """_to_float32 should produce normalised float32 output."""

    def test_recorder_converts_to_float32(self, mock_pyaudio):
        """Verify the conversion produces float32 values in [-1, 1]."""
        recorder = Recorder()

        # Build a chunk that contains both positive and negative int16 values.
        raw = struct.pack("<4h", -32768, -16384, 0, 32767)
        result = recorder._to_float32(raw, WHISPER_RATE)

        assert result.dtype == np.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0
        # Spot-check a known conversion
        np.testing.assert_almost_equal(result[2], 0.0)

        recorder.cleanup()


class TestCleanup:
    """cleanup() should be safe to call multiple times."""

    def test_cleanup_twice(self, mock_pyaudio):
        pa_instance, _ = mock_pyaudio
        recorder = Recorder()
        recorder.cleanup()
        recorder.cleanup()  # must not raise
        pa_instance.terminate.assert_called_once()
