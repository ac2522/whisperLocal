"""Tests for audio.recorder.Recorder (PyAudio is mocked)."""

import os
import struct
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audio.recorder import Recorder

# The recorder always records at 16 kHz; the PipeWire ALSA PCM resamples.
RATE = 16000
CHUNK_DURATION_MS = 30
_CHUNK_SAMPLES = int(RATE * CHUNK_DURATION_MS / 1000)


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

    with patch("audio.recorder.pyaudio.PyAudio", return_value=fake_pa_instance):
        yield fake_pa_instance, fake_stream


@pytest.fixture
def no_pipewire_node(monkeypatch):
    """Ensure PIPEWIRE_NODE is unset before the test starts."""
    monkeypatch.delenv("PIPEWIRE_NODE", raising=False)


def _stop_after_first_read(recorder, stream, chunk=None):
    """Make the mocked stream stop the recorder after its first read."""
    payload = chunk or _make_fake_chunk()

    def _read(*args, **kwargs):
        recorder.stop()
        return payload

    stream.read.side_effect = _read


# -----------------------------------------------------------------------
# Recorder construction
# -----------------------------------------------------------------------

class TestRecorderInit:
    """Recorder takes a target_node (PipeWire node name), not a device index."""

    def test_default_target_node_is_none(self, mock_pyaudio):
        recorder = Recorder()
        assert recorder.target_node is None
        recorder.cleanup()

    def test_target_node_is_stored(self, mock_pyaudio):
        recorder = Recorder(target_node="alsa_input.usb-foo.analog-stereo")
        assert recorder.target_node == "alsa_input.usb-foo.analog-stereo"
        recorder.cleanup()

    def test_target_node_is_mutable(self, mock_pyaudio):
        recorder = Recorder(target_node="old.node")
        recorder.target_node = "new.node"
        assert recorder.target_node == "new.node"
        recorder.cleanup()


# -----------------------------------------------------------------------
# PIPEWIRE_NODE environment handling
# -----------------------------------------------------------------------

class TestPipewireNodeEnv:
    """The recorder routes audio via the PIPEWIRE_NODE env var around pa.open."""

    def test_env_set_during_open_and_removed_after(
            self, mock_pyaudio, no_pipewire_node):
        pa_instance, stream = mock_pyaudio
        recorder = Recorder(target_node="my.target.node")
        _stop_after_first_read(recorder, stream)

        seen_env = {}

        def _open(**kwargs):
            seen_env["value"] = os.environ.get("PIPEWIRE_NODE")
            return stream

        pa_instance.open.side_effect = _open

        recorder.record_button_mode()

        assert seen_env["value"] == "my.target.node"
        assert "PIPEWIRE_NODE" not in os.environ
        recorder.cleanup()

    def test_preexisting_env_value_is_restored(self, mock_pyaudio, monkeypatch):
        monkeypatch.setenv("PIPEWIRE_NODE", "xyz")
        pa_instance, stream = mock_pyaudio
        recorder = Recorder(target_node="my.target.node")
        _stop_after_first_read(recorder, stream)

        seen_env = {}

        def _open(**kwargs):
            seen_env["value"] = os.environ.get("PIPEWIRE_NODE")
            return stream

        pa_instance.open.side_effect = _open

        recorder.record_button_mode()

        assert seen_env["value"] == "my.target.node"
        assert os.environ.get("PIPEWIRE_NODE") == "xyz"
        recorder.cleanup()

    def test_open_failure_restores_env_and_propagates(
            self, mock_pyaudio, no_pipewire_node):
        pa_instance, _stream = mock_pyaudio
        pa_instance.open.side_effect = OSError("[Errno -9996] Invalid device")

        recorder = Recorder(target_node="my.target.node")
        with pytest.raises(OSError):
            recorder.record_button_mode()

        assert "PIPEWIRE_NODE" not in os.environ
        assert not recorder.is_recording
        recorder.cleanup()

    def test_open_failure_restores_preexisting_env(
            self, mock_pyaudio, monkeypatch):
        monkeypatch.setenv("PIPEWIRE_NODE", "xyz")
        pa_instance, _stream = mock_pyaudio
        pa_instance.open.side_effect = OSError("boom")

        recorder = Recorder(target_node="my.target.node")
        with pytest.raises(OSError):
            recorder.record_button_mode()

        assert os.environ.get("PIPEWIRE_NODE") == "xyz"
        assert not recorder.is_recording
        recorder.cleanup()

    def test_none_target_does_not_touch_env_or_pass_device_index(
            self, mock_pyaudio, no_pipewire_node):
        pa_instance, stream = mock_pyaudio
        recorder = Recorder(target_node=None)
        _stop_after_first_read(recorder, stream)

        seen_env = {}

        def _open(**kwargs):
            seen_env["value"] = os.environ.get("PIPEWIRE_NODE")
            return stream

        pa_instance.open.side_effect = _open

        recorder.record_button_mode()

        assert seen_env["value"] is None
        assert "PIPEWIRE_NODE" not in os.environ
        _args, kwargs = pa_instance.open.call_args
        assert "input_device_index" not in kwargs
        recorder.cleanup()

    def test_open_always_uses_default_device_at_16k(self, mock_pyaudio):
        """Even with a target node, no input_device_index and rate is 16000."""
        pa_instance, stream = mock_pyaudio
        recorder = Recorder(target_node="my.target.node")
        _stop_after_first_read(recorder, stream)

        recorder.record_button_mode()

        _args, kwargs = pa_instance.open.call_args
        assert "input_device_index" not in kwargs
        assert kwargs["rate"] == 16000
        assert kwargs["input"] is True
        recorder.cleanup()

    def test_silence_mode_sets_and_restores_env(
            self, mock_pyaudio, no_pipewire_node):
        pa_instance, stream = mock_pyaudio
        recorder = Recorder(target_node="my.target.node")

        seen_env = {}

        def _open(**kwargs):
            seen_env["value"] = os.environ.get("PIPEWIRE_NODE")
            return stream

        pa_instance.open.side_effect = _open

        fake_vad = MagicMock()
        fake_vad.chunk_samples = 512
        fake_vad.sample_rate = 16000

        call_count = [0]

        def _is_speech(*a, **kw):
            call_count[0] += 1
            if call_count[0] >= 3:
                recorder.stop()
            return False

        fake_vad.is_speech.side_effect = _is_speech

        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD", return_value=fake_vad):
            recorder.record_silence_mode(vad_aggressiveness=1, break_length=1)

        assert seen_env["value"] == "my.target.node"
        assert "PIPEWIRE_NODE" not in os.environ
        recorder.cleanup()

    def test_silence_mode_open_failure_restores_env(
            self, mock_pyaudio, no_pipewire_node):
        pa_instance, _stream = mock_pyaudio
        pa_instance.open.side_effect = OSError("[Errno -9993] Illegal combination")

        recorder = Recorder(target_node="my.target.node")
        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD") as mock_vad_cls:
            mock_vad = MagicMock()
            mock_vad.chunk_samples = 512
            mock_vad.sample_rate = 16000
            mock_vad_cls.return_value = mock_vad
            with pytest.raises(OSError):
                recorder.record_silence_mode()

        assert "PIPEWIRE_NODE" not in os.environ
        assert not recorder.is_recording
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
        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD") as mock_vad_cls:
            mock_vad = MagicMock()
            mock_vad.chunk_samples = 512
            mock_vad.sample_rate = 16000
            mock_vad_cls.return_value = mock_vad
            with pytest.raises(OSError):
                recorder.record_silence_mode()

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
        result = recorder._to_float32(raw)

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


class TestRecordSilenceModeSilero:
    """Silero-based silence-mode recording."""

    @pytest.fixture
    def mock_pyaudio_stream(self):
        """Fake PyAudio stream for scripted silence-mode runs."""
        with patch("audio.recorder.pyaudio.PyAudio") as pa_cls:
            pa = MagicMock()
            pa_cls.return_value = pa
            stream = MagicMock()
            pa.open.return_value = stream
            yield pa, stream

    def test_surfaces_download_failure(self, mock_pyaudio_stream):
        pa, stream = mock_pyaudio_stream
        with patch("audio.recorder.ensure_vad_model",
                   side_effect=RuntimeError("network down")):
            rec = Recorder()
            with pytest.raises(RuntimeError, match="network down"):
                rec.record_silence_mode(vad_aggressiveness=1, break_length=1)

    def test_resets_vad_before_use(self, mock_pyaudio_stream):
        pa, stream = mock_pyaudio_stream
        # One 30ms chunk of silence then stop flag trips
        stream.read.side_effect = [b"\x00" * 960] * 200

        fake_vad = MagicMock()
        fake_vad.is_speech.return_value = False
        fake_vad.sample_rate = 16000
        fake_vad.chunk_samples = 512

        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD", return_value=fake_vad):
            rec = Recorder()

            # Stop after a few reads so the test terminates
            call_count = [0]

            def counting_is_speech(*a, **kw):
                call_count[0] += 1
                if call_count[0] >= 3:
                    rec.stop()
                return False
            fake_vad.is_speech.side_effect = counting_is_speech

            rec.record_silence_mode(vad_aggressiveness=1, break_length=1)
            fake_vad.reset.assert_called_once()

    def test_stops_after_break_length_of_silence_post_speech(self, mock_pyaudio_stream):
        """Scripted: 1 s of speech then 2 s of silence with break_length=2."""
        pa, stream = mock_pyaudio_stream
        # 30ms chunks at 16kHz = 480 int16 samples = 960 bytes
        stream.read.return_value = b"\x00" * 960

        fake_vad = MagicMock()
        fake_vad.chunk_samples = 512
        fake_vad.sample_rate = 16000

        # Build a response sequence: first N calls return True (speech),
        # then False forever (silence). Recorder should stop after
        # break_length seconds of silence post-speech.
        speech_responses = [True] * 30  # ~1s of speech
        silence_responses = [False] * 1000  # plenty of silence
        fake_vad.is_speech.side_effect = speech_responses + silence_responses

        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD", return_value=fake_vad):
            rec = Recorder()
            audio = rec.record_silence_mode(vad_aggressiveness=1, break_length=2)

        # is_speech was called enough times to reach break_length silence
        # after the speech block. At 512-sample chunks in 16kHz, break_length=2s
        # = 2 * 16000 / 512 ≈ 62.5 silent calls. Plus 30 speech calls → ~93.
        assert fake_vad.is_speech.call_count >= 90
        assert fake_vad.is_speech.call_count < 200
        # audio is float32 in [-1, 1] at 16 kHz
        assert audio.dtype == np.float32
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_stream_cleaned_up_on_vad_error(self, mock_pyaudio_stream):
        pa, stream = mock_pyaudio_stream
        with patch("audio.recorder.ensure_vad_model",
                   side_effect=RuntimeError("model download failed")):
            rec = Recorder()
            with pytest.raises(RuntimeError):
                rec.record_silence_mode(vad_aggressiveness=1, break_length=1)
        # Stream was never opened because ensure_vad_model failed first.
        pa.open.assert_not_called()
