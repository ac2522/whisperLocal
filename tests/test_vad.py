"""Tests for audio.vad — Silero VAD integration."""

import pytest

from audio.vad import AGGRESSIVENESS_TO_THRESHOLD, aggressiveness_to_threshold


class TestAggressivenessMapping:
    def test_level_0_maps_to_0_3(self):
        assert aggressiveness_to_threshold(0) == 0.3

    def test_level_1_maps_to_0_5(self):
        assert aggressiveness_to_threshold(1) == 0.5

    def test_level_2_maps_to_0_7(self):
        assert aggressiveness_to_threshold(2) == 0.7

    def test_level_3_maps_to_0_9(self):
        assert aggressiveness_to_threshold(3) == 0.9

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="aggressiveness"):
            aggressiveness_to_threshold(4)
        with pytest.raises(ValueError, match="aggressiveness"):
            aggressiveness_to_threshold(-1)

    def test_constant_covers_all_levels(self):
        assert set(AGGRESSIVENESS_TO_THRESHOLD.keys()) == {0, 1, 2, 3}


import os
from unittest.mock import patch

from audio.vad import SILERO_MODEL_URL, ensure_vad_model


class TestEnsureVADModel:
    def test_skips_download_if_file_exists(self, tmp_path):
        existing = tmp_path / "silero_vad.onnx"
        existing.write_bytes(b"\x00" * 100)
        with patch("audio.vad.urllib.request.urlretrieve") as urlretrieve:
            result = ensure_vad_model(str(existing))
        urlretrieve.assert_not_called()
        assert result == str(existing)

    def test_downloads_to_partial_then_renames(self, tmp_path):
        dest = tmp_path / "silero_vad.onnx"
        partial = tmp_path / "silero_vad.onnx.partial"

        def fake_retrieve(url, filename):
            # Simulate urlretrieve writing the file at the target
            with open(filename, "wb") as f:
                f.write(b"\x00" * 100)

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=fake_retrieve) as urlretrieve:
            result = ensure_vad_model(str(dest))

        urlretrieve.assert_called_once()
        call_args = urlretrieve.call_args
        assert call_args.args[0] == SILERO_MODEL_URL
        assert call_args.args[1] == str(partial)
        assert dest.exists()
        assert not partial.exists()
        assert result == str(dest)

    def test_creates_parent_directory(self, tmp_path):
        dest = tmp_path / "nested" / "dir" / "silero_vad.onnx"

        def fake_retrieve(url, filename):
            with open(filename, "wb") as f:
                f.write(b"\x00")

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=fake_retrieve):
            ensure_vad_model(str(dest))

        assert dest.exists()

    def test_bare_filename_does_not_crash_on_mkdir(self, tmp_path, monkeypatch):
        """A dest_path with no directory component must not crash os.makedirs."""
        monkeypatch.chdir(tmp_path)

        def fake_retrieve(url, filename):
            with open(filename, "wb") as f:
                f.write(b"\x00")

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=fake_retrieve):
            # Bare filename — os.path.dirname returns ""
            result = ensure_vad_model("silero_vad.onnx")
        assert result == "silero_vad.onnx"
        assert (tmp_path / "silero_vad.onnx").exists()

    def test_cleans_up_partial_on_failure(self, tmp_path):
        dest = tmp_path / "silero_vad.onnx"
        partial = tmp_path / "silero_vad.onnx.partial"

        def failing_retrieve(url, filename):
            # Simulate a partially-written download before the error
            with open(filename, "wb") as f:
                f.write(b"\x00" * 50)
            raise RuntimeError("network down")

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=failing_retrieve):
            with pytest.raises(RuntimeError, match="network down"):
                ensure_vad_model(str(dest))

        assert not dest.exists()
        assert not partial.exists()

import numpy as np
from unittest.mock import MagicMock

from audio.vad import SileroVAD


@pytest.fixture
def mock_onnx_session():
    """Patch onnxruntime.InferenceSession with a MagicMock that
    returns scripted probabilities + a state tensor."""
    with patch("audio.vad.onnxruntime.InferenceSession") as cls:
        session = MagicMock()
        # Default: probability 0.8, state is just zeros
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        cls.return_value = session
        yield cls, session


class TestSileroVADConstruction:
    def test_loads_model(self, mock_onnx_session):
        cls, _ = mock_onnx_session
        SileroVAD("/tmp/silero_vad.onnx")
        cls.assert_called_once()
        args, kwargs = cls.call_args
        assert args[0] == "/tmp/silero_vad.onnx"
        assert kwargs["providers"] == ["CPUExecutionProvider"]

    def test_custom_providers(self, mock_onnx_session):
        cls, _ = mock_onnx_session
        SileroVAD("/tmp/silero_vad.onnx", providers=["CUDAExecutionProvider"])
        assert cls.call_args.kwargs["providers"] == ["CUDAExecutionProvider"]

    def test_sample_rate_and_chunk_samples(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.sample_rate == 16000
        assert vad.chunk_samples == 512


class TestSileroVADIsSpeech:
    def test_accepts_correct_input_shape(self, mock_onnx_session):
        _, session = mock_onnx_session
        vad = SileroVAD("/tmp/silero_vad.onnx")
        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk, threshold=0.5)
        feeds = session.run.call_args.args[1]
        assert feeds["input"].shape == (1, 512)
        assert feeds["input"].dtype == np.float32

    def test_passes_sample_rate_as_int64(self, mock_onnx_session):
        _, session = mock_onnx_session
        vad = SileroVAD("/tmp/silero_vad.onnx")
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        sr = session.run.call_args.args[1]["sr"]
        assert int(sr) == 16000
        assert sr.dtype == np.int64

    def test_threads_state_between_calls(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.full((2, 1, 128), 0.42, dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")

        # First call: state is zeros
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        first_state = session.run.call_args.args[1]["state"]
        assert np.all(first_state == 0.0)

        # Second call: state is the output from the first call
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        second_state = session.run.call_args.args[1]["state"]
        assert np.allclose(second_state, 0.42)

    def test_threshold_comparison_is_inclusive(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.5]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5) is True

    def test_probability_below_threshold_returns_false(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.4]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5) is False


class TestSileroVADInputValidation:
    def test_wrong_sample_count_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="512"):
            vad.is_speech(np.zeros(480, dtype=np.float32), threshold=0.5)

    def test_wrong_dtype_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="float32"):
            vad.is_speech(np.zeros(512, dtype=np.float64), threshold=0.5)

    def test_non_1d_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="1-D"):
            vad.is_speech(np.zeros((1, 512), dtype=np.float32), threshold=0.5)


class TestSileroVADReset:
    def test_reset_zeros_state(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.full((2, 1, 128), 0.42, dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")

        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        # state is now 0.42
        vad.reset()
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        # After reset, first call should have used zero state
        state = session.run.call_args_list[-1].args[1]["state"]
        assert np.all(state == 0.0)
