"""Tests for engine.parakeet_engine.ParakeetEngine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.parakeet_engine import ParakeetEngine


@pytest.fixture
def mock_load_model():
    """Patch onnx_asr.load_model so no real ONNX session is created."""
    with patch("engine.parakeet_engine.onnx_asr.load_model") as m:
        fake = MagicMock()
        fake.recognize.return_value = "hello world"
        m.return_value = fake
        yield m, fake


class TestConstruction:
    def test_loads_model_from_path(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        load.assert_called_once()
        kwargs = load.call_args.kwargs
        assert kwargs["path"] == "/tmp/parakeet-tdt-0.6b-v3-int8"

    def test_passes_cuda_then_cpu_providers(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        kwargs = load.call_args.kwargs
        assert kwargs["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_uses_correct_model_id_for_v3(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/some/path/parakeet-tdt-0.6b-v3-int8")
        args = load.call_args.args
        assert args[0] == "nemo-parakeet-tdt-0.6b-v3"

    def test_uses_correct_model_id_for_v2(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/some/path/parakeet-tdt-0.6b-v2-int8")
        args = load.call_args.args
        assert args[0] == "nemo-parakeet-tdt-0.6b-v2"


class TestIsLoaded:
    def test_is_loaded_after_construction(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        assert engine.is_loaded() is True

    def test_is_loaded_false_after_unload(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        assert engine.is_loaded() is False


class TestTranscribe:
    def test_transcribe_passes_float32_array(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        audio = np.zeros(16000, dtype=np.float32)
        engine.transcribe(audio)
        called_arg = fake.recognize.call_args.args[0]
        assert called_arg.dtype == np.float32

    def test_transcribe_converts_int16_bytes_to_float32(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        # 1000 samples of int16 silence
        raw = (np.zeros(1000, dtype=np.int16)).tobytes()
        engine.transcribe(raw)
        called_arg = fake.recognize.call_args.args[0]
        assert called_arg.dtype == np.float32
        assert called_arg.shape == (1000,)

    def test_transcribe_normalizes_int16_ndarray(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        # Peak int16 value should normalize to ~1.0, not 32767.0
        audio = np.full(1000, 16384, dtype=np.int16)
        engine.transcribe(audio)
        called_arg = fake.recognize.call_args.args[0]
        assert called_arg.dtype == np.float32
        assert abs(called_arg.max() - 0.5) < 0.001

    def test_transcribe_passes_sample_rate_16000(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert fake.recognize.call_args.kwargs["sample_rate"] == 16000

    def test_transcribe_returns_text(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = "  hello   world  "
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        # Whitespace collapsed and stripped
        assert text == "hello world"

    def test_transcribe_handles_list_result(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = ["hello world"]
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "hello world"

    def test_transcribe_raises_when_unloaded(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.transcribe(np.zeros(16000, dtype=np.float32))


class TestUnload:
    def test_unload_is_idempotent(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        engine.unload()  # must not raise

    def test_context_manager_unloads(self, mock_load_model):
        with ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8") as engine:
            assert engine.is_loaded()
        assert not engine.is_loaded()


class TestReload:
    def test_reload_swaps_model(self, mock_load_model):
        load, _ = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.reload("/tmp/parakeet-tdt-0.6b-v2-int8")
        # Two load_model calls total: initial + reload
        assert load.call_count == 2
        assert load.call_args.kwargs["path"] == "/tmp/parakeet-tdt-0.6b-v2-int8"

    def test_reload_accepts_extra_kwargs_ignored(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        # WhisperEngine takes language=; ParakeetEngine should accept and ignore.
        engine.reload("/tmp/parakeet-tdt-0.6b-v3-int8", language="en")
