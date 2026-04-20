"""Tests for engine.whisper_engine.WhisperEngine."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.whisper_engine import WhisperEngine

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "models", "ggml-base.bin"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)

requires_model = pytest.mark.skipif(
    not os.path.isfile(MODEL_PATH),
    reason=f"Model file not found: {MODEL_PATH}",
)


@requires_model
def test_engine_loads_model():
    """Model loads successfully and is_loaded() returns True."""
    engine = WhisperEngine(MODEL_PATH)
    try:
        assert engine.is_loaded()
    finally:
        engine.unload()


@requires_model
def test_engine_unload():
    """After unload(), the internal model is None."""
    engine = WhisperEngine(MODEL_PATH)
    engine.unload()
    assert not engine.is_loaded()


@requires_model
def test_engine_unload_is_idempotent():
    """Calling unload() twice must not raise."""
    engine = WhisperEngine(MODEL_PATH)
    engine.unload()
    engine.unload()  # second call -- should be a safe no-op


@requires_model
def test_engine_transcribe_silence():
    """Transcribing silence returns a string (possibly empty or whitespace)."""
    engine = WhisperEngine(MODEL_PATH)
    try:
        # 1 second of silence at 16 kHz
        silence = np.zeros(16000, dtype=np.float32)
        result = engine.transcribe(silence)
        assert isinstance(result, str)
    finally:
        engine.unload()


def test_engine_transcribe_requires_loaded_model():
    """transcribe() raises RuntimeError when no model is loaded."""
    engine = WhisperEngine.__new__(WhisperEngine)
    engine._model = None
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


@requires_model
def test_engine_context_manager():
    """The model is unloaded after exiting a ``with`` block."""
    with WhisperEngine(MODEL_PATH) as engine:
        assert engine.is_loaded()
    assert not engine.is_loaded()


@requires_model
def test_engine_reload_model():
    """reload() unloads the old model and loads a new one."""
    engine = WhisperEngine(MODEL_PATH)
    try:
        assert engine.is_loaded()
        # Reload with the same model file (just verifying the cycle works).
        engine.reload(MODEL_PATH)
        assert engine.is_loaded()
    finally:
        engine.unload()


@pytest.fixture
def mock_pwcpp_model():
    with patch("engine.whisper_engine.Model") as cls:
        fake = MagicMock()
        seg = MagicMock()
        seg.text = "hello world"
        fake.transcribe.return_value = [seg]
        cls.return_value = fake
        yield cls, fake


class TestWhisperEngineVocabulary:
    def test_no_vocabulary_omits_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000)
        # initial_prompt must not be a kwarg at all — pywhispercpp treats
        # any passed string as a prompt, including empty string, so omission
        # is important.
        assert "initial_prompt" not in fake.transcribe.call_args.kwargs

    def test_empty_vocabulary_omits_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000, vocabulary=[])
        assert "initial_prompt" not in fake.transcribe.call_args.kwargs

    def test_vocabulary_sets_initial_prompt(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        engine.transcribe(b"\x00" * 32000, vocabulary=["Avrillo", "SDLT"])
        prompt = fake.transcribe.call_args.kwargs["initial_prompt"]
        assert "Avrillo" in prompt
        assert "SDLT" in prompt

    def test_overlong_vocabulary_truncated(self, mock_pwcpp_model):
        _, fake = mock_pwcpp_model
        engine = WhisperEngine("/tmp/fake.bin")
        long_vocab = [f"word{i}" for i in range(100)]  # > 200 chars combined
        engine.transcribe(b"\x00" * 32000, vocabulary=long_vocab)
        prompt = fake.transcribe.call_args.kwargs["initial_prompt"]
        assert len(prompt) <= 200
        # And it must end on a complete word
        last_word = prompt.split(", ")[-1]
        assert last_word in long_vocab
