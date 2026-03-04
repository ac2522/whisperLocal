"""Tests for engine.whisper_engine.WhisperEngine."""

import os

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
