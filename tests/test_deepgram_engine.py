"""Tests for engine.deepgram_engine.DeepgramEngine."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_sdk(monkeypatch):
    """Inject a fake `deepgram` module so the lazy import resolves to our mock."""
    fake_module = MagicMock(name="deepgram_module")
    fake_client_cls = MagicMock(name="DeepgramClient")
    fake_client = MagicMock(name="DeepgramClientInstance")

    # Build response object matching response.results.channels[0].alternatives[0].transcript
    fake_response = SimpleNamespace(
        results=SimpleNamespace(
            channels=[
                SimpleNamespace(
                    alternatives=[SimpleNamespace(transcript="  hello world  ")]
                )
            ]
        )
    )
    fake_client.listen.v1.media.transcribe_file.return_value = fake_response
    fake_client_cls.return_value = fake_client
    fake_module.DeepgramClient = fake_client_cls

    monkeypatch.setitem(sys.modules, "deepgram", fake_module)
    yield fake_client_cls, fake_client


@pytest.fixture
def with_key(monkeypatch):
    """Make config.api_keys.get_deepgram_key() return a usable key."""
    with patch("config.api_keys.get_deepgram_key", return_value="test-key"):
        yield


class TestConstruction:
    def test_loads_client_with_key(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        cls, _ = fake_sdk
        DeepgramEngine()
        cls.assert_called_once_with(api_key="test-key")

    def test_raises_when_no_key(self, fake_sdk, monkeypatch):
        with patch("config.api_keys.get_deepgram_key", return_value=None):
            from engine.deepgram_engine import DeepgramEngine
            with pytest.raises(RuntimeError, match="no API key"):
                DeepgramEngine()

    def test_is_loaded_after_construction(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        assert eng.is_loaded() is True


class TestTranscribe:
    def test_strips_and_returns_transcript(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        assert eng.transcribe(audio) == "hello world"

    def test_passes_model_and_language(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine(model="nova-3", language="en")
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio)
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["model"] == "nova-3"
        assert kwargs["language"] == "en"
        assert kwargs["smart_format"] is True
        assert kwargs["encoding"] == "linear16"
        assert kwargs["sample_rate"] == 16000
        assert kwargs["channels"] == 1

    def test_passes_keyterm_when_vocabulary_supplied(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio, vocabulary=["Avrillo", "conveyancing"])
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["keyterm"] == ["Avrillo", "conveyancing"]

    def test_omits_keyterm_when_vocabulary_empty(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio, vocabulary=[])
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert "keyterm" not in kwargs

    def test_omits_keyterm_when_vocabulary_none(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio)
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert "keyterm" not in kwargs

    def test_float32_audio_is_converted_to_int16_bytes(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        # 0.5 in float should become 16384 in int16 (0.5 * 32768).
        audio = np.full(4, 0.5, dtype=np.float32)
        eng.transcribe(audio)
        sent = client.listen.v1.media.transcribe_file.call_args.kwargs["request"]
        assert isinstance(sent, bytes)
        assert len(sent) == 8  # 4 samples * 2 bytes/int16
        assert np.frombuffer(sent, dtype=np.int16).tolist() == [16384] * 4

    def test_int16_bytes_audio_passes_through(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        raw = (np.array([100, -100, 200], dtype=np.int16)).tobytes()
        eng.transcribe(raw)
        sent = client.listen.v1.media.transcribe_file.call_args.kwargs["request"]
        assert sent == raw

    def test_unexpected_response_shape_raises(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        client.listen.v1.media.transcribe_file.return_value = SimpleNamespace(
            results=SimpleNamespace(channels=[])
        )
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Unexpected Deepgram response"):
            eng.transcribe(audio)


class TestLifecycle:
    def test_unload_drops_client(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        eng.unload()
        assert eng.is_loaded() is False

    def test_reload_reinitialises(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        cls, _ = fake_sdk
        eng = DeepgramEngine()
        eng.reload()
        assert cls.call_count == 2
        assert eng.is_loaded() is True
