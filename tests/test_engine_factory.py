"""Tests for engine.factory.make_engine."""

from unittest.mock import MagicMock, patch

import pytest

from engine.factory import make_engine


@pytest.fixture
def patched_engines():
    """Patch both engine constructors so no real models load."""
    with patch("engine.factory.WhisperEngine") as w, \
         patch("engine.factory.ParakeetEngine") as p:
        w.return_value = MagicMock(name="WhisperEngineInstance")
        p.return_value = MagicMock(name="ParakeetEngineInstance")
        yield w, p


def test_returns_whisper_for_bin_file(tmp_path, patched_engines):
    w, p = patched_engines
    bin_path = tmp_path / "ggml-base.bin"
    bin_path.write_bytes(b"\x00")
    make_engine(str(bin_path))
    w.assert_called_once_with(str(bin_path))
    p.assert_not_called()


def test_returns_parakeet_for_directory_with_encoder_onnx(tmp_path, patched_engines):
    w, p = patched_engines
    model_dir = tmp_path / "parakeet-tdt-0.6b-v3-int8"
    model_dir.mkdir()
    (model_dir / "encoder-model.int8.onnx").write_bytes(b"\x00")
    make_engine(str(model_dir))
    p.assert_called_once_with(str(model_dir))
    w.assert_not_called()


def test_returns_parakeet_for_directory_with_unquantized_encoder(tmp_path, patched_engines):
    w, p = patched_engines
    model_dir = tmp_path / "parakeet-tdt-0.6b-v3"
    model_dir.mkdir()
    (model_dir / "encoder-model.onnx").write_bytes(b"\x00")
    make_engine(str(model_dir))
    p.assert_called_once_with(str(model_dir))
    w.assert_not_called()


def test_passes_through_extra_kwargs(tmp_path, patched_engines):
    w, _ = patched_engines
    bin_path = tmp_path / "ggml-base.bin"
    bin_path.write_bytes(b"\x00")
    make_engine(str(bin_path), language="fr", n_threads=8)
    w.assert_called_once_with(str(bin_path), language="fr", n_threads=8)


def test_raises_for_unrecognized_file(tmp_path, patched_engines):
    bogus = tmp_path / "model.txt"
    bogus.write_text("hi")
    with pytest.raises(ValueError, match="Unrecognized model"):
        make_engine(str(bogus))


def test_raises_for_directory_without_encoder(tmp_path, patched_engines):
    model_dir = tmp_path / "empty-dir"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("foo")
    with pytest.raises(ValueError, match="Unrecognized model"):
        make_engine(str(model_dir))


def test_raises_for_missing_path(patched_engines):
    with pytest.raises(ValueError, match="does not exist"):
        make_engine("/no/such/path/here")


class TestCloudDispatch:
    def test_returns_deepgram_for_cloud_uri(self):
        with patch("engine.factory.WhisperEngine") as w, \
             patch("engine.factory.ParakeetEngine") as p, \
             patch("engine.deepgram_engine.DeepgramEngine") as d:
            d.return_value = MagicMock(name="DeepgramEngineInstance")
            from engine.factory import make_engine
            make_engine("cloud://deepgram-nova-3")
            d.assert_called_once_with(model="nova-3", language="en")
            w.assert_not_called()
            p.assert_not_called()

    def test_raises_for_unknown_cloud_provider(self):
        from engine.factory import make_engine
        with pytest.raises(ValueError, match="Unknown cloud provider"):
            make_engine("cloud://nope-not-real")

    def test_cloud_uri_does_not_check_filesystem(self):
        # cloud:// branch must run before the os.path.exists check.
        with patch("engine.deepgram_engine.DeepgramEngine") as d:
            d.return_value = MagicMock()
            from engine.factory import make_engine
            # No exception even though "cloud://deepgram-nova-3" is not a path.
            make_engine("cloud://deepgram-nova-3")
            d.assert_called_once()
