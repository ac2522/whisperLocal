"""Tests for engine.model_manager.ModelManager."""

import os

import pytest

from engine.model_manager import AVAILABLE_MODELS, ModelManager

# ---------------------------------------------------------------------------
# The real models/ directory shipped with the project.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@pytest.fixture
def manager():
    """ModelManager pointed at the real models/ directory."""
    return ModelManager(MODELS_DIR)


# ---------------------------------------------------------------------------
# Tests that use the real models/ directory
# ---------------------------------------------------------------------------


class TestListDownloaded:
    """list_downloaded() should report models present on disk."""

    def test_list_downloaded_models(self, manager):
        downloaded = manager.list_downloaded()
        names = [m["name"] for m in downloaded]
        assert "ggml-base.bin" in names
        assert "ggml-small.bin" in names

    def test_list_downloaded_includes_size(self, manager):
        downloaded = manager.list_downloaded()
        for model in downloaded:
            assert "size_mb" in model
            assert isinstance(model["size_mb"], float)
            assert model["size_mb"] > 0

    def test_list_downloaded_has_required_keys(self, manager):
        downloaded = manager.list_downloaded()
        for model in downloaded:
            assert "name" in model
            assert "path" in model
            assert "size_mb" in model
            assert "description" in model


class TestListAvailable:
    """list_available() should return the static catalogue."""

    def test_list_available_models(self, manager):
        available = manager.list_available()
        assert len(available) == len(AVAILABLE_MODELS)
        names = [m["name"] for m in available]
        assert "ggml-base.bin" in names
        assert "ggml-large-v3-turbo-q8_0.bin" in names

    def test_list_available_has_required_keys(self, manager):
        available = manager.list_available()
        for model in available:
            assert "name" in model
            assert "size_mb" in model
            assert "description" in model


class TestIsDownloaded:
    """is_downloaded() should check file existence."""

    def test_is_downloaded(self, manager):
        assert manager.is_downloaded("ggml-base.bin") is True
        assert manager.is_downloaded("ggml-small.bin") is True
        assert manager.is_downloaded("ggml-nonexistent.bin") is False


class TestGetModelPath:
    """get_model_path() should return the path or raise."""

    def test_get_model_path(self, manager):
        path = manager.get_model_path("ggml-base.bin")
        assert os.path.isfile(path)
        assert path.endswith("ggml-base.bin")

    def test_get_model_path_not_downloaded(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.get_model_path("ggml-nonexistent.bin")


class TestDeleteModel:
    """delete_model() should remove the file from disk."""

    def test_delete_model(self, tmp_path):
        """Use a temporary directory so we don't delete real models."""
        mm = ModelManager(str(tmp_path))
        fake_model = tmp_path / "ggml-fake.bin"
        fake_model.write_bytes(b"\x00" * 1024)
        assert fake_model.exists()

        mm.delete_model("ggml-fake.bin")
        assert not fake_model.exists()

    def test_delete_model_nonexistent(self, tmp_path):
        """Deleting a model that doesn't exist should not raise."""
        mm = ModelManager(str(tmp_path))
        mm.delete_model("ggml-nonexistent.bin")  # should not raise


class TestConstructor:
    """Constructor should create the models_dir if it doesn't exist."""

    def test_creates_models_dir(self, tmp_path):
        new_dir = str(tmp_path / "new_models")
        assert not os.path.exists(new_dir)
        ModelManager(new_dir)
        assert os.path.isdir(new_dir)
