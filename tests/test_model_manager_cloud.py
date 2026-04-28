"""Tests for ModelManager handling of cloud (sentinel-URI) models."""

from unittest.mock import patch

import pytest

from engine.model_manager import ModelManager


@pytest.fixture
def mm(tmp_path):
    return ModelManager(str(tmp_path))


@pytest.fixture
def mm_with_local(tmp_path):
    """ModelManager seeded with one local Whisper file."""
    (tmp_path / "ggml-base.bin").write_bytes(b"\x00")
    return ModelManager(str(tmp_path))


class TestCatalog:
    def test_catalog_contains_deepgram_nova_3(self):
        names = [m["name"] for m in ModelManager.list_available()]
        assert "deepgram-nova-3" in names

    def test_cloud_entry_has_type_cloud(self):
        entry = next(
            m for m in ModelManager.list_available()
            if m["name"] == "deepgram-nova-3"
        )
        assert entry["type"] == "cloud"
        assert entry["cloud_uri"] == "cloud://deepgram-nova-3"


class TestListDownloaded:
    def test_includes_cloud_entry_with_empty_dir(self, mm):
        names = [m["name"] for m in mm.list_downloaded()]
        assert "deepgram-nova-3" in names

    def test_cloud_entry_has_uri_path(self, mm):
        entry = next(
            m for m in mm.list_downloaded() if m["name"] == "deepgram-nova-3"
        )
        assert entry["path"] == "cloud://deepgram-nova-3"
        assert entry["type"] == "cloud"

    def test_cloud_entry_listed_alongside_local(self, mm_with_local):
        names = [m["name"] for m in mm_with_local.list_downloaded()]
        assert "ggml-base.bin" in names
        assert "deepgram-nova-3" in names


class TestIsDownloaded:
    def test_true_for_cloud_name(self, mm):
        assert mm.is_downloaded("deepgram-nova-3") is True


class TestGetModelPath:
    def test_returns_cloud_uri_for_cloud_name(self, mm):
        assert mm.get_model_path("deepgram-nova-3") == "cloud://deepgram-nova-3"


class TestDownloadModel:
    def test_raises_for_cloud_entry(self, mm):
        with pytest.raises(ValueError, match="does not need to be downloaded"):
            mm.download_model("deepgram-nova-3")


class TestDeleteModel:
    def test_no_op_for_cloud_entry(self, mm):
        # Should not raise, should not affect anything else.
        mm.delete_model("deepgram-nova-3")
        assert mm.is_downloaded("deepgram-nova-3") is True
