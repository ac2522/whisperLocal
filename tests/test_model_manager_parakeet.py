"""Tests for ModelManager handling of Parakeet (directory-based) models."""

import os
from unittest.mock import patch

import pytest

from engine.model_manager import ModelManager


@pytest.fixture
def manager_with_models(tmp_path):
    """ModelManager pointing at a temp dir seeded with one fake Whisper file
    and one fake Parakeet directory."""
    # Whisper-style: single .bin
    (tmp_path / "ggml-base.bin").write_bytes(b"\x00" * 1024)
    # Parakeet-style: directory with encoder-model.int8.onnx
    pdir = tmp_path / "parakeet-tdt-0.6b-v3-int8"
    pdir.mkdir()
    (pdir / "encoder-model.int8.onnx").write_bytes(b"\x00" * 1024)
    (pdir / "decoder_joint-model.int8.onnx").write_bytes(b"\x00" * 1024)
    (pdir / "vocab.txt").write_text("a\nb\n")
    return ModelManager(str(tmp_path))


class TestListDownloadedMixed:
    def test_lists_both_whisper_and_parakeet(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        names = [m["name"] for m in downloaded]
        assert "ggml-base.bin" in names
        assert "parakeet-tdt-0.6b-v3-int8" in names

    def test_each_entry_has_type_field(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        types = {m["name"]: m["type"] for m in downloaded}
        assert types["ggml-base.bin"] == "whisper"
        assert types["parakeet-tdt-0.6b-v3-int8"] == "parakeet"

    def test_parakeet_entry_path_is_directory(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        entry = next(m for m in downloaded if m["type"] == "parakeet")
        assert os.path.isdir(entry["path"])

    def test_directory_without_encoder_is_ignored(self, tmp_path):
        # A random directory should not show up.
        (tmp_path / "ggml-base.bin").write_bytes(b"\x00")
        (tmp_path / "stray-dir").mkdir()
        (tmp_path / "stray-dir" / "vocab.txt").write_text("x")
        mm = ModelManager(str(tmp_path))
        names = [m["name"] for m in mm.list_downloaded()]
        assert "stray-dir" not in names


class TestIsDownloadedDirectory:
    def test_true_for_parakeet_directory(self, manager_with_models):
        assert manager_with_models.is_downloaded("parakeet-tdt-0.6b-v3-int8")

    def test_false_for_missing_directory(self, manager_with_models):
        assert not manager_with_models.is_downloaded("parakeet-tdt-0.6b-v2-int8")


class TestGetModelPathDirectory:
    def test_returns_directory_path(self, manager_with_models):
        path = manager_with_models.get_model_path("parakeet-tdt-0.6b-v3-int8")
        assert os.path.isdir(path)
        assert path.endswith("parakeet-tdt-0.6b-v3-int8")


class TestDeleteModelDirectory:
    def test_removes_directory_recursively(self, manager_with_models):
        manager_with_models.delete_model("parakeet-tdt-0.6b-v3-int8")
        assert not manager_with_models.is_downloaded("parakeet-tdt-0.6b-v3-int8")
