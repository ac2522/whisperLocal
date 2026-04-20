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
