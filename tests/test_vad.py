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
