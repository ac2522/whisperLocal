"""Tests for audio.device_manager.DeviceManager."""

from unittest.mock import MagicMock, patch

import pytest

from audio.device_manager import DeviceManager

# ------------------------------------------------------------------
# Shared mock device data
# ------------------------------------------------------------------

MOCK_DEVICES = {
    0: {
        "index": 0,
        "name": "Built-in Mic",
        "maxInputChannels": 2,
        "maxOutputChannels": 0,
        "defaultSampleRate": 44100.0,
    },
    1: {
        "index": 1,
        "name": "Rode Wireless MICRO",
        "maxInputChannels": 1,
        "maxOutputChannels": 0,
        "defaultSampleRate": 48000.0,
    },
    2: {
        "index": 2,
        "name": "HDMI Output",
        "maxInputChannels": 0,
        "maxOutputChannels": 2,
        "defaultSampleRate": 44100.0,
    },
}


@pytest.fixture
def mock_pyaudio():
    """Patch PyAudio so no real audio subsystem is needed."""
    pa_instance = MagicMock()
    pa_instance.get_device_count.return_value = len(MOCK_DEVICES)

    def _device_info_by_index(index):
        if index not in MOCK_DEVICES:
            raise IOError(f"Invalid device index {index}")
        return MOCK_DEVICES[index]

    pa_instance.get_device_info_by_index.side_effect = _device_info_by_index
    pa_instance.get_default_input_device_info.return_value = MOCK_DEVICES[0]

    with patch("audio.device_manager.pyaudio") as mock_mod:
        mock_mod.PyAudio.return_value = pa_instance
        yield pa_instance


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestListInputDevices:
    """list_input_devices should only return devices with input channels."""

    def test_list_input_devices(self, mock_pyaudio):
        dm = DeviceManager()
        devices = dm.list_input_devices()

        # HDMI Output (index 2) has maxInputChannels=0 and must be excluded
        assert len(devices) == 2
        assert devices[0] == {
            "index": 0,
            "name": "Built-in Mic",
            "channels": 2,
            "sample_rate": 44100.0,
        }
        assert devices[1] == {
            "index": 1,
            "name": "Rode Wireless MICRO",
            "channels": 1,
            "sample_rate": 48000.0,
        }


class TestGetDefaultDevice:
    """get_default_device should return the system default input device."""

    def test_get_default_device(self, mock_pyaudio):
        dm = DeviceManager()
        device = dm.get_default_device()

        assert device is not None
        assert device == {
            "index": 0,
            "name": "Built-in Mic",
            "channels": 2,
            "sample_rate": 44100.0,
        }


class TestGetDeviceByIndex:
    """get_device_by_index should return a specific device."""

    def test_get_device_by_index(self, mock_pyaudio):
        dm = DeviceManager()
        device = dm.get_device_by_index(1)

        assert device == {
            "index": 1,
            "name": "Rode Wireless MICRO",
            "channels": 1,
            "sample_rate": 48000.0,
        }

    def test_get_device_by_index_fallback(self, mock_pyaudio):
        """An invalid index should fall back to the default input device."""
        dm = DeviceManager()
        device = dm.get_device_by_index(999)

        # Should fall back to Built-in Mic (default)
        assert device == {
            "index": 0,
            "name": "Built-in Mic",
            "channels": 2,
            "sample_rate": 44100.0,
        }

    def test_get_device_by_index_output_only_fallback(self, mock_pyaudio):
        """A device with no input channels should fall back to default."""
        dm = DeviceManager()
        device = dm.get_device_by_index(2)

        # HDMI Output has 0 input channels -> fallback
        assert device == {
            "index": 0,
            "name": "Built-in Mic",
            "channels": 2,
            "sample_rate": 44100.0,
        }
