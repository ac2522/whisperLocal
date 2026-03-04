"""Audio device discovery and selection using PyAudio."""

import pyaudio


class DeviceManager:
    """Enumerate and select audio input devices."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_pa(self) -> None:
        """Re-initialize PyAudio if it has been terminated."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

    def list_input_devices(self) -> list[dict]:
        """Return a list of dicts for every device with input channels.

        Each dict contains the keys: index, name, channels, sample_rate.
        Devices that raise an exception when queried are silently skipped.
        """
        self._ensure_pa()
        devices: list[dict] = []
        for i in range(self._pa.get_device_count()):
            try:
                info = self._pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append(self._to_dict(i, info))
            except Exception:
                continue
        return devices

    def get_default_device(self) -> dict | None:
        """Return the default input device as a dict, or *None* on error."""
        self._ensure_pa()
        try:
            info = self._pa.get_default_input_device_info()
            return self._to_dict(info["index"], info)
        except Exception:
            return None

    def get_device_by_index(self, index: int) -> dict:
        """Return the device at *index*.

        If the index is invalid or the device has no input channels the
        method falls back to the default input device.
        """
        self._ensure_pa()
        try:
            info = self._pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                return self._to_dict(index, info)
        except Exception:
            pass

        # Fallback to default device
        return self.get_default_device()

    def cleanup(self) -> None:
        """Terminate the PyAudio instance. Safe to call multiple times."""
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(index: int, info: dict) -> dict:
        return {
            "index": index,
            "name": info.get("name", ""),
            "channels": info.get("maxInputChannels", 0),
            "sample_rate": info.get("defaultSampleRate", 0.0),
        }
