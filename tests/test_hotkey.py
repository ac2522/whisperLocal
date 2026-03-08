"""Tests for hotkey parsing and keyboard device detection."""

from unittest.mock import MagicMock, patch

import pytest

from ui.main_window import MainWindow, _find_keyboard_devices


class TestParseHotkey:
    """Tests for MainWindow._parse_hotkey static method."""

    def test_ctrl_alt_shift_l(self):
        mods, key = MainWindow._parse_hotkey("Ctrl+Alt+Shift+L")
        assert mods == {"ctrl", "alt", "shift"}
        assert key == "l"

    def test_ctrl_alt_z(self):
        mods, key = MainWindow._parse_hotkey("Ctrl+Alt+Z")
        assert mods == {"ctrl", "alt"}
        assert key == "z"

    def test_single_key(self):
        mods, key = MainWindow._parse_hotkey("F12")
        assert mods == set()
        assert key == "f12"

    def test_ctrl_shift(self):
        mods, key = MainWindow._parse_hotkey("Ctrl+Shift+S")
        assert mods == {"ctrl", "shift"}
        assert key == "s"

    def test_case_insensitive(self):
        mods, key = MainWindow._parse_hotkey("CTRL+alt+SHIFT+a")
        assert mods == {"ctrl", "alt", "shift"}
        assert key == "a"

    def test_super_modifier(self):
        mods, key = MainWindow._parse_hotkey("Super+Z")
        assert mods == {"super"}
        assert key == "z"

    def test_control_alias(self):
        mods, key = MainWindow._parse_hotkey("Control+Alt+X")
        assert mods == {"ctrl", "alt"}
        assert key == "x"

    def test_meta_alias(self):
        mods, key = MainWindow._parse_hotkey("Meta+A")
        assert mods == {"super"}
        assert key == "a"

    def test_spaces_around_plus(self):
        mods, key = MainWindow._parse_hotkey("Ctrl + Alt + S")
        assert mods == {"ctrl", "alt"}
        assert key == "s"


def _make_mock_device(path, name, has_key_a=False, has_rel=False, has_btn_left=False):
    """Create a mock evdev InputDevice."""
    dev = MagicMock()
    dev.path = path
    dev.name = name

    caps = {}
    if has_key_a:
        keys = [30]  # KEY_A
        if has_btn_left:
            keys.append(0x110)  # BTN_LEFT
        caps[1] = keys  # EV_KEY
    if has_rel:
        caps[2] = [0, 1]  # EV_REL: REL_X, REL_Y

    dev.capabilities.return_value = caps
    return dev


class TestFindKeyboardDevices:
    """Tests for _find_keyboard_devices."""

    def test_prefers_keyboard_over_mouse(self):
        keyboard = _make_mock_device("/dev/input/event3", "AT keyboard", has_key_a=True)
        mouse = _make_mock_device("/dev/input/event23", "Logitech Mouse", has_key_a=True, has_rel=True, has_btn_left=True)

        with patch("ui.main_window.evdev") as mock_evdev:
            mock_evdev.list_devices.return_value = ["/dev/input/event23", "/dev/input/event3"]
            mock_evdev.InputDevice.side_effect = lambda p: mouse if p == "/dev/input/event23" else keyboard
            result = _find_keyboard_devices()

        assert result == ["/dev/input/event3"]

    def test_only_mouse_used_as_fallback(self):
        mouse = _make_mock_device("/dev/input/event23", "Logitech Mouse", has_key_a=True, has_rel=True, has_btn_left=True)

        with patch("ui.main_window.evdev") as mock_evdev:
            mock_evdev.list_devices.return_value = ["/dev/input/event23"]
            mock_evdev.InputDevice.side_effect = lambda p: mouse
            result = _find_keyboard_devices()

        assert result == ["/dev/input/event23"]

    def test_no_suitable_device_returns_empty(self):
        audio = _make_mock_device("/dev/input/event5", "HDA Audio", has_key_a=False)

        with patch("ui.main_window.evdev") as mock_evdev:
            mock_evdev.list_devices.return_value = ["/dev/input/event5"]
            mock_evdev.InputDevice.side_effect = lambda p: audio
            result = _find_keyboard_devices()

        assert result == []

    def test_multiple_keyboards_returns_all(self):
        kb1 = _make_mock_device("/dev/input/event3", "AT keyboard", has_key_a=True)
        kb2 = _make_mock_device("/dev/input/event4", "USB keyboard", has_key_a=True)

        with patch("ui.main_window.evdev") as mock_evdev:
            mock_evdev.list_devices.return_value = ["/dev/input/event3", "/dev/input/event4"]
            mock_evdev.InputDevice.side_effect = lambda p: kb1 if p == "/dev/input/event3" else kb2
            result = _find_keyboard_devices()

        assert "/dev/input/event3" in result
        assert "/dev/input/event4" in result

    def test_mouse_excluded_from_keyboard_list(self):
        """A device with EV_REL (relative axes) is a mouse, not included with keyboards."""
        mouse = _make_mock_device("/dev/input/event10", "Trackball", has_key_a=True, has_rel=True)
        keyboard = _make_mock_device("/dev/input/event3", "AT keyboard", has_key_a=True)

        with patch("ui.main_window.evdev") as mock_evdev:
            mock_evdev.list_devices.return_value = ["/dev/input/event10", "/dev/input/event3"]
            mock_evdev.InputDevice.side_effect = lambda p: mouse if p == "/dev/input/event10" else keyboard
            result = _find_keyboard_devices()

        assert result == ["/dev/input/event3"]
