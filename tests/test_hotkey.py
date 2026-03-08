"""Tests for hotkey parsing in MainWindow."""

import pytest

from ui.main_window import MainWindow


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
