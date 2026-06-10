"""Tests for audio.device_manager.DeviceManager (PipeWire pw-dump based)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audio.device_manager import DeviceManager

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pw_dump_live.json"

# Node names / descriptions present in the live capture
WIRELESS_NODE = "alsa_input.usb-R__DE_Wireless_MICRO_201RXB2504903983-01.analog-stereo"
WIRELESS_DESC = "Wireless MICRO Analog Stereo"
HEADPHONES_DESC = "Raptor Lake-P/U/H cAVS Headphones Stereo Microphone"
DIGITAL_DESC = "Raptor Lake-P/U/H cAVS Digital Microphone"


@pytest.fixture
def live_dump():
    """The real captured pw-dump output (123 objects, 3 Audio/Source nodes)."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def dm_live(monkeypatch, live_dump):
    """DeviceManager whose _pw_dump returns the live capture."""
    monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: live_dump)
    return DeviceManager()


def _node(obj_id, name, desc, media_class="Audio/Source"):
    """Build a minimal pw-dump node object."""
    return {
        "id": obj_id,
        "type": "PipeWire:Interface:Node",
        "info": {
            "props": {
                "media.class": media_class,
                "node.name": name,
                "node.description": desc,
            }
        },
    }


def _default_metadata(entries):
    """Build a minimal pw-dump 'default' Metadata object."""
    return {
        "id": 31,
        "type": "PipeWire:Interface:Metadata",
        "props": {"metadata.name": "default"},
        "metadata": entries,
    }


def _make_dm(monkeypatch, dump):
    monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: dump)
    return DeviceManager()


# ------------------------------------------------------------------
# _pw_dump
# ------------------------------------------------------------------

class TestPwDump:
    """_pw_dump should run pw-dump and degrade gracefully on every failure."""

    def test_success_returns_parsed_json(self):
        result_obj = MagicMock(returncode=0, stdout='[{"id": 1}]')
        with patch("audio.device_manager.subprocess.run",
                   return_value=result_obj) as mock_run:
            dm = DeviceManager()
            assert dm._pw_dump() == [{"id": 1}]

        args, kwargs = mock_run.call_args
        assert args[0] == ["pw-dump"]
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") == 2.0

    def test_pw_dump_not_installed_returns_none(self):
        with patch("audio.device_manager.subprocess.run",
                   side_effect=FileNotFoundError("pw-dump")):
            dm = DeviceManager()
            assert dm._pw_dump() is None

    def test_timeout_returns_none(self):
        with patch("audio.device_manager.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(
                       cmd=["pw-dump"], timeout=2.0)):
            dm = DeviceManager()
            assert dm._pw_dump() is None

    def test_nonzero_returncode_returns_none(self):
        result_obj = MagicMock(returncode=1, stdout='[{"id": 1}]')
        with patch("audio.device_manager.subprocess.run",
                   return_value=result_obj):
            dm = DeviceManager()
            assert dm._pw_dump() is None

    def test_bad_json_returns_none(self):
        result_obj = MagicMock(returncode=0, stdout="this is not json {")
        with patch("audio.device_manager.subprocess.run",
                   return_value=result_obj):
            dm = DeviceManager()
            assert dm._pw_dump() is None

    def test_empty_output_returns_none(self):
        result_obj = MagicMock(returncode=0, stdout="")
        with patch("audio.device_manager.subprocess.run",
                   return_value=result_obj):
            dm = DeviceManager()
            assert dm._pw_dump() is None


# ------------------------------------------------------------------
# list_sources
# ------------------------------------------------------------------

class TestListSources:
    """list_sources should return only media.class == 'Audio/Source' nodes."""

    def test_live_dump_finds_three_sources(self, dm_live):
        sources = dm_live.list_sources()
        assert len(sources) == 3
        descs = {s["description"] for s in sources}
        assert descs == {WIRELESS_DESC, HEADPHONES_DESC, DIGITAL_DESC}

    def test_source_dict_shape(self, dm_live):
        sources = dm_live.list_sources()
        wireless = [s for s in sources if s["node_name"] == WIRELESS_NODE]
        assert len(wireless) == 1
        src = wireless[0]
        assert set(src.keys()) == {"node_name", "description", "id"}
        assert src["description"] == WIRELESS_DESC
        assert isinstance(src["id"], int)

    def test_excludes_non_source_classes(self, monkeypatch):
        dump = [
            _node(1, "mic", "Real Mic", media_class="Audio/Source"),
            _node(2, "speakers", "Speakers", media_class="Audio/Sink"),
            _node(3, "monitor", "Monitor of Speakers",
                  media_class="Audio/Source/Virtual"),
            _node(4, "stream", "Some App", media_class="Stream/Output/Audio"),
        ]
        dm = _make_dm(monkeypatch, dump)
        sources = dm.list_sources()
        assert sources == [
            {"node_name": "mic", "description": "Real Mic", "id": 1},
        ]

    def test_returns_empty_list_when_pw_dump_fails(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: None)
        dm = DeviceManager()
        assert dm.list_sources() == []

    def test_objects_without_props_are_skipped(self, monkeypatch):
        dump = [
            {"id": 1, "type": "PipeWire:Interface:Core"},
            {"id": 2, "type": "PipeWire:Interface:Node", "info": {}},
            _node(3, "mic", "Mic"),
        ]
        dm = _make_dm(monkeypatch, dump)
        assert [s["node_name"] for s in dm.list_sources()] == ["mic"]


# ------------------------------------------------------------------
# get_default_source
# ------------------------------------------------------------------

class TestGetDefaultSource:
    """get_default_source should follow the 'default.audio.source' metadata."""

    def test_live_dump_default_is_wireless_micro(self, dm_live):
        default = dm_live.get_default_source()
        assert default is not None
        # Must come from "default.audio.source" (the Wireless MICRO),
        # NOT "default.configured.audio.source" (an absent Scarlett).
        assert default["node_name"] == WIRELESS_NODE
        assert default["description"] == WIRELESS_DESC

    def test_returns_none_when_pw_dump_fails(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: None)
        dm = DeviceManager()
        assert dm.get_default_source() is None

    def test_returns_none_without_default_metadata(self, monkeypatch):
        dump = [_node(1, "mic", "Mic")]
        dm = _make_dm(monkeypatch, dump)
        assert dm.get_default_source() is None

    def test_ignores_configured_only_metadata(self, monkeypatch):
        """Only 'default.configured.audio.source' present -> None."""
        dump = [
            _node(1, "mic", "Mic"),
            _default_metadata([
                {"subject": 0, "key": "default.configured.audio.source",
                 "type": "Spa:String:JSON", "value": {"name": "mic"}},
            ]),
        ]
        dm = _make_dm(monkeypatch, dump)
        assert dm.get_default_source() is None

    def test_returns_none_when_named_node_not_a_source(self, monkeypatch):
        dump = [
            _node(1, "mic", "Mic"),
            _default_metadata([
                {"subject": 0, "key": "default.audio.source",
                 "type": "Spa:String:JSON", "value": {"name": "gone-mic"}},
            ]),
        ]
        dm = _make_dm(monkeypatch, dump)
        assert dm.get_default_source() is None


# ------------------------------------------------------------------
# find_source
# ------------------------------------------------------------------

class TestFindSource:
    def test_finds_existing_node(self, dm_live):
        src = dm_live.find_source(WIRELESS_NODE)
        assert src is not None
        assert src["node_name"] == WIRELESS_NODE
        assert src["description"] == WIRELESS_DESC

    def test_returns_none_for_unknown_node(self, dm_live):
        assert dm_live.find_source("alsa_input.does-not-exist") is None

    def test_returns_none_when_pw_dump_fails(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: None)
        dm = DeviceManager()
        assert dm.find_source(WIRELESS_NODE) is None


# ------------------------------------------------------------------
# resolve_target
# ------------------------------------------------------------------

class TestResolveTarget:
    """Central mic-resolution logic."""

    def test_none_saved_node_uses_default_description(self, dm_live):
        node, label = dm_live.resolve_target(None, None)
        assert node is None
        assert label == WIRELESS_DESC

    def test_none_saved_node_without_default(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: None)
        dm = DeviceManager()
        assert dm.resolve_target(None, None) == (None, "System Default")

    def test_saved_node_found(self, dm_live):
        node, label = dm_live.resolve_target(WIRELESS_NODE, "My saved label")
        assert node == WIRELESS_NODE
        assert label == WIRELESS_DESC

    def test_saved_node_missing_falls_back_with_label(self, dm_live):
        node, label = dm_live.resolve_target(
            "alsa_input.usb-Focusrite_Scarlett_2i4_USB-00.analog-stereo",
            "Scarlett 2i4 USB",
        )
        assert node is None
        assert label == (
            f"{WIRELESS_DESC} (fallback — Scarlett 2i4 USB not connected)"
        )

    def test_saved_node_missing_without_label_uses_node_name(self, dm_live):
        node, label = dm_live.resolve_target("gone.node", None)
        assert node is None
        assert label == f"{WIRELESS_DESC} (fallback — gone.node not connected)"

    def test_saved_node_missing_and_no_default(self, monkeypatch):
        dump = [_node(1, "other", "Other Mic")]  # no default metadata
        dm = _make_dm(monkeypatch, dump)
        node, label = dm.resolve_target("gone.node", "Old Mic")
        assert node is None
        assert label == "System Default (fallback — Old Mic not connected)"


# ------------------------------------------------------------------
# refresh / cleanup compatibility no-ops
# ------------------------------------------------------------------

class TestCompatibilityNoOps:
    def test_refresh_and_cleanup_do_not_raise(self, monkeypatch):
        monkeypatch.setattr(DeviceManager, "_pw_dump", lambda self: None)
        dm = DeviceManager()
        assert dm.refresh() is None
        assert dm.cleanup() is None
        dm.cleanup()  # safe to call repeatedly
