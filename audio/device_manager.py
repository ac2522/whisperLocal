"""Audio source discovery via PipeWire (pw-dump).

Replaces the old PyAudio/ALSA enumeration, which surfaced raw ``hw:``
devices (exclusive-access traps that fail with ``-9985 Device
unavailable`` while PipeWire holds the hardware) and meaningless
aliases (``default``, ``pipewire``).  PipeWire's node list contains
exactly the real microphones, with stable names and human-readable
descriptions.
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

PW_DUMP_TIMEOUT = 2.0


class DeviceManager:
    """Enumerate PipeWire audio sources and resolve the recording target.

    Selection is stored by ``node.name`` (stable across hotplug/reboot),
    never by index.  All methods degrade gracefully when ``pw-dump`` is
    unavailable: the app then behaves as if only the system default
    exists, which is the pre-PipeWire-aware behaviour.
    """

    def _pw_dump(self):
        """Run ``pw-dump`` and return the parsed object list, or None."""
        try:
            result = subprocess.run(
                ["pw-dump"],
                capture_output=True,
                text=True,
                timeout=PW_DUMP_TIMEOUT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("pw-dump unavailable: %s", e)
            return None
        if result.returncode != 0:
            logger.warning("pw-dump exited with %d", result.returncode)
            return None
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning("pw-dump produced invalid JSON: %s", e)
            return None
        if not data:
            logger.warning("pw-dump produced no objects")
            return None
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sources(self):
        """Return every real microphone as a list of dicts.

        Each dict has the keys ``node_name``, ``description`` and ``id``.
        Only nodes with ``media.class == "Audio/Source"`` are included,
        which excludes sink monitors, playback streams and virtual
        devices.  Returns ``[]`` when PipeWire can't be queried.
        """
        data = self._pw_dump()
        if data is None:
            return []
        return self._extract_sources(data)

    def get_default_source(self):
        """Return the current system-default source dict, or None.

        Reads the ``default.audio.source`` key from PipeWire's "default"
        metadata object (the *current* default — not
        ``default.configured.audio.source``, which names the user's
        preference even when that device is unplugged).
        """
        data = self._pw_dump()
        if data is None:
            return None

        default_name = None
        for obj in data:
            if "Metadata" not in obj.get("type", ""):
                continue
            if obj.get("props", {}).get("metadata.name") != "default":
                continue
            for entry in obj.get("metadata", []):
                if entry.get("key") == "default.audio.source":
                    default_name = (entry.get("value") or {}).get("name")
                    break
            break

        if not default_name:
            return None
        for src in self._extract_sources(data):
            if src["node_name"] == default_name:
                return src
        return None

    def find_source(self, node_name):
        """Return the source dict for *node_name*, or None if absent."""
        for src in self.list_sources():
            if src["node_name"] == node_name:
                return src
        return None

    def resolve_target(self, saved_node, saved_label):
        """Resolve the saved mic choice against currently-present sources.

        Returns ``(target_node_or_None, human_label)``.  ``None`` as the
        target means "record from the system default".  When the saved
        mic isn't connected the label says so explicitly, so a fallback
        recording is never silent about which mic it used.
        """
        default = self.get_default_source()
        default_desc = default["description"] if default else "System Default"

        if saved_node is None:
            return None, default_desc

        found = self.find_source(saved_node)
        if found is not None:
            return saved_node, found["description"]

        return None, (
            f"{default_desc} (fallback — "
            f"{saved_label or saved_node} not connected)"
        )

    # ------------------------------------------------------------------
    # Compatibility no-ops
    # ------------------------------------------------------------------

    def refresh(self):
        """No-op. pw-dump is re-run on every query, so nothing is cached."""

    def cleanup(self):
        """No-op. No resources are held between queries."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(data):
        sources = []
        for obj in data:
            props = obj.get("info", {}).get("props", {})
            if props.get("media.class") != "Audio/Source":
                continue
            sources.append({
                "node_name": props.get("node.name", ""),
                "description": props.get("node.description", ""),
                "id": obj.get("id"),
            })
        return sources
