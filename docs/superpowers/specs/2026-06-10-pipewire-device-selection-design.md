# PipeWire-Native Microphone Selection — Design

**Date:** 2026-06-10
**Status:** Approved, awaiting implementation
**Owner:** ac2522

## 1. Problem & goal

The Settings → Audio → Input Device dropdown is populated from PortAudio's raw ALSA
enumeration. On the user's machine that yields a confusing and dangerous list:

| Entry | Reality |
|---|---|
| System Default | OK — follows the PipeWire default source |
| `sof-hda-dsp (hw:1,0)` | Headset-jack mic, **raw hardware** |
| `sof-hda-dsp (hw:1,6)` / `(hw:1,7)` | Built-in mic array, raw, two near-duplicates |
| `CalDigit Thunderbolt 3 Audio (hw:2,0)` | Dock's own audio chip (not the Scarlett) |
| `Scarlett 2i4 USB (hw:3,0)` | Scarlett, **raw hardware** |
| `pipewire` | PipeWire's ALSA compatibility PCM |
| `default` | Alias of the same PCM |

Raw `hw:` entries are traps: they bypass PipeWire, require exclusive device access,
and produce `OSError [Errno -9985] Device unavailable` when PipeWire (or a leaked
stream) holds the device — observed in production on 2026-06-09. `default` /
`pipewire` / "System Default" are three labels for one thing. Numeric PyAudio
indexes also shift between hotplugs, so a saved index can silently point at a
different device.

**User's workflow (drives the requirements):** mics rotate constantly — Scarlett 2i4
when docked (via CalDigit), Rode wireless USB receiver frequently, built-in laptop
mic as fallback. The selected mic will often *not* be connected.

**Goals:**
1. Dropdown lists only real microphones with human-friendly names, plus System Default.
2. Selection is stable across replug/reboot (stored by name, not index).
3. Recording always goes through PipeWire's shared route — exclusive-access
   failures become impossible.
4. When the chosen mic is absent, auto-fall back to the system default and *show*
   which mic is actually in use.
5. A recording that transcribes to empty text notifies the user ("No speech
   detected") instead of silently doing nothing — the 2026-06-09 muted-source
   incident was undiagnosable from the UI.

**Non-goals (v1):**
- Per-recording device picker / quick-switch outside Settings.
- Volume/mute control or unmute-on-record (the app will not mutate PipeWire state).
- Supporting non-PipeWire systems (PulseAudio-only, bare ALSA). If `pw-dump` is
  missing the app degrades to System Default-only, which is today's safe behaviour.
- Output device selection.

## 2. Architecture

Three touched layers, one new data flow:

```
pw-dump (subprocess, JSON)
      │
      ▼
DeviceManager.list_sources() ──► SettingsDialog dropdown (label=description, data=node_name)
DeviceManager.get_default_source()      │ saves
DeviceManager.find_source(node_name)    ▼
      │                    settings: audio_device_node (str|None)
      ▼                                 │
MainWindow._resolve_mic() ──────────────┤
      │ (target node or None)           │
      ▼                                 ▼
Recorder(target_node=...) ── sets PIPEWIRE_NODE env ──► pa.open(PortAudio default)
                                                         └─► PipeWire links stream
                                                             to target (or default)
```

### 2.1 DeviceManager (audio/device_manager.py — rewritten)

Drops PyAudio entirely. Public API:

- `list_sources() -> list[dict]` — runs `pw-dump` with a short timeout (2 s),
  parses JSON, returns `[{"node_name": str, "description": str, "id": int}]` for
  every node with `media.class == "Audio/Source"` (this class excludes sink
  monitors by construction). Returns `[]` on any failure (missing binary, timeout,
  parse error) — logged at WARNING, never raised.
- `get_default_source() -> dict | None` — reads the `default.audio.source` key
  from PipeWire's `Metadata` object in the same `pw-dump` output, resolves it to
  a source dict.
- `find_source(node_name: str) -> dict | None` — membership check used for
  fallback resolution.

One `pw-dump` invocation per call; callers are interactive (dialog open, recording
start, settings apply) so ~50 ms subprocess cost is fine. No caching to invalidate.

`refresh()` and `cleanup()` remain as no-ops for call-site compatibility, then call
sites are cleaned up to stop using them.

### 2.2 Settings (config/settings.py)

- New key `audio_device_node: str | None = None` (None ⇒ system default).
- Migration in the existing settings-load migration path: delete legacy
  `audio_device_index` and `audio_device_name` keys if present. An old integer
  index cannot be mapped to a node reliably, so it maps to None; the user re-picks
  their mic once.

### 2.3 Settings dialog (ui/settings_dialog.py)

- Combo populated from `DeviceManager.list_sources()`:
  - Item 0: `"System Default (currently: <default description>)"`, data `None`.
    If the default can't be determined: plain `"System Default"`.
  - One item per source: label `description`, data `node_name`.
  - If the saved `audio_device_node` is not among current sources (mic unplugged),
    append a greyed-but-selectable `"<saved name> (not connected)"` item carrying
    the saved node_name, so opening Settings doesn't silently discard the choice.
- The existing refresh-on-popup behaviour stays (re-runs `pw-dump`), so a
  just-plugged mic appears when the dropdown opens.
- Saving writes `audio_device_node`. MainWindow's "device changed → recreate
  recorder" comparison switches to this key.

### 2.4 Recorder (audio/recorder.py)

- Constructor becomes `Recorder(target_node: str | None = None)`. PyAudio device
  index parameters, `_validate_device`, and `_pick_sample_rate`'s device branch are
  removed — the stream always opens PortAudio's **default device** (the PipeWire
  ALSA PCM, shared access).
- Stream opening (both modes) is wrapped:
  1. If `target_node` is set, `os.environ["PIPEWIRE_NODE"] = target_node` before
     `pa.open()`; restore/remove the variable in a `finally` immediately after
     `pa.open()` returns. The plugin reads the variable during open, so the window
     is a few milliseconds; recordings are serialized by `is_recording`, so the
     process-global env is not a race in practice.
  2. If PipeWire can't honour the target (node vanished between check and open),
     it links to the default source instead — observed PipeWire behaviour, and
     exactly the fallback semantics we want.
- `MainWindow` recreates the Recorder when the setting changes (existing pattern).

**Verified 2026-06-10:** `PIPEWIRE_NODE=<node.name>` (name string — numeric IDs do
**not** work) with the `pipewire` ALSA PCM steers the capture link to the named
source. Verified via `pw-dump` link inspection while recording.

**Risk & contingency:** verification used a subprocess (`arecord`); in-process
`os.environ` mutation before `pa.open()` uses the same plugin mechanism but must be
proven first inside the PyInstaller bundle (Task 1 of the plan: a throwaway harness
that opens a PyAudio stream with the env var set and asserts the link target via
`pw-dump`). If the bundled PortAudio/ALSA stack ignores it, fallback plumbing:
enumerate PortAudio for the `pipewire` PCM and open it with an ALSA config override
(`pipewire:NODE=<name>` style custom PCM definition via `ALSA_CONFIG_PATH` drop-in).
Same route, more plumbing; the public Recorder API is unchanged either way.

### 2.5 Mic visibility & resolution (ui/main_window.py)

- New helper `_resolve_mic() -> tuple[str | None, str]` returning
  `(target_node_or_None, human_label)`:
  - Setting is None → `(None, "<default description>")`.
  - Saved node present → `(node_name, "<description>")`.
  - Saved node absent → `(None, "<default description> (fallback — <saved> not connected)")`.
- Called when recording starts and after settings apply; result shown in the
  existing status label area as `Mic: <label>` and logged at INFO (the 2026-06-09
  incident had zero log evidence of which device recorded silence).

### 2.6 No-speech feedback (ui/main_window.py)

In `_transcribe_and_emit`, when transcription returns empty/whitespace text — or
the Whisper silence token `[BLANK_AUDIO]`, which today is pasted verbatim (no
filtering exists; this work adds it): show a tray notification
"No speech detected — check your microphone" (via the existing `tray_icon`
`showMessage` pattern) and log at INFO with the resolved mic label. No notification
when text is produced.

## 3. Error handling summary

| Failure | Behaviour |
|---|---|
| `pw-dump` missing/fails/times out | Dropdown shows System Default only; recording uses default route; WARNING logged |
| Saved mic not connected at record time | Fallback to default; status label + log say so |
| Mic vanishes between resolve and stream open | PipeWire links to default; same as above on next status refresh |
| Default source muted / silent audio | Recording proceeds; empty transcript triggers "No speech detected" notification |
| All mics gone (no sources at all) | `pa.open` on default PCM fails → existing error-signal path shows the error |

## 4. Testing

- **Unit (pytest, no hardware):**
  - `pw-dump` JSON parsing against a captured fixture (sources, metadata default,
    monitors excluded, malformed JSON, empty output, binary missing → `[]`).
  - Settings migration: legacy keys removed, `audio_device_node` defaulted.
  - `_resolve_mic` matrix: None / present / absent saved node.
  - Recorder env handling with mocked PyAudio: `PIPEWIRE_NODE` set during open,
    restored after, absent when `target_node=None`, restored on open failure.
- **Manual (user, post-deploy):** record via Scarlett docked, Rode receiver,
  built-in mic; unplug selected mic and confirm fallback label + working recording;
  mute default source and confirm "No speech detected" notification.

## 5. Out-of-repo notes

- `pw-dump` ships with `pipewire-bin` on Ubuntu — already present (the app already
  assumes a PipeWire/Wayland/GNOME host for evdev + ydotool).
- PyInstaller: no new bundled deps; `pw-dump` is invoked from the host system.
  PyAudio remains bundled (still does the actual capture).
