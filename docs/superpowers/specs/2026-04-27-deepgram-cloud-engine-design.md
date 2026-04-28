# Deepgram Cloud Transcription Engine — Design

**Date:** 2026-04-27
**Status:** Implemented 2026-04-28 (commits f604b93..a487061; awaiting Task 8 manual smoke test)
**Owner:** ac2522

> **Implementation note (added 2026-04-28):** §3's `transcribe()` outline shows
> `sample_rate` and `channels` as direct top-level kwargs to
> `client.listen.v1.media.transcribe_file(...)`. This was correct against the
> v5 SDK research the spec was based on, but `deepgram-sdk` v7 (the version
> that resolved against `>=5.0`) made `transcribe_file` keyword-only with no
> `**kwargs`, dropping those two parameters. The shipped engine routes them
> through `request_options={"additional_query_parameters": {"sample_rate":
> 16000, "channels": 1}}` instead, which surfaces them as the same query
> string on the wire. See commit b4a31a2 and the
> `TestSignatureCompatibility` tests for the regression guard. Behaviour is
> unchanged from the user's perspective.

## 1. Goal & scope

Add Deepgram (Nova-3) as a third selectable transcription engine alongside the existing local Whisper and Parakeet backends. The user picks "[Cloud] Deepgram Nova-3" from the same Settings → Model & Compute dropdown they already use to pick local models. Authentication is via the user's own API key, stored in the OS keyring or supplied through the `DEEPGRAM_API_KEY` environment variable. No existing local-engine functionality is removed or degraded.

**Non-goals (v1):**
- Other cloud providers (ElevenLabs Scribe, OpenAI Whisper API, etc.).
- Streaming / WebSocket transcription. The existing flow is record-then-transcribe; batch fits naturally.
- Diarization, sentiment, summarisation, custom Deepgram options beyond punctuation/keyterm.
- Language picker UI. Hardcoded to `language="en"` to match the current Whisper default.
- Auto-fallback from cloud to local on errors. Cloud failures surface in the error panel and the recording is dropped (matches existing local-engine error behaviour).
- Plain-text storage of API keys in `settings.json`.

## 2. Library choice & dependencies

- **`deepgram-sdk>=5.0`** (PyPI: `deepgram-sdk`) — official Python SDK. Used for the Listen v1 REST API only. Exposes `DeepgramClient(api_key=...).listen.v1.media.transcribe_file(request=audio_bytes, model="nova-3", ...)`. Auto-discovers `DEEPGRAM_API_KEY` from the environment when constructed without arguments. Brings `httpx` and `pydantic`, both small and broadly compatible.
- **`keyring>=25.0`** — standard cross-platform keyring access. On the user's GNOME/Wayland setup it routes through `SecretService` (gnome-keyring-daemon) via `secretstorage` + `jeepney`, both pulled in transitively as backend deps.

Both become hard requirements in `requirements.txt`. The Deepgram SDK is small enough that lazy install isn't worth the complexity, but module-level import is still avoided (see §3).

Approximate install footprint added: `deepgram-sdk` + `httpx` + `pydantic` ≈ 30 MB; `keyring` + `secretstorage` + `jeepney` ≈ 1 MB. No GPU/ML dependencies.

## 3. Engine abstraction

The existing `WhisperEngine` and `ParakeetEngine` define the implicit engine contract:

- `__init__(model_path, **kwargs)` (Whisper/Parakeet) — replaced by parameterless `__init__(model="nova-3", language="en")` for Deepgram, since there is no model file path.
- `transcribe(audio_data, *, vocabulary=None) -> str`
- `is_loaded() -> bool`
- `unload() -> None`
- `reload(*args, **kwargs) -> None`
- context-manager support (`__enter__` / `__exit__`)

Add `engine/deepgram_engine.py` with a `DeepgramEngine` class that matches this shape. Duck-typed; no formal `Protocol`/`ABC` — same rationale as the Parakeet design: three implementations still don't justify formalising a contract.

### Lazy import of the Deepgram SDK

Following the same defensive pattern as `ParakeetEngine` and `SileroVAD`, the `from deepgram import DeepgramClient` import is performed inside `_load()`, not at module top level. This keeps the SDK (and its `httpx`/`pydantic` chain) off the application's startup path, so a stale install or partial dep tree never blocks the app from launching with a local model selected.

### Class outline

```python
class DeepgramEngine:
    SAMPLE_RATE = 16000

    def __init__(self, model: str = "nova-3", language: str = "en"):
        self._model = model
        self._language = language
        self._client = None
        self._load()
        atexit.register(self.unload)

    def _load(self):
        from deepgram import DeepgramClient
        from config.api_keys import get_deepgram_key
        api_key = get_deepgram_key()
        if not api_key:
            raise RuntimeError(
                "Deepgram selected but no API key found. Set DEEPGRAM_API_KEY "
                "or save a key in Settings → Model & Compute → Deepgram API Key."
            )
        self._client = DeepgramClient(api_key=api_key)

    def is_loaded(self) -> bool:
        return self._client is not None

    def unload(self) -> None:
        # No persistent connection — just drop the client reference.
        self._client = None

    def reload(self, *_, **__) -> None:
        self.unload()
        self._load()

    def transcribe(self, audio_data, *, vocabulary=None) -> str:
        if self._client is None:
            raise RuntimeError("Deepgram client not initialised")

        pcm = _to_int16_pcm_bytes(audio_data)

        kwargs = {
            "model": self._model,
            "smart_format": True,
            "language": self._language,
            "encoding": "linear16",
            "sample_rate": self.SAMPLE_RATE,
            "channels": 1,
        }
        if vocabulary:
            kwargs["keyterm"] = list(vocabulary)

        resp = self._client.listen.v1.media.transcribe_file(request=pcm, **kwargs)
        try:
            return resp.results.channels[0].alternatives[0].transcript.strip()
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected Deepgram response shape: {e}"
            ) from e
```

### Audio conversion helper

Recorder produces either `bytes` (int16 PCM) or `numpy.ndarray` (float32 or int16). Helper coerces to int16 PCM bytes:

```python
def _to_int16_pcm_bytes(audio_data) -> bytes:
    if isinstance(audio_data, (bytes, bytearray)):
        return bytes(audio_data)
    arr = audio_data
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr.tobytes()
```

Sent as raw `linear16` PCM with `sample_rate=16000` and `channels=1` parameters — no WAV wrapper needed.

## 4. API key storage

Add `config/api_keys.py`:

```python
import logging
import os

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "whisperLocal"
DEEPGRAM_USERNAME = "deepgram"

def get_deepgram_key() -> str | None:
    env = os.environ.get("DEEPGRAM_API_KEY")
    if env:
        return env
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, DEEPGRAM_USERNAME)
    except Exception:
        logger.warning("keyring unavailable", exc_info=True)
        return None

def set_deepgram_key(value: str) -> None:
    import keyring
    keyring.set_password(KEYRING_SERVICE, DEEPGRAM_USERNAME, value)

def clear_deepgram_key() -> None:
    import keyring
    try:
        keyring.delete_password(KEYRING_SERVICE, DEEPGRAM_USERNAME)
    except keyring.errors.PasswordDeleteError:
        pass

def has_deepgram_key() -> bool:
    return get_deepgram_key() is not None

def get_key_source() -> str | None:
    """Return 'env', 'keyring', or None — for the Settings status label."""
    if os.environ.get("DEEPGRAM_API_KEY"):
        return "env"
    try:
        import keyring
        if keyring.get_password(KEYRING_SERVICE, DEEPGRAM_USERNAME):
            return "keyring"
    except Exception:
        pass
    return None
```

Design rules:

- **Env var wins.** Lets users override per-shell, per-systemd-unit, or in CI without touching the keyring.
- **Never logged.** Any logging at the call site logs key *presence*, not value.
- **Never stored in `settings.json`.** Plain JSON is world-readable on a single-user system but trivially shoulder-surfed; keyring entries require an unlocked session keyring (which on GNOME the user has already by being logged in).
- `keyring` import is local to each function (matches the rest of the codebase's defensive-import pattern) — a missing/broken keyring backend never blocks app startup or local-model use.

## 5. Model catalog & manager

Extend `engine/model_manager.py`:

### Schema change to `AVAILABLE_MODELS`

Add a new `type` value: `"cloud"`. Cloud entries carry a `cloud_uri` instead of `url` / `hf_repo`:

```python
{"name": "deepgram-nova-3", "type": "cloud", "size_mb": 0,
 "cloud_uri": "cloud://deepgram-nova-3",
 "description": "Deepgram Nova-3 (Cloud) — requires API key, ~$0.46/hr"},
```

### `ModelManager` behaviour for cloud entries

- `list_downloaded()` — append cloud entries to the result *after* the on-disk scan, regardless of filesystem state. Each entry's `path` is the `cloud_uri` ("cloud://deepgram-nova-3").
- `is_downloaded(name)` — return `True` for any cloud entry.
- `get_model_path(name)` — return the `cloud_uri` for cloud entries; existing path lookup for local models.
- `download_model(name)` — raise `ValueError("Cloud models do not need to be downloaded")` when called on a cloud entry. The settings UI prevents this from happening (see §7) but the manager defends defensively.
- `delete_model(name)` — no-op for cloud entries.

The catalog stays a single source of truth for both engine-aware code and the dropdown UI.

## 6. Factory dispatch

Extend `engine/factory.py`. The new branch comes *first* so cloud URIs aren't accidentally treated as filesystem paths by `os.path.exists`:

```python
def make_engine(model_path: str, **kwargs):
    if model_path.startswith("cloud://"):
        provider = model_path.removeprefix("cloud://")
        if provider == "deepgram-nova-3":
            from engine.deepgram_engine import DeepgramEngine
            return DeepgramEngine(model="nova-3", language="en")
        raise ValueError(f"Unknown cloud provider: {provider!r}")

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    # ... existing whisper/parakeet dispatch unchanged ...
```

## 7. Settings dialog UI

On the existing **Model & Compute** tab, add a new group below the "Compute" group:

```
┌─ Deepgram API Key ─────────────────────────────────────┐
│ Status: ● Configured   (or ○ Not set)                  │
│ [_________________________________________]  [Save]    │
│ [Clear stored key]                                     │
│ Tip: env var DEEPGRAM_API_KEY overrides stored key     │
└────────────────────────────────────────────────────────┘
```

Widgets:
- `QLineEdit` with `setEchoMode(QLineEdit.Password)` for the key entry.
- "Save" button → calls `set_deepgram_key(line_edit.text())`, refreshes the status label, clears the field.
- "Clear stored key" button → confirmation dialog, then `clear_deepgram_key()`, refreshes status.
- Status label refreshes from `has_deepgram_key()` (env + keyring). The label distinguishes between "set via env var" and "set via keyring" so the user understands precedence.

### Dropdown rendering

Extend `_format_model_label`:

```python
@staticmethod
def _format_model_label(model: dict) -> str:
    t = model.get("type")
    if t == "cloud":
        return f"[Cloud] {model['name']}"
    engine_tag = "Parakeet" if t == "parakeet" else "Whisper"
    return f"[{engine_tag}] {model['name']} ({model['size_mb']} MB)"
```

### Hide cloud entries from the Download/Delete combos

The "Download Model" combo (`_download_combo`) and the Delete button operate on local files only. Skip cloud entries when populating that combo, and short-circuit `_delete_model` when the selected model is type `"cloud"` (defense against future changes to the displayed dropdown).

### Pre-save warning

In `_save`, when the newly-selected `model_size` is a cloud entry and `has_deepgram_key()` is False, show a `QMessageBox.warning` ("Deepgram requires an API key — set one or transcription will fail"). Don't block the save; the warning gives the user a chance to add the key without losing their selection.

## 8. MainWindow integration

Three small changes in `ui/main_window.py`:

1. **`_create_engine`** — skip the GPU→CPU retry block for cloud paths. Cloud failures aren't backend-fallback-recoverable (a missing key on retry is still a missing key):
   ```python
   def _create_engine(self, model_path: str):
       if model_path.startswith("cloud://"):
           # Cloud engines have no compute backend; bypass GPU/CPU retry.
           self._active_compute_backend = "cloud"
           return make_engine(model_path)
       requested = (self.settings.get("compute_backend", "auto") or "auto").lower()
       self._apply_compute_backend()
       try:
           return make_engine(model_path)
       except Exception:
           # ... existing CPU-fallback logic unchanged ...
   ```

2. **`_detect_compute_backend`** — when the active engine is `DeepgramEngine`, return `"Cloud"` instead of CPU/CUDA/Vulkan:
   ```python
   from engine.deepgram_engine import DeepgramEngine
   if isinstance(self.engine, DeepgramEngine):
       return "Cloud"
   ```
   Status label then reads e.g. `Model: deepgram-nova-3 | Cloud | Mic: USB`.

3. **`_apply_compute_backend`** — no change. It still runs harmlessly for cloud (the env vars it sets only affect ggml/onnxruntime).

## 9. Vocabulary, errors, and edge cases

### Vocabulary

`custom_vocabulary` already flows through `transcribe(audio, vocabulary=...)` from `MainWindow._transcribe_and_emit`. `DeepgramEngine` maps it directly to the `keyterm` param. Per Deepgram docs, Nova-3 supports up to 500 tokens (~100 words) of keyterms. The existing UI text says "biases the decoder", which is accurate for both Whisper (`initial_prompt`) and Deepgram (`keyterm`); no copy change needed.

### Error surface

All Deepgram errors propagate up unchanged. `MainWindow._transcribe_and_emit` catches `Exception` and routes to `error_signal` → `ErrorPanel`. Concrete cases:

| Failure | Source | User-visible message |
|---|---|---|
| Missing key | `DeepgramEngine._load` | "Deepgram selected but no API key found…" |
| Wrong key (HTTP 401) | SDK | Raised exception text |
| No network | `httpx.ConnectError` | Raised exception text |
| Rate limit (HTTP 429) | SDK | Raised exception text |
| Audio too short / empty | SDK or empty-result handling | Raised exception or empty transcript (silently skipped, matches local engines) |

No retry queue, no degraded-mode fallback (per §1 non-goals).

### `_load_initial_engine` start-up

When the saved `model_size` is `"deepgram-nova-3"` and no key is configured, `DeepgramEngine.__init__` raises. The existing fallback in `_load_initial_engine` then tries the first downloaded local model, mirroring the current "saved model file missing" behaviour. The user gets a non-fatal warning logged and an empty status until they fix the key in Settings.

### Settings migration

None needed. Existing installations have `model_size == "ggml-*.bin"` or a Parakeet directory name — neither collides with `"deepgram-nova-3"`. New users who pick Deepgram first get the value written through the existing `_save` flow.

### PyInstaller spec changes

`whisper2text.spec` needs:
- `hiddenimports += ["keyring.backends.SecretService", "secretstorage", "jeepney", "jeepney.io.blocking"]`
- `datas += copy_metadata("keyring")` so `keyring`'s entry-point lookup finds backends inside the frozen app
- `hiddenimports += ["deepgram"]` (defensive — most of the SDK is plain-Python and PyInstaller picks it up automatically, but the lazy import in `_load` makes the static analyser miss it)

## 10. Testing

Manual smoke tests sufficient for v1 (matches how Parakeet and Silero VAD were verified). No mocked SDK tests — Deepgram has no useful local stub, and the failure surface is small enough to exercise interactively:

- Pick `[Cloud] Deepgram Nova-3` with no key → record → expect error panel "no API key found".
- Save a valid key in Settings → record short clip → expect transcript identical (modulo capitalisation) to a Whisper run on the same clip.
- Save an invalid key → record → expect error panel with 401 text.
- Disconnect network → record → expect error panel with connection error.
- Vocabulary → set `["Avrillo", "conveyancing"]`, record clip containing those terms, confirm spelling preserved.
- Switch back to a local model → record → confirm Whisper/Parakeet still works (engine swap clean).
- Status label reads `Model: deepgram-nova-3 | Cloud | Mic: …` while cloud is active.

## 11. Build sequence

The implementation plan (separate doc) will sequence work roughly as:

1. `config/api_keys.py` + `keyring` to `requirements.txt` — testable in isolation.
2. `engine/deepgram_engine.py` + `deepgram-sdk` to `requirements.txt` — testable via a one-off script before any UI wiring.
3. `engine/model_manager.py` cloud-entry support + `engine/factory.py` cloud branch.
4. `ui/settings_dialog.py` API key group + dropdown rendering tweaks + download-combo filtering.
5. `ui/main_window.py` `_create_engine` and `_detect_compute_backend` adjustments.
6. `whisper2text.spec` PyInstaller hidden-imports/metadata.
7. Manual smoke tests (§10) on the deployed binary.
