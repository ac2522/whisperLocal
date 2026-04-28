# Deepgram Cloud Transcription Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Deepgram Nova-3 as a third selectable transcription engine alongside the existing local Whisper and Parakeet engines, with the user's API key stored in the OS keyring (or supplied via `DEEPGRAM_API_KEY`), and engine selection routed through the existing Settings model dropdown via a `cloud://` URI sentinel.

**Architecture:** A new `DeepgramEngine` class duck-types the engine contract (`transcribe`, `unload`, `reload`, `is_loaded`). The model catalog gains a `type="cloud"` entry whose "path" is a `cloud://deepgram-nova-3` sentinel; `make_engine` recognises this prefix and constructs the cloud engine without touching the filesystem. API keys live in `config/api_keys.py`, which prefers the env var and falls back to `keyring`. The Settings dialog grows a Deepgram API Key group and renders the cloud entry in the existing Transcription Model dropdown. No existing local-engine functionality changes.

**Tech Stack:** Python 3.10, PyQt5, `deepgram-sdk>=5.0` (REST client; lazy-imported), `keyring>=25.0` (Secret Service / GNOME Keyring backend), pytest + unittest.mock for tests.

**Spec reference:** `docs/superpowers/specs/2026-04-27-deepgram-cloud-engine-design.md` (commit 3995562)

**File map:**
- Create: `config/api_keys.py`, `engine/deepgram_engine.py`
- Create: `tests/test_api_keys.py`, `tests/test_deepgram_engine.py`, `tests/test_model_manager_cloud.py`
- Modify: `engine/model_manager.py`, `engine/factory.py`, `tests/test_engine_factory.py`
- Modify: `ui/settings_dialog.py`, `ui/main_window.py`
- Modify: `requirements.txt`, `whisper2text.spec`

---

## Task 1: API key storage module (TDD)

**Files:**
- Create: `tests/test_api_keys.py`
- Create: `config/api_keys.py`
- Modify: `requirements.txt`

Tiny module with five functions: `get_deepgram_key`, `set_deepgram_key`, `clear_deepgram_key`, `has_deepgram_key`, `get_key_source`. Env var wins; keyring is fallback. Imports `keyring` lazily inside each function so a broken backend never blocks app startup or local-engine use.

- [ ] **Step 1: Add `keyring` to requirements.txt**

Append at the end of `requirements.txt`:

```
# Cloud transcription (Deepgram) — API key storage in OS keyring
keyring>=25.0
```

- [ ] **Step 2: Install into the venv**

Run:
```bash
source whisper_env/bin/activate
pip install -r requirements.txt
```

Expected: `Successfully installed keyring-... secretstorage-... jeepney-...`. The `secretstorage` and `jeepney` packages come in transitively as the Linux Secret Service backend.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_api_keys.py` with this full content:

```python
"""Tests for config.api_keys."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure DEEPGRAM_API_KEY does not leak from the developer's shell."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    yield


@pytest.fixture
def fake_keyring(monkeypatch):
    """Patch the imported `keyring` module inside config.api_keys.

    The real module is imported lazily *inside each function*, so we
    monkeypatch sys.modules so the lazy import resolves to our fake.
    """
    fake = MagicMock()
    fake._store = {}

    def _set(service, user, value):
        fake._store[(service, user)] = value
    def _get(service, user):
        return fake._store.get((service, user))
    def _del(service, user):
        if (service, user) not in fake._store:
            from keyring.errors import PasswordDeleteError
            raise PasswordDeleteError("not found")
        del fake._store[(service, user)]

    fake.set_password = _set
    fake.get_password = _get
    fake.delete_password = _del

    import keyring as _real
    fake.errors = _real.errors

    monkeypatch.setitem(__import__("sys").modules, "keyring", fake)
    yield fake


class TestGetKey:
    def test_returns_env_when_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "env-value")
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "env-value"

    def test_returns_keyring_when_env_missing(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "ring-value"
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "ring-value"

    def test_env_wins_over_keyring(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "env-value")
        fake_keyring._store[("whisperLocal", "deepgram")] = "ring-value"
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() == "env-value"

    def test_returns_none_when_neither_present(self, fake_keyring):
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() is None

    def test_returns_none_when_keyring_raises(self, monkeypatch):
        # Simulate a keyring backend that blows up on any access.
        fake = MagicMock()
        fake.get_password.side_effect = RuntimeError("no backend")
        monkeypatch.setitem(__import__("sys").modules, "keyring", fake)
        from config.api_keys import get_deepgram_key
        assert get_deepgram_key() is None


class TestSetClear:
    def test_set_writes_to_keyring(self, fake_keyring):
        from config.api_keys import set_deepgram_key
        set_deepgram_key("new-value")
        assert fake_keyring._store[("whisperLocal", "deepgram")] == "new-value"

    def test_clear_removes_from_keyring(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "old"
        from config.api_keys import clear_deepgram_key
        clear_deepgram_key()
        assert ("whisperLocal", "deepgram") not in fake_keyring._store

    def test_clear_is_idempotent(self, fake_keyring):
        from config.api_keys import clear_deepgram_key
        clear_deepgram_key()  # no entry yet — must not raise


class TestHasKey:
    def test_true_when_env_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is True

    def test_true_when_keyring_set(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "x"
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is True

    def test_false_when_neither(self, fake_keyring):
        from config.api_keys import has_deepgram_key
        assert has_deepgram_key() is False


class TestGetKeySource:
    def test_returns_env_when_env_set(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        from config.api_keys import get_key_source
        assert get_key_source() == "env"

    def test_returns_keyring_when_only_keyring_set(self, fake_keyring):
        fake_keyring._store[("whisperLocal", "deepgram")] = "x"
        from config.api_keys import get_key_source
        assert get_key_source() == "keyring"

    def test_env_takes_precedence(self, monkeypatch, fake_keyring):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
        fake_keyring._store[("whisperLocal", "deepgram")] = "y"
        from config.api_keys import get_key_source
        assert get_key_source() == "env"

    def test_returns_none_when_nothing_set(self, fake_keyring):
        from config.api_keys import get_key_source
        assert get_key_source() is None
```

- [ ] **Step 4: Run tests — expect failure**

Run:
```bash
pytest tests/test_api_keys.py -v
```

Expected: collection error / `ModuleNotFoundError: No module named 'config.api_keys'`.

- [ ] **Step 5: Implement `config/api_keys.py`**

Create `config/api_keys.py` with this full content:

```python
"""API key storage for cloud transcription providers.

Resolution order: DEEPGRAM_API_KEY env var, then OS keyring entry
(service "whisperLocal", username "deepgram"). The `keyring` module
is imported lazily inside each function so a missing or broken
backend never blocks app startup or local-engine use.
"""

import logging
import os

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "whisperLocal"
DEEPGRAM_USERNAME = "deepgram"


def get_deepgram_key() -> str | None:
    """Return the configured Deepgram API key, or None.

    Env var wins over the keyring so users can override per shell or
    per systemd unit without touching stored secrets.
    """
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
    """Persist *value* into the OS keyring."""
    import keyring
    keyring.set_password(KEYRING_SERVICE, DEEPGRAM_USERNAME, value)


def clear_deepgram_key() -> None:
    """Remove the stored keyring entry. No-op if there is none."""
    import keyring
    try:
        keyring.delete_password(KEYRING_SERVICE, DEEPGRAM_USERNAME)
    except keyring.errors.PasswordDeleteError:
        pass


def has_deepgram_key() -> bool:
    """Return True if a key is available from any source."""
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

- [ ] **Step 6: Run tests — expect pass**

Run:
```bash
pytest tests/test_api_keys.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config/api_keys.py tests/test_api_keys.py
git commit -m "Add config.api_keys for Deepgram API key storage (env + keyring)"
```

---

## Task 2: DeepgramEngine class (TDD)

**Files:**
- Create: `tests/test_deepgram_engine.py`
- Create: `engine/deepgram_engine.py`
- Modify: `requirements.txt`

The engine wraps `deepgram-sdk`'s `client.listen.v1.media.transcribe_file` and exposes the same shape as `WhisperEngine` / `ParakeetEngine`. The SDK is imported lazily inside `_load()` (matching the Parakeet pattern). Audio comes in as a numpy float32 array (Recorder's output) or bytes; we coerce to int16 PCM and send with `encoding="linear16"`, `sample_rate=16000`, `channels=1` so Deepgram doesn't need to autodetect.

- [ ] **Step 1: Add `deepgram-sdk` to requirements.txt**

Append at the end of `requirements.txt`:

```
# Cloud transcription (Deepgram Nova-3 via Listen v1 REST API)
deepgram-sdk>=5.0
```

- [ ] **Step 2: Install into the venv**

Run:
```bash
source whisper_env/bin/activate
pip install -r requirements.txt
```

Expected: `Successfully installed deepgram-sdk-... httpx-... pydantic-... ...`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_deepgram_engine.py` with this full content:

```python
"""Tests for engine.deepgram_engine.DeepgramEngine."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_sdk(monkeypatch):
    """Inject a fake `deepgram` module so the lazy import resolves to our mock."""
    fake_module = MagicMock(name="deepgram_module")
    fake_client_cls = MagicMock(name="DeepgramClient")
    fake_client = MagicMock(name="DeepgramClientInstance")

    # Build response object matching response.results.channels[0].alternatives[0].transcript
    fake_response = SimpleNamespace(
        results=SimpleNamespace(
            channels=[
                SimpleNamespace(
                    alternatives=[SimpleNamespace(transcript="  hello world  ")]
                )
            ]
        )
    )
    fake_client.listen.v1.media.transcribe_file.return_value = fake_response
    fake_client_cls.return_value = fake_client
    fake_module.DeepgramClient = fake_client_cls

    monkeypatch.setitem(sys.modules, "deepgram", fake_module)
    yield fake_client_cls, fake_client


@pytest.fixture
def with_key(monkeypatch):
    """Make config.api_keys.get_deepgram_key() return a usable key."""
    with patch("config.api_keys.get_deepgram_key", return_value="test-key"):
        yield


class TestConstruction:
    def test_loads_client_with_key(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        cls, _ = fake_sdk
        DeepgramEngine()
        cls.assert_called_once_with(api_key="test-key")

    def test_raises_when_no_key(self, fake_sdk, monkeypatch):
        with patch("config.api_keys.get_deepgram_key", return_value=None):
            from engine.deepgram_engine import DeepgramEngine
            with pytest.raises(RuntimeError, match="no API key"):
                DeepgramEngine()

    def test_is_loaded_after_construction(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        assert eng.is_loaded() is True


class TestTranscribe:
    def test_strips_and_returns_transcript(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        assert eng.transcribe(audio) == "hello world"

    def test_passes_model_and_language(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine(model="nova-3", language="en")
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio)
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["model"] == "nova-3"
        assert kwargs["language"] == "en"
        assert kwargs["smart_format"] is True
        assert kwargs["encoding"] == "linear16"
        assert kwargs["sample_rate"] == 16000
        assert kwargs["channels"] == 1

    def test_passes_keyterm_when_vocabulary_supplied(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio, vocabulary=["Avrillo", "conveyancing"])
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert kwargs["keyterm"] == ["Avrillo", "conveyancing"]

    def test_omits_keyterm_when_vocabulary_empty(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio, vocabulary=[])
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert "keyterm" not in kwargs

    def test_omits_keyterm_when_vocabulary_none(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        eng.transcribe(audio)
        kwargs = client.listen.v1.media.transcribe_file.call_args.kwargs
        assert "keyterm" not in kwargs

    def test_float32_audio_is_converted_to_int16_bytes(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        # 0.5 in float should become 16384 in int16 (0.5 * 32768).
        audio = np.full(4, 0.5, dtype=np.float32)
        eng.transcribe(audio)
        sent = client.listen.v1.media.transcribe_file.call_args.kwargs["request"]
        assert isinstance(sent, bytes)
        assert len(sent) == 8  # 4 samples * 2 bytes/int16
        assert np.frombuffer(sent, dtype=np.int16).tolist() == [16384] * 4

    def test_int16_bytes_audio_passes_through(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        eng = DeepgramEngine()
        raw = (np.array([100, -100, 200], dtype=np.int16)).tobytes()
        eng.transcribe(raw)
        sent = client.listen.v1.media.transcribe_file.call_args.kwargs["request"]
        assert sent == raw

    def test_unexpected_response_shape_raises(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        _, client = fake_sdk
        client.listen.v1.media.transcribe_file.return_value = SimpleNamespace(
            results=SimpleNamespace(channels=[])
        )
        eng = DeepgramEngine()
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="Unexpected Deepgram response"):
            eng.transcribe(audio)


class TestLifecycle:
    def test_unload_drops_client(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        eng = DeepgramEngine()
        eng.unload()
        assert eng.is_loaded() is False

    def test_reload_reinitialises(self, fake_sdk, with_key):
        from engine.deepgram_engine import DeepgramEngine
        cls, _ = fake_sdk
        eng = DeepgramEngine()
        eng.reload()
        assert cls.call_count == 2
        assert eng.is_loaded() is True
```

- [ ] **Step 4: Run tests — expect failure**

Run:
```bash
pytest tests/test_deepgram_engine.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine.deepgram_engine'`.

- [ ] **Step 5: Implement `engine/deepgram_engine.py`**

Create `engine/deepgram_engine.py` with this full content:

```python
"""Wrapper around deepgram-sdk providing a managed cloud transcription engine.

Mirrors the public shape of WhisperEngine / ParakeetEngine so callers can
swap engines through engine.factory.make_engine.
"""

import atexit
import gc
import logging

import numpy as np

# NOTE: ``deepgram`` (and its httpx/pydantic chain) is deliberately NOT
# imported at module level. Following the same defensive pattern as
# ParakeetEngine and SileroVAD, the heavy import lives inside _load() so
# a stale or partial install never blocks the app from launching with a
# local model selected.

logger = logging.getLogger(__name__)


def _to_int16_pcm_bytes(audio_data) -> bytes:
    """Coerce numpy/bytes audio into int16 PCM bytes at 16 kHz mono."""
    if isinstance(audio_data, (bytes, bytearray)):
        return bytes(audio_data)
    arr = audio_data
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr.tobytes()


class DeepgramEngine:
    """Cloud transcription via Deepgram's Listen v1 REST API.

    Parameters
    ----------
    model : str, optional
        Deepgram model identifier (default ``"nova-3"``).
    language : str, optional
        BCP-47 language code (default ``"en"``).
    """

    SAMPLE_RATE = 16000

    def __init__(self, model: str = "nova-3", language: str = "en"):
        self._model = model
        self._language = language
        self._client = None
        self._load()
        atexit.register(self.unload)

    def _load(self) -> None:
        # Lazy import — see module docstring.
        from deepgram import DeepgramClient
        from config.api_keys import get_deepgram_key

        api_key = get_deepgram_key()
        if not api_key:
            raise RuntimeError(
                "Deepgram selected but no API key found. Set DEEPGRAM_API_KEY "
                "or save a key in Settings → Model & Compute → Deepgram API Key."
            )
        logger.info("Initialising Deepgram client (model=%s, language=%s)",
                    self._model, self._language)
        self._client = DeepgramClient(api_key=api_key)

    def is_loaded(self) -> bool:
        return self._client is not None

    def unload(self) -> None:
        if self._client is not None:
            logger.info("Releasing Deepgram client")
            self._client = None
            gc.collect()

    def reload(self, *_args, **_kwargs) -> None:
        """Re-read the API key and rebuild the client."""
        self.unload()
        self._load()

    def transcribe(self, audio_data, *, vocabulary: list[str] | None = None) -> str:
        """Send audio to Deepgram and return the cleaned transcript.

        Errors from the SDK (auth, network, rate limit) propagate to the
        caller. MainWindow routes them to the error panel.
        """
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

        response = self._client.listen.v1.media.transcribe_file(
            request=pcm, **kwargs
        )
        try:
            transcript = response.results.channels[0].alternatives[0].transcript
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected Deepgram response shape: {e}"
            ) from e
        return transcript.strip()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
```

- [ ] **Step 6: Run tests — expect pass**

Run:
```bash
pytest tests/test_deepgram_engine.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt engine/deepgram_engine.py tests/test_deepgram_engine.py
git commit -m "Add DeepgramEngine wrapping deepgram-sdk Listen v1 REST API"
```

---

## Task 3: Cloud entries in ModelManager (TDD)

**Files:**
- Create: `tests/test_model_manager_cloud.py`
- Modify: `engine/model_manager.py`

Add a synthetic cloud entry to `AVAILABLE_MODELS`. `list_downloaded()` always includes cloud entries (regardless of disk state); `is_downloaded()` returns True; `get_model_path()` returns `cloud://deepgram-nova-3`; `download_model()` raises a clear error; `delete_model()` is a no-op.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_manager_cloud.py` with this full content:

```python
"""Tests for ModelManager handling of cloud (sentinel-URI) models."""

from unittest.mock import patch

import pytest

from engine.model_manager import ModelManager


@pytest.fixture
def mm(tmp_path):
    return ModelManager(str(tmp_path))


@pytest.fixture
def mm_with_local(tmp_path):
    """ModelManager seeded with one local Whisper file."""
    (tmp_path / "ggml-base.bin").write_bytes(b"\x00")
    return ModelManager(str(tmp_path))


class TestCatalog:
    def test_catalog_contains_deepgram_nova_3(self):
        names = [m["name"] for m in ModelManager.list_available()]
        assert "deepgram-nova-3" in names

    def test_cloud_entry_has_type_cloud(self):
        entry = next(
            m for m in ModelManager.list_available()
            if m["name"] == "deepgram-nova-3"
        )
        assert entry["type"] == "cloud"
        assert entry["cloud_uri"] == "cloud://deepgram-nova-3"


class TestListDownloaded:
    def test_includes_cloud_entry_with_empty_dir(self, mm):
        names = [m["name"] for m in mm.list_downloaded()]
        assert "deepgram-nova-3" in names

    def test_cloud_entry_has_uri_path(self, mm):
        entry = next(
            m for m in mm.list_downloaded() if m["name"] == "deepgram-nova-3"
        )
        assert entry["path"] == "cloud://deepgram-nova-3"
        assert entry["type"] == "cloud"

    def test_cloud_entry_listed_alongside_local(self, mm_with_local):
        names = [m["name"] for m in mm_with_local.list_downloaded()]
        assert "ggml-base.bin" in names
        assert "deepgram-nova-3" in names


class TestIsDownloaded:
    def test_true_for_cloud_name(self, mm):
        assert mm.is_downloaded("deepgram-nova-3") is True


class TestGetModelPath:
    def test_returns_cloud_uri_for_cloud_name(self, mm):
        assert mm.get_model_path("deepgram-nova-3") == "cloud://deepgram-nova-3"


class TestDownloadModel:
    def test_raises_for_cloud_entry(self, mm):
        with pytest.raises(ValueError, match="do not need to be downloaded"):
            mm.download_model("deepgram-nova-3")


class TestDeleteModel:
    def test_no_op_for_cloud_entry(self, mm):
        # Should not raise, should not affect anything else.
        mm.delete_model("deepgram-nova-3")
        assert mm.is_downloaded("deepgram-nova-3") is True
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
pytest tests/test_model_manager_cloud.py -v
```

Expected: most tests fail. The catalog tests fail because `deepgram-nova-3` is not in `AVAILABLE_MODELS`; the others fail because cloud entries aren't recognised.

- [ ] **Step 3: Add the cloud entry to `AVAILABLE_MODELS`**

In `engine/model_manager.py`, append a new entry to `AVAILABLE_MODELS` after the last Parakeet entry (around line 68, before the closing `]`):

```python
    # ── Cloud (Deepgram) ─────────────────────────────────────────────
    {"name": "deepgram-nova-3", "type": "cloud", "size_mb": 0,
     "cloud_uri": "cloud://deepgram-nova-3",
     "description": "Deepgram Nova-3 (Cloud) — requires API key, ~$0.46/hr"},
```

- [ ] **Step 4: Update `list_downloaded`, `is_downloaded`, `get_model_path`**

Replace the body of `ModelManager.list_downloaded` in `engine/model_manager.py` with:

```python
    def list_downloaded(self) -> list[dict]:
        """Return a list of dicts for every downloaded model.

        Recognizes three on-disk shapes:
          * Whisper: any single ``.bin`` file in ``models_dir``.
          * Parakeet: any subdirectory containing ``encoder-model*.onnx``.
          * Cloud: synthetic catalog entries (always "downloaded").

        Each dict has keys: name, path, size_mb, description, type.
        """
        results = []
        for filename in sorted(os.listdir(self.models_dir)):
            full_path = os.path.join(self.models_dir, filename)

            if os.path.isfile(full_path) and filename.endswith(".bin"):
                size_bytes = os.path.getsize(full_path)
                model_type = "whisper"
            elif os.path.isdir(full_path) and glob.glob(
                os.path.join(full_path, "encoder-model*.onnx")
            ):
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for dirpath, _, files in os.walk(full_path)
                    for f in files
                )
                model_type = "parakeet"
            else:
                continue

            size_mb = round(size_bytes / (1024 * 1024), 1)
            known = _MODELS_BY_NAME.get(filename)
            description = known["description"] if known else filename

            results.append({
                "name": filename,
                "path": full_path,
                "size_mb": size_mb,
                "description": description,
                "type": model_type,
            })

        # Append cloud entries — always available regardless of disk state.
        for m in AVAILABLE_MODELS:
            if m.get("type") == "cloud":
                results.append({
                    "name": m["name"],
                    "path": m["cloud_uri"],
                    "size_mb": 0,
                    "description": m["description"],
                    "type": "cloud",
                })
        return results
```

Replace `is_downloaded` and `get_model_path` in the same file:

```python
    def is_downloaded(self, model_name: str) -> bool:
        """Return True if ``model_name`` exists on disk or is a cloud entry."""
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            return True
        path = os.path.join(self.models_dir, model_name)
        return os.path.isfile(path) or os.path.isdir(path)

    def get_model_path(self, model_name: str) -> str:
        """Return the full path or cloud URI for ``model_name``.

        Raises ``FileNotFoundError`` if a local model has not been downloaded.
        Cloud entries always resolve to their ``cloud_uri``.
        """
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            return known["cloud_uri"]
        path = os.path.join(self.models_dir, model_name)
        if not (os.path.isfile(path) or os.path.isdir(path)):
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {self.models_dir}"
            )
        return path
```

- [ ] **Step 5: Update `download_model` and `delete_model`**

In `ModelManager.download_model` in `engine/model_manager.py`, add a guard at the very top of the method (before reading `_MODELS_BY_NAME`):

```python
    def download_model(self, model_name: str, progress_callback=None) -> str:
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            raise ValueError(
                f"Cloud model '{model_name}' does not need to be downloaded"
            )
        model_type = (known or {}).get("type", "whisper")
        # ... rest of the existing method body unchanged ...
```

In `ModelManager.delete_model`, add a guard at the top:

```python
    def delete_model(self, model_name: str) -> None:
        """Delete the model file or directory from ``models_dir``.

        No-op for cloud entries (nothing to delete on disk).
        """
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            return
        path = os.path.join(self.models_dir, model_name)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
```

- [ ] **Step 6: Run cloud tests — expect pass**

Run:
```bash
pytest tests/test_model_manager_cloud.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 7: Run existing model_manager tests — expect no regressions**

Run:
```bash
pytest tests/test_model_manager.py tests/test_model_manager_parakeet.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add engine/model_manager.py tests/test_model_manager_cloud.py
git commit -m "ModelManager: support cloud:// sentinel entries (Deepgram Nova-3)"
```

---

## Task 4: Factory dispatch for cloud:// URIs (TDD)

**Files:**
- Modify: `engine/factory.py`
- Modify: `tests/test_engine_factory.py`

Add a branch at the top of `make_engine` that recognises `cloud://` and dispatches to `DeepgramEngine` without touching the filesystem. The branch comes first so `os.path.exists` never sees the URI.

- [ ] **Step 1: Add the failing tests**

Append the following test class to the end of `tests/test_engine_factory.py`:

```python


class TestCloudDispatch:
    def test_returns_deepgram_for_cloud_uri(self):
        with patch("engine.factory.WhisperEngine") as w, \
             patch("engine.factory.ParakeetEngine") as p, \
             patch("engine.deepgram_engine.DeepgramEngine") as d:
            d.return_value = MagicMock(name="DeepgramEngineInstance")
            from engine.factory import make_engine
            make_engine("cloud://deepgram-nova-3")
            d.assert_called_once_with(model="nova-3", language="en")
            w.assert_not_called()
            p.assert_not_called()

    def test_raises_for_unknown_cloud_provider(self):
        from engine.factory import make_engine
        with pytest.raises(ValueError, match="Unknown cloud provider"):
            make_engine("cloud://nope-not-real")

    def test_cloud_uri_does_not_check_filesystem(self):
        # cloud:// branch must run before the os.path.exists check.
        with patch("engine.deepgram_engine.DeepgramEngine") as d:
            d.return_value = MagicMock()
            from engine.factory import make_engine
            # No exception even though "cloud://deepgram-nova-3" is not a path.
            make_engine("cloud://deepgram-nova-3")
            d.assert_called_once()
```

- [ ] **Step 2: Run tests — expect failure**

Run:
```bash
pytest tests/test_engine_factory.py -v
```

Expected: the three new tests fail (`make_engine` doesn't recognise `cloud://`); existing tests still pass.

- [ ] **Step 3: Add the cloud branch to `make_engine`**

Replace the body of `engine/factory.py` with:

```python
"""Factory that picks a transcription engine based on model path or URI.

Whisper: a single .bin file (whisper.cpp GGML format).
Parakeet: a directory containing encoder-model*.onnx (onnx-asr format).
Cloud: a sentinel URI like 'cloud://deepgram-nova-3'.
"""

import glob
import os

from engine.parakeet_engine import ParakeetEngine
from engine.whisper_engine import WhisperEngine


def make_engine(model_path: str, **kwargs):
    """Return a transcription engine appropriate for the given model path or URI."""
    if model_path.startswith("cloud://"):
        provider = model_path.removeprefix("cloud://")
        if provider == "deepgram-nova-3":
            from engine.deepgram_engine import DeepgramEngine
            return DeepgramEngine(model="nova-3", language="en")
        raise ValueError(f"Unknown cloud provider: {provider!r}")

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    if os.path.isfile(model_path) and model_path.lower().endswith(".bin"):
        return WhisperEngine(model_path, **kwargs)

    if os.path.isdir(model_path):
        encoders = glob.glob(os.path.join(model_path, "encoder-model*.onnx"))
        if encoders:
            return ParakeetEngine(model_path, **kwargs)

    raise ValueError(
        f"Unrecognized model at {model_path!r}: expected a .bin file (Whisper) "
        f"or a directory containing encoder-model*.onnx (Parakeet)."
    )
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```bash
pytest tests/test_engine_factory.py -v
```

Expected: all tests pass (existing 7 + new 3 = 10).

- [ ] **Step 5: Commit**

```bash
git add engine/factory.py tests/test_engine_factory.py
git commit -m "Factory: dispatch cloud:// URIs to DeepgramEngine"
```

---

## Task 5: Settings dialog — Deepgram API Key group + cloud-aware dropdown

**Files:**
- Modify: `ui/settings_dialog.py`

The settings dialog gets:
1. A new "Deepgram API Key" group on the Model & Compute tab (status label, password field, Save button, Clear button).
2. `_format_model_label` extended to render `[Cloud] deepgram-nova-3`.
3. The Download Model combo skips cloud entries.
4. `_delete_model` short-circuits for cloud entries.
5. `_save` warns (non-blocking) when the user picks Deepgram with no key configured.

PyQt5 dialog code is interactively verified rather than unit-tested (matching the rest of `ui/`).

- [ ] **Step 1: Add the api_keys import**

In `ui/settings_dialog.py`, after the existing `from PyQt5...` import block (the one ending with `QWidget,` and a closing paren near line 26), add a blank line and then:

```python
from config import api_keys
```

No other imports change.

- [ ] **Step 2: Extend `_format_model_label` for cloud entries**

Replace the existing `_format_model_label` static method in `ui/settings_dialog.py` with:

```python
    @staticmethod
    def _format_model_label(model: dict) -> str:
        """Format dropdown label per engine type."""
        t = model.get("type")
        if t == "cloud":
            return f"[Cloud] {model['name']}"
        engine_tag = "Parakeet" if t == "parakeet" else "Whisper"
        return f"[{engine_tag}] {model['name']} ({model['size_mb']} MB)"
```

- [ ] **Step 3: Filter cloud entries from the Download combo**

In `_build_model_group`, replace the loop that populates `self._download_combo` (around line 160) with:

```python
        for m in self._model_manager.list_available():
            if m.get("type") == "cloud":
                continue  # Cloud entries don't need downloading.
            self._download_combo.addItem(self._format_model_label(m), m["name"])
```

- [ ] **Step 4: Add a guard in `_delete_model`**

Replace the body of `_delete_model` in `ui/settings_dialog.py` with:

```python
    def _delete_model(self):
        """Delete the currently selected downloaded model after confirmation."""
        model_name = self._model_combo.currentData()
        if model_name is None:
            return

        # Cloud entries have nothing to delete on disk.
        from engine.model_manager import _MODELS_BY_NAME
        known = _MODELS_BY_NAME.get(model_name)
        if known and known.get("type") == "cloud":
            QMessageBox.information(
                self,
                "Cloud model",
                "Cloud models are not stored locally and cannot be deleted.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {model_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._model_manager.delete_model(model_name)
            self._refresh_model_list()
```

- [ ] **Step 5: Add the `_build_deepgram_group` builder**

Insert the following method into `SettingsDialog` (anywhere in the UI-construction-helpers block; suggest just below `_build_compute_group`):

```python
    def _build_deepgram_group(self):
        group = QGroupBox("Deepgram API Key")
        vbox = QVBoxLayout()

        self._deepgram_status_label = QLabel()
        vbox.addWidget(self._deepgram_status_label)

        row = QHBoxLayout()
        self._deepgram_key_edit = QLineEdit()
        self._deepgram_key_edit.setEchoMode(QLineEdit.Password)
        self._deepgram_key_edit.setPlaceholderText("Paste API key, then press Save")
        row.addWidget(self._deepgram_key_edit)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_deepgram_key)
        row.addWidget(save_btn)
        vbox.addLayout(row)

        clear_btn = QPushButton("Clear stored key")
        clear_btn.clicked.connect(self._clear_deepgram_key)
        vbox.addWidget(clear_btn)

        tip = QLabel("Tip: env var DEEPGRAM_API_KEY overrides the stored key.")
        tip.setStyleSheet("color: gray;")
        vbox.addWidget(tip)

        group.setLayout(vbox)
        self._refresh_deepgram_status()
        return group

    def _refresh_deepgram_status(self) -> None:
        source = api_keys.get_key_source()
        if source == "env":
            self._deepgram_status_label.setText(
                "Status: ● Configured (via DEEPGRAM_API_KEY env var)"
            )
        elif source == "keyring":
            self._deepgram_status_label.setText(
                "Status: ● Configured (in OS keyring)"
            )
        else:
            self._deepgram_status_label.setText("Status: ○ Not set")

    def _save_deepgram_key(self) -> None:
        value = self._deepgram_key_edit.text().strip()
        if not value:
            QMessageBox.warning(
                self, "Empty key", "Enter a key before pressing Save."
            )
            return
        try:
            api_keys.set_deepgram_key(value)
        except Exception as exc:
            QMessageBox.critical(
                self, "Keyring error", f"Failed to save key:\n{exc}"
            )
            return
        self._deepgram_key_edit.clear()
        self._refresh_deepgram_status()
        QMessageBox.information(self, "Saved", "Deepgram API key saved.")

    def _clear_deepgram_key(self) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Remove the stored Deepgram API key?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            api_keys.clear_deepgram_key()
        except Exception as exc:
            QMessageBox.critical(
                self, "Keyring error", f"Failed to clear key:\n{exc}"
            )
            return
        self._refresh_deepgram_status()
```

- [ ] **Step 6: Add the new group to the Model & Compute tab**

In `__init__`, find the block that builds `tab_model` (around line 99-104) and replace it with:

```python
        # --- Tab 1: Model & Compute ---
        tab_model = QWidget()
        tab_model_layout = QVBoxLayout(tab_model)
        tab_model_layout.addWidget(self._build_model_group())
        tab_model_layout.addWidget(self._build_compute_group())
        tab_model_layout.addWidget(self._build_deepgram_group())
        tab_model_layout.addStretch()
        tabs.addTab(tab_model, "Model && Compute")
```

- [ ] **Step 7: Warn in `_save` when picking Deepgram with no key**

Find `_save` and replace its existing "Model" block (the first `model_name = self._model_combo.currentData() ...` lines) with:

```python
        # Model
        model_name = self._model_combo.currentData()
        if model_name is not None:
            self._settings.set("model_size", model_name)
            from engine.model_manager import _MODELS_BY_NAME
            known = _MODELS_BY_NAME.get(model_name)
            if (
                known
                and known.get("type") == "cloud"
                and not api_keys.has_deepgram_key()
            ):
                QMessageBox.warning(
                    self,
                    "Deepgram API key not set",
                    "You selected Deepgram Nova-3 but no API key is configured. "
                    "Set DEEPGRAM_API_KEY or save a key in this dialog before "
                    "recording, otherwise transcription will fail.",
                )
```

- [ ] **Step 8: Smoke-test the dialog interactively**

Run:
```bash
source whisper_env/bin/activate
python whisper2text.py
```

Open Settings → Model & Compute. Verify:
- The Transcription Model dropdown lists `[Cloud] deepgram-nova-3` after the local entries.
- The Download Model dropdown does NOT show the cloud entry.
- The Deepgram API Key group renders with a password field, Save, and Clear buttons.
- Status reads "○ Not set" if the keyring is empty.
- Saving a junk value (e.g. "abc") flips the status to "● Configured (in OS keyring)".
- Selecting `[Cloud] deepgram-nova-3` and pressing Save with no key shows the warning dialog.
- Pressing Clear stored key flips the status back.
- Setting `DEEPGRAM_API_KEY=xyz` in the launching shell and reopening the dialog shows the env-var status.

Close the app afterwards. (Stale junk keys are harmless; clear if desired.)

- [ ] **Step 9: Commit**

```bash
git add ui/settings_dialog.py
git commit -m "Settings: Deepgram API Key group + cloud-aware Transcription Model dropdown"
```

---

## Task 6: MainWindow — bypass GPU retry + show "Cloud" in status

**Files:**
- Modify: `ui/main_window.py`

`_create_engine` must skip the GPU→CPU retry loop for cloud paths (cloud failures aren't backend-recoverable). `_detect_compute_backend` must return `"Cloud"` when the active engine is `DeepgramEngine` so the status label reads naturally.

- [ ] **Step 1: Import DeepgramEngine**

In `ui/main_window.py`, find the existing imports near the top (around line 51-52) and add the new import alongside the existing engine imports:

```python
from engine.model_manager import ModelManager
from engine.factory import make_engine
from engine.parakeet_engine import ParakeetEngine
from engine.deepgram_engine import DeepgramEngine
```

- [ ] **Step 2: Bypass GPU retry for cloud paths in `_create_engine`**

Replace the body of `_create_engine` in `ui/main_window.py` with:

```python
    def _create_engine(self, model_path: str):
        """Create an engine via make_engine, retrying on CPU if a GPU backend
        fails. Cloud engines have no compute backend and bypass the retry."""
        if model_path.startswith("cloud://"):
            self._active_compute_backend = "cloud"
            return make_engine(model_path)

        requested = (self.settings.get('compute_backend', 'auto') or 'auto').lower()
        self._apply_compute_backend()
        try:
            return make_engine(model_path)
        except Exception:
            if getattr(self, '_active_compute_backend', 'cpu') == 'cpu':
                raise
            logger.warning(
                "Engine failed on backend '%s', retrying on CPU",
                getattr(self, '_active_compute_backend', 'unknown'),
                exc_info=True,
            )
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['GGML_VK_DISABLE'] = '1'
            self._active_compute_backend = 'cpu'
            if requested != 'auto':
                self.settings.set('compute_backend', 'cpu')
                self.settings.save()
            return make_engine(model_path)
```

- [ ] **Step 3: Show "Cloud" in `_detect_compute_backend`**

Replace the body of `_detect_compute_backend` in `ui/main_window.py` with:

```python
    def _detect_compute_backend(self):
        """Return a human-readable backend string, with fallback annotations.

        Prefers ``_active_compute_backend`` (set by ``_create_engine`` after
        any GPU → CPU fallback) over the user's saved setting, so the label
        reflects what actually ended up running rather than what was asked.
        """
        if isinstance(self.engine, DeepgramEngine):
            return "Cloud"

        backend = getattr(self, '_active_compute_backend', None)
        if backend is None:
            backend = self.settings.get("compute_backend", "cpu")

        engine_is_parakeet = isinstance(self.engine, ParakeetEngine)

        if backend == "vulkan":
            if engine_is_parakeet:
                return "Vulkan (Parakeet on CPU)"
            return "Vulkan"

        if backend == "cuda":
            if engine_is_parakeet:
                active = self.engine.get_active_provider()
                if active is not None and active != "CUDAExecutionProvider":
                    return "CUDA (CPU fallback)"
            return "CUDA"

        return "CPU"
```

- [ ] **Step 4: Run unit tests — expect no regressions**

Run:
```bash
pytest tests/ -v
```

Expected: all tests pass. (No new tests for MainWindow; existing engine/factory/manager tests should be unaffected.)

- [ ] **Step 5: Commit**

```bash
git add ui/main_window.py
git commit -m "MainWindow: bypass GPU retry for cloud paths, show \"Cloud\" in status"
```

---

## Task 7: PyInstaller spec — bundle keyring backends and Deepgram SDK

**Files:**
- Modify: `whisper2text.spec`

Lazy imports and entry-point lookups need help inside a frozen bundle. `keyring` discovers backends through entry points (so `copy_metadata` is required) and PyInstaller's static analyser misses lazy imports inside functions.

- [ ] **Step 1: Add the keyring + deepgram bundling block**

In `whisper2text.spec`, find the existing `xet_hiddens` block (around lines 64-71) and add the following block immediately after it (before the `# ── Hidden imports ──` divider):

```python
# keyring (used for Deepgram API key storage). Backends are loaded via
# entry points, so we need the package metadata in the bundle plus the
# Linux SecretService backend's transitive deps.
from PyInstaller.utils.hooks import copy_metadata
keyring_datas, keyring_bins, keyring_hiddens = collect_all('keyring')
datas += keyring_datas
binaries += keyring_bins
datas += copy_metadata('keyring')

# deepgram-sdk is imported lazily inside engine.deepgram_engine._load,
# so PyInstaller's static analysis misses it.
deepgram_datas, deepgram_bins, deepgram_hiddens = collect_all('deepgram')
datas += deepgram_datas
binaries += deepgram_bins
```

- [ ] **Step 2: Add the new hidden imports**

Replace the existing `hiddenimports = (...)` block (around lines 73-86) with:

```python
hiddenimports = (
    pwcpp_hiddens
    + evdev_hiddens
    + pyaudio_hiddens
    + ort_hiddens
    + onnxasr_hiddens
    + hf_hiddens
    + xet_hiddens
    + keyring_hiddens
    + deepgram_hiddens
    + collect_submodules('config')
    + collect_submodules('engine')
    + collect_submodules('audio')
    + collect_submodules('ui')
    + [
        'pyperclip', 'pkg_resources', 'setuptools',
        # Linux Secret Service backend for keyring.
        'keyring.backends.SecretService',
        'secretstorage',
        'jeepney',
        'jeepney.io.blocking',
    ]
)
```

- [ ] **Step 3: Build the bundle**

Run:
```bash
source whisper_env/bin/activate
pyinstaller whisper2text.spec --noconfirm
```

Expected: `Building completed successfully` at the end. No `ModuleNotFoundError` warnings for `keyring` or `deepgram`.

- [ ] **Step 4: Smoke-test the frozen binary's keyring**

Run:
```bash
./dist/whisper2text/whisper2text 2>&1 | head -30 &
APP_PID=$!
sleep 5
kill $APP_PID 2>/dev/null
```

Expected: app launches, no `RuntimeError: No recommended backend was available` in the first 30 log lines. (If you see one, the `copy_metadata("keyring")` call wasn't picked up — re-run pyinstaller with `--clean`.)

Then manually open Settings → Model & Compute and verify the Deepgram API Key group shows "Status: ○ Not set" rather than a backend error.

- [ ] **Step 5: Commit**

```bash
git add whisper2text.spec
git commit -m "PyInstaller: bundle keyring backends + deepgram-sdk for cloud engine"
```

---

## Task 8: End-to-end manual smoke test

**Files:**
- None (verification only)

Verify the integrated feature against a live Deepgram account. Requires a real API key; ~10 min of audio worth of API calls (~$0.08 of usage at $0.46/hr).

- [ ] **Step 1: Save a real Deepgram API key**

Open the app, Settings → Model & Compute → Deepgram API Key. Paste a real key, press Save. Status should read "● Configured (in OS keyring)".

- [ ] **Step 2: Pick the cloud model and record**

Settings → Transcription Model → `[Cloud] deepgram-nova-3` → Save. The status bar at the bottom should read `Model: deepgram-nova-3 | Cloud | Mic: …`. Record a 5–10 second clip via the hotkey or Record button. Expect a transcript identical (modulo capitalisation) to a Whisper run on the same speaker.

- [ ] **Step 3: Test custom vocabulary mapping**

Open Settings → Vocabulary, enter `Avrillo`, `conveyancing`, `SDLT` (one per line). Save. Record a clip using those words. Expect each spelled correctly.

- [ ] **Step 4: Test missing-key error path**

Open Settings → Model & Compute → Deepgram API Key → Clear stored key. Attempt to record. Expect the error panel to read "Deepgram selected but no API key found…" (or similar), and no transcript appended.

- [ ] **Step 5: Test invalid-key error path**

Save a junk key like `dg_invalid_xxxxxxx`. Record. Expect a 401-ish error in the error panel.

- [ ] **Step 6: Test offline error path**

Disable network (e.g. `nmcli networking off` or unplug Ethernet/Wi-Fi). Record. Expect a connection error in the error panel. Re-enable networking afterwards.

- [ ] **Step 7: Test switching back to a local model**

Pick `[Whisper] ggml-base.bin` (or whatever local model is downloaded) → Save. Record. Expect Whisper transcription, status reads `Model: ggml-base.bin | <CPU/CUDA/Vulkan> | Mic: …`. Confirms cloud → local engine swap is clean.

- [ ] **Step 8: Test env-var override**

Quit the app. Re-launch with `DEEPGRAM_API_KEY=xxxxxxxx ./dist/whisper2text/whisper2text` (or via systemd drop-in). Open Settings → Deepgram API Key. Status should read "● Configured (via DEEPGRAM_API_KEY env var)".

- [ ] **Step 9: Final commit (if anything was tweaked)**

If any small fixes came out of the smoke test, commit them with a message like `Fix <issue> found in smoke test`. Otherwise this task ends here.
