# Parakeet ASR Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVIDIA Parakeet TDT 0.6B (v2 English / v3 multilingual, ONNX) as a selectable transcription engine alongside the existing whisper.cpp engine, with CUDA acceleration on RTX A1000 / RTX 4090 and graceful CPU fallback.

**Architecture:** Add a `ParakeetEngine` class that mirrors `WhisperEngine`'s duck-typed shape (`transcribe`, `unload`, `reload`, `is_loaded`, context-manager). A small `make_engine(model_path)` factory inspects the path (file vs directory) to pick the right class. `ModelManager` is extended (not split): each `AVAILABLE_MODELS` entry gets a `type` field; Parakeet entries download as full HF repo snapshots into directories. The Settings UI relabels the existing Whisper Model dropdown to "Transcription Model" and prefixes each entry with `[Whisper]` or `[Parakeet]`. No formal `Protocol`/`ABC` — promote later if a third engine appears.

**Tech Stack:** Python 3.10, PyQt5, `onnx-asr` (Parakeet wrapper), `onnxruntime-gpu` (CUDA 12 bundled DLLs), `huggingface_hub` (model snapshot downloader), pytest + unittest.mock for tests.

**Spec reference:** `docs/superpowers/specs/2026-04-20-parakeet-engine-design.md` (commit a732416)

**File map:**
- Create: `engine/parakeet_engine.py`, `engine/factory.py`
- Create: `tests/test_parakeet_engine.py`, `tests/test_engine_factory.py`, `tests/test_model_manager_parakeet.py`
- Modify: `engine/model_manager.py`, `ui/main_window.py`, `ui/settings_dialog.py`, `requirements.txt`

---

## Task 1: Add Python dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the three new dependencies**

Open `requirements.txt` and append at the end:

```
# Parakeet ASR (NVIDIA TDT 0.6B v2/v3 via ONNX Runtime)
# Install with GPU: pip install onnxruntime-gpu>=1.19 (bundles CUDA 12 DLLs)
# Install with CPU only: replace onnxruntime-gpu with onnxruntime
onnx-asr[hub]>=0.7.0
onnxruntime-gpu>=1.19.0
huggingface_hub>=0.24.0
```

The `[hub]` extra pulls in `huggingface_hub` automatically, but we list it explicitly so it's also available to `ModelManager` regardless of how onnx-asr is installed.

- [ ] **Step 2: Install into the project's venv**

Run:
```bash
source whisper_env/bin/activate
pip install -r requirements.txt
```

Expected: `Successfully installed onnx-asr-... onnxruntime-gpu-... huggingface_hub-...`. If pip warns that `onnxruntime` (CPU build) is also installed and conflicts, run `pip uninstall -y onnxruntime` first, then re-run install.

- [ ] **Step 3: Verify the imports work**

Run:
```bash
python -c "import onnx_asr, onnxruntime, huggingface_hub; print(onnxruntime.get_available_providers())"
```

Expected output should include `'CUDAExecutionProvider'` and `'CPUExecutionProvider'` (CUDA only appears if the NVIDIA driver is present — on machines without CUDA only CPU appears, which is fine).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add onnx-asr, onnxruntime-gpu, huggingface_hub for Parakeet support"
```

---

## Task 2: ParakeetEngine class (TDD)

**Files:**
- Create: `tests/test_parakeet_engine.py`
- Create: `engine/parakeet_engine.py`

The engine wraps `onnx_asr.load_model()` and exposes the same shape as `WhisperEngine`: `transcribe(audio)`, `unload()`, `reload(path)`, `is_loaded()`, plus context-manager support. Audio in is `bytes` (int16 PCM, 16 kHz mono) or a `numpy` float32 array, matching `WhisperEngine.transcribe`'s contract.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parakeet_engine.py` with the full file content below:

```python
"""Tests for engine.parakeet_engine.ParakeetEngine."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engine.parakeet_engine import ParakeetEngine


@pytest.fixture
def mock_load_model():
    """Patch onnx_asr.load_model so no real ONNX session is created."""
    with patch("engine.parakeet_engine.onnx_asr.load_model") as m:
        fake = MagicMock()
        fake.recognize.return_value = "hello world"
        m.return_value = fake
        yield m, fake


class TestConstruction:
    def test_loads_model_from_path(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        load.assert_called_once()
        kwargs = load.call_args.kwargs
        assert kwargs["path"] == "/tmp/parakeet-tdt-0.6b-v3-int8"

    def test_passes_cuda_then_cpu_providers(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        kwargs = load.call_args.kwargs
        assert kwargs["providers"] == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_uses_correct_model_id_for_v3(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/some/path/parakeet-tdt-0.6b-v3-int8")
        args = load.call_args.args
        assert args[0] == "nemo-parakeet-tdt-0.6b-v3"

    def test_uses_correct_model_id_for_v2(self, mock_load_model):
        load, _ = mock_load_model
        ParakeetEngine("/some/path/parakeet-tdt-0.6b-v2-int8")
        args = load.call_args.args
        assert args[0] == "nemo-parakeet-tdt-0.6b-v2"


class TestIsLoaded:
    def test_is_loaded_after_construction(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        assert engine.is_loaded() is True

    def test_is_loaded_false_after_unload(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        assert engine.is_loaded() is False


class TestTranscribe:
    def test_transcribe_passes_float32_array(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        audio = np.zeros(16000, dtype=np.float32)
        engine.transcribe(audio)
        called_arg = fake.recognize.call_args.args[0]
        assert called_arg.dtype == np.float32

    def test_transcribe_converts_int16_bytes_to_float32(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        # 1000 samples of int16 silence
        raw = (np.zeros(1000, dtype=np.int16)).tobytes()
        engine.transcribe(raw)
        called_arg = fake.recognize.call_args.args[0]
        assert called_arg.dtype == np.float32
        assert called_arg.shape == (1000,)

    def test_transcribe_passes_sample_rate_16000(self, mock_load_model):
        _, fake = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert fake.recognize.call_args.kwargs["sample_rate"] == 16000

    def test_transcribe_returns_text(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = "  hello   world  "
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        # Whitespace collapsed and stripped
        assert text == "hello world"

    def test_transcribe_handles_list_result(self, mock_load_model):
        _, fake = mock_load_model
        fake.recognize.return_value = ["hello world"]
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        text = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "hello world"

    def test_transcribe_raises_when_unloaded(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.transcribe(np.zeros(16000, dtype=np.float32))


class TestUnload:
    def test_unload_is_idempotent(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.unload()
        engine.unload()  # must not raise

    def test_context_manager_unloads(self, mock_load_model):
        with ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8") as engine:
            assert engine.is_loaded()
        assert not engine.is_loaded()


class TestReload:
    def test_reload_swaps_model(self, mock_load_model):
        load, _ = mock_load_model
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        engine.reload("/tmp/parakeet-tdt-0.6b-v2-int8")
        # Two load_model calls total: initial + reload
        assert load.call_count == 2
        assert load.call_args.kwargs["path"] == "/tmp/parakeet-tdt-0.6b-v2-int8"

    def test_reload_accepts_extra_kwargs_ignored(self, mock_load_model):
        engine = ParakeetEngine("/tmp/parakeet-tdt-0.6b-v3-int8")
        # WhisperEngine takes language=; ParakeetEngine should accept and ignore.
        engine.reload("/tmp/parakeet-tdt-0.6b-v3-int8", language="en")
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run:
```bash
pytest tests/test_parakeet_engine.py -v
```

Expected: ImportError / ModuleNotFoundError on `engine.parakeet_engine` for every test.

- [ ] **Step 3: Implement ParakeetEngine**

Create `engine/parakeet_engine.py` with the full content below:

```python
"""Wrapper around onnx-asr providing a managed Parakeet model lifecycle.

Mirrors the public shape of WhisperEngine so callers can swap engines
through engine.factory.make_engine.
"""

import atexit
import gc
import logging
import os
import re

import numpy as np
import onnx_asr

logger = logging.getLogger(__name__)

# onnx-asr expects an architecture identifier; we infer it from the
# directory name so the same logic works for any saved Parakeet variant.
_MODEL_ID_BY_KEYWORD = (
    ("v3", "nemo-parakeet-tdt-0.6b-v3"),
    ("v2", "nemo-parakeet-tdt-0.6b-v2"),
)
_DEFAULT_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"


def _infer_model_id(model_path: str) -> str:
    name = os.path.basename(os.path.normpath(model_path)).lower()
    for keyword, model_id in _MODEL_ID_BY_KEYWORD:
        if keyword in name:
            return model_id
    return _DEFAULT_MODEL_ID


class ParakeetEngine:
    """Load, manage, and transcribe with an ONNX Parakeet model.

    Parameters
    ----------
    model_path : str
        Filesystem path to a directory containing the ONNX model files
        (encoder, decoder/joint, preprocessor_config, vocab).
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_path: str, **_ignored):
        self._model_path = model_path
        self._model = None
        self._load(model_path)
        atexit.register(self.unload)

    def _load(self, model_path: str) -> None:
        model_id = _infer_model_id(model_path)
        logger.info("Loading Parakeet model %s from %s", model_id, model_path)
        self._model = onnx_asr.load_model(
            model_id,
            path=model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info("Parakeet model loaded successfully")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is not None:
            logger.info("Unloading Parakeet model")
            del self._model
            self._model = None
            gc.collect()

    def reload(self, model_path: str, **_ignored) -> None:
        self.unload()
        self._model_path = model_path
        self._load(model_path)

    def transcribe(self, audio_data) -> str:
        if self._model is None:
            raise RuntimeError("No model loaded")

        if isinstance(audio_data, (bytes, bytearray)):
            audio_data = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        result = self._model.recognize(audio_data, sample_rate=self.SAMPLE_RATE)
        if isinstance(result, list):
            result = result[0] if result else ""
        text = str(result)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run:
```bash
pytest tests/test_parakeet_engine.py -v
```

Expected: all tests pass (the `mock_load_model` fixture means no real ONNX runtime is needed).

- [ ] **Step 5: Commit**

```bash
git add engine/parakeet_engine.py tests/test_parakeet_engine.py
git commit -m "Add ParakeetEngine wrapper around onnx-asr"
```

---

## Task 3: Engine factory

**Files:**
- Create: `tests/test_engine_factory.py`
- Create: `engine/factory.py`

The factory inspects a model path and returns the right engine class. Selection rules: file ending in `.bin` → `WhisperEngine`; directory containing a file matching `encoder-model*.onnx` → `ParakeetEngine`; anything else → `ValueError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine_factory.py`:

```python
"""Tests for engine.factory.make_engine."""

from unittest.mock import MagicMock, patch

import pytest

from engine.factory import make_engine


@pytest.fixture
def patched_engines():
    """Patch both engine constructors so no real models load."""
    with patch("engine.factory.WhisperEngine") as w, \
         patch("engine.factory.ParakeetEngine") as p:
        w.return_value = MagicMock(name="WhisperEngineInstance")
        p.return_value = MagicMock(name="ParakeetEngineInstance")
        yield w, p


def test_returns_whisper_for_bin_file(tmp_path, patched_engines):
    w, p = patched_engines
    bin_path = tmp_path / "ggml-base.bin"
    bin_path.write_bytes(b"\x00")
    make_engine(str(bin_path))
    w.assert_called_once_with(str(bin_path))
    p.assert_not_called()


def test_returns_parakeet_for_directory_with_encoder_onnx(tmp_path, patched_engines):
    w, p = patched_engines
    model_dir = tmp_path / "parakeet-tdt-0.6b-v3-int8"
    model_dir.mkdir()
    (model_dir / "encoder-model.int8.onnx").write_bytes(b"\x00")
    make_engine(str(model_dir))
    p.assert_called_once_with(str(model_dir))
    w.assert_not_called()


def test_returns_parakeet_for_directory_with_unquantized_encoder(tmp_path, patched_engines):
    w, p = patched_engines
    model_dir = tmp_path / "parakeet-tdt-0.6b-v3"
    model_dir.mkdir()
    (model_dir / "encoder-model.onnx").write_bytes(b"\x00")
    make_engine(str(model_dir))
    p.assert_called_once_with(str(model_dir))
    w.assert_not_called()


def test_passes_through_extra_kwargs(tmp_path, patched_engines):
    w, _ = patched_engines
    bin_path = tmp_path / "ggml-base.bin"
    bin_path.write_bytes(b"\x00")
    make_engine(str(bin_path), language="fr", n_threads=8)
    w.assert_called_once_with(str(bin_path), language="fr", n_threads=8)


def test_raises_for_unrecognized_file(tmp_path, patched_engines):
    bogus = tmp_path / "model.txt"
    bogus.write_text("hi")
    with pytest.raises(ValueError, match="Unrecognized model"):
        make_engine(str(bogus))


def test_raises_for_directory_without_encoder(tmp_path, patched_engines):
    model_dir = tmp_path / "empty-dir"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("foo")
    with pytest.raises(ValueError, match="Unrecognized model"):
        make_engine(str(model_dir))


def test_raises_for_missing_path(patched_engines):
    with pytest.raises(ValueError, match="does not exist"):
        make_engine("/no/such/path/here")
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run:
```bash
pytest tests/test_engine_factory.py -v
```

Expected: ImportError on `engine.factory` for all tests.

- [ ] **Step 3: Implement the factory**

Create `engine/factory.py`:

```python
"""Factory that picks a transcription engine based on model path.

Whisper: a single .bin file (whisper.cpp GGML format).
Parakeet: a directory containing encoder-model*.onnx (onnx-asr format).
"""

import glob
import os

from engine.parakeet_engine import ParakeetEngine
from engine.whisper_engine import WhisperEngine


def make_engine(model_path: str, **kwargs):
    """Return a transcription engine appropriate for the given model path."""
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

- [ ] **Step 4: Run the tests to confirm they pass**

Run:
```bash
pytest tests/test_engine_factory.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/factory.py tests/test_engine_factory.py
git commit -m "Add engine factory that picks Whisper or Parakeet by model path"
```

---

## Task 4: Add type field and Parakeet entries to AVAILABLE_MODELS

**Files:**
- Modify: `engine/model_manager.py:6-55`
- Modify: `tests/test_model_manager.py`

Add a `"type": "whisper"` field to every existing entry and append two `"type": "parakeet"` entries with HuggingFace repo IDs.

- [ ] **Step 1: Modify the AVAILABLE_MODELS list**

Open `engine/model_manager.py`. Replace the entire `AVAILABLE_MODELS = [...]` block (lines 6-55) with the version below. The change is: every existing entry gets `"type": "whisper"`; two new Parakeet entries are appended.

```python
AVAILABLE_MODELS = [
    # ── Standard Whisper (ggerganov/whisper.cpp) ──────────────────────
    {"name": "ggml-base.bin", "type": "whisper", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
     "description": "Base (~142MB) — good starting point"},
    {"name": "ggml-base.en.bin", "type": "whisper", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
     "description": "Base English-only (~142MB)"},
    {"name": "ggml-small.bin", "type": "whisper", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
     "description": "Small (~466MB) — better accuracy"},
    {"name": "ggml-small.en.bin", "type": "whisper", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
     "description": "Small English-only (~466MB)"},
    {"name": "ggml-medium.bin", "type": "whisper", "size_mb": 1533,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
     "description": "Medium (~1.5GB) — high accuracy"},
    {"name": "ggml-medium-q5_0.bin", "type": "whisper", "size_mb": 539,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
     "description": "Medium quantized Q5 (~539MB)"},
    {"name": "ggml-large-v3.bin", "type": "whisper", "size_mb": 3095,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
     "description": "Large V3 (~3GB) — best accuracy"},
    {"name": "ggml-large-v3-q5_0.bin", "type": "whisper", "size_mb": 1080,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
     "description": "Large V3 quantized Q5 (~1GB)"},
    {"name": "ggml-large-v3-turbo.bin", "type": "whisper", "size_mb": 1624,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
     "description": "Large V3 Turbo (~1.6GB) — fast + accurate"},
    {"name": "ggml-large-v3-turbo-q5_0.bin", "type": "whisper", "size_mb": 574,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
     "description": "Large V3 Turbo Q5 (~574MB)"},
    {"name": "ggml-large-v3-turbo-q8_0.bin", "type": "whisper", "size_mb": 874,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
     "description": "Large V3 Turbo Q8 (~874MB)"},

    # ── Distil-Whisper (distilled, 5-6x faster) ──────────────────────
    {"name": "ggml-distil-small.en.bin", "type": "whisper", "size_mb": 334,
     "url": "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin",
     "description": "Distil Small EN (~334MB) — 4x faster than large-v2"},
    {"name": "ggml-distil-medium.en.bin", "type": "whisper", "size_mb": 789,
     "url": "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin",
     "description": "Distil Medium EN (~789MB) — 4x faster, <1% WER loss"},
    {"name": "ggml-distil-large-v3.bin", "type": "whisper", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
     "description": "Distil Large V3 (~756MB) — 5x faster than large-v3"},
    {"name": "ggml-distil-large-v3.5.bin", "type": "whisper", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
     "description": "Distil Large V3.5 (~756MB) — latest distilled, best speed/accuracy"},

    # ── NVIDIA Parakeet (TDT 0.6B, ONNX INT8) ────────────────────────
    {"name": "parakeet-tdt-0.6b-v2-int8", "type": "parakeet", "size_mb": 600,
     "hf_repo": "istupakov/parakeet-tdt-0.6b-v2-onnx",
     "hf_revision": "main",
     "description": "Parakeet TDT v2 INT8 (~600MB) — English-only, very fast on GPU"},
    {"name": "parakeet-tdt-0.6b-v3-int8", "type": "parakeet", "size_mb": 670,
     "hf_repo": "istupakov/parakeet-tdt-0.6b-v3-onnx",
     "hf_revision": "main",
     "description": "Parakeet TDT v3 INT8 (~670MB) — multilingual (25 EU langs), fast on GPU"},
]
```

- [ ] **Step 2: Update existing model_manager test for the type field**

Open `tests/test_model_manager.py`. Inside `class TestListAvailable`, replace `test_list_available_has_required_keys` with:

```python
    def test_list_available_has_required_keys(self, manager):
        available = manager.list_available()
        for model in available:
            assert "name" in model
            assert "type" in model
            assert model["type"] in ("whisper", "parakeet")
            assert "size_mb" in model
            assert "description" in model

    def test_list_available_includes_parakeet_entries(self, manager):
        available = manager.list_available()
        names = [m["name"] for m in available]
        assert "parakeet-tdt-0.6b-v2-int8" in names
        assert "parakeet-tdt-0.6b-v3-int8" in names
```

- [ ] **Step 3: Run the model_manager tests**

Run:
```bash
pytest tests/test_model_manager.py -v
```

Expected: all tests pass, including the two new ones.

- [ ] **Step 4: Commit**

```bash
git add engine/model_manager.py tests/test_model_manager.py
git commit -m "Add type field to AVAILABLE_MODELS and Parakeet entries"
```

---

## Task 5: ModelManager directory support

**Files:**
- Modify: `engine/model_manager.py` (methods `list_downloaded`, `is_downloaded`, `get_model_path`, `delete_model`)
- Create: `tests/test_model_manager_parakeet.py`

`list_downloaded` and friends currently assume single `.bin` files. Extend them so directories that look like Parakeet models are recognized too. The detection rule mirrors `engine.factory`: a directory containing `encoder-model*.onnx`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_manager_parakeet.py`:

```python
"""Tests for ModelManager handling of Parakeet (directory-based) models."""

import os
from unittest.mock import patch

import pytest

from engine.model_manager import ModelManager


@pytest.fixture
def manager_with_models(tmp_path):
    """ModelManager pointing at a temp dir seeded with one fake Whisper file
    and one fake Parakeet directory."""
    # Whisper-style: single .bin
    (tmp_path / "ggml-base.bin").write_bytes(b"\x00" * 1024)
    # Parakeet-style: directory with encoder-model.int8.onnx
    pdir = tmp_path / "parakeet-tdt-0.6b-v3-int8"
    pdir.mkdir()
    (pdir / "encoder-model.int8.onnx").write_bytes(b"\x00" * 1024)
    (pdir / "decoder_joint-model.int8.onnx").write_bytes(b"\x00" * 1024)
    (pdir / "vocab.txt").write_text("a\nb\n")
    return ModelManager(str(tmp_path))


class TestListDownloadedMixed:
    def test_lists_both_whisper_and_parakeet(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        names = [m["name"] for m in downloaded]
        assert "ggml-base.bin" in names
        assert "parakeet-tdt-0.6b-v3-int8" in names

    def test_each_entry_has_type_field(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        types = {m["name"]: m["type"] for m in downloaded}
        assert types["ggml-base.bin"] == "whisper"
        assert types["parakeet-tdt-0.6b-v3-int8"] == "parakeet"

    def test_parakeet_entry_path_is_directory(self, manager_with_models):
        downloaded = manager_with_models.list_downloaded()
        entry = next(m for m in downloaded if m["type"] == "parakeet")
        assert os.path.isdir(entry["path"])

    def test_directory_without_encoder_is_ignored(self, tmp_path):
        # A random directory should not show up.
        (tmp_path / "ggml-base.bin").write_bytes(b"\x00")
        (tmp_path / "stray-dir").mkdir()
        (tmp_path / "stray-dir" / "vocab.txt").write_text("x")
        mm = ModelManager(str(tmp_path))
        names = [m["name"] for m in mm.list_downloaded()]
        assert "stray-dir" not in names


class TestIsDownloadedDirectory:
    def test_true_for_parakeet_directory(self, manager_with_models):
        assert manager_with_models.is_downloaded("parakeet-tdt-0.6b-v3-int8")

    def test_false_for_missing_directory(self, manager_with_models):
        assert not manager_with_models.is_downloaded("parakeet-tdt-0.6b-v2-int8")


class TestGetModelPathDirectory:
    def test_returns_directory_path(self, manager_with_models):
        path = manager_with_models.get_model_path("parakeet-tdt-0.6b-v3-int8")
        assert os.path.isdir(path)
        assert path.endswith("parakeet-tdt-0.6b-v3-int8")


class TestDeleteModelDirectory:
    def test_removes_directory_recursively(self, manager_with_models):
        manager_with_models.delete_model("parakeet-tdt-0.6b-v3-int8")
        assert not manager_with_models.is_downloaded("parakeet-tdt-0.6b-v3-int8")
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run:
```bash
pytest tests/test_model_manager_parakeet.py -v
```

Expected: failures because `list_downloaded` doesn't see the directory; `is_downloaded` and `get_model_path` only check `os.path.isfile`; `delete_model` doesn't recurse.

- [ ] **Step 3: Add top-level imports and modify the four methods**

Open `engine/model_manager.py`. At the top of the file, just below `import urllib.request`, add:

```python
import glob
import shutil
```

Replace the entire `list_downloaded` method (currently at lines 72-97) with:

```python
    def list_downloaded(self) -> list[dict]:
        """Return a list of dicts for every downloaded model.

        Recognizes two on-disk shapes:
          * Whisper: any single ``.bin`` file in ``models_dir``.
          * Parakeet: any subdirectory containing ``encoder-model*.onnx``.

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
        return results
```

Replace `is_downloaded` (currently at lines 107-109) with:

```python
    def is_downloaded(self, model_name: str) -> bool:
        """Return True if ``model_name`` exists as a file or directory."""
        path = os.path.join(self.models_dir, model_name)
        return os.path.isfile(path) or os.path.isdir(path)
```

Replace `get_model_path` (currently at lines 111-121) with:

```python
    def get_model_path(self, model_name: str) -> str:
        """Return the full path for ``model_name`` (file or directory).

        Raises ``FileNotFoundError`` if the model has not been downloaded.
        """
        path = os.path.join(self.models_dir, model_name)
        if not (os.path.isfile(path) or os.path.isdir(path)):
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {self.models_dir}"
            )
        return path
```

Replace `delete_model` (currently at lines 127-131) with:

```python
    def delete_model(self, model_name: str) -> None:
        """Delete the model file or directory from ``models_dir``."""
        path = os.path.join(self.models_dir, model_name)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
```

- [ ] **Step 4: Run both model_manager test files**

Run:
```bash
pytest tests/test_model_manager.py tests/test_model_manager_parakeet.py -v
```

Expected: all tests pass — old tests still pass (Whisper behavior unchanged), new tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/model_manager.py tests/test_model_manager_parakeet.py
git commit -m "ModelManager: support directory-based Parakeet models"
```

---

## Task 6: Parakeet download via huggingface_hub

**Files:**
- Modify: `engine/model_manager.py` (`download_model`)
- Modify: `tests/test_model_manager_parakeet.py`

For Parakeet entries, dispatch to `huggingface_hub.snapshot_download` instead of `urllib.request.urlretrieve`. Stage downloads into a `<name>.partial` directory and `os.rename` on success, mirroring the existing whisper safety pattern.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_model_manager_parakeet.py`:

```python
class TestDownloadParakeet:
    def test_dispatches_to_snapshot_download(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        with patch("engine.model_manager.snapshot_download") as snap, \
             patch("os.rename") as rename:
            # snapshot_download writes to local_dir; emulate with a no-op.
            snap.return_value = str(tmp_path / "parakeet-tdt-0.6b-v3-int8.partial")
            mm.download_model("parakeet-tdt-0.6b-v3-int8")

        snap.assert_called_once()
        kwargs = snap.call_args.kwargs
        assert kwargs["repo_id"] == "istupakov/parakeet-tdt-0.6b-v3-onnx"
        assert kwargs["revision"] == "main"
        assert kwargs["local_dir"].endswith("parakeet-tdt-0.6b-v3-int8.partial")
        rename.assert_called_once()

    def test_progress_callback_invoked_with_completion(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        seen = []

        def cb(percent, downloaded, total):
            seen.append(percent)

        with patch("engine.model_manager.snapshot_download"), \
             patch("os.rename"):
            mm.download_model("parakeet-tdt-0.6b-v3-int8", progress_callback=cb)

        # We don't expose per-byte progress; just ensure 0 and 100 are reported.
        assert seen[0] == 0
        assert seen[-1] == 100

    def test_cleans_up_partial_dir_on_failure(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        partial = tmp_path / "parakeet-tdt-0.6b-v3-int8.partial"
        partial.mkdir()  # simulate a partial that snapshot_download started

        def boom(**_):
            raise RuntimeError("network down")

        with patch("engine.model_manager.snapshot_download", side_effect=boom):
            with pytest.raises(RuntimeError, match="network down"):
                mm.download_model("parakeet-tdt-0.6b-v3-int8")

        assert not partial.exists()
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run:
```bash
pytest tests/test_model_manager_parakeet.py::TestDownloadParakeet -v
```

Expected: AttributeError or import errors — `snapshot_download` is not imported in `model_manager.py` yet, and `download_model` does not branch on type.

- [ ] **Step 3: Modify `download_model` to dispatch on type**

Open `engine/model_manager.py`. Add this import near the top (after the `import shutil` line added in Task 5):

```python
from huggingface_hub import snapshot_download
```

Replace the existing `download_model` method with the version below:

```python
    def download_model(self, model_name: str, progress_callback=None) -> str:
        """Download ``model_name`` into ``models_dir``.

        Whisper models stream a single ``.bin`` file from the URL in the
        catalog. Parakeet models snapshot a HuggingFace repo into a
        subdirectory. Both flows stage into ``<name>.partial`` and rename
        atomically on success so incomplete downloads are never visible.
        """
        known = _MODELS_BY_NAME.get(model_name)
        model_type = (known or {}).get("type", "whisper")

        if model_type == "parakeet":
            return self._download_parakeet(known, model_name, progress_callback)
        return self._download_whisper(known, model_name, progress_callback)

    def _download_whisper(self, known, model_name, progress_callback) -> str:
        if known and "url" in known:
            url = known["url"]
        else:
            url = (
                "https://huggingface.co/ggerganov/whisper.cpp/"
                f"resolve/main/{model_name}"
            )
        dest_path = os.path.join(self.models_dir, model_name)
        partial_path = dest_path + ".partial"

        def _reporthook(block_num: int, block_size: int, total_size: int):
            if progress_callback is None:
                return
            downloaded = block_num * block_size
            percent = (
                min(downloaded / total_size * 100, 100.0)
                if total_size > 0
                else 0.0
            )
            progress_callback(percent, downloaded, total_size)

        try:
            urllib.request.urlretrieve(url, partial_path, reporthook=_reporthook)
            os.rename(partial_path, dest_path)
        except Exception:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise
        return dest_path

    def _download_parakeet(self, known, model_name, progress_callback) -> str:
        if not known or "hf_repo" not in known:
            raise ValueError(
                f"Parakeet model '{model_name}' has no hf_repo entry in the catalog"
            )

        dest_path = os.path.join(self.models_dir, model_name)
        partial_path = dest_path + ".partial"

        if progress_callback is not None:
            progress_callback(0, 0, 0)

        try:
            snapshot_download(
                repo_id=known["hf_repo"],
                revision=known.get("hf_revision", "main"),
                local_dir=partial_path,
            )
            os.rename(partial_path, dest_path)
        except Exception:
            if os.path.isdir(partial_path):
                shutil.rmtree(partial_path, ignore_errors=True)
            raise

        if progress_callback is not None:
            progress_callback(100, 0, 0)
        return dest_path
```

- [ ] **Step 4: Run all model_manager tests**

Run:
```bash
pytest tests/test_model_manager.py tests/test_model_manager_parakeet.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/model_manager.py tests/test_model_manager_parakeet.py
git commit -m "ModelManager: download Parakeet models via huggingface_hub"
```

---

## Task 7: Wire MainWindow to use the engine factory

**Files:**
- Modify: `ui/main_window.py:50` (import)
- Modify: `ui/main_window.py:206-245` (`_load_initial_engine`)
- Modify: `ui/main_window.py:639-662` (`_reload_engine`)

The two places that currently instantiate `WhisperEngine` directly become `make_engine` calls. The factory's path-sniffing handles the dispatch.

- [ ] **Step 1: Replace the WhisperEngine import**

Open `ui/main_window.py`. Find the line:

```python
from engine.whisper_engine import WhisperEngine
```

Replace it with:

```python
from engine.factory import make_engine
```

- [ ] **Step 2: Update `_load_initial_engine`**

In `_load_initial_engine`, change the two places that call `WhisperEngine(...)`:

Replace:
```python
                return WhisperEngine(model_path)
```
with:
```python
                return make_engine(model_path)
```

(both occurrences — one in the saved-model branch around line 226, one in the fallback branch around line 240).

- [ ] **Step 3: Update `_reload_engine`**

Replace:
```python
            if self.engine is not None and self.engine.is_loaded():
                self.update_status_signal.emit("Reloading model...")
                self.engine.reload(model_path)
            else:
                self.update_status_signal.emit("Loading model...")
                self.engine = WhisperEngine(model_path)
```

with:
```python
            # Always rebuild via the factory so cross-engine switches
            # (Whisper <-> Parakeet) work correctly. Model load cost is
            # incurred either way; calling reload() in-place would only save
            # an object alloc.
            already_loaded = self.engine is not None and self.engine.is_loaded()
            self.update_status_signal.emit(
                "Reloading model..." if already_loaded else "Loading model..."
            )
            if self.engine is not None:
                self.engine.unload()
            self.engine = make_engine(model_path)
```

- [ ] **Step 4: Run the existing test suite to confirm nothing regressed**

Run:
```bash
pytest tests/ -v --ignore=tests/test_autopaste_e2e.py --ignore=tests/test_paste_realistic.py
```

Expected: all tests pass. (We skip the two interactive autopaste tests; they don't relate to engines.)

- [ ] **Step 5: Manual smoke check**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from engine.factory import make_engine
print('factory imports cleanly')
"
```

Expected: prints `factory imports cleanly`. If you get an import error from `onnx_asr`, return to Task 1 and confirm the install.

- [ ] **Step 6: Commit**

```bash
git add ui/main_window.py
git commit -m "MainWindow: use engine factory to load Whisper or Parakeet"
```

---

## Task 8: SettingsDialog label and prefix updates

**Files:**
- Modify: `ui/settings_dialog.py:118` (label text)
- Modify: `ui/settings_dialog.py:113-143` (`_build_model_group`)
- Modify: `ui/settings_dialog.py:268-281` (`_refresh_model_list`)

Show users which engine each entry uses, but otherwise keep the dialog flow identical.

- [ ] **Step 1: Add a small helper at the top of the SettingsDialog class**

Open `ui/settings_dialog.py`. Inside `class SettingsDialog`, just above `_build_model_group`, add:

```python
    @staticmethod
    def _format_model_label(model: dict) -> str:
        """Format '[Whisper] ggml-base.bin (142 MB)' or the Parakeet equivalent."""
        engine_tag = "Parakeet" if model.get("type") == "parakeet" else "Whisper"
        return f"[{engine_tag}] {model['name']} ({model['size_mb']} MB)"
```

- [ ] **Step 2: Update `_build_model_group` to use the helper and rename the label**

In `_build_model_group`, replace:

```python
        # Whisper Model dropdown (downloaded models)
        vbox.addWidget(QLabel("Whisper Model"))
        self._model_combo = QComboBox()
        vbox.addWidget(self._model_combo)
        self._refresh_model_list()

        # Download Model dropdown + Download button
        vbox.addWidget(QLabel("Download Model"))
        dl_row = QHBoxLayout()
        self._download_combo = QComboBox()
        for m in self._model_manager.list_available():
            label = f"{m['name']} ({m['size_mb']} MB)"
            self._download_combo.addItem(label, m["name"])
        dl_row.addWidget(self._download_combo)
```

with:

```python
        # Transcription Model dropdown (downloaded models, both engines)
        vbox.addWidget(QLabel("Transcription Model"))
        self._model_combo = QComboBox()
        vbox.addWidget(self._model_combo)
        self._refresh_model_list()

        # Download Model dropdown + Download button (both engines)
        vbox.addWidget(QLabel("Download Model"))
        dl_row = QHBoxLayout()
        self._download_combo = QComboBox()
        for m in self._model_manager.list_available():
            self._download_combo.addItem(self._format_model_label(m), m["name"])
        dl_row.addWidget(self._download_combo)
```

- [ ] **Step 3: Update `_refresh_model_list` to use the helper**

Replace the body of `_refresh_model_list` with:

```python
    def _refresh_model_list(self):
        """Repopulate the Transcription Model dropdown with downloaded models."""
        self._model_combo.clear()
        current_model = self._settings.get("model_size")
        select_idx = 0

        for i, m in enumerate(self._model_manager.list_downloaded()):
            self._model_combo.addItem(self._format_model_label(m), m["name"])
            if m["name"] == current_model:
                select_idx = i

        if self._model_combo.count() > 0:
            self._model_combo.setCurrentIndex(select_idx)
```

- [ ] **Step 4: Manual smoke — open the settings dialog**

Run the app and open Settings. Verify:
- The "Whisper Model" label now reads "Transcription Model".
- Each entry in both dropdowns is prefixed with `[Whisper]` or `[Parakeet]`.
- Selecting a Whisper entry and saving still works as before.

```bash
python whisper2text.py &
sleep 3 && echo "open Settings via the gear button, verify, then close the app"
```

(Test interactively; if you can't run a UI right now, defer this manual step until you're at the machine.)

- [ ] **Step 5: Commit**

```bash
git add ui/settings_dialog.py
git commit -m "SettingsDialog: rename to Transcription Model, prefix entries with engine"
```

---

## Task 9: Status label fallback indicators

**Files:**
- Modify: `ui/main_window.py:317-344` (`_detect_compute_backend`, `_update_status_label`)

When the active engine is Parakeet but the user picked `vulkan` (which Parakeet can't honor), the status label should read `Vulkan (Parakeet on CPU)`. When the user picked `cuda` but ONNX Runtime fell back to CPU at session creation, it should read `CUDA (CPU fallback)`. Other cases unchanged.

For runtime-fallback detection, we ask `onnxruntime` which providers are available at module load time and compare to what the engine class expects.

- [ ] **Step 1: Add a helper to detect ONNX runtime CUDA availability**

Open `ui/main_window.py`. Just above `class MainWindow`, add:

```python
def _onnx_cuda_available() -> bool:
    """Return True if ONNX Runtime reports CUDAExecutionProvider available."""
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False
```

- [ ] **Step 2: Replace `_detect_compute_backend`**

Replace the existing `_detect_compute_backend` method (lines 317-324) with:

```python
    def _detect_compute_backend(self):
        """Return a human-readable backend string, with fallback annotations."""
        from engine.parakeet_engine import ParakeetEngine

        backend = self.settings.get("compute_backend", "cpu")
        engine_is_parakeet = isinstance(self.engine, ParakeetEngine)

        if backend == "vulkan":
            if engine_is_parakeet:
                return "Vulkan (Parakeet on CPU)"
            return "Vulkan"

        if backend == "cuda":
            if engine_is_parakeet and not _onnx_cuda_available():
                return "CUDA (CPU fallback)"
            return "CUDA"

        return "CPU"
```

- [ ] **Step 3: Run the test suite**

Run:
```bash
pytest tests/ -v --ignore=tests/test_autopaste_e2e.py --ignore=tests/test_paste_realistic.py
```

Expected: all tests still pass — this change only affects the status label.

- [ ] **Step 4: Commit**

```bash
git add ui/main_window.py
git commit -m "MainWindow: surface Parakeet CPU-fallback in status label"
```

---

## Task 10: End-to-end smoke test (manual, requires GPU + network)

**Goal:** Prove a fresh Parakeet download, engine switch, and live recording produce a transcription.

This task is interactive — there's no automated assertion. Do it on whichever machine has a working NVIDIA driver (RTX A1000 laptop or RTX 4090 desktop).

- [ ] **Step 1: Stop any running instance and launch the app**

```bash
pkill -f whisper2text 2>/dev/null; sleep 1; rm -f ~/.whisper2text/app.lock
python whisper2text.py
```

- [ ] **Step 2: Download Parakeet v3 via Settings**

In the running app:
1. Click the gear (Settings).
2. In the "Download Model" dropdown pick `[Parakeet] parakeet-tdt-0.6b-v3-int8`.
3. Click Download. The progress dialog should jump to 100% once the snapshot finishes (per-byte progress isn't reported).
4. Check `~/.whisper2text/models/parakeet-tdt-0.6b-v3-int8/` exists and contains at least `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `vocab.txt`.

- [ ] **Step 3: Switch active engine to Parakeet**

In Settings → "Transcription Model" select `[Parakeet] parakeet-tdt-0.6b-v3-int8`. Set Compute Backend to `cuda`. Save.

The status label should show:
```
Model: parakeet-tdt-0.6b-v3-int8 | CUDA | Mic: ...
```

If you see `CUDA (CPU fallback)`, your driver/onnxruntime-gpu install isn't healthy — re-check Task 1.

- [ ] **Step 4: Record a sample**

Press the hotkey or click Record. Speak a short sentence. Stop the recording. Confirm:
- A transcript button appears within ~1 s of stopping (Parakeet on CUDA is sub-realtime).
- The text is reasonable (not gibberish, not a Whisper hallucination like "Thank you").

- [ ] **Step 5: Switch back to a Whisper model and record**

Settings → switch model to a downloaded `[Whisper] ggml-*.bin`. Save. Record again. Confirm Whisper still transcribes correctly — no regression from the engine factory change.

- [ ] **Step 6: Commit a note in the spec/plan if anything surprised you**

If everything worked as expected, no commit needed. If you hit a quirk worth recording for future work (e.g. v2 outperforms v3 on UK English in your testing), append a short paragraph to the spec under a new "## 8. Implementation notes" section and commit it.
