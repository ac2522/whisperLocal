# Parakeet ASR Engine — Design

**Date:** 2026-04-20
**Status:** Approved (brainstorming complete; awaiting spec review before implementation plan)
**Owner:** ac2522

## 1. Goal & scope

Add NVIDIA Parakeet (TDT 0.6B v2 and v3, ONNX variants) as a selectable transcription engine alongside the existing Whisper backend. Both engines coexist permanently — the user picks per-model in the Settings dialog. The feature must work on either target machine (RTX A1000 Laptop GPU, 4 GB VRAM / RTX 4090, 24 GB VRAM) and degrade gracefully to CPU on systems without CUDA. No existing Whisper functionality is removed or degraded.

**Non-goals:** language-specific tuning, streaming/partial transcription, holding multiple models loaded at once, Parakeet fine-tuning, replacing or hiding Whisper.

## 2. Library choice & dependencies

We avoid the NeMo/PyTorch stack (3–5 GB, GPU-only realistic perf). Instead:

- **`onnx-asr`** (PyPI: `onnx-asr`, repo `istupakov/onnx-asr`) — thin Python wrapper around community-converted ONNX Parakeet models. Exposes `load_model()` and a `recognize(audio)` call that fits our existing `transcribe(audio)` contract.
- **`onnxruntime-gpu`** — provides `CUDAExecutionProvider` and falls back automatically to `CPUExecutionProvider` when CUDA is unavailable. Bundles its own CUDA 12 runtime DLLs so the only host requirement is a working NVIDIA driver. Both target machines already have one (they run whisper.cpp CUDA today).
- **`huggingface_hub`** — used for snapshot-downloading Parakeet model directories. Small dep, no transitive ML stack.

All three become hard requirements in `requirements.txt`. Hard-dep approach is consistent with how `pywhispercpp` is currently installed; lazy-install would complicate the existing PyInstaller build.

Approximate footprint added at install: `onnxruntime-gpu` ~500 MB + `onnx-asr` + `huggingface_hub` ~10 MB combined. Per Parakeet model on disk: ~670 MB (INT8 quantized).

## 3. Engine abstraction

The current `WhisperEngine` (`engine/whisper_engine.py`) already exposes the implicit "engine" shape:

- `__init__(model_path, **kwargs)`
- `transcribe(audio_data) -> str`  (accepts `bytes` int16 PCM or `np.ndarray` float32)
- `unload() -> None`
- `reload(model_path, language=None) -> None`
- `is_loaded() -> bool`
- context-manager support (`__enter__`/`__exit__`)

Add `engine/parakeet_engine.py` with a `ParakeetEngine` class that matches the same shape. **Duck-typed; no formal `Protocol`/`ABC`.** Two implementations don't justify formalising a contract — we can promote it later if a third engine appears.

Add `engine/factory.py`:

```python
def make_engine(model_path: str, **kwargs):
    """Return a transcription engine appropriate for the given model path.

    A whisper.cpp GGML file (single .bin) returns WhisperEngine.
    A directory containing encoder-model*.onnx returns ParakeetEngine.
    """
```

Sniff order:
1. If `os.path.isfile(model_path)` and ends with `.bin` → `WhisperEngine`.
2. If `os.path.isdir(model_path)` and contains a file matching `encoder-model*.onnx` → `ParakeetEngine`.
3. Otherwise raise `ValueError` with both paths-checked messages.

`MainWindow` calls `make_engine(...)` in the two places that currently instantiate `WhisperEngine` (`_load_initial_engine`, `_reload_engine`). The `self.engine` attribute is no longer typed to `WhisperEngine` specifically.

`reload()` semantics differ between engines (Whisper carries language, Parakeet doesn't). To keep the call sites simple, `reload(model_path)` becomes the canonical signature — extra kwargs ignored if the backend doesn't use them. If the new model path implies a different engine class, MainWindow tears down the old one and constructs a new one via `make_engine()` rather than calling `reload()` across types.

## 4. Model manager

Extend `engine/model_manager.py` rather than split it. One `ModelManager` continues to handle the on-disk models directory (`~/.whisper2text/models`).

### Schema change to `AVAILABLE_MODELS`

Each entry gains a `"type"` field — `"whisper"` or `"parakeet"`. Whisper entries keep `"url"` (single-file download); Parakeet entries carry `"hf_repo"` (HuggingFace repo ID, full directory snapshot).

```python
{"name": "parakeet-tdt-0.6b-v3-int8", "type": "parakeet",
 "size_mb": 670,
 "hf_repo": "istupakov/parakeet-tdt-0.6b-v3-onnx",
 "hf_revision": "main",
 "description": "Parakeet TDT v3 INT8 (~670 MB) — multilingual, fast on GPU"},
{"name": "parakeet-tdt-0.6b-v2-int8", "type": "parakeet",
 "size_mb": 600,
 "hf_repo": "istupakov/parakeet-tdt-0.6b-v2-onnx",
 "hf_revision": "main",
 "description": "Parakeet TDT v2 INT8 (~600 MB) — English-only, very fast"},
```

(Final HF repo IDs and exact sizes verified during implementation — both repos exist as of 2026-04.)

### Disk layout

- Whisper unchanged: `models/ggml-large-v3-turbo.bin`
- Parakeet stored as a directory: `models/parakeet-tdt-0.6b-v3-int8/{encoder-model.int8.onnx, decoder_joint-model.int8.onnx, preprocessor_config.json, vocab.txt, ...}`

### Behaviour changes

- `list_downloaded()` walks `models_dir`. For each entry:
  - `*.bin` file → whisper, as today.
  - Directory containing an `encoder-model*.onnx` → parakeet.
  Returned dicts gain a `"type"` field.
- `is_downloaded(name)` checks file-or-directory existence.
- `get_model_path(name)` returns the file path for whisper, the directory path for parakeet.
- `download_model(name, progress_callback)` dispatches on the entry's `type`:
  - whisper → existing `urllib.request.urlretrieve` flow with `.partial` rename-on-success.
  - parakeet → `huggingface_hub.snapshot_download(repo_id=hf_repo, revision=hf_revision, local_dir=<models_dir>/<name>.partial, ...)` then `os.rename()` to final name on success. Progress is reported as a coarse 0/50/100 (snapshot_download doesn't expose per-byte progress without extra plumbing — acceptable for first cut; documented as a known limitation).
- `delete_model(name)` removes file or recursively removes directory.

## 5. Settings UI

Minimal changes — the engine is implied by model selection, no new combo box.

- Existing "Whisper Model" label/dropdown → "Transcription Model". Items now include both engine types, prefixed for clarity:
  - `[Whisper] ggml-large-v3-turbo.bin (1624 MB)`
  - `[Parakeet] parakeet-tdt-0.6b-v3-int8 (670 MB)`
- "Download Model" dropdown identical formatting; selecting a Parakeet entry triggers the HF snapshot download path.
- Compute backend combo is unchanged. Interpretation extends:
  - `cpu` — Whisper uses whisper.cpp CPU build; Parakeet uses `CPUExecutionProvider`.
  - `cuda` — both engines use CUDA.
  - `vulkan` — only meaningful to whisper.cpp. If a Parakeet model is loaded under this setting, Parakeet uses `CPUExecutionProvider` (Vulkan isn't an ONNX Runtime provider for our use). We log this once per session at INFO level.
- `MainWindow._update_status_label()` continues to show `Model: ... | <backend> | Mic: ...`. The `<backend>` segment is augmented in two specific cases:
  - When user-selected backend is `vulkan` AND active engine is Parakeet → display reads `Vulkan (Parakeet on CPU)`.
  - When user-selected backend is `cuda` but the engine reported a runtime fallback to CPU (driver mismatch, OOM at session create) → display reads `CUDA (CPU fallback)`.
  - In all other cases (`cpu` selected, `cuda` selected and honored, `vulkan` selected with Whisper engine) the label is unchanged from today.

No changes to hotkey, audio, recording, or auto-paste sections.

## 6. Runtime behaviour, errors, testing

### Runtime

- `WhisperEngine.transcribe()` keeps its current hallucination-cleanup regexes (large-v3-turbo "thank you", `[Silence]` brackets). These are Whisper-specific and stay in `WhisperEngine` only.
- `ParakeetEngine.transcribe()` does lighter cleanup — collapse extra whitespace, strip leading/trailing whitespace. Parakeet's failure modes (rare repetitions, occasional language mis-ID on v3) aren't fixable with regex; we leave them alone.
- Audio contract is identical: 16 kHz mono. Recorder already produces this. Parakeet engine accepts the same `bytes`-or-`ndarray` input and converts internally.
- Engine is instantiated in a background thread (`_reload_engine`) — unchanged. ONNX Runtime model load takes ~1–3 s on first load; the existing "Loading model..." status label covers it.

### Errors

- Existing `error_signal` / `_on_error` flow handles all surfaced exceptions; no new pathway needed.
- Parakeet-specific error cases:
  - Model directory missing files → `_reload_engine` catches, surfaces `Model 'X' is corrupt or incomplete; please re-download`.
  - `onnxruntime` cannot create CUDA session (driver mismatch, OOM) → engine constructor catches once, retries with CPU provider, logs WARNING, status label appends `(CPU fallback)`. No user-blocking error.
  - `huggingface_hub` download network error → existing download error dialog catches via the propagated exception.
- No try/except swallowing: any exception we don't explicitly handle propagates to the existing `_handle_exception` hook.

### Testing

New unit tests in `tests/`:

- `test_parakeet_engine.py` — mocks `onnx_asr.load_model`. Verifies:
  - `ParakeetEngine(path)` calls `load_model` with the right kwargs.
  - `transcribe(int16_bytes)` produces float32 ndarray and calls `recognize`.
  - `transcribe(float32_ndarray)` passes through unchanged.
  - `unload()` is idempotent.
  - `is_loaded()` reflects state.
  - Whitespace cleanup is applied to returned text.
- `test_engine_factory.py` — for each fixture (`.bin` file, dir with `encoder-model.onnx`, unrelated path), `make_engine` returns the correct class or raises `ValueError`.
- `test_model_manager_parakeet.py` — extends existing model_manager tests:
  - `list_downloaded()` reports both whisper files and parakeet directories with correct `type`.
  - `is_downloaded()` true for both file and directory.
  - `delete_model()` removes directory recursively.
  - `download_model()` for parakeet calls `snapshot_download` (mocked) with correct args.

Existing tests for `WhisperEngine` and the rest of `ModelManager` must continue passing unchanged.

Integration test (opt-in, skipped without `WHISPERLOCAL_GPU_TESTS=1`): load each downloaded model and transcribe a short fixture WAV, asserting non-empty output. Not run in CI by default.

## 7. Out of scope (explicit non-decisions to revisit later)

- Streaming / chunked Parakeet inference for ultra-long recordings.
- Per-byte download progress for Parakeet (HF snapshot_download limitation).
- A formal `TranscriptionEngine` Protocol/ABC — wait until a third engine appears.
- Migration of existing Whisper-only setups to default-Parakeet — Whisper remains default; Parakeet is opt-in via Settings.
- Bundling Parakeet model files in the PyInstaller dist — they download on first selection like Whisper models do today.
