# Silero VAD Integration — Design

**Date:** 2026-04-20
**Status:** Approved for implementation
**Owner:** ac2522

## 1. Goal & scope

Replace the existing `webrtcvad`-based voice-activity detection in `Recorder.record_silence_mode` with [Silero VAD v5](https://github.com/snakers4/silero-vad). Silero is a neural VAD that substantially outperforms WebRTC VAD (which is a 2012-era DSP model) on quiet voices and in noisy environments. The change is internal: existing Settings UI, recording modes, and public `Recorder` API stay the same.

**Non-goals:**
- Streaming / partial transcription.
- Using Silero for anything other than silence-mode stop detection.
- User-facing tuning knobs beyond the existing 0–3 aggressiveness spinner.
- GPU acceleration for the VAD itself (Silero is 1.8 MB — CPU is faster once loaded than paying the GPU-dispatch overhead per 32 ms chunk).

## 2. Dependencies

- **Remove:** `webrtcvad>=2.0.10` from `requirements.txt`.
- **Add:** nothing. Silero is shipped as a single `silero_vad.onnx` file (~1.8 MB) and loaded via the `onnxruntime-gpu` package already added for Parakeet.

## 3. Model distribution

The ONNX model is downloaded on first use rather than bundled in the git repo.

- **Source URL:** `https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx`
- **Destination:** `~/.whisper2text/vad/silero_vad.onnx`
- **Staging:** download to `silero_vad.onnx.partial` then `os.rename` on success (same atomic pattern used by `ModelManager` for Whisper/Parakeet models).
- **Failure:** if the download fails, `record_silence_mode` surfaces the error via the existing `error_signal` path. No silent degradation.
- **Re-download:** if the local file exists we reuse it. A future task can add version-checking if the upstream model is bumped.

## 4. Module structure

### `audio/vad.py` (new)

```python
class SileroVAD:
    def __init__(self, model_path: str, providers: list[str] | None = None): ...
    def is_speech(self, chunk_f32: np.ndarray, threshold: float) -> bool: ...
    def reset(self) -> None: ...
    @property
    def sample_rate(self) -> int: ...  # always 16000
    @property
    def chunk_samples(self) -> int: ...  # always 512
```

- `__init__` creates an `onnxruntime.InferenceSession` with `["CPUExecutionProvider"]` by default; accepts a `providers` override for tests/future use.
- `is_speech(chunk_f32, threshold)`:
  - Asserts `chunk_f32.shape == (512,)` and `chunk_f32.dtype == np.float32`.
  - Runs inference with inputs `{"input": chunk[None, :], "sr": np.array(16000, dtype=np.int64), "state": self._state}`.
  - Updates `self._state` from the output.
  - Returns `probability >= threshold`.
- `reset()` sets `self._state = np.zeros((2, 1, 128), dtype=np.float32)` (Silero v5 initial state shape).

### `audio/vad.py` module-level

```python
AGGRESSIVENESS_TO_THRESHOLD: dict[int, float] = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9}
SILERO_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

def aggressiveness_to_threshold(aggressiveness: int) -> float: ...
def ensure_vad_model(dest_path: str) -> str: ...  # downloads if missing, returns dest_path
```

## 5. Recorder integration

`audio/recorder.py` changes:

1. Remove `import webrtcvad`.
2. Import `SileroVAD`, `aggressiveness_to_threshold`, `ensure_vad_model` from `audio.vad`.
3. Add `VAD_MODEL_PATH = os.path.expanduser("~/.whisper2text/vad/silero_vad.onnx")` constant (or route through `config.paths`).
4. In `record_silence_mode`:
   - Before entering the loop, call `ensure_vad_model(VAD_MODEL_PATH)` and construct `vad = SileroVAD(VAD_MODEL_PATH)`. Call `vad.reset()` to start with a clean LSTM state.
   - Convert the `vad_aggressiveness` parameter (0–3) to a float threshold via `aggressiveness_to_threshold`.
   - Keep recording chunks at hw-native 30 ms for PyAudio latency.
   - Introduce a rolling `np.float32` buffer. For each recorded chunk, resample to 16 kHz (existing code), convert `int16 → float32 [-1, 1]` (divide by 32768), append to buffer. While the buffer has ≥ 512 samples, take the first 512, call `vad.is_speech(...)`, slide remainder to front.
   - Replace chunk-counting silence heuristic with **sample-accurate** tracking:
     - Track `total_samples_16k` (cumulative 16 kHz sample count fed into the VAD).
     - On every positive `is_speech`, set `last_speech_idx = total_samples_16k`.
     - Stop recording when `speech_detected and (total_samples_16k - last_speech_idx) >= break_length * 16000`.
   - On exception during VAD model init or download, stop the stream cleanly and re-raise so `MainWindow._record_silence` surfaces it via `error_signal`.

`record_button_mode` is unchanged.

## 6. Behaviour, errors, testing

### Behaviour deltas users will notice

- **Quieter voices detected reliably.** WebRTC VAD was poor on low-amplitude speech; Silero is strong.
- **Slightly more permissive at threshold 0** (aggressiveness 0 maps to 0.3 which still triggers on fairly ambient noise). User can raise to 1–2 for typical office use.
- Minor latency change: VAD runs at 32 ms cadence on the 16 kHz buffer rather than per-30-ms-chunk, so `break_length` stops are ~2 ms finer-grained. Imperceptible.
- The `vad_aggressiveness` setting range (0–3) and default (1) are preserved.

### Error surfaces

- **First run, no network:** `ensure_vad_model` raises `RuntimeError` with a clear message. `MainWindow._record_silence` catches and displays it via `error_signal`; recording never starts.
- **Model file present but corrupt:** `onnxruntime` raises at `InferenceSession()` construction. Re-raised, surfaced same way. Operator can delete `~/.whisper2text/vad/silero_vad.onnx` to retrigger download.
- **Device stream error during VAD accumulation:** existing `try/finally` around `stream.read` covers this; state and partial audio are discarded.

### Tests (TDD, comprehensive coverage)

New test file `tests/test_vad.py`:

- **`TestAggressivenessMapping`** — covers all four integer inputs producing expected float thresholds; out-of-range inputs raise `ValueError`.
- **`TestSileroVADIsSpeech`** — patches `onnxruntime.InferenceSession`; asserts:
  - Input tensor shape `(1, 512)`, dtype `float32`.
  - `sr` input tensor is scalar `int64` equal to 16000.
  - State tensor is threaded through (first call uses zeros, subsequent calls use previous output state).
  - Threshold comparison uses `>=` (not `>`), so a probability exactly at the threshold returns `True`.
- **`TestSileroVADReset`** — `reset()` zeros the state tensor; state persists across `is_speech` calls until `reset()` is called.
- **`TestSileroVADInputValidation`** — non-512 chunk raises `ValueError`; non-float32 raises `ValueError`.
- **`TestEnsureVADModel`** — mocks `urllib.request.urlretrieve`:
  - First call when file absent triggers download to `.partial` and renames.
  - Second call when file present is a no-op (no HTTP request).
  - Network failure cleans up `.partial`.

Extend `tests/test_recorder.py`:

- **`TestRecordSilenceModeSilero`** — patches `SileroVAD` (returns scripted speech/silence bool sequence) and `ensure_vad_model`. Feeds a scripted audio stream, asserts:
  - VAD is reset before use.
  - Stop fires exactly after `break_length` seconds of silence (sample-accurate).
  - Returned audio matches the captured frames (float32, 16 kHz, in [-1, 1]).
  - If `ensure_vad_model` raises, `record_silence_mode` re-raises and the pyaudio stream is torn down cleanly.

Existing `Recorder.record_button_mode` tests must continue to pass unchanged.

## 7. Out of scope

- Bundling the `silero_vad.onnx` in the repo (rejected — network download is fine and keeps the repo clean).
- Exposing the float threshold directly in Settings (keep 0–3 UX).
- Adaptive / auto-calibrated thresholds.
- Using Silero for wake-word style always-on detection.
- Running Silero on GPU (per-inference overhead exceeds savings for a 1.8 MB model).
