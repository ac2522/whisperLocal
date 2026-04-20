# Silero VAD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace `webrtcvad` with Silero VAD v5 in `Recorder.record_silence_mode`, preserving the existing 0–3 aggressiveness UX and all public `Recorder` methods.

**Architecture:** New `audio/vad.py` module hosts a `SileroVAD` class (ONNX via existing `onnxruntime-gpu`), an aggressiveness→threshold mapping, and a download-on-first-use helper. `Recorder.record_silence_mode` calls into this module with sample-accurate silence tracking.

**Tech Stack:** Python 3.12, `numpy`, `onnxruntime` (already installed), stdlib `urllib.request`, pytest + unittest.mock.

**Spec reference:** `docs/superpowers/specs/2026-04-20-silero-vad-design.md` (commit 0e8e4b9)

**File map:**
- Create: `audio/vad.py`, `tests/test_vad.py`
- Modify: `audio/recorder.py`, `tests/test_recorder.py`, `requirements.txt`

**Execution venv:** `venv/` (Python 3.12). `whisper_env/` is broken on this machine — always use `venv/bin/python`, `venv/bin/pytest`.

---

## Task 1: Add `aggressiveness_to_threshold` helper (TDD)

**Files:**
- Create: `audio/vad.py`
- Create: `tests/test_vad.py`

Smallest atom of the new module — pure function, no I/O, no onnxruntime yet. Establishes the module file and tests the simplest surface.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vad.py` with:

```python
"""Tests for audio.vad — Silero VAD integration."""

import pytest

from audio.vad import AGGRESSIVENESS_TO_THRESHOLD, aggressiveness_to_threshold


class TestAggressivenessMapping:
    def test_level_0_maps_to_0_3(self):
        assert aggressiveness_to_threshold(0) == 0.3

    def test_level_1_maps_to_0_5(self):
        assert aggressiveness_to_threshold(1) == 0.5

    def test_level_2_maps_to_0_7(self):
        assert aggressiveness_to_threshold(2) == 0.7

    def test_level_3_maps_to_0_9(self):
        assert aggressiveness_to_threshold(3) == 0.9

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="aggressiveness"):
            aggressiveness_to_threshold(4)
        with pytest.raises(ValueError, match="aggressiveness"):
            aggressiveness_to_threshold(-1)

    def test_constant_covers_all_levels(self):
        assert set(AGGRESSIVENESS_TO_THRESHOLD.keys()) == {0, 1, 2, 3}
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
venv/bin/pytest tests/test_vad.py -v
```

Expected: ImportError on `audio.vad`.

- [ ] **Step 3: Create `audio/vad.py` with the helper**

```python
"""Silero VAD v5 integration: ONNX-based neural voice activity detection.

Replaces the older webrtcvad DSP model in Recorder.record_silence_mode.
"""

AGGRESSIVENESS_TO_THRESHOLD: dict[int, float] = {
    0: 0.3,
    1: 0.5,
    2: 0.7,
    3: 0.9,
}


def aggressiveness_to_threshold(aggressiveness: int) -> float:
    """Map the Settings 0-3 aggressiveness spinner to a Silero probability threshold.

    Higher aggressiveness means "only report speech when very confident", so it
    maps to a higher threshold. Preserves the numeric UX users already have.
    """
    if aggressiveness not in AGGRESSIVENESS_TO_THRESHOLD:
        raise ValueError(
            f"aggressiveness must be 0-3, got {aggressiveness!r}"
        )
    return AGGRESSIVENESS_TO_THRESHOLD[aggressiveness]
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
venv/bin/pytest tests/test_vad.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add audio/vad.py tests/test_vad.py
git commit -m "$(cat <<'EOF'
Add Silero VAD aggressiveness->threshold mapping

First slice of the Silero VAD module. Pure function with no I/O —
lets subsequent tasks build the onnxruntime-backed SileroVAD class
on top of a tested foundation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `ensure_vad_model` downloader (TDD)

**Files:**
- Modify: `audio/vad.py`
- Modify: `tests/test_vad.py`

Downloads the Silero v5 ONNX file on first use with atomic staging. Mirrors the `ModelManager` pattern.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_vad.py`:

```python
import os
from unittest.mock import patch

from audio.vad import SILERO_MODEL_URL, ensure_vad_model


class TestEnsureVADModel:
    def test_skips_download_if_file_exists(self, tmp_path):
        existing = tmp_path / "silero_vad.onnx"
        existing.write_bytes(b"\x00" * 100)
        with patch("audio.vad.urllib.request.urlretrieve") as urlretrieve:
            result = ensure_vad_model(str(existing))
        urlretrieve.assert_not_called()
        assert result == str(existing)

    def test_downloads_to_partial_then_renames(self, tmp_path):
        dest = tmp_path / "silero_vad.onnx"
        partial = tmp_path / "silero_vad.onnx.partial"

        def fake_retrieve(url, filename):
            # Simulate urlretrieve writing the file at the target
            with open(filename, "wb") as f:
                f.write(b"\x00" * 100)

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=fake_retrieve) as urlretrieve:
            result = ensure_vad_model(str(dest))

        urlretrieve.assert_called_once()
        call_args = urlretrieve.call_args
        assert call_args.args[0] == SILERO_MODEL_URL
        assert call_args.args[1] == str(partial)
        assert dest.exists()
        assert not partial.exists()
        assert result == str(dest)

    def test_creates_parent_directory(self, tmp_path):
        dest = tmp_path / "nested" / "dir" / "silero_vad.onnx"

        def fake_retrieve(url, filename):
            with open(filename, "wb") as f:
                f.write(b"\x00")

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=fake_retrieve):
            ensure_vad_model(str(dest))

        assert dest.exists()

    def test_cleans_up_partial_on_failure(self, tmp_path):
        dest = tmp_path / "silero_vad.onnx"
        partial = tmp_path / "silero_vad.onnx.partial"

        def failing_retrieve(url, filename):
            # Simulate a partially-written download before the error
            with open(filename, "wb") as f:
                f.write(b"\x00" * 50)
            raise RuntimeError("network down")

        with patch("audio.vad.urllib.request.urlretrieve", side_effect=failing_retrieve):
            with pytest.raises(RuntimeError, match="network down"):
                ensure_vad_model(str(dest))

        assert not dest.exists()
        assert not partial.exists()
```

- [ ] **Step 2: Run, confirm the new tests fail**

```bash
venv/bin/pytest tests/test_vad.py::TestEnsureVADModel -v
```

Expected: ImportError or AttributeError on `ensure_vad_model`, `SILERO_MODEL_URL`.

- [ ] **Step 3: Implement the downloader**

In `audio/vad.py`, add below the existing helper:

```python
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

SILERO_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)


def ensure_vad_model(dest_path: str) -> str:
    """Ensure the Silero VAD ONNX model is present at ``dest_path``.

    Downloads from ``SILERO_MODEL_URL`` on first use, staging into a
    ``.partial`` file and renaming atomically on success. Returns the
    final path. Raises the underlying exception on download failure
    after cleaning up any partial file.
    """
    if os.path.isfile(dest_path):
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    partial = dest_path + ".partial"

    try:
        logger.info("Downloading Silero VAD model to %s", dest_path)
        urllib.request.urlretrieve(SILERO_MODEL_URL, partial)
        os.rename(partial, dest_path)
    except Exception:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        raise

    logger.info("Silero VAD model downloaded successfully")
    return dest_path
```

Move `import os`, `import logging`, `import urllib.request` to the top of the file so they appear only once. Tidy imports alphabetically.

- [ ] **Step 4: Run tests, confirm pass**

```bash
venv/bin/pytest tests/test_vad.py -v
```

Expected: 10 passed (6 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add audio/vad.py tests/test_vad.py
git commit -m "$(cat <<'EOF'
Add Silero VAD model downloader with atomic staging

Download-on-first-use mirrors the ModelManager pattern: stage into
.partial, rename atomically on success, clean up on failure. Parent
directory created if absent. Repeat calls with an existing file are a
no-op — no HTTP request.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `SileroVAD` class (TDD)

**Files:**
- Modify: `audio/vad.py`
- Modify: `tests/test_vad.py`

The core class. Loads the ONNX session, runs inference with threaded LSTM state, exposes `is_speech`, `reset`, `sample_rate`, `chunk_samples`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_vad.py`:

```python
import numpy as np
from unittest.mock import MagicMock

from audio.vad import SileroVAD


@pytest.fixture
def mock_onnx_session():
    """Patch onnxruntime.InferenceSession with a MagicMock that
    returns scripted probabilities + a state tensor."""
    with patch("audio.vad.onnxruntime.InferenceSession") as cls:
        session = MagicMock()
        # Default: probability 0.8, state is just zeros
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        cls.return_value = session
        yield cls, session


class TestSileroVADConstruction:
    def test_loads_model(self, mock_onnx_session):
        cls, _ = mock_onnx_session
        SileroVAD("/tmp/silero_vad.onnx")
        cls.assert_called_once()
        args, kwargs = cls.call_args
        assert args[0] == "/tmp/silero_vad.onnx"
        assert kwargs["providers"] == ["CPUExecutionProvider"]

    def test_custom_providers(self, mock_onnx_session):
        cls, _ = mock_onnx_session
        SileroVAD("/tmp/silero_vad.onnx", providers=["CUDAExecutionProvider"])
        assert cls.call_args.kwargs["providers"] == ["CUDAExecutionProvider"]

    def test_sample_rate_and_chunk_samples(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.sample_rate == 16000
        assert vad.chunk_samples == 512


class TestSileroVADIsSpeech:
    def test_accepts_correct_input_shape(self, mock_onnx_session):
        _, session = mock_onnx_session
        vad = SileroVAD("/tmp/silero_vad.onnx")
        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk, threshold=0.5)
        feeds = session.run.call_args.args[1]
        assert feeds["input"].shape == (1, 512)
        assert feeds["input"].dtype == np.float32

    def test_passes_sample_rate_as_int64(self, mock_onnx_session):
        _, session = mock_onnx_session
        vad = SileroVAD("/tmp/silero_vad.onnx")
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        sr = session.run.call_args.args[1]["sr"]
        assert int(sr) == 16000
        assert sr.dtype == np.int64

    def test_threads_state_between_calls(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.full((2, 1, 128), 0.42, dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")

        # First call: state is zeros
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        first_state = session.run.call_args.args[1]["state"]
        assert np.all(first_state == 0.0)

        # Second call: state is the output from the first call
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        second_state = session.run.call_args.args[1]["state"]
        assert np.allclose(second_state, 0.42)

    def test_threshold_comparison_is_inclusive(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.5]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5) is True

    def test_probability_below_threshold_returns_false(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.4]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")
        assert vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5) is False


class TestSileroVADInputValidation:
    def test_wrong_sample_count_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="512"):
            vad.is_speech(np.zeros(480, dtype=np.float32), threshold=0.5)

    def test_wrong_dtype_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="float32"):
            vad.is_speech(np.zeros(512, dtype=np.float64), threshold=0.5)

    def test_non_1d_raises(self, mock_onnx_session):
        vad = SileroVAD("/tmp/silero_vad.onnx")
        with pytest.raises(ValueError, match="1-D"):
            vad.is_speech(np.zeros((1, 512), dtype=np.float32), threshold=0.5)


class TestSileroVADReset:
    def test_reset_zeros_state(self, mock_onnx_session):
        _, session = mock_onnx_session
        session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.full((2, 1, 128), 0.42, dtype=np.float32),
        ]
        vad = SileroVAD("/tmp/silero_vad.onnx")

        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        # state is now 0.42
        vad.reset()
        vad.is_speech(np.zeros(512, dtype=np.float32), threshold=0.5)
        # After reset, first call should have used zero state
        state = session.run.call_args_list[-1].args[1]["state"]
        assert np.all(state == 0.0)
```

- [ ] **Step 2: Run, confirm the new tests fail**

```bash
venv/bin/pytest tests/test_vad.py -v
```

Expected: `AttributeError: module 'audio.vad' has no attribute 'SileroVAD'`.

- [ ] **Step 3: Implement `SileroVAD`**

Append to `audio/vad.py`:

```python
import numpy as np
import onnxruntime


class SileroVAD:
    """Neural voice-activity detector wrapping Silero VAD v5.

    Stateful: maintains LSTM hidden state across ``is_speech`` calls so
    predictions benefit from temporal context. Call ``reset()`` at the
    start of each recording session.
    """

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512
    _STATE_SHAPE = (2, 1, 128)

    def __init__(self, model_path: str, providers: list[str] | None = None):
        providers = providers or ["CPUExecutionProvider"]
        self._session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._state: np.ndarray
        self.reset()

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def chunk_samples(self) -> int:
        return self.CHUNK_SAMPLES

    def reset(self) -> None:
        """Clear the LSTM hidden state. Call at the start of a new recording."""
        self._state = np.zeros(self._STATE_SHAPE, dtype=np.float32)

    def is_speech(self, chunk_f32: np.ndarray, threshold: float) -> bool:
        """Return True iff the model's speech probability meets ``threshold``.

        ``chunk_f32`` must be a 1-D float32 array of exactly 512 samples at
        16 kHz, normalised to [-1, 1].
        """
        if chunk_f32.ndim != 1:
            raise ValueError(
                f"chunk must be 1-D, got shape {chunk_f32.shape}"
            )
        if chunk_f32.shape[0] != self.CHUNK_SAMPLES:
            raise ValueError(
                f"chunk must have exactly 512 samples, got {chunk_f32.shape[0]}"
            )
        if chunk_f32.dtype != np.float32:
            raise ValueError(
                f"chunk must be float32, got {chunk_f32.dtype}"
            )

        feeds = {
            "input": chunk_f32[np.newaxis, :],
            "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
            "state": self._state,
        }
        prob, new_state = self._session.run(None, feeds)
        self._state = new_state
        return float(prob[0, 0]) >= threshold
```

- [ ] **Step 4: Run tests, confirm all pass**

```bash
venv/bin/pytest tests/test_vad.py -v
```

Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add audio/vad.py tests/test_vad.py
git commit -m "$(cat <<'EOF'
Add SileroVAD class wrapping the v5 ONNX model

Stateful LSTM-backed neural VAD. Runs inference with correctly-shaped
inputs (float32, 1x512 at 16 kHz, sr int64), threads hidden state
across calls, and exposes reset() for recording-session boundaries.
Input validation raises ValueError for wrong shape/dtype so contract
violations surface immediately rather than producing garbage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire Silero into `Recorder.record_silence_mode`

**Files:**
- Modify: `audio/recorder.py`
- Modify: `tests/test_recorder.py`
- Modify: `requirements.txt`

Replace the `webrtcvad`-based loop with a sample-accurate Silero-based loop.

- [ ] **Step 1: Write the new recorder test first**

Open `tests/test_recorder.py`. Read the file to see the existing test conventions and fixtures. Append a new test class at the end:

```python
from unittest.mock import MagicMock, patch


class TestRecordSilenceModeSilero:
    """Silero-based silence-mode recording."""

    @pytest.fixture
    def mock_pyaudio_stream(self):
        """Fake PyAudio stream yielding a scripted sequence of chunks."""
        with patch("audio.recorder.pyaudio.PyAudio") as pa_cls:
            pa = MagicMock()
            pa_cls.return_value = pa
            # device validation + sample rate picks
            pa.get_device_info_by_index.return_value = {
                "index": 0, "name": "mock", "maxInputChannels": 1,
                "defaultSampleRate": 16000.0,
            }
            pa.is_format_supported.return_value = True
            stream = MagicMock()
            pa.open.return_value = stream
            yield pa, stream

    def test_surfaces_download_failure(self, mock_pyaudio_stream):
        from audio.recorder import Recorder
        pa, stream = mock_pyaudio_stream
        with patch("audio.recorder.ensure_vad_model",
                   side_effect=RuntimeError("network down")):
            rec = Recorder(device_index=0)
            with pytest.raises(RuntimeError, match="network down"):
                rec.record_silence_mode(vad_aggressiveness=1, break_length=1)

    def test_resets_vad_before_use(self, mock_pyaudio_stream):
        from audio.recorder import Recorder
        pa, stream = mock_pyaudio_stream
        # One 30ms chunk of silence then stop flag trips
        stream.read.side_effect = [b"\x00" * 960] * 200

        fake_vad = MagicMock()
        fake_vad.is_speech.return_value = False  # never sees speech → loop body exits when stop set
        fake_vad.sample_rate = 16000
        fake_vad.chunk_samples = 512

        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD", return_value=fake_vad):
            rec = Recorder(device_index=0)

            # Stop after a few reads so the test terminates
            call_count = [0]
            original_is_speech = fake_vad.is_speech
            def counting_is_speech(*a, **kw):
                call_count[0] += 1
                if call_count[0] >= 3:
                    rec.stop()
                return False
            fake_vad.is_speech.side_effect = counting_is_speech

            rec.record_silence_mode(vad_aggressiveness=1, break_length=1)
            fake_vad.reset.assert_called_once()

    def test_stops_after_break_length_of_silence_post_speech(self, mock_pyaudio_stream):
        """Scripted: 1 s of speech then 2 s of silence with break_length=2."""
        from audio.recorder import Recorder
        pa, stream = mock_pyaudio_stream
        # 30ms chunks at 16kHz = 480 int16 samples = 960 bytes
        stream.read.return_value = b"\x00" * 960

        fake_vad = MagicMock()
        fake_vad.chunk_samples = 512
        fake_vad.sample_rate = 16000

        # Build a response sequence: first N calls return True (speech),
        # then False forever (silence). Recorder should stop after
        # break_length seconds of silence post-speech.
        speech_responses = [True] * 30  # ~1s of speech
        silence_responses = [False] * 1000  # plenty of silence
        fake_vad.is_speech.side_effect = speech_responses + silence_responses

        with patch("audio.recorder.ensure_vad_model"), \
             patch("audio.recorder.SileroVAD", return_value=fake_vad):
            rec = Recorder(device_index=0)
            audio = rec.record_silence_mode(vad_aggressiveness=1, break_length=2)

        # is_speech was called enough times to reach break_length silence
        # after the speech block. At 512-sample chunks in 16kHz, break_length=2s
        # = 2 * 16000 / 512 ≈ 62.5 silent calls. Plus 30 speech calls → ~93.
        assert fake_vad.is_speech.call_count >= 90
        assert fake_vad.is_speech.call_count < 200
        # audio is float32 in [-1, 1] at 16 kHz
        assert audio.dtype == np.float32
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0

    def test_stream_cleaned_up_on_vad_error(self, mock_pyaudio_stream):
        from audio.recorder import Recorder
        pa, stream = mock_pyaudio_stream
        with patch("audio.recorder.ensure_vad_model",
                   side_effect=RuntimeError("model download failed")):
            rec = Recorder(device_index=0)
            with pytest.raises(RuntimeError):
                rec.record_silence_mode(vad_aggressiveness=1, break_length=1)
        # Stream was never opened because ensure_vad_model failed first.
        pa.open.assert_not_called()
```

Add `import numpy as np` at the top of the file if not already imported.

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
venv/bin/pytest tests/test_recorder.py::TestRecordSilenceModeSilero -v
```

Expected: errors referencing `ensure_vad_model` / `SileroVAD` not importable from `audio.recorder` (because they aren't imported there yet).

- [ ] **Step 3: Rewrite `record_silence_mode`**

Open `audio/recorder.py`. Replace `import webrtcvad` with:

```python
from audio.vad import SileroVAD, aggressiveness_to_threshold, ensure_vad_model
```

Add (below existing imports):

```python
# Location of the Silero VAD model on disk. Downloaded on first use.
VAD_MODEL_PATH = os.path.expanduser("~/.whisper2text/vad/silero_vad.onnx")
```

Also add `import os` if not already present (it is).

Replace the entire body of `record_silence_mode` with the version below:

```python
    def record_silence_mode(
        self, vad_aggressiveness: int = 1, break_length: int = 5
    ) -> np.ndarray:
        """Record speech, stopping after ``break_length`` seconds of silence.

        Uses Silero VAD v5 (neural, ONNX) for speech detection. The 0-3
        ``vad_aggressiveness`` maps to an internal probability threshold
        (see audio.vad.AGGRESSIVENESS_TO_THRESHOLD).
        """
        # Ensure the VAD model is available BEFORE opening the PyAudio stream,
        # so a download failure doesn't leave an orphan stream.
        ensure_vad_model(VAD_MODEL_PATH)
        vad = SileroVAD(VAD_MODEL_PATH)
        vad.reset()
        threshold = aggressiveness_to_threshold(vad_aggressiveness)

        with self._lock:
            self._recording = True

        hw_rate = self._hw_rate
        chunk = int(hw_rate * CHUNK_DURATION_MS / 1000)

        stream_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": hw_rate,
            "input": True,
            "frames_per_buffer": chunk,
        }
        if self._device_index is not None:
            stream_kwargs["input_device_index"] = self._device_index

        try:
            stream = self._pa.open(**stream_kwargs)
        except Exception:
            with self._lock:
                self._recording = False
            raise

        frames: list[bytes] = []
        # Buffer of 16 kHz float32 samples fed into Silero VAD.
        vad_buffer = np.empty(0, dtype=np.float32)
        total_samples_16k = 0
        last_speech_idx = 0
        speech_detected = False
        silence_samples_threshold = break_length * WHISPER_RATE

        try:
            while self.is_recording:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

                samples = np.frombuffer(data, dtype=np.int16)
                if hw_rate != WHISPER_RATE:
                    samples = _resample(samples, hw_rate, WHISPER_RATE)
                chunk_f32 = samples.astype(np.float32) / 32768.0
                vad_buffer = np.concatenate([vad_buffer, chunk_f32])

                while vad_buffer.shape[0] >= vad.chunk_samples:
                    window = vad_buffer[: vad.chunk_samples]
                    vad_buffer = vad_buffer[vad.chunk_samples :]
                    total_samples_16k += vad.chunk_samples

                    if vad.is_speech(window, threshold):
                        speech_detected = True
                        last_speech_idx = total_samples_16k
                    elif speech_detected and (
                        total_samples_16k - last_speech_idx >= silence_samples_threshold
                    ):
                        with self._lock:
                            self._recording = False
                        break
        finally:
            with self._lock:
                self._recording = False
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        raw = b"".join(frames)
        return self._to_float32(raw, hw_rate)
```

- [ ] **Step 4: Drop webrtcvad from requirements.txt**

Remove the line `webrtcvad>=2.0.10` from `requirements.txt`.

- [ ] **Step 5: Uninstall webrtcvad from the active venv**

```bash
venv/bin/pip uninstall -y webrtcvad
```

Expected: `Successfully uninstalled webrtcvad-2.0.10`. Do NOT re-run pip install — the only removed dep is webrtcvad.

- [ ] **Step 6: Run the full test suite**

```bash
venv/bin/pytest tests/ -v --ignore=tests/test_autopaste_e2e.py --ignore=tests/test_paste_realistic.py
```

Expected: all tests pass including the four new `TestRecordSilenceModeSilero` tests. The existing `TestRecordSilenceMode` (webrtcvad-era) tests, if any, may need to be updated or removed — check `tests/test_recorder.py` for references to `webrtcvad` or `vad_aggressiveness` that still assume the old API. Remove tests that specifically verify webrtcvad internals (they're no longer meaningful). Keep tests that verify `Recorder` public behaviour — those should still pass with Silero.

If an existing test fails because it was asserting webrtcvad-specific behaviour, delete that test and note in the commit message which tests were removed and why.

- [ ] **Step 7: Commit**

```bash
git add audio/recorder.py audio/vad.py tests/test_recorder.py tests/test_vad.py requirements.txt
git commit -m "$(cat <<'EOF'
Replace webrtcvad with Silero VAD in silence-detection mode

Recorder.record_silence_mode now uses Silero VAD v5 via ONNX Runtime.
Chunk accumulation decouples PyAudio's hw-native 30 ms cadence from
Silero's required 512-sample (32 ms at 16 kHz) windows. Silence
cutoff is now sample-accurate rather than chunk-counted.

The 0-3 aggressiveness setting is preserved and mapped internally to
a Silero probability threshold. webrtcvad is removed from the
dependency list.

Download-on-first-use keeps the repo lean; a network failure surfaces
via the existing error_signal path and the recording never starts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Manual smoke test

- [ ] **Step 1: First-run download**

Delete any pre-existing VAD file so the download path is exercised:

```bash
rm -f ~/.whisper2text/vad/silero_vad.onnx
```

- [ ] **Step 2: Launch the app**

```bash
pkill -f whisper2text 2>/dev/null; sleep 1; rm -f ~/.whisper2text/app.lock
venv/bin/python whisper2text.py
```

- [ ] **Step 3: Record in silence mode**

Ensure `Settings → Recording Mode = silence` and `VAD Aggressiveness = 1`. Press Record (or the hotkey), speak a sentence, then stay silent. Recording should stop automatically ~5 s after you finish speaking (default `break_length`).

- [ ] **Step 4: Verify the download occurred**

```bash
ls -lh ~/.whisper2text/vad/silero_vad.onnx
```

Expected: a ~1.8 MB file.

- [ ] **Step 5: Record again — second time should be instant**

No download delay on the second recording. Transcription works normally.

- [ ] **Step 6: Optional — try aggressiveness 0 vs 3**

Compare behaviour on a quiet voice. Aggressiveness 0 (threshold 0.3) should trigger more eagerly; 3 (threshold 0.9) should require clear, loud speech.

No code changes in this task. Report any oddities.
