"""Silero VAD v5 integration: ONNX-based neural voice activity detection.

Replaces the older webrtcvad DSP model in Recorder.record_silence_mode.
"""

import logging
import os
import urllib.request

import numpy as np
import onnxruntime

logger = logging.getLogger(__name__)

AGGRESSIVENESS_TO_THRESHOLD: dict[int, float] = {
    0: 0.3,
    1: 0.5,
    2: 0.7,
    3: 0.9,
}

SILERO_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)


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


def ensure_vad_model(dest_path: str) -> str:
    """Ensure the Silero VAD ONNX model is present at ``dest_path``.

    Downloads from ``SILERO_MODEL_URL`` on first use, staging into a
    ``.partial`` file and renaming atomically on success. Returns the
    final path. Raises the underlying exception on download failure
    after cleaning up any partial file.
    """
    if os.path.isfile(dest_path):
        return dest_path

    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
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
