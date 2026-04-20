"""Silero VAD v5 integration: ONNX-based neural voice activity detection.

Replaces the older webrtcvad DSP model in Recorder.record_silence_mode.
"""

import logging
import os
import urllib.request

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
