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
