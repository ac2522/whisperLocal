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
        elif np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / 32768.0
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
