"""Thin wrapper around pywhispercpp providing a managed Whisper model lifecycle."""

import atexit
import gc
import logging

import numpy as np
from pywhispercpp.model import Model

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Load, manage and transcribe with a whisper.cpp model.

    Parameters
    ----------
    model_path : str
        Filesystem path to a ``ggml-*.bin`` model file.
    language : str, optional
        Language code passed to whisper.cpp (default ``"en"``).
    n_threads : int, optional
        Number of CPU threads used for inference (default ``4``).
    """

    def __init__(self, model_path: str, language: str = "en", n_threads: int = 4):
        self._language = language
        self._n_threads = n_threads
        self._model = None

        self._load_model(model_path)

        # Ensure the model is freed when the process exits.
        atexit.register(self.unload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Instantiate the underlying ``pywhispercpp`` model."""
        logger.info("Loading whisper model from %s", model_path)
        self._model = Model(
            model_path,
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            n_threads=self._n_threads,
            language=self._language,
        )
        logger.info("Model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return ``True`` if a model is currently loaded."""
        return self._model is not None

    def unload(self) -> None:
        """Release the model and reclaim memory.

        This method is idempotent -- calling it on an already-unloaded
        engine is a safe no-op.
        """
        if self._model is not None:
            logger.info("Unloading whisper model")
            del self._model
            self._model = None
            gc.collect()

    def reload(self, model_path: str, language: str | None = None) -> None:
        """Unload the current model and load a new one.

        Parameters
        ----------
        model_path : str
            Path to the new model file.
        language : str or None, optional
            If provided, override the language for the new model.
        """
        self.unload()
        if language is not None:
            self._language = language
        self._load_model(model_path)

    def transcribe(self, audio_data) -> str:
        """Transcribe audio data and return the full text.

        Parameters
        ----------
        audio_data : numpy.ndarray or bytes
            If a numpy ``float32`` array, used directly.  If ``bytes``
            (assumed int16 PCM), it is converted to float32 by dividing
            by ``32768.0``.

        Returns
        -------
        str
            Concatenation of all segment texts separated by a single
            space.

        Raises
        ------
        RuntimeError
            If no model is currently loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Convert raw int16 bytes to float32 numpy array.
        if isinstance(audio_data, (bytes, bytearray)):
            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        segments = self._model.transcribe(audio_data)
        return " ".join(seg.text for seg in segments)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
