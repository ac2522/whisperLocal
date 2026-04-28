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

        Parameters
        ----------
        audio_data : numpy.ndarray or bytes
            Float32 numpy array in [-1, 1], int16 numpy array, or raw
            int16 PCM bytes. Coerced to little-endian int16 PCM at
            ``SAMPLE_RATE`` mono before being sent.
        vocabulary : list[str] or None, optional
            Domain-specific words/phrases sent to Deepgram via the
            ``keyterm`` parameter to bias the recogniser. If None or
            empty, no keyterms are sent.

        Returns
        -------
        str
            Cleaned transcript text (whitespace stripped).

        Raises
        ------
        RuntimeError
            If the client is not initialised, or if the response shape
            from Deepgram is missing the expected ``results.channels[0]
            .alternatives[0].transcript`` path.
        Exception
            Any SDK-level error (auth failure, network error, rate
            limit, etc.) propagates unchanged to the caller; MainWindow
            routes these to the error panel.
        """
        if self._client is None:
            raise RuntimeError("Deepgram client not initialised")

        pcm = _to_int16_pcm_bytes(audio_data)

        # deepgram-sdk v7 does NOT accept sample_rate / channels as
        # top-level kwargs on transcribe_file. The Listen REST API still
        # needs them as query-string params for raw linear16 PCM (which
        # has no header), so we route them through request_options'
        # additional_query_parameters escape hatch.
        kwargs = {
            "model": self._model,
            "smart_format": True,
            "language": self._language,
            "encoding": "linear16",
            "request_options": {
                "additional_query_parameters": {
                    "sample_rate": self.SAMPLE_RATE,
                    "channels": 1,
                },
            },
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
