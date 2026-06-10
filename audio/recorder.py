"""Audio recording module with button-mode and silence-detection modes."""

import contextlib
import logging
import os
import threading

import numpy as np
import pyaudio

from audio.vad import SileroVAD, aggressiveness_to_threshold, ensure_vad_model

logger = logging.getLogger(__name__)

# Location of the Silero VAD model on disk. Downloaded on first use.
VAD_MODEL_PATH = os.path.expanduser("~/.whisper2text/vad/silero_vad.onnx")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
WHISPER_RATE = 16000          # Whisper expects 16 kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_DURATION_MS = 30


@contextlib.contextmanager
def _pipewire_target(node_name):
    """Point the PipeWire ALSA plugin at *node_name* for the enclosed open().

    The plugin reads ``PIPEWIRE_NODE`` when the PCM is opened; outside that
    window the variable must not linger or it would leak into unrelated
    subprocesses (ydotool, pw-dump).  Recordings are serialized by
    ``Recorder.is_recording`` so the process-global variable is not racy.
    If the named node has vanished, PipeWire links the stream to the
    default source instead — exactly the fallback behaviour we want.
    """
    if not node_name:
        yield
        return
    prior = os.environ.get("PIPEWIRE_NODE")
    os.environ["PIPEWIRE_NODE"] = node_name
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("PIPEWIRE_NODE", None)
        else:
            os.environ["PIPEWIRE_NODE"] = prior


class Recorder:
    """Record audio from the microphone.

    Always opens PortAudio's default input device, which routes through
    PipeWire's shared ALSA PCM — never a raw ``hw:`` device, so recording
    can't fail with exclusive-access errors.  A specific microphone is
    selected by setting :attr:`target_node` to a PipeWire ``node.name``
    before recording starts.

    Supports two recording modes:
      * **button mode** -- record while a button is held, stop on release.
      * **silence mode** -- start on speech, stop after prolonged silence.

    Parameters
    ----------
    target_node : str or None
        PipeWire source ``node.name`` to record from.  ``None`` records
        from the system default source.
    """

    def __init__(self, target_node=None):
        self._pa = pyaudio.PyAudio()
        self.target_node = target_node
        self._recording = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Thread-safe recording flag
    # ------------------------------------------------------------------
    def stop(self):
        """Signal the recording loop to finish (thread-safe)."""
        with self._lock:
            self._recording = False

    @property
    def is_recording(self):
        """Return ``True`` while a recording session is active."""
        with self._lock:
            return self._recording

    # ------------------------------------------------------------------
    # Stream handling
    # ------------------------------------------------------------------
    def _open_stream(self, chunk):
        """Open the capture stream, targeting :attr:`target_node` if set.

        On failure the recording flag is cleared and the exception
        propagates to the caller.
        """
        stream_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": WHISPER_RATE,
            "input": True,
            "frames_per_buffer": chunk,
        }
        try:
            with _pipewire_target(self.target_node):
                return self._pa.open(**stream_kwargs)
        except Exception:
            with self._lock:
                self._recording = False
            raise

    # ------------------------------------------------------------------
    # Recording modes
    # ------------------------------------------------------------------
    def record_button_mode(self) -> np.ndarray:
        """Record until :meth:`stop` is called from another thread.

        Returns
        -------
        np.ndarray
            Recorded audio as float32 samples normalised to [-1, 1] at 16 kHz.
        """
        with self._lock:
            self._recording = True

        chunk = int(WHISPER_RATE * CHUNK_DURATION_MS / 1000)
        stream = self._open_stream(chunk)

        frames: list[bytes] = []

        try:
            while self.is_recording:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
        finally:
            with self._lock:
                self._recording = False
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        raw = b"".join(frames)
        return self._to_float32(raw)

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

        chunk = int(WHISPER_RATE * CHUNK_DURATION_MS / 1000)
        stream = self._open_stream(chunk)

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
                chunk_f32 = samples.astype(np.float32) / 32768.0
                vad_buffer = np.concatenate([vad_buffer, chunk_f32])

                while vad_buffer.shape[0] >= vad.chunk_samples:
                    window = vad_buffer[: vad.chunk_samples]
                    vad_buffer = vad_buffer[vad.chunk_samples :]
                    total_samples_16k += vad.chunk_samples

                    try:
                        is_speech = vad.is_speech(window, threshold)
                    except Exception:
                        logger.exception(
                            "Silero VAD inference failed at sample %d "
                            "(threshold=%s, window shape=%s)",
                            total_samples_16k, threshold, window.shape,
                        )
                        raise

                    if is_speech:
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
        return self._to_float32(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float32(raw_bytes: bytes) -> np.ndarray:
        """Convert raw int16 PCM bytes to a float32 numpy array in [-1, 1]."""
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def cleanup(self):
        """Terminate PyAudio.  Safe to call more than once."""
        with self._lock:
            self._recording = False
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
