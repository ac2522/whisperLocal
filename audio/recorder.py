"""Audio recording module with button-mode and silence-detection modes."""

import logging
import threading

import os

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


def _validate_device(pa: pyaudio.PyAudio, device_index: int | None) -> int | None:
    """Validate that device_index is a usable input device.

    Returns the device_index if valid, or None (system default) if the device
    doesn't exist or can't be used for mono input.  This prevents PortAudio
    native crashes (SEGV) that occur when open() is called with an invalid device.
    """
    if device_index is None:
        return None
    try:
        info = pa.get_device_info_by_index(device_index)
        if info.get("maxInputChannels", 0) < CHANNELS:
            logger.warning(
                "Device %d (%s) has no input channels, falling back to default",
                device_index, info.get("name", "unknown"),
            )
            return None
        return device_index
    except Exception as e:
        logger.warning(
            "Device index %d is invalid (%s), falling back to default",
            device_index, e,
        )
        return None


def _pick_sample_rate(pa: pyaudio.PyAudio, device_index: int | None) -> int:
    """Return WHISPER_RATE if the device supports it, otherwise its native rate."""
    if device_index is not None:
        try:
            pa.is_format_supported(
                WHISPER_RATE,
                input_device=device_index,
                input_channels=CHANNELS,
                input_format=FORMAT,
            )
            return WHISPER_RATE
        except ValueError:
            info = pa.get_device_info_by_index(device_index)
            return int(info["defaultSampleRate"])
    return WHISPER_RATE


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 audio from *src_rate* to *dst_rate* using linear interpolation."""
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    n_samples = int(duration * dst_rate)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(audio.dtype)


class Recorder:
    """Record audio from the microphone.

    Supports two recording modes:
      * **button mode** -- record while a button is held, stop on release.
      * **silence mode** -- start on speech, stop after prolonged silence.

    Parameters
    ----------
    device_index : int or None
        PyAudio input device index.  ``None`` uses the system default.
    """

    def __init__(self, device_index=None):
        self._pa = pyaudio.PyAudio()
        self._device_index = _validate_device(self._pa, device_index)
        self._recording = False
        self._lock = threading.Lock()
        self._hw_rate = _pick_sample_rate(self._pa, self._device_index)

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
        return self._to_float32(raw, hw_rate)

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_float32(raw_bytes: bytes, src_rate: int) -> np.ndarray:
        """Convert raw int16 PCM bytes to a float32 numpy array in [-1, 1] at 16 kHz."""
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        if src_rate != WHISPER_RATE:
            samples = _resample(samples, src_rate, WHISPER_RATE)
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
