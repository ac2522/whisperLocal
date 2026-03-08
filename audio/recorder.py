"""Audio recording module with button-mode and silence-detection modes."""

import threading

import numpy as np
import pyaudio
import webrtcvad

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
WHISPER_RATE = 16000          # Whisper expects 16 kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_DURATION_MS = 30


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
        self._device_index = device_index
        self._recording = False
        self._lock = threading.Lock()
        self._hw_rate = _pick_sample_rate(self._pa, device_index)

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
        """Record speech, stopping after *break_length* seconds of silence.

        The method waits for voice activity before it starts accumulating
        audio, then continues until *break_length* consecutive seconds of
        silence are detected.

        Parameters
        ----------
        vad_aggressiveness : int
            WebRTC VAD aggressiveness (0-3).  Higher = more aggressive
            filtering of non-speech.
        break_length : int
            Seconds of silence after speech that trigger a stop.

        Returns
        -------
        np.ndarray
            Recorded audio as float32 samples normalised to [-1, 1] at 16 kHz.
        """
        vad = webrtcvad.Vad(vad_aggressiveness)

        with self._lock:
            self._recording = True

        hw_rate = self._hw_rate
        chunk = int(hw_rate * CHUNK_DURATION_MS / 1000)

        # VAD needs 16 kHz audio; if recording at a different rate we
        # resample each chunk before passing it to the VAD.
        vad_chunk = int(WHISPER_RATE * CHUNK_DURATION_MS / 1000)

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

        speech_detected = False
        silent_chunks = 0
        chunks_per_second = 1000 // CHUNK_DURATION_MS
        silence_threshold = break_length * chunks_per_second

        try:
            while self.is_recording:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

                # Resample chunk to 16 kHz for VAD if needed
                if hw_rate != WHISPER_RATE:
                    samples = np.frombuffer(data, dtype=np.int16)
                    resampled = _resample(samples, hw_rate, WHISPER_RATE)
                    vad_data = resampled.astype(np.int16).tobytes()
                else:
                    vad_data = data

                is_speech = vad.is_speech(vad_data, WHISPER_RATE)

                if is_speech:
                    speech_detected = True
                    silent_chunks = 0
                elif speech_detected:
                    silent_chunks += 1
                    if silent_chunks >= silence_threshold:
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
