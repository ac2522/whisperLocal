# Whisper2Text Performance Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor whisper2text from slow PyTorch whisper to fast whisper.cpp (pywhispercpp with CUDA), split into modules, add error resilience and UI improvements.

**Architecture:** Modular split: `config/` (settings), `engine/` (whisper.cpp wrapper + model manager), `audio/` (recording + device selection), `ui/` (PyQt5 main window, settings dialog, error panel). Entry point remains `whisper2text.py`.

**Tech Stack:** Python 3.12, pywhispercpp (CUDA build), PyQt5, PyAudio, webrtcvad, pyperclip, pynput

---

## Pre-Implementation: Environment Setup

### Task 0: Create Python venv and install dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Step 1: Create fresh venv (old whisper_env uses Python 3.10, we need 3.12)**

Run:
```bash
cd ~/Development/whisperLocal
python3 -m venv venv
source venv/bin/activate
```
Expected: New venv created, prompt shows `(venv)`

**Step 2: Install system deps if missing**

Run:
```bash
sudo apt install -y portaudio19-dev python3-dev cmake build-essential
```
Expected: System libraries installed

**Step 3: Install pywhispercpp with CUDA**

Run:
```bash
GGML_CUDA=1 pip install pywhispercpp
```
Expected: Builds from source with CUDA support. If CUDA build fails, fall back to `pip install pywhispercpp` (CPU-only) and note it.

**Step 4: Install remaining dependencies**

Run:
```bash
pip install PyQt5 pyaudio webrtcvad pyperclip pynput numpy
```
Expected: All packages install successfully

**Step 5: Update requirements.txt**

Replace contents of `requirements.txt` with:
```
# Audio processing
pyaudio>=0.2.11
webrtcvad>=2.0.10
numpy>=1.24.0

# Speech recognition (whisper.cpp)
# Install with CUDA: GGML_CUDA=1 pip install pywhispercpp
pywhispercpp>=1.2.0

# GUI
PyQt5>=5.15.0

# Utilities
pyperclip>=1.8.2
pynput>=1.7.6
```

**Step 6: Update .gitignore**

Add to `.gitignore`:
```
# Models (large binary files)
models/*.bin
```

**Step 7: Verify pywhispercpp loads and can use a model**

Run:
```bash
python -c "
from pywhispercpp.model import Model
m = Model('models/ggml-base.bin', redirect_whispercpp_logs_to=None)
print('Model loaded successfully')
print(Model.system_info())
del m
print('Model freed')
"
```
Expected: Model loads, system info shows CUDA if available, model freed without error.

**Step 8: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: update deps for whisper.cpp backend"
```

---

## Phase 1: Core Infrastructure (config + engine)

### Task 1: Settings Manager module

**Files:**
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `tests/__init__.py`
- Create: `tests/test_settings.py`

**Step 1: Create test directory and test file**

Create `tests/__init__.py` (empty).

Create `tests/test_settings.py`:
```python
import json
import os
import tempfile
import pytest
from config.settings import SettingsManager


@pytest.fixture
def temp_settings_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def settings(temp_settings_dir):
    return SettingsManager(settings_dir=temp_settings_dir)


def test_default_values(settings):
    assert settings.get('model_size') == 'base'
    assert settings.get('vad_aggressiveness') == 1
    assert settings.get('recording_mode') == 'silence'
    assert settings.get('break_length') == 5
    assert settings.get('auto_paste') is False
    assert settings.get('transcripts') == []


def test_set_and_get(settings):
    settings.set('model_size', 'small')
    assert settings.get('model_size') == 'small'


def test_save_and_load(temp_settings_dir):
    s1 = SettingsManager(settings_dir=temp_settings_dir)
    s1.set('model_size', 'medium')
    s1.set('transcripts', ['hello', 'world'])
    s1.save()

    s2 = SettingsManager.__new__(SettingsManager)
    s2._lock = __import__('threading').Lock()
    s2._settings_dir = temp_settings_dir
    s2._settings_file = os.path.join(temp_settings_dir, 'settings.json')
    s2._settings = {}
    s2._load_settings()

    assert s2.get('model_size') == 'medium'
    assert s2.get('transcripts') == ['hello', 'world']


def test_get_unknown_key_returns_default(settings):
    assert settings.get('nonexistent', 'fallback') == 'fallback'


def test_get_all(settings):
    settings.set('model_size', 'large')
    all_settings = settings.get_all()
    assert all_settings['model_size'] == 'large'
    assert 'vad_aggressiveness' in all_settings


def test_backward_compatible_with_existing_settings(tmp_path):
    """Test that existing settings.json from old app is loaded correctly."""
    settings_file = tmp_path / 'settings.json'
    settings_file.write_text(json.dumps({
        "transcripts": ["old transcript"],
        "vad_aggressiveness": 2,
        "model_size": "small",
        "padding_duration_ms": 1000,
        "recording_mode": "button",
        "break_length": 3,
        "auto_paste": True
    }))
    s = SettingsManager(settings_dir=str(tmp_path))
    assert s.get('model_size') == 'small'
    assert s.get('recording_mode') == 'button'
    assert s.get('transcripts') == ["old transcript"]
    assert s.get('auto_paste') is True
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/Development/whisperLocal && source venv/bin/activate && python -m pytest tests/test_settings.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'config'`

**Step 3: Implement SettingsManager**

Create `config/__init__.py` (empty).

Create `config/settings.py`:
```python
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULTS = {
    'model_size': 'base',
    'vad_aggressiveness': 1,
    'padding_duration_ms': 1000,
    'recording_mode': 'silence',
    'break_length': 5,
    'auto_paste': False,
    'transcripts': [],
    'audio_device_index': None,
    'audio_device_name': None,
}


class SettingsManager:
    def __init__(self, settings_dir=None):
        self._lock = threading.Lock()
        self._settings_dir = settings_dir or os.path.join(Path.home(), '.whisper2text')
        os.makedirs(self._settings_dir, exist_ok=True)
        self._settings_file = os.path.join(self._settings_dir, 'settings.json')
        self._settings = dict(DEFAULTS)
        self._load_settings()

    def _load_settings(self):
        if os.path.exists(self._settings_file):
            try:
                with open(self._settings_file, 'r') as f:
                    saved = json.load(f)
                self._settings.update(saved)
                logger.info("Settings loaded from %s", self._settings_file)
            except Exception:
                logger.error("Failed to load settings", exc_info=True)

    def get(self, key, default=None):
        with self._lock:
            if default is not None:
                return self._settings.get(key, default)
            return self._settings.get(key, DEFAULTS.get(key))

    def set(self, key, value):
        with self._lock:
            self._settings[key] = value

    def get_all(self):
        with self._lock:
            return dict(self._settings)

    def save(self):
        with self._lock:
            try:
                with open(self._settings_file, 'w') as f:
                    json.dump(self._settings, f, indent=4)
                logger.info("Settings saved to %s", self._settings_file)
            except Exception:
                logger.error("Failed to save settings", exc_info=True)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/Development/whisperLocal && python -m pytest tests/test_settings.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add config/ tests/
git commit -m "feat: extract SettingsManager into config module with tests"
```

---

### Task 2: Whisper Engine module

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/whisper_engine.py`
- Create: `tests/test_engine.py`

**Step 1: Write failing tests**

Create `tests/test_engine.py`:
```python
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from engine.whisper_engine import WhisperEngine

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
BASE_MODEL = os.path.join(MODELS_DIR, 'ggml-base.bin')


@pytest.fixture
def engine():
    """Create engine with base model if available, skip otherwise."""
    if not os.path.exists(BASE_MODEL):
        pytest.skip("ggml-base.bin not available")
    e = WhisperEngine(model_path=BASE_MODEL)
    yield e
    e.unload()


def test_engine_loads_model(engine):
    assert engine.is_loaded()


def test_engine_unload(engine):
    engine.unload()
    assert not engine.is_loaded()


def test_engine_unload_is_idempotent(engine):
    engine.unload()
    engine.unload()  # Should not raise
    assert not engine.is_loaded()


def test_engine_transcribe_silence(engine):
    """Transcribing silence should return empty or near-empty text."""
    silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of silence
    result = engine.transcribe(silence)
    assert isinstance(result, str)


def test_engine_transcribe_requires_loaded_model():
    e = WhisperEngine.__new__(WhisperEngine)
    e._model = None
    with pytest.raises(RuntimeError, match="No model loaded"):
        e.transcribe(np.zeros(16000, dtype=np.float32))


def test_engine_context_manager():
    if not os.path.exists(BASE_MODEL):
        pytest.skip("ggml-base.bin not available")
    with WhisperEngine(model_path=BASE_MODEL) as e:
        assert e.is_loaded()
    assert not e.is_loaded()


def test_engine_reload_model():
    """Switching models should unload old model first."""
    if not os.path.exists(BASE_MODEL):
        pytest.skip("ggml-base.bin not available")
    e = WhisperEngine(model_path=BASE_MODEL)
    assert e.is_loaded()
    e.reload(BASE_MODEL)  # Reload same model
    assert e.is_loaded()
    e.unload()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'engine'`

**Step 3: Implement WhisperEngine**

Create `engine/__init__.py` (empty).

Create `engine/whisper_engine.py`:
```python
import atexit
import gc
import logging
import numpy as np
from pywhispercpp.model import Model

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, model_path, language='en', n_threads=4):
        self._model = None
        self._model_path = model_path
        self._language = language
        self._n_threads = n_threads
        self._load(model_path)
        atexit.register(self.unload)

    def _load(self, model_path):
        logger.info("Loading model: %s", model_path)
        self._model = Model(
            model_path,
            n_threads=self._n_threads,
            language=self._language,
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        self._model_path = model_path
        logger.info("Model loaded successfully")

    def is_loaded(self):
        return self._model is not None

    def unload(self):
        if self._model is not None:
            logger.info("Unloading model")
            del self._model
            self._model = None
            gc.collect()
            logger.info("Model unloaded, memory freed")

    def reload(self, model_path, language=None):
        self.unload()
        if language:
            self._language = language
        self._load(model_path)

    def transcribe(self, audio_data):
        if self._model is None:
            raise RuntimeError("No model loaded")
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._model.transcribe(audio_data)
        return ' '.join(seg.text for seg in segments).strip()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()

    def __del__(self):
        self.unload()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine.py -v`
Expected: All 7 tests PASS (or skip if model not available)

**Step 5: Commit**

```bash
git add engine/ tests/test_engine.py
git commit -m "feat: add WhisperEngine wrapping pywhispercpp with CUDA"
```

---

### Task 3: Model Manager module

**Files:**
- Create: `engine/model_manager.py`
- Create: `tests/test_model_manager.py`

**Step 1: Write failing tests**

Create `tests/test_model_manager.py`:
```python
import os
import pytest
from engine.model_manager import ModelManager

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


@pytest.fixture
def manager():
    return ModelManager(models_dir=MODELS_DIR)


def test_list_downloaded_models(manager):
    models = manager.list_downloaded()
    # We know ggml-base.bin and ggml-small.bin exist
    names = [m['name'] for m in models]
    assert 'ggml-base.bin' in names
    assert 'ggml-small.bin' in names


def test_list_downloaded_includes_size(manager):
    models = manager.list_downloaded()
    for m in models:
        assert 'size_mb' in m
        assert m['size_mb'] > 0


def test_list_available_models(manager):
    available = manager.list_available()
    names = [m['name'] for m in available]
    assert 'ggml-base.bin' in names
    assert 'ggml-small.bin' in names
    assert 'ggml-medium.bin' in names
    assert 'ggml-large-v3-turbo.bin' in names
    assert 'ggml-medium-q5_0.bin' in names


def test_is_downloaded(manager):
    assert manager.is_downloaded('ggml-base.bin')
    assert not manager.is_downloaded('ggml-nonexistent.bin')


def test_get_model_path(manager):
    path = manager.get_model_path('ggml-base.bin')
    assert os.path.exists(path)
    assert path.endswith('ggml-base.bin')


def test_get_model_path_not_downloaded(manager):
    with pytest.raises(FileNotFoundError):
        manager.get_model_path('ggml-nonexistent.bin')


def test_delete_model(manager, tmp_path):
    """Test deleting a model (using a temp copy)."""
    tmp_manager = ModelManager(models_dir=str(tmp_path))
    fake_model = tmp_path / 'ggml-test.bin'
    fake_model.write_bytes(b'fake model data')
    assert tmp_manager.is_downloaded('ggml-test.bin')
    tmp_manager.delete_model('ggml-test.bin')
    assert not tmp_manager.is_downloaded('ggml-test.bin')
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'engine.model_manager'`

**Step 3: Implement ModelManager**

Create `engine/model_manager.py`:
```python
import logging
import os
import urllib.request
from typing import Callable, Optional

logger = logging.getLogger(__name__)

HUGGINGFACE_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

AVAILABLE_MODELS = [
    {"name": "ggml-base.bin", "size_mb": 142, "description": "Base model (~142MB)"},
    {"name": "ggml-base.en.bin", "size_mb": 142, "description": "Base English-only (~142MB)"},
    {"name": "ggml-small.bin", "size_mb": 466, "description": "Small model (~466MB)"},
    {"name": "ggml-small.en.bin", "size_mb": 466, "description": "Small English-only (~466MB)"},
    {"name": "ggml-small-q5_1.bin", "size_mb": 181, "description": "Small quantized Q5 (~181MB)"},
    {"name": "ggml-small.en-q5_1.bin", "size_mb": 181, "description": "Small English Q5 (~181MB)"},
    {"name": "ggml-small-q8_0.bin", "size_mb": 264, "description": "Small quantized Q8 (~264MB)"},
    {"name": "ggml-medium.bin", "size_mb": 1533, "description": "Medium model (~1.5GB)"},
    {"name": "ggml-medium.en.bin", "size_mb": 1533, "description": "Medium English-only (~1.5GB)"},
    {"name": "ggml-medium-q5_0.bin", "size_mb": 539, "description": "Medium quantized Q5 (~539MB)"},
    {"name": "ggml-medium.en-q5_0.bin", "size_mb": 539, "description": "Medium English Q5 (~539MB)"},
    {"name": "ggml-medium-q8_0.bin", "size_mb": 836, "description": "Medium quantized Q8 (~836MB)"},
    {"name": "ggml-large-v3.bin", "size_mb": 3095, "description": "Large V3 (~3GB, may not fit 6GB VRAM)"},
    {"name": "ggml-large-v3-q5_0.bin", "size_mb": 1080, "description": "Large V3 quantized Q5 (~1GB)"},
    {"name": "ggml-large-v3-turbo.bin", "size_mb": 1624, "description": "Large V3 Turbo (~1.6GB, fits 6GB VRAM)"},
    {"name": "ggml-large-v3-turbo-q5_0.bin", "size_mb": 574, "description": "Large V3 Turbo Q5 (~574MB)"},
    {"name": "ggml-large-v3-turbo-q8_0.bin", "size_mb": 874, "description": "Large V3 Turbo Q8 (~874MB)"},
]


class ModelManager:
    def __init__(self, models_dir):
        self._models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def list_downloaded(self):
        result = []
        for f in sorted(os.listdir(self._models_dir)):
            if f.endswith('.bin'):
                path = os.path.join(self._models_dir, f)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                desc = ""
                for m in AVAILABLE_MODELS:
                    if m['name'] == f:
                        desc = m['description']
                        break
                result.append({
                    'name': f,
                    'path': path,
                    'size_mb': round(size_mb, 1),
                    'description': desc,
                })
        return result

    def list_available(self):
        return list(AVAILABLE_MODELS)

    def is_downloaded(self, model_name):
        return os.path.exists(os.path.join(self._models_dir, model_name))

    def get_model_path(self, model_name):
        path = os.path.join(self._models_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    def delete_model(self, model_name):
        path = os.path.join(self._models_dir, model_name)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted model: %s", model_name)

    def download_model(self, model_name, progress_callback: Optional[Callable] = None):
        url = f"{HUGGINGFACE_BASE}/{model_name}"
        dest = os.path.join(self._models_dir, model_name)
        partial = dest + '.partial'
        logger.info("Downloading %s from %s", model_name, url)

        def _reporthook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                progress_callback(percent, downloaded, total_size)

        try:
            urllib.request.urlretrieve(url, partial, reporthook=_reporthook)
            os.rename(partial, dest)
            logger.info("Downloaded %s successfully", model_name)
        except Exception:
            if os.path.exists(partial):
                os.remove(partial)
            logger.error("Failed to download %s", model_name, exc_info=True)
            raise
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add engine/model_manager.py tests/test_model_manager.py
git commit -m "feat: add ModelManager for listing, downloading, deleting ggml models"
```

---

### Task 4: Audio Device Manager

**Files:**
- Create: `audio/__init__.py`
- Create: `audio/device_manager.py`
- Create: `tests/test_device_manager.py`

**Step 1: Write failing tests**

Create `tests/test_device_manager.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from audio.device_manager import DeviceManager


@pytest.fixture
def mock_pyaudio():
    with patch('audio.device_manager.pyaudio') as mock_pa:
        instance = MagicMock()
        mock_pa.PyAudio.return_value = instance
        instance.get_device_count.return_value = 3
        instance.get_device_info_by_index.side_effect = [
            {'index': 0, 'name': 'Built-in Mic', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0},
            {'index': 1, 'name': 'Rode Wireless MICRO', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0},
            {'index': 2, 'name': 'HDMI Output', 'maxInputChannels': 0, 'defaultSampleRate': 44100.0},
        ]
        instance.get_default_input_device_info.return_value = {
            'index': 0, 'name': 'Built-in Mic', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0
        }
        yield mock_pa, instance


def test_list_input_devices(mock_pyaudio):
    dm = DeviceManager()
    devices = dm.list_input_devices()
    # Should only return devices with input channels > 0
    assert len(devices) == 2
    names = [d['name'] for d in devices]
    assert 'Built-in Mic' in names
    assert 'Rode Wireless MICRO' in names
    assert 'HDMI Output' not in names
    dm.cleanup()


def test_get_default_device(mock_pyaudio):
    dm = DeviceManager()
    default = dm.get_default_device()
    assert default['name'] == 'Built-in Mic'
    dm.cleanup()


def test_get_device_by_index(mock_pyaudio):
    dm = DeviceManager()
    device = dm.get_device_by_index(1)
    assert device['name'] == 'Rode Wireless MICRO'
    dm.cleanup()


def test_get_device_by_index_fallback(mock_pyaudio):
    """If saved index doesn't exist, fall back to default."""
    dm = DeviceManager()
    device = dm.get_device_by_index(99)  # doesn't exist
    assert device['name'] == 'Built-in Mic'  # falls back to default
    dm.cleanup()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_device_manager.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'audio'`

**Step 3: Implement DeviceManager**

Create `audio/__init__.py` (empty).

Create `audio/device_manager.py`:
```python
import logging
import pyaudio

logger = logging.getLogger(__name__)


class DeviceManager:
    def __init__(self):
        self._pa = pyaudio.PyAudio()

    def list_input_devices(self):
        devices = []
        for i in range(self._pa.get_device_count()):
            try:
                info = self._pa.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    devices.append({
                        'index': info['index'],
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate']),
                    })
            except Exception:
                logger.warning("Could not query device %d", i, exc_info=True)
        return devices

    def get_default_device(self):
        try:
            info = self._pa.get_default_input_device_info()
            return {
                'index': info['index'],
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': int(info['defaultSampleRate']),
            }
        except Exception:
            logger.error("Could not get default input device", exc_info=True)
            return None

    def get_device_by_index(self, index):
        try:
            info = self._pa.get_device_info_by_index(index)
            if info.get('maxInputChannels', 0) > 0:
                return {
                    'index': info['index'],
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                }
        except Exception:
            logger.warning("Device index %d not found, falling back to default", index)
        return self.get_default_device()

    def cleanup(self):
        if self._pa:
            self._pa.terminate()
            self._pa = None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_device_manager.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add audio/ tests/test_device_manager.py
git commit -m "feat: add DeviceManager for audio input device selection"
```

---

### Task 5: Audio Recorder module

**Files:**
- Create: `audio/recorder.py`
- Create: `tests/test_recorder.py`

**Step 1: Write failing tests**

Create `tests/test_recorder.py`:
```python
import struct
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from audio.recorder import Recorder


def make_audio_frame(sample_rate=16000, duration_ms=30, is_speech=True):
    """Generate a PCM16 audio frame."""
    n_samples = int(sample_rate * duration_ms / 1000)
    if is_speech:
        # Generate a sine wave (speech-like)
        t = np.linspace(0, duration_ms / 1000, n_samples)
        samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    else:
        samples = np.zeros(n_samples, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def mock_audio():
    with patch('audio.recorder.pyaudio') as mock_pa:
        pa_instance = MagicMock()
        mock_pa.PyAudio.return_value = pa_instance
        mock_pa.paInt16 = 8  # pyaudio constant
        stream = MagicMock()
        pa_instance.open.return_value = stream
        pa_instance.get_sample_size.return_value = 2
        yield mock_pa, pa_instance, stream


def test_recorder_button_mode_returns_audio(mock_audio):
    _, pa_instance, stream = mock_audio
    frames = [make_audio_frame() for _ in range(10)]
    call_count = [0]
    def read_side_effect(*args, **kwargs):
        if call_count[0] >= len(frames):
            return frames[-1]
        result = frames[call_count[0]]
        call_count[0] += 1
        return result
    stream.read.side_effect = read_side_effect

    r = Recorder(device_index=0)
    # Simulate stopping after a few reads
    import threading
    def stop_after():
        import time
        time.sleep(0.1)
        r.stop()
    threading.Thread(target=stop_after, daemon=True).start()
    audio = r.record_button_mode()
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    r.cleanup()


def test_recorder_converts_to_float32(mock_audio):
    """Audio should be returned as float32 normalized to [-1, 1]."""
    _, pa_instance, stream = mock_audio
    # One frame of max-amplitude signal
    samples = np.array([32767, -32768, 0], dtype=np.int16)
    stream.read.return_value = samples.tobytes()

    r = Recorder(device_index=0)
    import threading
    def stop_after():
        import time
        time.sleep(0.05)
        r.stop()
    threading.Thread(target=stop_after, daemon=True).start()
    audio = r.record_button_mode()
    assert audio.dtype == np.float32
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0
    r.cleanup()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'audio.recorder'`

**Step 3: Implement Recorder**

Create `audio/recorder.py`:
```python
import logging
import threading
import numpy as np
import pyaudio
import webrtcvad

logger = logging.getLogger(__name__)

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)


class Recorder:
    def __init__(self, device_index=None):
        self._pa = pyaudio.PyAudio()
        self._device_index = device_index
        self._recording = False
        self._lock = threading.Lock()

    def _open_stream(self):
        kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        if self._device_index is not None:
            kwargs['input_device_index'] = self._device_index
        return self._pa.open(**kwargs)

    def stop(self):
        with self._lock:
            self._recording = False

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    def record_button_mode(self):
        with self._lock:
            self._recording = True
        stream = self._open_stream()
        frames = []
        try:
            while True:
                with self._lock:
                    if not self._recording:
                        break
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
        return self._to_float32(b''.join(frames))

    def record_silence_mode(self, vad_aggressiveness=1, break_length=5):
        with self._lock:
            self._recording = True
        vad = webrtcvad.Vad(vad_aggressiveness)
        num_silent_chunks = int(break_length * 1000 / CHUNK_DURATION_MS)
        stream = self._open_stream()
        frames = []
        silence_count = 0
        triggered = False
        try:
            while True:
                with self._lock:
                    if not self._recording:
                        break
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                is_speech = vad.is_speech(data, RATE)
                if is_speech:
                    if not triggered:
                        triggered = True
                    frames.append(data)
                    silence_count = 0
                elif triggered:
                    silence_count += 1
                    frames.append(data)
                    if silence_count > num_silent_chunks:
                        break
        finally:
            stream.stop_stream()
            stream.close()
        return self._to_float32(b''.join(frames))

    def _to_float32(self, raw_bytes):
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        return audio

    def cleanup(self):
        if self._pa:
            self._pa.terminate()
            self._pa = None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_recorder.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add audio/recorder.py tests/test_recorder.py
git commit -m "feat: add Recorder with button and silence modes, float32 output"
```

---

## Phase 2: Process Lock & Logging

### Task 6: Process lock file

**Files:**
- Create: `config/process_lock.py`
- Create: `tests/test_process_lock.py`

**Step 1: Write failing tests**

Create `tests/test_process_lock.py`:
```python
import os
import pytest
from config.process_lock import ProcessLock


def test_acquire_and_release(tmp_path):
    lock = ProcessLock(str(tmp_path / 'app.lock'))
    assert lock.acquire()
    assert os.path.exists(str(tmp_path / 'app.lock'))
    lock.release()
    assert not os.path.exists(str(tmp_path / 'app.lock'))


def test_double_acquire_same_process(tmp_path):
    lock = ProcessLock(str(tmp_path / 'app.lock'))
    assert lock.acquire()
    # Same process can re-acquire
    assert lock.acquire()
    lock.release()


def test_stale_lock_is_overridden(tmp_path):
    lock_file = tmp_path / 'app.lock'
    lock_file.write_text('99999999')  # PID that doesn't exist
    lock = ProcessLock(str(lock_file))
    assert lock.acquire()  # Should override stale lock
    lock.release()


def test_is_locked_by_another(tmp_path):
    lock_file = tmp_path / 'app.lock'
    # Write current PID (simulating another instance)
    lock_file.write_text(str(os.getpid()))
    lock = ProcessLock(str(lock_file))
    # Same PID, so it should allow acquisition
    assert lock.acquire()
    lock.release()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_process_lock.py -v`
Expected: FAIL

**Step 3: Implement ProcessLock**

Create `config/process_lock.py`:
```python
import logging
import os

logger = logging.getLogger(__name__)


class ProcessLock:
    def __init__(self, lock_path):
        self._lock_path = lock_path

    def acquire(self):
        if os.path.exists(self._lock_path):
            try:
                with open(self._lock_path, 'r') as f:
                    pid = int(f.read().strip())
                if pid == os.getpid():
                    return True
                # Check if PID is still running
                try:
                    os.kill(pid, 0)
                    logger.warning("Another instance is running (PID %d)", pid)
                    return False
                except OSError:
                    logger.info("Removing stale lock file (PID %d no longer running)", pid)
            except (ValueError, IOError):
                logger.info("Removing invalid lock file")

        with open(self._lock_path, 'w') as f:
            f.write(str(os.getpid()))
        return True

    def release(self):
        try:
            if os.path.exists(self._lock_path):
                with open(self._lock_path, 'r') as f:
                    pid = int(f.read().strip())
                if pid == os.getpid():
                    os.remove(self._lock_path)
        except Exception:
            logger.error("Error releasing lock", exc_info=True)

    def is_locked_by_another(self):
        if not os.path.exists(self._lock_path):
            return False
        try:
            with open(self._lock_path, 'r') as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                return False
            os.kill(pid, 0)
            return True
        except (OSError, ValueError, IOError):
            return False
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_process_lock.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add config/process_lock.py tests/test_process_lock.py
git commit -m "feat: add ProcessLock to prevent double-launch"
```

---

### Task 7: Logging setup with rotating file handler

**Files:**
- Create: `config/logging_setup.py`

**Step 1: Implement logging setup**

Create `config/logging_setup.py`:
```python
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 2


def setup_logging(settings_dir=None, level=logging.INFO):
    settings_dir = settings_dir or os.path.join(Path.home(), '.whisper2text')
    os.makedirs(settings_dir, exist_ok=True)
    log_file = os.path.join(settings_dir, 'app.log')

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console_handler)

    return log_file
```

**Step 2: Commit**

```bash
git add config/logging_setup.py
git commit -m "feat: add rotating log file setup"
```

---

## Phase 3: UI Modules

### Task 8: Error Panel widget

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/error_panel.py`

**Step 1: Implement error panel**

Create `ui/__init__.py` (empty).

Create `ui/error_panel.py`:
```python
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont


class QtLogHandler(logging.Handler):
    """Logging handler that emits to the error panel."""
    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def emit(self, record):
        try:
            msg = self.format(record)
            self._panel.append_log(msg, record.levelno)
        except Exception:
            pass


class ErrorPanel(QWidget):
    MAX_ENTRIES = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entry_count = 0
        self._collapsed = True
        self._init_ui()
        self._install_log_handler()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header bar
        header = QHBoxLayout()
        self._toggle_btn = QPushButton("Show Logs")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.clicked.connect(self._toggle)
        self._status_label = QLabel("")
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFlat(True)
        self._clear_btn.clicked.connect(self._clear)

        header.addWidget(self._toggle_btn)
        header.addWidget(self._status_label)
        header.addStretch()
        header.addWidget(self._clear_btn)
        layout.addLayout(header)

        # Log text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(200)
        self._text.setFont(QFont("monospace", 9))
        self._text.setVisible(False)
        layout.addWidget(self._text)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def _install_log_handler(self):
        handler = QtLogHandler(self)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(handler)

    @pyqtSlot()
    def _toggle(self):
        self._collapsed = not self._collapsed
        self._text.setVisible(not self._collapsed)
        self._toggle_btn.setText("Hide Logs" if not self._collapsed else "Show Logs")

    @pyqtSlot()
    def _clear(self):
        self._text.clear()
        self._entry_count = 0
        self._status_label.setText("")

    def append_log(self, message, level):
        color = {
            logging.ERROR: '#cc0000',
            logging.WARNING: '#cc7700',
            logging.INFO: '#333333',
            logging.DEBUG: '#888888',
        }.get(level, '#333333')

        self._text.append(f'<span style="color:{color}">{message}</span>')
        self._entry_count += 1

        # Trim old entries
        if self._entry_count > self.MAX_ENTRIES:
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor)
            cursor.removeSelectedText()
            self._entry_count -= 1

        if level >= logging.ERROR:
            self._status_label.setText("Error occurred - check logs")
            self._status_label.setStyleSheet("color: red;")
            # Auto-expand on error
            if self._collapsed:
                self._toggle()

        # Auto-scroll to bottom
        scrollbar = self._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
```

**Step 2: Commit**

```bash
git add ui/
git commit -m "feat: add collapsible ErrorPanel widget with log handler"
```

---

### Task 9: Settings Dialog

**Files:**
- Create: `ui/settings_dialog.py`

**Step 1: Implement settings dialog**

Create `ui/settings_dialog.py`:
```python
import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class DownloadThread(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_manager, model_name):
        super().__init__()
        self._manager = model_manager
        self._model_name = model_name

    def run(self):
        try:
            def on_progress(percent, downloaded, total):
                self.progress.emit(percent)
            self._manager.download_model(self._model_name, progress_callback=on_progress)
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))


class SettingsDialog(QDialog):
    def __init__(self, settings_manager, model_manager, device_manager, parent=None):
        super().__init__(parent)
        self._settings = settings_manager
        self._model_manager = model_manager
        self._device_manager = device_manager
        self._download_thread = None
        self.setWindowTitle('Settings')
        self.setMinimumWidth(400)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === Model Group ===
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()

        model_label = QLabel("Whisper Model:")
        self._model_combo = QComboBox()
        self._refresh_model_list()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self._model_combo)

        download_layout = QHBoxLayout()
        self._download_combo = QComboBox()
        for m in self._model_manager.list_available():
            label = f"{m['name']} ({m['size_mb']}MB)"
            self._download_combo.addItem(label, m['name'])
        download_btn = QPushButton("Download")
        download_btn.clicked.connect(self._download_model)
        delete_btn = QPushButton("Delete Selected Model")
        delete_btn.clicked.connect(self._delete_model)
        download_layout.addWidget(self._download_combo, stretch=1)
        download_layout.addWidget(download_btn)

        model_layout.addWidget(QLabel("Download Model:"))
        model_layout.addLayout(download_layout)
        model_layout.addWidget(delete_btn)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # === Audio Group ===
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout()

        device_label = QLabel("Input Device:")
        self._device_combo = QComboBox()
        self._device_combo.addItem("System Default", None)
        for dev in self._device_manager.list_input_devices():
            self._device_combo.addItem(dev['name'], dev['index'])
        saved_index = self._settings.get('audio_device_index')
        if saved_index is not None:
            for i in range(self._device_combo.count()):
                if self._device_combo.itemData(i) == saved_index:
                    self._device_combo.setCurrentIndex(i)
                    break
        audio_layout.addWidget(device_label)
        audio_layout.addWidget(self._device_combo)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # === Recording Group ===
        rec_group = QGroupBox("Recording")
        rec_layout = QVBoxLayout()

        vad_label = QLabel("VAD Aggressiveness (0-3):")
        self._vad_spin = QSpinBox()
        self._vad_spin.setRange(0, 3)
        self._vad_spin.setValue(self._settings.get('vad_aggressiveness'))

        mode_label = QLabel("Recording Mode:")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(['silence', 'button'])
        self._mode_combo.setCurrentText(self._settings.get('recording_mode'))

        break_label = QLabel("Silence Duration to Stop (seconds):")
        self._break_spin = QSpinBox()
        self._break_spin.setRange(1, 30)
        self._break_spin.setValue(self._settings.get('break_length'))

        padding_label = QLabel("Padding Duration (ms):")
        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(100, 5000)
        self._padding_spin.setValue(self._settings.get('padding_duration_ms'))

        self._auto_paste_check = QCheckBox("Automatically paste after copying")
        self._auto_paste_check.setChecked(self._settings.get('auto_paste'))

        rec_layout.addWidget(vad_label)
        rec_layout.addWidget(self._vad_spin)
        rec_layout.addWidget(mode_label)
        rec_layout.addWidget(self._mode_combo)
        rec_layout.addWidget(break_label)
        rec_layout.addWidget(self._break_spin)
        rec_layout.addWidget(padding_label)
        rec_layout.addWidget(self._padding_spin)
        rec_layout.addWidget(self._auto_paste_check)
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # === Save Button ===
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

    def _refresh_model_list(self):
        self._model_combo.clear()
        for m in self._model_manager.list_downloaded():
            self._model_combo.addItem(f"{m['name']} ({m['size_mb']}MB)", m['name'])
        saved_model = self._settings.get('model_size')
        # Try to match saved model name to a downloaded ggml file
        for i in range(self._model_combo.count()):
            item_data = self._model_combo.itemData(i)
            if item_data and saved_model in item_data:
                self._model_combo.setCurrentIndex(i)
                break

    def _download_model(self):
        model_name = self._download_combo.currentData()
        if self._model_manager.is_downloaded(model_name):
            QMessageBox.information(self, "Info", f"{model_name} is already downloaded.")
            return

        progress = QProgressDialog(f"Downloading {model_name}...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        self._download_thread = DownloadThread(self._model_manager, model_name)
        self._download_thread.progress.connect(progress.setValue)
        self._download_thread.finished_ok.connect(lambda: self._on_download_done(progress, model_name))
        self._download_thread.error.connect(lambda e: self._on_download_error(progress, e))
        progress.canceled.connect(self._download_thread.terminate)
        self._download_thread.start()

    def _on_download_done(self, progress, model_name):
        progress.close()
        self._refresh_model_list()
        QMessageBox.information(self, "Success", f"{model_name} downloaded successfully.")

    def _on_download_error(self, progress, error_msg):
        progress.close()
        QMessageBox.critical(self, "Error", f"Download failed: {error_msg}")

    def _delete_model(self):
        model_name = self._model_combo.currentData()
        if not model_name:
            return
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete {model_name}? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._model_manager.delete_model(model_name)
            self._refresh_model_list()

    def _save(self):
        model_name = self._model_combo.currentData()
        device_index = self._device_combo.currentData()
        device_name = self._device_combo.currentText()

        self._settings.set('model_size', model_name or 'ggml-base.bin')
        self._settings.set('audio_device_index', device_index)
        self._settings.set('audio_device_name', device_name)
        self._settings.set('vad_aggressiveness', self._vad_spin.value())
        self._settings.set('recording_mode', self._mode_combo.currentText())
        self._settings.set('break_length', self._break_spin.value())
        self._settings.set('padding_duration_ms', self._padding_spin.value())
        self._settings.set('auto_paste', self._auto_paste_check.isChecked())
        self._settings.save()

        self.accept()

    def get_model_changed(self):
        """Returns the new model name if it changed from what was saved."""
        current = self._model_combo.currentData()
        saved = self._settings.get('model_size')
        if current and current != saved:
            return current
        return None
```

**Step 2: Commit**

```bash
git add ui/settings_dialog.py
git commit -m "feat: add SettingsDialog with model download, device selector, recording options"
```

---

### Task 10: Main Window (full refactor)

**Files:**
- Create: `ui/main_window.py`

**Step 1: Implement main window**

Create `ui/main_window.py`:
```python
import atexit
import logging
import os
import platform
import signal
import sys
import threading
import time

import numpy as np
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QSizePolicy, QStyle, QSystemTrayIcon,
    QMenu, QAction, QMessageBox, QLabel, QStatusBar
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSignal

from config.settings import SettingsManager
from config.process_lock import ProcessLock
from engine.whisper_engine import WhisperEngine
from engine.model_manager import ModelManager
from audio.recorder import Recorder
from audio.device_manager import DeviceManager
from ui.error_panel import ErrorPanel
from ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)

ICON_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_NORMAL = os.path.join(ICON_BASE, 'icon.png')
ICON_RECORDING = os.path.join(ICON_BASE, 'icon_recording.png')
MODELS_DIR = os.path.join(ICON_BASE, 'models')


class MainWindow(QWidget):
    hotkey_signal = pyqtSignal()
    transcript_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    recording_stopped_signal = pyqtSignal()
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech to Text")
        self.setWindowIcon(QIcon(ICON_NORMAL))
        self.resize(400, 600)
        self._is_quitting = False

        # Core components
        self._settings = SettingsManager()
        self._model_manager = ModelManager(MODELS_DIR)
        self._device_manager = DeviceManager()
        self._keyboard_controller = Controller()

        # Load engine
        self._engine = None
        self._load_engine()

        # Recorder
        device_index = self._settings.get('audio_device_index')
        self._recorder = Recorder(device_index=device_index)

        # State
        self._transcripts = self._settings.get('transcripts', [])
        self._max_transcripts = 10
        self._last_clicked_button = None
        self._recording_lock = threading.Lock()

        # Connect signals
        self.transcript_signal.connect(self._add_transcript)
        self.error_signal.connect(self._show_error)
        self.recording_stopped_signal.connect(self._on_recording_stopped)
        self.hotkey_signal.connect(self._toggle_recording)
        self.update_status_signal.connect(self._update_status_text)

        # Build UI
        self._init_ui()

        # System tray
        self._setup_tray()

        # Global hotkey
        threading.Thread(target=self._start_hotkey_listener, daemon=True).start()

        # Cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        sys.excepthook = self._handle_exception

    def _load_engine(self):
        model_name = self._settings.get('model_size', 'ggml-base.bin')
        # Handle old-style model names (e.g., 'base' -> 'ggml-base.bin')
        if not model_name.endswith('.bin'):
            model_name = f'ggml-{model_name}.bin'
        try:
            model_path = self._model_manager.get_model_path(model_name)
            self._engine = WhisperEngine(model_path=model_path)
            logger.info("Engine loaded with model: %s", model_name)
        except FileNotFoundError:
            logger.error("Model %s not found. Available: %s",
                        model_name, [m['name'] for m in self._model_manager.list_downloaded()])
            # Try to load any available model
            downloaded = self._model_manager.list_downloaded()
            if downloaded:
                fallback = downloaded[0]['path']
                logger.info("Falling back to %s", fallback)
                self._engine = WhisperEngine(model_path=fallback)
                self._settings.set('model_size', downloaded[0]['name'])
                self._settings.save()
            else:
                logger.error("No models available!")
                self._engine = None
        except Exception:
            logger.error("Failed to load engine", exc_info=True)
            self._engine = None

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top bar with settings
        top_bar = QHBoxLayout()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(spacer)
        settings_btn = QPushButton()
        settings_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        settings_btn.setFlat(True)
        settings_btn.setToolTip('Settings')
        settings_btn.clicked.connect(self._open_settings)
        top_bar.addWidget(settings_btn)
        layout.addLayout(top_bar)

        # Transcript scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._transcript_widget = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript_widget)
        self._transcript_layout.setAlignment(Qt.AlignTop)
        self._scroll_area.setWidget(self._transcript_widget)
        self._scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._scroll_area)

        # Record button
        self._record_btn = QPushButton('Record')
        self._record_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self._record_btn)

        # Status bar
        self._status_label = QLabel()
        self._update_status()
        layout.addWidget(self._status_label)

        # Error panel
        self._error_panel = ErrorPanel()
        layout.addWidget(self._error_panel)

        # Load saved transcripts
        for text in self._transcripts[-self._max_transcripts:]:
            self._create_transcript_button(text)

    def _update_status(self):
        model = self._settings.get('model_size', '?')
        device = self._settings.get('audio_device_name', 'System Default')
        gpu = "GPU" if self._engine and self._engine.is_loaded() else "CPU"
        self._status_label.setText(f"Model: {model} | {gpu} | Mic: {device}")
        self._status_label.setStyleSheet("color: gray; font-size: 10px;")

    def _update_status_text(self, text):
        self._status_label.setText(text)

    def _setup_tray(self):
        self._tray = QSystemTrayIcon(self)
        icon = QIcon(ICON_NORMAL) if os.path.exists(ICON_NORMAL) else self.style().standardIcon(QStyle.SP_MediaPlay)
        self._tray.setIcon(icon)
        self._tray.setToolTip("Speech to Text")

        menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        hide_action = QAction("Hide", self)
        hide_action.triggered.connect(self.hide)
        self._record_action = QAction("Start Recording", self)
        self._record_action.triggered.connect(self._toggle_recording)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._open_settings)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)

        menu.addAction(show_action)
        menu.addAction(hide_action)
        menu.addSeparator()
        menu.addAction(self._record_action)
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.show()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.show() if not self.isVisible() else self.hide()

    def _toggle_recording(self):
        with self._recording_lock:
            if self._recorder.is_recording:
                self._recorder.stop()
                return

        if self._engine is None or not self._engine.is_loaded():
            self.error_signal.emit("No model loaded. Open Settings to select a model.")
            return

        self._record_btn.setText('Stop Recording')
        self._record_action.setText("Stop Recording")
        self._update_tray_icon(recording=True)

        mode = self._settings.get('recording_mode')
        if mode == 'button':
            threading.Thread(target=self._record_button, daemon=True).start()
        else:
            self._record_btn.setEnabled(False)
            threading.Thread(target=self._record_silence, daemon=True).start()

    def _record_button(self):
        try:
            audio = self._recorder.record_button_mode()
            if audio is not None and len(audio) > 0:
                text = self._engine.transcribe(audio)
                if text and text.strip():
                    self.transcript_signal.emit(text)
        except Exception as e:
            logger.error("Button mode recording error", exc_info=True)
            self.error_signal.emit(str(e))
        finally:
            self.recording_stopped_signal.emit()

    def _record_silence(self):
        try:
            vad = self._settings.get('vad_aggressiveness')
            brk = self._settings.get('break_length')
            audio = self._recorder.record_silence_mode(vad_aggressiveness=vad, break_length=brk)
            if audio is not None and len(audio) > 0:
                text = self._engine.transcribe(audio)
                if text and text.strip():
                    self.transcript_signal.emit(text)
        except Exception as e:
            logger.error("Silence mode recording error", exc_info=True)
            self.error_signal.emit(str(e))
        finally:
            self.recording_stopped_signal.emit()

    def _add_transcript(self, text):
        if not text.strip():
            return
        self._transcripts.append(text)
        self._create_transcript_button(text)

        while self._transcript_layout.count() > self._max_transcripts:
            item = self._transcript_layout.takeAt(0)
            if item and item.widget():
                if item.widget() == self._last_clicked_button:
                    self._last_clicked_button = None
                item.widget().deleteLater()

        try:
            pyperclip.copy(text)
            if self._settings.get('auto_paste'):
                self._paste_text()
        except Exception:
            logger.error("Failed to copy/paste", exc_info=True)

        self._settings.set('transcripts', self._transcripts)
        self._settings.save()

    def _create_transcript_button(self, text):
        btn = QPushButton(text[:50] + '...' if len(text) > 50 else text)
        btn.setToolTip(text)
        btn.setStyleSheet(self._button_style())
        btn.clicked.connect(lambda: self._on_transcript_click(btn, text))
        self._transcript_layout.addWidget(btn)
        self._highlight_button(btn)

    def _on_transcript_click(self, button, text):
        try:
            pyperclip.copy(text)
            if self._settings.get('auto_paste'):
                self._paste_text()
            self._highlight_button(button)
        except Exception:
            logger.error("Failed to copy/paste", exc_info=True)

    def _paste_text(self):
        try:
            time.sleep(0.1)
            if platform.system() == 'Darwin':
                with self._keyboard_controller.pressed(Key.cmd):
                    self._keyboard_controller.press('v')
                    self._keyboard_controller.release('v')
            else:
                with self._keyboard_controller.pressed(Key.ctrl):
                    self._keyboard_controller.press('v')
                    self._keyboard_controller.release('v')
        except Exception:
            logger.error("Failed to paste", exc_info=True)

    def _highlight_button(self, button):
        if self._last_clicked_button:
            self._last_clicked_button.setStyleSheet(self._button_style(selected=False))
        button.setStyleSheet(self._button_style(selected=True))
        self._last_clicked_button = button

    def _button_style(self, selected=False):
        base = """
        QPushButton {
            border: 1px solid blue;
            border-radius: 5px;
            text-align: left;
            padding: 5px;
            background-color: white;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        """
        if selected:
            return base + "QPushButton { background-color: rgba(0, 0, 255, 50); }"
        return base

    def _on_recording_stopped(self):
        self._record_btn.setText('Record')
        self._record_btn.setEnabled(True)
        self._record_action.setText("Start Recording")
        self._update_tray_icon(recording=False)

    def _update_tray_icon(self, recording=False):
        if recording:
            icon_path = ICON_RECORDING if os.path.exists(ICON_RECORDING) else None
        else:
            icon_path = ICON_NORMAL if os.path.exists(ICON_NORMAL) else None
        icon = QIcon(icon_path) if icon_path else self.style().standardIcon(
            QStyle.SP_MediaStop if recording else QStyle.SP_MediaPlay
        )
        self._tray.setIcon(icon)
        self.setWindowIcon(icon)

    def _open_settings(self):
        dialog = SettingsDialog(self._settings, self._model_manager, self._device_manager, self)
        old_model = self._settings.get('model_size')
        if dialog.exec_():
            new_model = self._settings.get('model_size')
            if new_model != old_model and new_model:
                # Reload engine with new model
                self._record_btn.setEnabled(False)
                self._record_btn.setText("Loading model...")
                threading.Thread(target=self._reload_engine, args=(new_model,), daemon=True).start()

            # Update recorder with new device
            device_index = self._settings.get('audio_device_index')
            self._recorder.cleanup()
            self._recorder = Recorder(device_index=device_index)
            self._update_status()

    def _reload_engine(self, model_name):
        try:
            if not model_name.endswith('.bin'):
                model_name = f'ggml-{model_name}.bin'
            model_path = self._model_manager.get_model_path(model_name)
            if self._engine:
                self._engine.reload(model_path)
            else:
                self._engine = WhisperEngine(model_path=model_path)
            logger.info("Model reloaded: %s", model_name)
            self.update_status_signal.emit(f"Model loaded: {model_name}")
        except Exception as e:
            logger.error("Failed to reload model", exc_info=True)
            self.error_signal.emit(f"Failed to load model: {e}")
        finally:
            self.recording_stopped_signal.emit()

    def _show_error(self, message):
        logger.error("UI error: %s", message)

    def _start_hotkey_listener(self):
        ctrl_pressed = False
        alt_pressed = False
        shift_pressed = False

        def on_press(key):
            nonlocal ctrl_pressed, alt_pressed, shift_pressed
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    ctrl_pressed = True
                elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    alt_pressed = True
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    shift_pressed = True
                elif hasattr(key, 'char') and key.char and key.char.lower() == 'l':
                    if ctrl_pressed and alt_pressed and shift_pressed:
                        self.hotkey_signal.emit()
            except Exception:
                logger.error("Hotkey on_press error", exc_info=True)

        def on_release(key):
            nonlocal ctrl_pressed, alt_pressed, shift_pressed
            try:
                if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    ctrl_pressed = False
                elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
                    alt_pressed = False
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    shift_pressed = False
            except Exception:
                logger.error("Hotkey on_release error", exc_info=True)

        try:
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            self._keyboard_listener = listener
        except Exception:
            logger.error("Could not start hotkey listener", exc_info=True)

    def _handle_exception(self, exctype, value, tb):
        logger.error("Uncaught exception", exc_info=(exctype, value, tb))
        self.error_signal.emit(f"Unexpected error: {value}")

    def _signal_handler(self, signum, frame):
        logger.info("Signal %d received, shutting down", signum)
        self._quit()

    def _cleanup(self):
        if self._engine:
            self._engine.unload()
        if self._recorder:
            self._recorder.cleanup()
        if self._device_manager:
            self._device_manager.cleanup()

    def _quit(self):
        self._is_quitting = True
        self._tray.hide()
        self._cleanup()
        try:
            self._keyboard_listener.stop()
        except Exception:
            pass
        self.close()
        QApplication.instance().quit()

    def closeEvent(self, event):
        self._is_quitting = True
        self._tray.hide()
        self._cleanup()
        try:
            self._keyboard_listener.stop()
        except Exception:
            pass
        event.accept()
```

**Step 2: Commit**

```bash
git add ui/main_window.py
git commit -m "feat: add MainWindow with all UI components integrated"
```

---

### Task 11: Refactor entry point (whisper2text.py)

**Files:**
- Modify: `whisper2text.py`

**Step 1: Rewrite entry point**

Replace the entire contents of `whisper2text.py` with:
```python
#!/usr/bin/env python3

import os
os.environ['ALSA_DEBUG'] = '0'

import sys
import signal
from pathlib import Path

from config.logging_setup import setup_logging
from config.process_lock import ProcessLock

SETTINGS_DIR = os.path.join(Path.home(), '.whisper2text')
LOCK_FILE = os.path.join(SETTINGS_DIR, 'app.lock')


def main():
    # Set up logging first
    setup_logging()

    # Check for existing instance
    lock = ProcessLock(LOCK_FILE)
    if not lock.acquire():
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.warning(None, "Already Running",
                          "Another instance of Speech to Text is already running.")
        sys.exit(1)

    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QIcon
        from ui.main_window import MainWindow, ICON_NORMAL

        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon(ICON_NORMAL))

        window = MainWindow()
        window.show()

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(app.exec_())
    finally:
        lock.release()


if __name__ == '__main__':
    main()
```

**Step 2: Test launch**

Run: `cd ~/Development/whisperLocal && source venv/bin/activate && python whisper2text.py`
Expected: App launches with the new modular backend. Check:
- Window appears with transcript list and record button
- Status bar shows model name and audio device
- Error panel is visible at bottom (collapsed)
- Settings dialog opens and shows model list, device list

**Step 3: Test recording with Rode mic**

In the app:
1. Open Settings, select "Rode Wireless MICRO" from audio device dropdown
2. Save settings
3. Click Record, speak, click Stop
4. Verify transcription appears

**Step 4: Test hotkey (Ctrl+Alt+Shift+L)**

Press Ctrl+Alt+Shift+L, speak, and verify recording starts.

**Step 5: Commit**

```bash
git add whisper2text.py
git commit -m "feat: refactor entry point to use modular architecture"
```

---

## Phase 4: Cleanup & Final Testing

### Task 12: Migrate existing user settings

**Files:**
- Modify: `config/settings.py`

**Step 1: Add migration for old settings format**

The old app stored `model_size` as just `'base'`. The new app uses `'ggml-base.bin'`. Add migration in `SettingsManager._load_settings()`:

After loading saved settings, add:
```python
# Migrate old model_size format
model_size = self._settings.get('model_size', '')
if model_size and not model_size.endswith('.bin'):
    self._settings['model_size'] = f'ggml-{model_size}.bin'
```

**Step 2: Test that old settings.json is loaded correctly**

Run existing test: `python -m pytest tests/test_settings.py::test_backward_compatible_with_existing_settings -v`
Expected: PASS (the test already covers this)

**Step 3: Commit**

```bash
git add config/settings.py
git commit -m "fix: migrate old model_size format to ggml-*.bin"
```

---

### Task 13: Download additional models

**Step 1: Download models the user wants**

Run these in the app's model download dialog, or manually:
```bash
cd ~/Development/whisperLocal
# Medium quantized (good speed/accuracy balance for 6GB VRAM)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin -P models/

# Large V3 Turbo quantized (best accuracy that fits 6GB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin -P models/
```

**Step 2: Test each model in the app**

For each downloaded model:
1. Open Settings, select the model
2. Save (observe model loading in status bar)
3. Record a test phrase
4. Verify transcription quality and speed

---

### Task 14: Full integration test

**Step 1: Clean start test**

```bash
# Remove settings to test fresh start
mv ~/.whisper2text/settings.json ~/.whisper2text/settings.json.bak

# Launch app
cd ~/Development/whisperLocal && source venv/bin/activate && python whisper2text.py
```

Expected: App starts with defaults (base model, system default mic)

**Step 2: Settings persistence test**

1. Change model to small, select Rode mic, enable auto-paste
2. Save settings
3. Close app
4. Reopen app
5. Verify all settings are preserved

**Step 3: Error recovery test**

1. Start recording
2. Unplug the Rode mic while recording
3. Verify app shows error in error panel but doesn't crash
4. Plug mic back in
5. Verify recording works again

**Step 4: Lock file test**

1. Start the app
2. Try to start another instance
3. Verify "Already Running" dialog appears

**Step 5: Restore backup settings if needed**

```bash
mv ~/.whisper2text/settings.json.bak ~/.whisper2text/settings.json
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "chore: final integration verification"
```

---

## Summary of Files

| File | Action | Description |
|------|--------|-------------|
| `whisper2text.py` | Modify | Slim entry point with lock file |
| `config/__init__.py` | Create | Package init |
| `config/settings.py` | Create | SettingsManager with defaults and migration |
| `config/process_lock.py` | Create | PID-based lock file |
| `config/logging_setup.py` | Create | Rotating log file setup |
| `engine/__init__.py` | Create | Package init |
| `engine/whisper_engine.py` | Create | pywhispercpp wrapper with CUDA |
| `engine/model_manager.py` | Create | Model download/list/delete |
| `audio/__init__.py` | Create | Package init |
| `audio/recorder.py` | Create | Button and silence mode recording |
| `audio/device_manager.py` | Create | Audio device listing and selection |
| `ui/__init__.py` | Create | Package init |
| `ui/main_window.py` | Create | Full PyQt5 main window |
| `ui/settings_dialog.py` | Create | Settings with model/device management |
| `ui/error_panel.py` | Create | Collapsible log panel |
| `requirements.txt` | Modify | Switch to pywhispercpp deps |
| `.gitignore` | Modify | Add models/*.bin |
| `tests/__init__.py` | Create | Test package init |
| `tests/test_settings.py` | Create | Settings unit tests |
| `tests/test_engine.py` | Create | Engine unit tests |
| `tests/test_model_manager.py` | Create | Model manager unit tests |
| `tests/test_device_manager.py` | Create | Device manager unit tests |
| `tests/test_recorder.py` | Create | Recorder unit tests |
