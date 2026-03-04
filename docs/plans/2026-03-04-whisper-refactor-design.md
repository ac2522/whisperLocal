# Whisper2Text Performance Refactor Design

## Problem

The current app uses OpenAI's PyTorch-based `whisper` package, which is the slowest inference option. On a laptop with an RTX A1000 (6GB VRAM), transcription is too slow. The app also crashes silently, making debugging difficult.

## Solution

Swap the backend to `whisper.cpp` via `pywhispercpp` (built with CUDA support), split the codebase into modules, and add error resilience + UI improvements.

## System Constraints

- Intel i7-13800H (14 cores, 20 threads)
- 64GB RAM
- NVIDIA RTX A1000 6GB VRAM, CUDA 13.0
- Rode "Wireless MICRO" USB mic (card 2)

## Architecture

```
whisperLocal/
├── whisper2text.py          # Main entry point
├── engine/
│   ├── __init__.py
│   ├── whisper_engine.py    # pywhispercpp wrapper with CUDA
│   └── model_manager.py     # Download, list, delete ggml models + VRAM cleanup
├── audio/
│   ├── __init__.py
│   ├── recorder.py          # PyAudio recording (button/silence modes)
│   └── device_manager.py    # List & select audio devices
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # Main PyQt5 window
│   ├── settings_dialog.py   # Settings with model/device selectors
│   └── error_panel.py       # Collapsible error/log panel
├── config/
│   ├── __init__.py
│   └── settings.py          # SettingsManager singleton
├── models/                  # Downloaded ggml model files
├── icon.png
├── icon_recording.png
└── requirements.txt
```

## Engine Design

### Whisper Engine (`engine/whisper_engine.py`)
- Wraps `pywhispercpp` with CUDA support (built with `GGML_CUDA=1`)
- Loads model on startup, keeps in memory
- `transcribe(audio_data: bytes) -> str` - direct in-memory transcription, no temp WAV files
- Explicit `unload()` method that frees model from VRAM/RAM
- Context manager support for guaranteed cleanup
- On model change: unload old model first, then load new
- `atexit` + signal handlers for crash cleanup

### Model Manager (`engine/model_manager.py`)
- Downloads ggml models from HuggingFace (ggerganov/whisper.cpp repo)
- Available models:
  - `ggml-base.bin` (~142MB)
  - `ggml-small.bin` (~466MB)
  - `ggml-medium.bin` (~1.5GB)
  - `ggml-large-v3.bin` (~3GB)
  - `ggml-large-v3-turbo.bin` (~1.5GB)
  - `ggml-medium-q5_0.bin` (~500MB) - quantized
  - `ggml-large-v3-turbo-q5_0.bin` (~500MB) - quantized
- Progress callback for download UI
- Delete model function with VRAM cleanup

### Memory Safety
- Single model loaded at a time
- Process lock file (`~/.whisper2text/app.lock`) prevents double-launch
- Startup check: if lock exists and PID is alive, show "already running" dialog

## Audio Design

### Device Manager (`audio/device_manager.py`)
- Lists available input devices via PyAudio
- Persists selected device in settings
- Falls back to system default if saved device not found

### Recorder (`audio/recorder.py`)
- Two modes: "silence" (VAD auto-stop) and "button" (manual stop)
- webrtcvad for voice activity detection
- Records to in-memory buffer (no temp WAV files)
- Passes raw PCM audio to whisper engine

## UI Design

### Main Window
- Same layout: transcript list + record button + settings gear
- Added: collapsible error panel at bottom
- Added: status bar (model name, GPU/CPU mode, audio device)
- Added: model loading progress indicator
- Transcript history persisted across sessions
- System tray with same functionality

### Settings Dialog
- All existing settings preserved
- Added: audio device dropdown
- Added: model selector with download/delete
- Added: model download dialog with progress

### Error Panel
- Collapsible at bottom of main window
- Log entries with severity + timestamps
- Clear button
- Rotating log file at `~/.whisper2text/app.log` (5MB max)

## Error Handling

- No silent crashes: exceptions caught at top level, shown in error panel
- Model cleanup on loading failure (VRAM freed)
- Graceful CUDA fallback to CPU with notification
- Lock file prevents double-launch
- SIGTERM/SIGINT handlers for orderly shutdown
- atexit handler as final safety net

## Models Download Guide

### Automatic (via app)
Models can be downloaded through the Settings > Download Models dialog.

### Manual Download
Models are hosted at: `https://huggingface.co/ggerganov/whisper.cpp/tree/main`

Download any ggml model file and place it in the `models/` directory:

```bash
# Standard models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -P models/
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin -P models/
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin -P models/
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin -P models/

# Turbo model
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin -P models/

# Quantized models (smaller, nearly same accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin -P models/
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin -P models/
```

For models requiring HuggingFace credentials:
```bash
# Install huggingface-cli
pip install huggingface-hub

# Login (one-time)
huggingface-cli login

# Then download as above, or use:
huggingface-cli download ggerganov/whisper.cpp ggml-large-v3.bin --local-dir models/
```

## Installation Guide

```bash
# 1. Clone/navigate to the repo
cd ~/Development/whisperLocal

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install system dependencies (Ubuntu)
sudo apt install portaudio19-dev python3-pyaudio python3-pyqt5

# 4. Install pywhispercpp with CUDA support
GGML_CUDA=1 pip install pywhispercpp

# 5. Install remaining Python dependencies
pip install -r requirements.txt

# 6. Run the app
python whisper2text.py
```

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (falls back to CPU)
- CUDA toolkit installed
- PortAudio development libraries
- PyQt5

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference backend | whisper.cpp (pywhispercpp) | Proven fast on this hardware via Vocalinux, lowest VRAM usage |
| UI framework | PyQt5 (keep) | User loves the existing UI |
| Architecture | Modular split | Better debugging, easier to add error panel and model management |
| Models | All sizes except tiny + turbo + quantized | Flexibility to find the right speed/accuracy balance |
| Error handling | In-app panel + log file | Visible errors, no silent crashes |
| Audio device | Selectable in settings | Support both Rode mic and laptop mic |
