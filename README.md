# whisperLocal

Local speech-to-text using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with CUDA GPU acceleration. PyQt5 desktop app with system tray, global hotkeys, and auto-paste.

Built for Linux with NVIDIA GPUs. No cloud APIs, everything runs locally.

## Features

- **Fast local transcription** via whisper.cpp (CUDA-accelerated)
- **Voice Activity Detection**: auto-stops recording after silence (WebRTC VAD)
- **Two recording modes**: silence detection or manual button/hotkey
- **Global hotkey**: `Ctrl+Alt+Shift+L` (configurable, works on Wayland via evdev)
- **System tray**: blue icon idle, red when recording
- **Auto-paste**: optionally types transcribed text into the focused window (via ydotool)
- **Model management**: download/delete ggml models from the Settings dialog
- **Crash recovery**: systemd service auto-restarts on failure
- **Persistent history**: last 10 transcriptions saved across sessions

## Requirements

- Linux (tested on Ubuntu 24.04)
- Python 3.10+
- NVIDIA GPU with CUDA support (falls back to CPU)
- CUDA toolkit installed
- PortAudio development libraries

## Installation

### 1. System dependencies

```bash
sudo apt install portaudio19-dev python3-pyaudio python3-pyqt5 ydotool
```

For global hotkeys, your user needs to be in the `input` group:

```bash
sudo usermod -aG input $USER
# Log out and back in
```

### 2. Clone and set up

```bash
git clone https://github.com/ac2522/whisperLocal.git
cd whisperLocal
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
# With CUDA GPU support (recommended):
GGML_CUDA=1 pip install pywhispercpp --no-binary pywhispercpp --no-cache-dir

# Install the rest:
pip install -r requirements.txt
```

### 4. Run from source (optional, to verify it works)

```bash
python3 whisper2text.py
```

On first run, open Settings and download a model (start with `ggml-base.bin`).

### 5. Install as desktop app

```bash
./install.sh
```

This builds a PyInstaller binary locally (matching your GPU drivers), installs it to `~/.local/share/whisperLocal/`, and sets up a systemd user service with auto-restart.

## Managing the service

```bash
systemctl --user status whisper2text      # Check status
systemctl --user restart whisper2text     # Restart
systemctl --user stop whisper2text        # Stop
journalctl --user -u whisper2text -f      # View logs
```

The service auto-starts on login and restarts on crash (up to 5 times per minute).

## Uninstalling

```bash
./uninstall.sh
```

This removes the binary, desktop entry, and systemd service. Your settings and models in `~/.whisper2text/` are preserved. To remove everything:

```bash
rm -rf ~/.whisper2text
```

## Usage

1. **Start recording**: click Record, press `Ctrl+Alt+Shift+L`, or use the tray menu
2. **Stop recording**: in silence mode, wait for silence; in button mode, click/press hotkey again
3. **Copy text**: click any transcription button to copy to clipboard
4. **Auto-paste**: enable in Settings to auto-type text into the focused window

## Settings

Access via the gear icon or tray menu:

- **Model**: select from downloaded ggml models (download more in Settings)
- **Compute backend**: CUDA, Vulkan, or CPU
- **Audio device**: select input microphone
- **Recording mode**: silence detection or manual button
- **VAD aggressiveness** (0-3): higher = more aggressive noise filtering
- **Silence duration**: seconds of silence before auto-stop
- **Hotkey**: configurable global keyboard shortcut
- **Auto-paste**: auto-type transcriptions via ydotool

## Models

Models are downloaded through the app's Settings dialog. Available models:

| Model | Size | Notes |
|-------|------|-------|
| ggml-base.bin | ~142 MB | Good starting point |
| ggml-small.bin | ~466 MB | Better accuracy |
| ggml-medium.bin | ~1.5 GB | High accuracy |
| ggml-large-v3.bin | ~3 GB | Best accuracy |
| ggml-large-v3-turbo.bin | ~1.5 GB | Fast + accurate |
| ggml-medium-q5_0.bin | ~500 MB | Quantized, nearly same accuracy |
| ggml-large-v3-turbo-q5_0.bin | ~500 MB | Quantized turbo |

Models are stored in `~/.whisper2text/models/` and are not included in the repo.

## Architecture

```
whisperLocal/
├── whisper2text.py          # Entry point
├── engine/
│   ├── whisper_engine.py    # pywhispercpp wrapper
│   └── model_manager.py     # Model download/management
├── audio/
│   ├── recorder.py          # Recording (button/silence modes)
│   └── device_manager.py    # Audio device selection
├── ui/
│   ├── main_window.py       # Main PyQt5 window + tray
│   ├── settings_dialog.py   # Settings with model/device selectors
│   └── error_panel.py       # Error/log display
├── config/
│   ├── settings.py          # Settings persistence
│   ├── paths.py             # Path resolution (PyInstaller-aware)
│   ├── logging_setup.py     # Logging configuration
│   └── process_lock.py      # Single-instance lock
├── packaging/
│   ├── whisper2text.desktop  # GNOME launcher
│   ├── whisper2text.service  # systemd user service
│   └── rthook_cuda.py       # PyInstaller CUDA runtime hook
├── install.sh               # Build + install
├── uninstall.sh              # Clean removal
└── whisper2text.spec         # PyInstaller build spec
```

## Deploying to another machine

No pre-built releases (CUDA libs are ~1 GB and driver-specific). Build locally on each machine:

```bash
git clone https://github.com/ac2522/whisperLocal.git
cd whisperLocal
python3 -m venv venv
source venv/bin/activate
GGML_CUDA=1 pip install pywhispercpp --no-binary pywhispercpp --no-cache-dir
pip install -r requirements.txt
./install.sh
```

## Troubleshooting

**No sound input**: ensure your user is in the `audio` group (`sudo usermod -aG audio $USER`)

**Global hotkey not working**: ensure your user is in the `input` group and ydotool is installed

**CUDA not detected**: verify `nvidia-smi` works, CUDA toolkit is installed, and pywhispercpp was built with `GGML_CUDA=1`

**PyAudio install fails**: install `portaudio19-dev` (`sudo apt install portaudio19-dev`)

## License

MIT
