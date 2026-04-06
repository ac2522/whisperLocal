# whisperLocal

Local speech-to-text powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (GGML). A PyQt5 desktop app for Linux with system tray integration, global hotkeys, and auto-paste.

Everything runs locally on your machine. No cloud APIs, no data leaves your computer. Supports NVIDIA CUDA, Vulkan, and CPU backends with automatic detection and fallback.

## Features

- **GPU-accelerated transcription** via whisper.cpp with CUDA (NVIDIA), Vulkan, or CPU fallback
- **Voice Activity Detection** (WebRTC VAD) auto-stops recording after configurable silence
- **Two recording modes**: silence detection or manual button/hotkey
- **Global hotkey**: `Ctrl+Alt+Shift+L` (configurable, works on Wayland via evdev)
- **System tray**: icon changes color when recording
- **Auto-paste**: optionally types transcribed text into the focused window (via ydotool)
- **Model management**: download, select, and delete GGML models from the Settings dialog
- **Crash recovery**: systemd service auto-restarts on failure
- **Persistent history**: last 10 transcriptions saved across sessions
- **Single-instance lock**: prevents duplicate launches

## Quick Start

```bash
git clone https://github.com/ac2522/whisperLocal.git
cd whisperLocal
./install.sh
```

The installer handles everything:
- Installs system dependencies (PortAudio, ydotool)
- Detects your GPU and offers to install the CUDA toolkit if needed
- Builds pywhispercpp with GPU support (or CPU fallback)
- Creates a Python venv and installs all dependencies
- Builds a standalone binary via PyInstaller
- Installs a desktop entry ("whisperLocal" in your app launcher)
- Sets up a systemd user service with auto-start and crash recovery
- Adds your user to the `input` group for global hotkeys (requires logout/login)

After install, open the app and go to **Settings** to download a model (start with `ggml-base.bin`).

## Requirements

- **Linux** (tested on Ubuntu 22.04 and 24.04)
- **Python 3.10+**
- **PortAudio** development libraries (`portaudio19-dev`)
- **NVIDIA GPU** with CUDA support (optional, falls back to CPU)
- **CUDA toolkit** 12.x or newer (optional, installer can set this up)

## Manual Installation

If you prefer to set things up yourself instead of using `install.sh`:

### 1. System dependencies

```bash
sudo apt install portaudio19-dev ydotool
```

For global hotkeys (evdev), add your user to the `input` group:

```bash
sudo usermod -aG input $USER
# Log out and back in for this to take effect
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
# With CUDA GPU support (recommended if you have an NVIDIA GPU):
GGML_CUDA=1 pip install pywhispercpp --no-binary pywhispercpp --no-cache-dir

# Or with Vulkan GPU support:
# GGML_VULKAN=1 pip install pywhispercpp --no-binary pywhispercpp --no-cache-dir

# Or CPU only:
# pip install pywhispercpp

# Install the rest:
pip install -r requirements.txt
```

**Note:** CUDA builds require `nvcc` 12.x or newer. If you have an older version from Ubuntu's repos (`nvidia-cuda-toolkit`), install a newer one from NVIDIA:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8   # or newer
```

### 4. Run from source

```bash
python3 whisper2text.py
```

On first run, open **Settings** and download a model.

### 5. Install as desktop app (optional)

```bash
./install.sh
```

## Managing the Service

```bash
systemctl --user status whisper2text      # Check status
systemctl --user restart whisper2text     # Restart
systemctl --user stop whisper2text        # Stop
journalctl --user -u whisper2text -f      # View logs
```

The service auto-starts on login and restarts on crash (up to 5 times per minute).

You can also run `whisperlocal` from the terminal after installation.

## Uninstalling

```bash
./uninstall.sh
```

This removes the binary, desktop entry, and systemd service. Your settings and models in `~/.whisper2text/` are preserved. To remove everything:

```bash
rm -rf ~/.whisper2text
```

## Usage

1. **Start recording**: click the Record button, press `Ctrl+Alt+Shift+L`, or use the tray menu
2. **Stop recording**: in silence mode, stop talking and wait; in button mode, click/press hotkey again
3. **Copy text**: click any transcription in the history to copy to clipboard
4. **Auto-paste**: enable in Settings to auto-type text into the focused window after transcription

## Settings

Access via the Settings button or tray menu:

| Setting | Description |
|---------|-------------|
| **Model** | Select from downloaded GGML models (download more in the dialog) |
| **Compute backend** | Auto (recommended), CUDA, Vulkan, or CPU |
| **Audio device** | Select input microphone |
| **Recording mode** | Silence detection (auto-stop) or manual button |
| **VAD aggressiveness** | 0-3, higher = more aggressive noise filtering |
| **Silence duration** | Seconds of silence before auto-stop (silence mode) |
| **Hotkey** | Global keyboard shortcut (e.g. `Ctrl+Alt+Shift+L`) |
| **Auto-paste** | Auto-type transcriptions into focused window via ydotool |

## Models

Models are downloaded through the Settings dialog. Available models:

| Model | Size | Notes |
|-------|------|-------|
| `ggml-base.bin` | ~142 MB | Good starting point, fast |
| `ggml-small.bin` | ~466 MB | Better accuracy |
| `ggml-medium.bin` | ~1.5 GB | High accuracy |
| `ggml-large-v3.bin` | ~3 GB | Best accuracy |
| `ggml-large-v3-turbo.bin` | ~1.5 GB | Fast + accurate (recommended with GPU) |
| `ggml-medium-q5_0.bin` | ~500 MB | Quantized, nearly same accuracy as medium |
| `ggml-large-v3-turbo-q5_0.bin` | ~500 MB | Quantized turbo, great balance |
| `ggml-distil-large-v3.bin` | ~756 MB | Distilled, English-optimized |

Models are stored in `~/.whisper2text/models/` and are **not** included in the repository.

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
│   ├── main_window.py       # Main PyQt5 window + tray + hotkey listener
│   ├── settings_dialog.py   # Settings with model/device/backend selectors
│   └── error_panel.py       # Thread-safe error/log display
├── config/
│   ├── settings.py          # Settings persistence (JSON)
│   ├── paths.py             # Path resolution (PyInstaller-aware)
│   ├── logging_setup.py     # Logging configuration
│   └── process_lock.py      # Single-instance lock
├── packaging/
│   ├── whisper2text.desktop  # Desktop launcher entry
│   ├── whisper2text.service  # systemd user service
│   └── rthook_cuda.py       # PyInstaller CUDA runtime hook
├── install.sh               # Full automated installer
├── uninstall.sh             # Clean removal
└── whisper2text.spec        # PyInstaller build spec
```

## Deploying to Another Machine

No pre-built releases are provided because CUDA libraries are driver-specific and ~1 GB. Each machine must build locally:

```bash
git clone https://github.com/ac2522/whisperLocal.git
cd whisperLocal
./install.sh
```

The installer detects your GPU and builds accordingly.

## Troubleshooting

**App won't start / crashes immediately**
- Check logs: `journalctl --user -u whisper2text -f`
- Remove stale lock: `rm -f ~/.whisper2text/app.lock`
- Restart: `systemctl --user restart whisper2text`

**Transcription is slow (several seconds)**
- You're likely running on CPU. Check Settings > Compute Backend.
- Verify CUDA is working: the log should show GPU initialization on startup.
- Ensure pywhispercpp was built with `GGML_CUDA=1`. Re-run `./install.sh` to rebuild.

**No sound input**
- Ensure your user is in the `audio` group: `sudo usermod -aG audio $USER`
- Check Settings > Audio Device and select the correct microphone.

**Global hotkey not working**
- Ensure your user is in the `input` group: `sudo usermod -aG input $USER` (then log out/in).
- The app will show a warning in the log panel if no keyboard devices are accessible.

**CUDA build fails with "parameter packs not expanded"**
- Your CUDA toolkit (`nvcc`) is too old. You need version 12.x or newer.
- Remove the old one: `sudo apt remove nvidia-cuda-toolkit`
- Install from NVIDIA's repo (the installer can do this for you).

**"ydotool not found" warning**
- Install ydotool: `sudo apt install ydotool`
- Auto-paste is optional; transcriptions are always copied to clipboard regardless.

**PyAudio install fails**
- Install PortAudio: `sudo apt install portaudio19-dev`

## License

MIT
