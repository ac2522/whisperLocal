# Whisper2Text

A desktop speech-to-text application using OpenAI's Whisper model with a PyQt5 GUI. Features real-time audio recording, automatic transcription, and clipboard integration.

## Features

- **Voice Activity Detection (VAD)**: Automatically detects speech using WebRTC VAD
- **Multiple Recording Modes**:
  - Silence mode: Automatically stops recording after a period of silence
  - Button mode: Manual start/stop control
- **Global Hotkey**: `Ctrl+Alt+Shift+L` to start/stop recording from anywhere
- **System Tray Integration**: Runs in the background with tray icon
- **Auto-paste**: Optionally paste transcribed text automatically
- **Persistent History**: Keeps last 10 transcriptions
- **Customizable Settings**: Adjust VAD sensitivity, model size, silence duration, and more

## Screenshots

The app displays:
- Recent transcriptions as clickable buttons
- Recording indicator in system tray
- Settings dialog for customization

## Requirements

- Python 3.8+
- PyQt5
- OpenAI Whisper
- PyAudio
- WebRTC VAD
- Additional dependencies (see requirements.txt)

## Installation

### Ubuntu/Linux

1. **Install system dependencies:**
```bash
sudo apt update
sudo apt install -y python3-pip python3-dev portaudio19-dev
```

2. **Clone the repository:**
```bash
git clone https://github.com/yourusername/whisper2text.git
cd whisper2text
```

3. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

5. **Run the application:**
```bash
python whisper2text.py
```

### macOS

1. **Install Homebrew** (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install system dependencies:**
```bash
brew install portaudio python3
```

3. **Clone the repository:**
```bash
git clone https://github.com/yourusername/whisper2text.git
cd whisper2text
```

4. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

5. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

6. **Grant accessibility permissions:**
   - Go to System Preferences → Security & Privacy → Privacy → Accessibility
   - Add Terminal or your Python executable to allow keyboard control

7. **Run the application:**
```bash
python whisper2text.py
```

### Windows

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)

2. **Clone the repository:**
```cmd
git clone https://github.com/yourusername/whisper2text.git
cd whisper2text
```

3. **Create a virtual environment:**
```cmd
python -m venv venv
venv\Scripts\activate
```

4. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

5. **Run the application:**
```cmd
python whisper2text.py
```

## Usage

### Basic Usage

1. **Launch the app:**
```bash
python whisper2text.py
```

2. **Start recording:**
   - Click the "Record" button, or
   - Press `Ctrl+Alt+Shift+L` (global hotkey), or
   - Use the system tray menu

3. **Stop recording:**
   - In silence mode: Wait for the configured silence duration (default 5 seconds)
   - In button mode: Click "Stop Recording" or press the hotkey again

4. **View transcriptions:**
   - Transcriptions appear as buttons in the main window
   - Click any transcription to copy it to clipboard
   - If auto-paste is enabled, it will paste automatically

### Command Line Options

```bash
# Use button mode instead of silence detection
python whisper2text.py --recording_mode button

# Set silence duration to 3 seconds
python whisper2text.py --break_length 3
```

### Settings

Access settings via the gear icon in the app:

- **VAD Aggressiveness** (0-3): Higher values filter more aggressively (0 = least aggressive, 3 = most aggressive)
- **Model Size**: Choose between `tiny`, `base`, `small`, `medium`, `large`
  - `tiny`: Fastest, least accurate
  - `base`: Good balance (default)
  - `large`: Best accuracy, slowest
- **Padding Duration**: Time to continue recording after silence is detected
- **Recording Mode**: `silence` or `button`
- **Silence Duration**: How long to wait in silence before stopping (silence mode only)
- **Auto-paste**: Automatically paste transcriptions after copying

## Building Standalone Executable

### Ubuntu/Linux

```bash
pip install pyinstaller
pyinstaller whisper2text.spec
```

The executable will be in `dist/whisper2text/whisper2text`

### macOS

```bash
pip install pyinstaller
pyinstaller --windowed --add-data "icon.png:." --add-data "icon_recording.png:." whisper2text.py
```

The app bundle will be in `dist/whisper2text.app`

## Troubleshooting

### Linux: No sound input

Make sure your user is in the `audio` group:
```bash
sudo usermod -a -G audio $USER
```
Log out and log back in for changes to take effect.

### macOS: Microphone permission denied

Grant microphone access:
- System Preferences → Security & Privacy → Privacy → Microphone
- Add Terminal or Python to the allowed apps

### macOS: Auto-paste not working

Grant accessibility permissions:
- System Preferences → Security & Privacy → Privacy → Accessibility
- Add Terminal or Python to the allowed apps

### General: PyAudio installation fails

**Linux:**
```bash
sudo apt install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
Download the appropriate `.whl` file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install with:
```cmd
pip install PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl
```

## Configuration

Settings are stored in `~/.whisper2text/settings.json` and include:
- VAD aggressiveness
- Whisper model size
- Recording mode preferences
- Recent transcriptions
- Auto-paste preference

## Architecture

- **GUI**: PyQt5 for cross-platform desktop interface
- **Audio Capture**: PyAudio for cross-platform audio recording
- **Voice Activity Detection**: WebRTC VAD for efficient speech detection
- **Transcription**: OpenAI Whisper for accurate speech-to-text
- **Clipboard**: pyperclip for cross-platform clipboard operations
- **Keyboard Control**: pynput for global hotkeys and auto-paste

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Known Issues

- First transcription may take longer as the Whisper model loads
- Large model sizes require significant RAM and GPU (if available)
- Global hotkey may not work in some desktop environments

## Roadmap

- [ ] Custom hotkey configuration
- [ ] Export transcriptions to file
- [ ] Language selection
- [ ] Multiple audio input device support
- [ ] Timestamp support in transcriptions
