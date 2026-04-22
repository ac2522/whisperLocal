#!/usr/bin/env bash
set -euo pipefail

# ── whisperLocal installer ─────────────────────────────────────────────
# Installs system dependencies, builds pywhispercpp with GPU support,
# builds a PyInstaller binary, and sets up a systemd user service.
# ────────────────────────────────────────────────────────────────────────

INSTALL_DIR="$HOME/.local/share/whisperLocal"
DATA_DIR="$HOME/.whisper2text"
MODELS_DIR="$DATA_DIR/models"
DESKTOP_DIR="$HOME/.local/share/applications"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="whisper2text"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Step 1: System dependencies ───────────────────────────────────────
info "Checking system dependencies..."

NEED_APT=()

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install it first."
    exit 1
fi

# PortAudio (required for PyAudio)
if ! dpkg -s portaudio19-dev &>/dev/null 2>&1; then
    NEED_APT+=(portaudio19-dev)
fi

# ydotool (auto-paste)
if ! command -v ydotool &>/dev/null; then
    NEED_APT+=(ydotool)
fi

if [ ${#NEED_APT[@]} -gt 0 ]; then
    info "Installing: ${NEED_APT[*]}"
    sudo apt-get install -y "${NEED_APT[@]}"
fi

# ── Step 2: CUDA toolkit ──────────────────────────────────────────────
# Check if we have a GPU and whether nvcc is modern enough.
GPU_BACKEND="cpu"

if command -v nvidia-smi &>/dev/null; then
    # Extract max CUDA version from driver (e.g. "13.0")
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)"
    DRIVER_MAJOR="${DRIVER_CUDA%%.*}"

    NVCC_OK=false
    if command -v nvcc &>/dev/null; then
        NVCC_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0)"
        if [ "$NVCC_VER" -ge 12 ]; then
            NVCC_OK=true
        fi
    fi

    if [ "$NVCC_OK" = false ] && [ -n "$DRIVER_CUDA" ]; then
        warn "NVIDIA GPU detected (driver supports CUDA $DRIVER_CUDA)"
        warn "but nvcc is missing or too old for GPU-accelerated builds."
        echo ""
        info "The CUDA toolkit can be installed from NVIDIA's repository."
        # Determine package name: prefer matching driver major, fall back to 12-8
        CUDA_PKG="cuda-toolkit-${DRIVER_MAJOR:-12}-0"
        read -rp "Install $CUDA_PKG now? (requires sudo + ~3 GB download) [Y/n] " answer
        if [[ ! "$answer" =~ ^[Nn]$ ]]; then
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d '.')/x86_64/cuda-keyring_1.1-1_all.deb"
            KEYRING_TMP="$(mktemp)"
            info "Downloading CUDA keyring..."
            wget -q "$KEYRING_URL" -O "$KEYRING_TMP"
            sudo dpkg -i "$KEYRING_TMP"
            rm -f "$KEYRING_TMP"
            sudo apt-get update
            sudo apt-get install -y "$CUDA_PKG"
            # Add to PATH for this session
            CUDA_BIN="/usr/local/cuda/bin"
            if [ -d "$CUDA_BIN" ]; then
                export PATH="$CUDA_BIN:$PATH"
            fi
            NVCC_OK=true
        fi
    fi

    if [ "$NVCC_OK" = true ]; then
        GPU_BACKEND="cuda"
    fi
fi

info "GPU backend: $GPU_BACKEND"

# ── Step 3: Input group (for global hotkeys via evdev) ────────────────
if ! groups | grep -qw input; then
    warn "Current user is not in the 'input' group."
    warn "Global hotkeys will not work without it."
    echo ""
    read -rp "Add $USER to the 'input' group now? (requires sudo) [Y/n] " answer
    if [[ ! "$answer" =~ ^[Nn]$ ]]; then
        sudo usermod -aG input "$USER"
        info "Added $USER to 'input' group. You must log out and back in for this to take effect."
    else
        warn "Skipped. Fix later with: sudo usermod -aG input \$USER  (then log out/in)"
    fi
fi

# ── Step 4: Python venv and dependencies ──────────────────────────────
info "Setting up Python environment..."

if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
fi
source "$SCRIPT_DIR/venv/bin/activate"

# Install/rebuild pywhispercpp with GPU support
if [ "$GPU_BACKEND" = "cuda" ]; then
    info "Building pywhispercpp with CUDA support (this may take a few minutes)..."
    GGML_CUDA=1 pip install pywhispercpp --no-binary pywhispercpp --no-cache-dir --force-reinstall
else
    info "Installing pywhispercpp (CPU only)..."
    pip install pywhispercpp
fi

# Install remaining dependencies
pip install -r "$SCRIPT_DIR/requirements.txt"

# Ensure PyInstaller and setuptools (for pkg_resources) are available
pip install pyinstaller 'setuptools<70'

# ── Step 5: Build with PyInstaller ────────────────────────────────────
info "Building whisper2text binary..."
cd "$SCRIPT_DIR"
python3 -m PyInstaller whisper2text.spec --clean --noconfirm

if [ ! -f "dist/whisper2text/whisper2text" ]; then
    error "Build failed: dist/whisper2text/whisper2text not found."
    exit 1
fi

info "Build successful."

# ── Step 6: Install to ~/.local/share/whisperLocal/ ───────────────────
info "Installing to $INSTALL_DIR ..."

# Stop the service if it's already running
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp -a dist/whisper2text/* "$INSTALL_DIR/"

# Copy icon to top-level for .desktop file (PyInstaller puts it in _internal/)
cp "$INSTALL_DIR/_internal/icon.png" "$INSTALL_DIR/whisper2text.png" 2>/dev/null || true

info "Installed $(du -sh "$INSTALL_DIR" | cut -f1) to $INSTALL_DIR"

# ── Step 7: Set up models directory ───────────────────────────────────
mkdir -p "$MODELS_DIR"

# Check for models in the old location (project-local models/)
if [ -d "$SCRIPT_DIR/models" ] && ls "$SCRIPT_DIR/models/"*.bin &>/dev/null 2>&1; then
    echo ""
    info "Found models in $SCRIPT_DIR/models/:"
    ls -lh "$SCRIPT_DIR/models/"*.bin 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
    echo ""
    read -rp "Move these models to $MODELS_DIR? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        mv "$SCRIPT_DIR/models/"*.bin "$MODELS_DIR/" 2>/dev/null || true
        info "Models moved to $MODELS_DIR"
    fi
fi

# ── Step 8: Fix libcuda.so ────────────────────────────────────────────
# The bundled libcuda.so may not match this machine's GPU driver.
# Replace it with the system's driver-matched version.
if [ "$GPU_BACKEND" = "cuda" ]; then
    info "Checking libcuda.so compatibility..."
    SYSTEM_LIBCUDA="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\./ {print $NF; exit}')"
    if [ -n "$SYSTEM_LIBCUDA" ]; then
        BUNDLED_LIBCUDA="$(find "$INSTALL_DIR/pywhispercpp.libs" -name 'libcuda*.so*' 2>/dev/null | head -1)"
        if [ -n "$BUNDLED_LIBCUDA" ]; then
            info "Replacing bundled libcuda with system version..."
            cp "$SYSTEM_LIBCUDA" "$BUNDLED_LIBCUDA"
        fi
    else
        warn "System libcuda.so not found. CUDA may not work at runtime."
    fi
fi

# ── Step 9: Create whisperlocal command ───────────────────────────────
info "Creating whisperlocal command..."
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/whisperlocal" << 'LAUNCHER'
#!/bin/bash
exec "$HOME/.local/share/whisperLocal/whisper2text" "$@"
LAUNCHER
chmod +x "$HOME/.local/bin/whisperlocal"

if ! echo "$PATH" | tr ':' '\n' | grep -q "$HOME/.local/bin"; then
    warn "\$HOME/.local/bin is not in your PATH."
    warn "Add this to your ~/.bashrc or ~/.zshrc:"
    warn '  export PATH="$HOME/.local/bin:$PATH"'
fi

# ── Step 10: Install .desktop file ────────────────────────────────────
info "Installing desktop entry..."
mkdir -p "$DESKTOP_DIR"
sed "s|%HOME%|$HOME|g" "$SCRIPT_DIR/packaging/whisper2text.desktop" \
    > "$DESKTOP_DIR/whisper2text.desktop"
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

# ── Step 11: Install and enable systemd service ──────────────────────
info "Installing systemd user service..."
mkdir -p "$SERVICE_DIR"
cp "$SCRIPT_DIR/packaging/whisper2text.service" "$SERVICE_DIR/"

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start "$SERVICE_NAME"

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
info "Installation complete!"
echo "========================================"
echo ""
echo "  Install location:  $INSTALL_DIR"
echo "  User data:         $DATA_DIR"
echo "  Models:            $MODELS_DIR"
echo "  GPU backend:       $GPU_BACKEND"
echo ""
echo "  Management commands:"
echo "    systemctl --user status $SERVICE_NAME    # Check status"
echo "    systemctl --user restart $SERVICE_NAME   # Restart"
echo "    systemctl --user stop $SERVICE_NAME      # Stop"
echo "    journalctl --user -u $SERVICE_NAME -f    # View logs"
echo ""
echo "  Models can be downloaded via the app's Settings dialog."
echo ""
if ! groups | grep -qw input; then
    echo -e "  ${YELLOW}NOTE: Log out and back in to activate the 'input' group for hotkeys.${NC}"
    echo ""
fi
