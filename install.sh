#!/usr/bin/env bash
set -euo pipefail

# ── whisper2text installer ──────────────────────────────────────────────
# Builds a PyInstaller binary locally (matching this machine's GPU drivers)
# and installs it as a systemd user service with crash recovery.
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

# ── Step 1: Check prerequisites ────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install it first."
    exit 1
fi

if ! command -v ydotool &>/dev/null; then
    warn "ydotool not found. Auto-paste feature will not work."
    warn "Install with: sudo apt install ydotool"
fi

if ! groups | grep -qw input; then
    warn "Current user is not in the 'input' group."
    warn "Global hotkeys (evdev) may not work."
    warn "Fix with: sudo usermod -aG input \$USER  (then log out/in)"
fi

# ── Step 2: Ensure PyInstaller is available ─────────────────────────────
info "Checking for PyInstaller..."

# Activate venv if present
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

if ! python3 -m PyInstaller --version &>/dev/null; then
    info "Installing PyInstaller..."
    pip install pyinstaller
fi

# ── Step 3: Build with PyInstaller ──────────────────────────────────────
info "Building whisper2text binary..."
cd "$SCRIPT_DIR"
python3 -m PyInstaller whisper2text.spec --clean --noconfirm

if [ ! -f "dist/whisper2text/whisper2text" ]; then
    error "Build failed: dist/whisper2text/whisper2text not found."
    exit 1
fi

info "Build successful."

# ── Step 4: Install to ~/.local/share/whisperLocal/ ─────────────────────
info "Installing to $INSTALL_DIR ..."

# Stop the service if it's already running
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp -a dist/whisper2text/* "$INSTALL_DIR/"

info "Installed $(du -sh "$INSTALL_DIR" | cut -f1) to $INSTALL_DIR"

# ── Step 5: Set up models directory ─────────────────────────────────────
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

# ── Step 6: Fix libcuda.so ──────────────────────────────────────────────
# The bundled libcuda.so may not match this machine's GPU driver.
# Replace it with the system's driver-matched version.
info "Checking libcuda.so compatibility..."

SYSTEM_LIBCUDA="$(ldconfig -p 2>/dev/null | grep 'libcuda.so\.' | head -1 | awk '{print $NF}')"

if [ -n "$SYSTEM_LIBCUDA" ]; then
    # Find the bundled libcuda in pywhispercpp.libs
    BUNDLED_LIBCUDA="$(find "$INSTALL_DIR/pywhispercpp.libs" -name 'libcuda*.so*' 2>/dev/null | head -1)"
    if [ -n "$BUNDLED_LIBCUDA" ]; then
        info "Replacing bundled libcuda with system version ($SYSTEM_LIBCUDA)..."
        cp "$SYSTEM_LIBCUDA" "$BUNDLED_LIBCUDA"
    fi
else
    warn "System libcuda.so not found via ldconfig. CUDA may not work."
    warn "Ensure NVIDIA drivers are installed."
fi

# ── Step 7: Install .desktop file ───────────────────────────────────────
info "Installing desktop entry..."
mkdir -p "$DESKTOP_DIR"

# Expand %HOME% placeholders in the desktop file
sed "s|%HOME%|$HOME|g" "$SCRIPT_DIR/packaging/whisper2text.desktop" \
    > "$DESKTOP_DIR/whisper2text.desktop"

# Update desktop database if available
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

# ── Step 8: Install and enable systemd service ──────────────────────────
info "Installing systemd user service..."
mkdir -p "$SERVICE_DIR"
cp "$SCRIPT_DIR/packaging/whisper2text.service" "$SERVICE_DIR/"

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start "$SERVICE_NAME"

# ── Done ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
info "Installation complete!"
echo "========================================"
echo ""
echo "  Install location:  $INSTALL_DIR"
echo "  User data:         $DATA_DIR"
echo "  Models:            $MODELS_DIR"
echo ""
echo "  Management commands:"
echo "    systemctl --user status $SERVICE_NAME    # Check status"
echo "    systemctl --user restart $SERVICE_NAME   # Restart"
echo "    systemctl --user stop $SERVICE_NAME      # Stop"
echo "    journalctl --user -u $SERVICE_NAME -f    # View logs"
echo ""
echo "  Models can be downloaded via the app's Settings dialog."
echo ""
