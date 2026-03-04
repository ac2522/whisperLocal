#!/usr/bin/env bash
set -euo pipefail

# ── whisper2text uninstaller ────────────────────────────────────────────
# Removes the installed binary, desktop entry, and systemd service.
# Leaves ~/.whisper2text/ (user data, models, settings) intact.
# ────────────────────────────────────────────────────────────────────────

INSTALL_DIR="$HOME/.local/share/whisperLocal"
DATA_DIR="$HOME/.whisper2text"
DESKTOP_FILE="$HOME/.local/share/applications/whisper2text.desktop"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="whisper2text"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── Step 1: Stop and disable systemd service ────────────────────────────
info "Stopping systemd service..."
systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true

# ── Step 2: Remove service file ─────────────────────────────────────────
if [ -f "$SERVICE_DIR/$SERVICE_NAME.service" ]; then
    rm "$SERVICE_DIR/$SERVICE_NAME.service"
    systemctl --user daemon-reload
    info "Removed systemd service."
fi

# ── Step 3: Remove desktop entry ────────────────────────────────────────
if [ -f "$DESKTOP_FILE" ]; then
    rm "$DESKTOP_FILE"
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    info "Removed desktop entry."
fi

# ── Step 4: Remove install directory ────────────────────────────────────
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    info "Removed $INSTALL_DIR"
fi

# ── Done ────────────────────────────────────────────────────────────────
echo ""
info "Uninstall complete."
echo ""
echo "  User data preserved at: $DATA_DIR"
echo "  (Contains settings, models, and logs)"
echo "  To remove everything: rm -rf $DATA_DIR"
echo ""
