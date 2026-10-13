#!/usr/bin/env bash
# Installs a clickable desktop launcher for the Syntropy2 Compressor GUI.
#
# It builds the GUI (release) if needed, makes sure the `nsc` codec binary exists, and writes a
# .desktop entry into ~/.local/share/applications pointing at the absolute binary paths — so the
# app shows up in your application menu and can be launched with a double-click.
set -euo pipefail

GUI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEC_DIR="$(cd "$GUI_DIR/.." && pwd)"
GUI_BIN="$GUI_DIR/target/release/nsc-gui"
NSC_BIN="$CODEC_DIR/target/release/nsc"

echo "codec dir : $CODEC_DIR"
echo "gui dir   : $GUI_DIR"

# Build the codec if the binary is missing.
if [[ ! -x "$NSC_BIN" ]]; then
  echo "building nsc codec…"
  ( cd "$CODEC_DIR" && cargo build --release )
fi
# Build the GUI if the binary is missing.
if [[ ! -x "$GUI_BIN" ]]; then
  echo "building nsc-gui…"
  ( cd "$GUI_DIR" && cargo build --release )
fi

APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
DESKTOP="$APPS_DIR/syntropy2-compressor.desktop"

cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Syntropy2 Compressor
Comment=Compress files with the Syntropy2 neural codec
Exec=env NSC_BIN=$NSC_BIN $GUI_BIN
Icon=package-x-generic
Terminal=false
Categories=Utility;Archiving;Compression;
EOF

chmod +x "$DESKTOP" 2>/dev/null || true
# Refresh the menu database if the tool is available (best-effort).
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true

echo
echo "Installed launcher: $DESKTOP"
echo "You can now find \"Syntropy2 Compressor\" in your application menu,"
echo "or launch it directly:  NSC_BIN=$NSC_BIN $GUI_BIN"
