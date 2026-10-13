#!/usr/bin/env bash
# Build and deploy `nsc` (datatype-aware lossless compressor + archiver) as a ready-to-use command,
# plus the optional neural text model registry (so models can be upgraded safely).
#
#   ./install.sh                 # build, install `nsc`, register the current model as default
#   INSTALL_GUI=1 ./install.sh   # also build and install the `nsc-gui` desktop app
#   PREFIX=/usr/local/bin ./install.sh
#   NSC_HOME=/opt/nsc ./install.sh
#   MODEL_ID=base-v2 BPC=1.28 ./install.sh   # register under a different id
#
# Re-runnable: rebuilds and reinstalls the binary; skips re-importing a model id that already exists
# (upgrade to a new model later with `nsc models add <new-id> ... --default`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

PREFIX="${PREFIX:-$HOME/.local/bin}"
NSC_HOME="${NSC_HOME:-$HOME/.nsc}"
MODEL_ID="${MODEL_ID:-base-v1}"
TIER="${TIER:-base}"
BPC="${BPC:-1.29}"
# Source artifacts to seed the registry with (override to register a freshly-exported model).
ART_MODEL="${ART_MODEL:-$HERE/artifacts/model.nsc}"
ART_O2="${ART_O2:-$HERE/artifacts/o2.nsc}"
ART_TOK="${ART_TOK:-$REPO/results/export/tokenizer.json}"

echo "==> Building release binary (cargo build --release)"
( cd "$HERE" && cargo build --release )
BIN_SRC="$HERE/target/release/nsc"
[ -x "$BIN_SRC" ] || { echo "build failed: $BIN_SRC not found" >&2; exit 1; }

echo "==> Installing binary -> $PREFIX/nsc"
mkdir -p "$PREFIX"
install -m 0755 "$BIN_SRC" "$PREFIX/nsc"

if [ "${INSTALL_GUI:-0}" = "1" ]; then
  echo "==> Building the desktop app (gui/)"
  ( cd "$HERE/gui" && cargo build --release )
  install -m 0755 "$HERE/gui/target/release/nsc-gui" "$PREFIX/nsc-gui"
  echo "    installed $PREFIX/nsc-gui"
fi

echo "==> Setting up registry -> $NSC_HOME"
mkdir -p "$NSC_HOME/models"
install -m 0644 "$HERE/scripts/tok.py" "$NSC_HOME/tok.py"

if [ -d "$NSC_HOME/models/$MODEL_ID" ]; then
  echo "    model '$MODEL_ID' already present — skipping import (use MODEL_ID=<new> to add another)"
else
  missing=""
  for f in "$ART_MODEL" "$ART_O2" "$ART_TOK"; do
    [ -f "$f" ] || missing="$missing $f"
  done
  if [ -n "$missing" ]; then
    # The context-mixing engine needs no model, so a missing neural model is not fatal:
    # everything compresses with NSC3; text just won't use the neural path until a model is added.
    echo "    neural model artifacts not found:$missing" >&2
    echo "    -> installing without a neural model (all files use the CM engine)." >&2
    echo "       Add one later: run export_int8.py + export_o2.py, then \`nsc models add ...\`" >&2
  else
    echo "==> Registering model '$MODEL_ID' (tier=$TIER, bpc=$BPC) as default"
    "$BIN_SRC" --home "$NSC_HOME" models add "$MODEL_ID" \
      --model-nsc "$ART_MODEL" --o2 "$ART_O2" --tokenizer "$ART_TOK" \
      --tier "$TIER" --bpc "$BPC" --created "$(date +%F)" --default
  fi
fi

echo ""
"$BIN_SRC" --home "$NSC_HOME" models list
echo ""
echo "==> Done. Verify with:  nsc doctor"
case ":$PATH:" in
  *":$PREFIX:"*) : ;;
  *) echo "    NOTE: $PREFIX is not on your PATH. Add:  export PATH=\"$PREFIX:\$PATH\"" ;;
esac
if [ "$NSC_HOME" != "$HOME/.nsc" ]; then
  echo "    NOTE: custom registry — export NSC_HOME=\"$NSC_HOME\" so nsc finds it."
fi
echo ""
echo "Quickstart:"
echo "    nsc compress   photo.png photo.nsc        # any file, bit-exact"
echo "    nsc decompress photo.nsc photo.png"
echo "    nsc pack   backup.nsca ~/project          # solid, deduplicated archive"
echo "    nsc unpack backup.nsca restore/"
echo "    nsc analyze file --exact                  # what nsc sees in a file"
