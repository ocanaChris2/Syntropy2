#!/usr/bin/env bash
# Install a trained model into the nsc registry (NOT as default).
#
#   tools/install_model.sh <export-dir> <model-id> [data-file] [tier]
#
#   <export-dir>  folder with best.pt + tokenizer.json (e.g. the downloaded runs/best)
#   <model-id>    registry id, e.g. mid-v1
#   [data-file]   training data the model was built on (default data/corpus_r2/corpus_r2.bin)
#   [tier]        informational tier name (default mid)
#
# Old .nsc files keep decoding: each one records the id + fingerprint of its model, and
# the previous models stay installed. Promote only after comparing on the benchmark:
#   nsc bench nsc-codec/bench/corpus/text nsc-codec/bench/corpus/deflate --model <model-id>
#   nsc models default <model-id>
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,14p "$0"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORT="$(realpath "$1")"; ID="$2"
DATA="$(realpath "${3:-$ROOT/data/corpus_r2/corpus_r2.bin}")"; TIER="${4:-mid}"
OUT="$ROOT/nsc-codec/artifacts/$ID"
[ -f "$EXPORT/tokenizer.json" ] || { echo "no tokenizer.json in $EXPORT" >&2; exit 1; }
mkdir -p "$OUT"
python3 "$ROOT/nsc-codec/scripts/export_int8.py" --export-dir "$EXPORT" --data "$DATA" --out "$OUT"
python3 "$ROOT/nsc-codec/scripts/export_o2.py"   --export-dir "$EXPORT" --data "$DATA" --out "$OUT"
BPC="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('test_bits_per_byte',0))" "$EXPORT/calibration.json" 2>/dev/null || echo 0)"
nsc models add "$ID" --model-nsc "$OUT/model.nsc" --o2 "$OUT/o2.nsc" --tokenizer "$EXPORT/tokenizer.json" \
    --tier "$TIER" --bpc "$BPC" --notes "corpus $(basename "$DATA")"
nsc models
echo "installed $ID (not default). Compare first, then: nsc models default $ID"
