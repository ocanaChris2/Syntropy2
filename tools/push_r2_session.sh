#!/usr/bin/env bash
# Stage a finished Kaggle session's runs/ folder into kaggle_r2/ and publish a new version of
# the private dataset christianocanab/syntropy2-r2, so the next session can resume.
#
#   tools/push_r2_session.sh <downloaded runs dir> ["version message"]
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$(realpath "$1")"; MSG="${2:-session $(date +%F)}"
[ -f "$SRC/ledger.json" ] && [ -f "$SRC/accumulated.json" ] || { echo "$SRC has no ledger.json/accumulated.json" >&2; exit 1; }
DST="$ROOT/kaggle_r2/runs"
rm -rf "$DST"; mkdir -p "$DST"
# everything needed to resume; current/ duplicates best/ and is rebuilt every phase
rsync -a --exclude 'current/' "$SRC"/ "$DST"/
# never leave a nested zip beside its extracted twin (Kaggle auto-extract collision)
find "$ROOT/kaggle_r2" -name '*.zip' -print -delete
python3 - "$DST" <<'PY'
import json, sys, os
d = sys.argv[1]
led = json.load(open(os.path.join(d, "ledger.json")))
print(f"ledger: {len(led)} phase(s), last step {led[-1]['step']:,}/{led[-1]['total_steps']:,}")
PY
kaggle datasets version -p "$ROOT/kaggle_r2" -r zip -m "$MSG"
