#!/usr/bin/env python3
"""Per-file rows of a scorecard: python3 bench/show.py <tag> [fmt,fmt...]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results", f"formats_{sys.argv[1]}.json")))
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for r in rows:
    if only and r["fmt"] not in only:
        continue
    n = r["bytes"]
    pct = lambda v: f"{100 * (1 - v / n):6.2f}%" if v else "     - "
    print(f"{r['path']:<46} {n:>10}  new {pct(r['new'])}  0.3.1 {pct(r['old'])}  xz {pct(r['xz'])}  zstd {pct(r['zstd'])}  {r['secs']:6.1f}s  {r['split']}{'' if r['ok'] else '  ROUNDTRIP FAILED'}")
