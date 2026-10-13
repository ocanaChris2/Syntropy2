#!/usr/bin/env python3
"""Compress, decompress and compare some corpus files one at a time, against nsc 0.3.1's cached
sizes (a scorecard category in pieces, for machines short on memory or time).

    python3 bench/check.py [--corpus corpus] <category/file ...>
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(HERE), "target", "release", "nsc")
corpus = sys.argv[sys.argv.index("--corpus") + 1] if "--corpus" in sys.argv else "corpus"
cache = json.load(open(os.path.join(HERE, "results", "formats_cache.json")))
bad = 0
for rel in [a for a in sys.argv[1:] if "/" in a]:
    path = os.path.join(HERE, corpus, rel)
    raw = open(path, "rb").read()
    old = cache.get(f"{rel}@{zlib.crc32(raw):08x}", {}).get("old5")
    with tempfile.TemporaryDirectory(prefix="nsc-check-") as d:
        out, back = os.path.join(d, "o.nsc"), os.path.join(d, "o.bin")
        t0 = time.time()
        r = subprocess.run([NSC, "compress", "-5", "-q", "--jobs", "2", path, out])
        secs = time.time() - t0
        ok = r.returncode == 0 and subprocess.run([NSC, "decompress", "-q", "--jobs", "2", out, back]).returncode == 0 \
            and subprocess.run(["cmp", "-s", path, back]).returncode == 0
        n = os.path.getsize(out) if os.path.exists(out) else 0
    worse = bool(old) and n > old * 1.003
    bad += (not ok) or worse
    delta = f"{100 * (n / old - 1):+.2f}% vs 0.3.1 ({old})" if old else "no baseline"
    print(f"{rel:<28} {len(raw):>10} -> {n:>9}  {delta}  {secs:5.0f}s  {'ok' if ok else 'ROUNDTRIP FAILED'}{'  REGRESSION' if worse else ''}", flush=True)
sys.exit(1 if bad else 0)
