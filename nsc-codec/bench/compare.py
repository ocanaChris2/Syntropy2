#!/usr/bin/env python3
"""Compare nsc benchmark runs against the classical baseline.

    python3 bench/compare.py bench/results/p0_classical.json bench/results/p1_L1.json [...more.json]

Prints, per corpus category, bits/byte for every codec, nsc's gain vs xz -9e and vs the best
classical codec, plus single-thread speeds and the P1 gate metrics (Silesia binaries vs xz).
"""
import json
import os
import sys
from collections import OrderedDict, defaultdict

runs = [json.load(open(p)) for p in sys.argv[1:]]
files = OrderedDict()
for r in runs:
    for f in r["files"]:
        key = os.path.relpath(f["path"], "bench/corpus") if "bench/corpus" in f["path"] else f["path"]
        e = files.setdefault(key, {"bytes": f["bytes"], "cat": f["category"], "res": {}})
        for m in f["results"]:
            if m["error"] is None:
                e["res"][m["codec"]] = m

codecs = []
for e in files.values():
    for c in e["res"]:
        if c not in codecs:
            codecs.append(c)
classical = [c for c in codecs if not c.startswith("nsc")]
ours = [c for c in codecs if c.startswith("nsc")]

def bpb(n, size): return 8.0 * n / size if size else 0.0

cats = OrderedDict()
for k, e in files.items():
    cats.setdefault(e["cat"], []).append((k, e))

w = max(len(c) for c in codecs) + 1
print(f"{'category':<10} {'MB':>8} " + " ".join(f"{c:>{w}}" for c in codecs) + "   nsc vs xz   vs best classical")
tot = defaultdict(int)
tot_in = 0
for cat, items in cats.items():
    size = sum(e["bytes"] for _, e in items if all(c in e["res"] for c in codecs))
    if size == 0:
        continue
    sums = {c: sum(e["res"][c]["bytes"] for _, e in items if all(cc in e["res"] for cc in codecs)) for c in codecs}
    for c in codecs:
        tot[c] += sums[c]
    tot_in += size
    best_cl = min(sums[c] for c in classical) if classical else None
    line = f"{cat:<10} {size / 1e6:8.2f} " + " ".join(f"{bpb(sums[c], size):>{w}.3f}" for c in codecs)
    if ours and "xz-9e" in sums:
        o = min(sums[c] for c in ours)
        line += f"   {100 * (o / sums['xz-9e'] - 1):+7.1f}%   {100 * (o / best_cl - 1):+7.1f}%"
    print(line)
print(f"{'TOTAL':<10} {tot_in / 1e6:8.2f} " + " ".join(f"{bpb(tot[c], tot_in):>{w}.3f}" for c in codecs), end="")
if ours and "xz-9e" in tot:
    o = min(tot[c] for c in ours)
    best_cl = min(tot[c] for c in classical)
    print(f"   {100 * (o / tot['xz-9e'] - 1):+7.1f}%   {100 * (o / best_cl - 1):+7.1f}%")
else:
    print()

# speeds (single-thread, whole corpus)
print("\nspeed (MB/s, whole corpus):")
for c in codecs:
    enc = sum(e["res"][c]["enc_s"] for e in files.values() if c in e["res"])
    dec = sum((e["res"][c]["dec_s"] or 0) for e in files.values() if c in e["res"])
    size = sum(e["bytes"] for e in files.values() if c in e["res"])
    peak = max((max(e["res"][c]["enc_rss_kb"] or 0, e["res"][c]["dec_rss_kb"] or 0) for e in files.values() if c in e["res"]), default=0)
    print(f"  {c:<10} enc {size / 1e6 / enc if enc else 0:7.2f}   dec {size / 1e6 / dec if dec else 0:7.2f}   peak RSS {peak // 1024} MB")

# gate: Silesia binaries vs xz
bins = ["silesia/mozilla", "silesia/mr", "silesia/ooffice", "silesia/osdb", "silesia/sao", "silesia/x-ray"]
have = [b for b in bins if b in files and "xz-9e" in files[b]["res"]]
if have and ours:
    print("\nSilesia binaries (mozilla, mr, ooffice, osdb, sao, x-ray):")
    xz = sum(files[b]["res"]["xz-9e"]["bytes"] for b in have)
    for c in ours:
        if all(c in files[b]["res"] for b in have):
            n = sum(files[b]["res"][c]["bytes"] for b in have)
            print(f"  {c}: {n} B vs xz {xz} B  ->  {100 * (n / xz - 1):+.1f}%")
bad = [(k, c) for k, e in files.items() for c, m in e["res"].items() if m.get("roundtrip") is False]
print(f"\nroundtrip failures: {len(bad)}", bad[:5])
