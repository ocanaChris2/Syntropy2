#!/usr/bin/env python3
"""Per-format scorecard over the format corpus (bench/formats/, see build_formats.py).

    python3 bench/formats.py [--nsc target/release/nsc] [--level 5] [--tag f1] [--only mp3,flac]
                             [--oracles] [--jobs 8] [--corpus corpus]

`--corpus corpus` runs the same comparison over the original benchmark corpus (bench/corpus, by
category) — the regression check against nsc 0.3.1.

For every file: the size under this build of nsc (with a decompress-and-compare check), under
nsc 0.3.1 (the baseline before the format specialists), under xz -9 and zstd -19, and — with
--oracles — under the best dedicated tool for its format (packMP3 for MPEG-1 Layer III, paq8px
for the rest up to 8 MB; never part of nsc, only yardsticks). Results of everything but the
build under test are cached in bench/results/formats_cache.json by file checksum.

Prints one row per format (percent saved; higher is better) and writes
bench/results/formats_<tag>.{txt,json}. The holdout files are reported apart.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ORACLES = os.path.join(HERE, "oracles", "bin")
CACHE_PATH = os.path.join(RESULTS, "formats_cache.json")


def arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


CORPUS = os.path.join(HERE, arg("--corpus", "formats"))
NSC = os.path.abspath(arg("--nsc", os.path.join(os.path.dirname(HERE), "target", "release", "nsc")))
LEVEL = arg("--level", "5")
TAG = arg("--tag", "run")
ONLY = set(arg("--only", "").split(",")) - {""}
WITH_ORACLES = "--oracles" in sys.argv
JOBS = int(arg("--jobs", "4"))
BASELINE = os.path.join(ORACLES, "nsc-0.3.1")


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)


def piped_size(cmd, path):
    with open(path, "rb") as f:
        r = subprocess.run(cmd, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return len(r.stdout) if r.returncode == 0 else None


def nsc_size(binary, path, check):
    """(compressed size or None, seconds, roundtrip ok, blocks coded a second time)."""
    with tempfile.TemporaryDirectory(prefix="nsc-formats-") as d:
        out, back = os.path.join(d, "o.nsc"), os.path.join(d, "o.bin")
        t0 = time.time()
        r = run([binary, "compress", f"-{LEVEL}", "-q", "--jobs", "2", path, out], env=dict(os.environ, NSC_TRACE="1"))
        secs = time.time() - t0
        if r.returncode != 0 or not os.path.exists(out):
            return None, secs, False, 0
        size = os.path.getsize(out)
        again = r.stderr.count(b"coding again")
        ok = True
        if check:
            r = run([binary, "decompress", "-q", "--jobs", "2", out, back])
            ok = r.returncode == 0 and run(["cmp", "-s", path, back]).returncode == 0
        return size, secs, ok, again


def oracle(fmt, path, size):
    """(tool name, size) of the dedicated yardstick for this file, or None."""
    with tempfile.TemporaryDirectory(prefix="nsc-oracle-") as d:
        if fmt == "mp3":
            tmp = os.path.join(d, "x.mp3")
            os.symlink(path, tmp)
            run([os.path.join(ORACLES, "packMP3"), "-np", tmp], cwd=d)
            out = os.path.join(d, "x.pmp")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                return "packMP3", os.path.getsize(out)
        if size <= 8 << 20 and os.path.exists(os.path.join(ORACLES, "paq8px")):
            out = os.path.join(d, "x.paq")
            r = run([os.path.join(ORACLES, "paq8px"), "-4", path, out])
            if r.returncode == 0 and os.path.exists(out):
                return "paq8px", os.path.getsize(out)
    return None


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(CORPUS, "MANIFEST.tsv"))]
    head, rows = rows[0], rows[1:]
    col = {name: k for k, name in enumerate(head)}
    fmt_col = col.get("format", col.get("category"))
    files = [dict(path=r[0], bytes=int(r[1]), crc=r[2], fmt=r[fmt_col], split=r[col["split"]] if "split" in col else "dev") for r in rows]
    if ONLY:
        files = [f for f in files if f["fmt"] in ONLY]

    def measure(f):
        p = os.path.join(CORPUS, f["path"])
        key = f"{f['path']}@{f['crc']}"
        c = cache.setdefault(key, {})
        if "xz" not in c:
            c["xz"] = piped_size(["xz", "-9", "-c"], p)
        if "zstd" not in c:
            c["zstd"] = piped_size(["zstd", "-19", "-q", "-c"], p)
        if f"old{LEVEL}" not in c and os.path.exists(BASELINE):
            c[f"old{LEVEL}"] = nsc_size(BASELINE, p, False)[0]
        if WITH_ORACLES and "oracle" not in c:
            c["oracle"] = oracle(f["fmt"], p, f["bytes"])
        size, secs, ok, again = nsc_size(NSC, p, True)
        return dict(f, new=size if ok else None, secs=secs, ok=ok, again=again, xz=c.get("xz"), zstd=c.get("zstd"), old=c.get(f"old{LEVEL}"), oracle=c.get("oracle"))

    with concurrent.futures.ThreadPoolExecutor(JOBS) as ex:
        res = list(ex.map(measure, files))
    json.dump(cache, open(CACHE_PATH, "w"), indent=0, sort_keys=True)

    def saved(n, total):
        return f"{100 * (1 - n / total):6.2f}%" if n is not None and total else "      -"

    lines = []
    for title, pick in (("all files", lambda r: True), ("holdout only", lambda r: r["split"] == "holdout")):
        sel = [r for r in res if pick(r)]
        lines.append(f"== {title}: level {LEVEL}, percent saved (higher is better)")
        lines.append(f"{'format':<6} {'files':>5} {'MB':>8} {'nsc new':>8} {'nsc 0.3.1':>9} {'xz -9':>8} {'zstd -19':>8} {'oracle':>8}  {'':10} {'MB/s':>6}")
        tot = dict(n=0, bytes=0, new=0, old=0, xz=0, zstd=0, secs=0.0)
        for fmt in sorted({r["fmt"] for r in sel}):
            rs = [r for r in sel if r["fmt"] == fmt]
            b = sum(r["bytes"] for r in rs)

            def col(k):
                return sum(r[k] if r[k] is not None else r["bytes"] for r in rs)
            orc = [r for r in rs if r["oracle"]]
            ob, osz = sum(r["bytes"] for r in orc), sum(r["oracle"][1] for r in orc)
            onames = "+".join(sorted({r["oracle"][0] for r in orc}))
            onew = sum(r["new"] or r["bytes"] for r in orc)
            ocol = f"{saved(osz, ob)}  {onames + (' (nsc ' + saved(onew, ob).strip() + ' on same)' if orc and len(orc) < len(rs) else ''):<10}" if orc else f"{'-':>8}  {'':10}"
            secs = sum(r["secs"] for r in rs)
            lines.append(f"{fmt:<6} {len(rs):>5} {b / 1e6:>8.2f} {saved(col('new'), b):>8} {saved(col('old'), b):>9} {saved(col('xz'), b):>8} {saved(col('zstd'), b):>8} {ocol} {b / 1e6 / max(secs, 1e-9):>6.2f}")
            tot["n"] += len(rs)
            tot["bytes"] += b
            tot["secs"] += secs
            for k in ("new", "old", "xz", "zstd"):
                tot[k] += col(k)
        b = tot["bytes"]
        lines.append(f"{'TOTAL':<6} {tot['n']:>5} {b / 1e6:>8.2f} {saved(tot['new'], b):>8} {saved(tot['old'], b):>9} {saved(tot['xz'], b):>8} {saved(tot['zstd'], b):>8}")
        lines.append("")
    worse = [(r["path"], r["old"], r["new"]) for r in res if r["new"] and r["old"] and r["new"] > r["old"] * 1.003]
    lines.append(f"files more than 0.3% larger than under nsc 0.3.1: {len(worse)}" + "".join(f"\n  {p}: {o} -> {n} ({100 * (n / o - 1):+.2f}%)" for p, o, n in worse))
    again = [r["path"] for r in res if r["again"]]
    lines.append(f"files with a block coded a second time (a stream or bytes cost more modelled than not): {len(again)}" + "".join(f"\n  {p}" for p in again[:20]))
    bad = [r["path"] for r in res if not r["ok"]]
    lines.append(f"roundtrip: {len(res) - len(bad)}/{len(res)} OK" + (f"  FAILED: {bad}" if bad else ""))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(RESULTS, f"formats_{TAG}.txt"), "w").write(text + "\n")
    json.dump(res, open(os.path.join(RESULTS, f"formats_{TAG}.json"), "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
