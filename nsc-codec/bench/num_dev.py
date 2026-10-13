#!/usr/bin/env python3
"""Development loop for the number model of text (src/cm/num.rs): code each file of a set of
logs, tables, JSON, XML and plain text (the first megabyte of the larger ones) as one text
segment, and compare with a saved result.

    python3 bench/num_dev.py [--save name] [--against name] [--jobs 3] [--level 5] [--num HEX] [--no-num] [--dense]
    python3 bench/num_dev.py --round [--num HEX] [--tolerance 0.0003]
    python3 bench/num_dev.py --grow [--num HEX] [--tolerance 0.0003]

`--num HEX` keeps only those contexts of the model (bits 62 and 63: its mixer selector and
refinement); `--no-num` leaves it off. `--round` tries the model without each context in turn and
prints the masks ranked: one round of a greedy elimination, to run again with the best mask;
`--grow` tries adding each context left out instead (one round of a greedy selection).
The measure gives every file the same weight (its size against the saved baseline). `--dense`
looks only at the files the model is switched on for (6% digits or more).
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NSC = os.path.join(ROOT, "target", "release", "nsc")
HEADS = os.path.join(HERE, "cache", "numdev")
CAP = 1 << 20
FILES = [
    "formats/log/Apache_2k.log", "formats/log/BGL_2k.log", "formats/log/HDFS_2k.log", "formats/log/Linux_2k.log", "formats/log/Spark_2k.log",
    "formats/log/Zookeeper_2k.log", "formats/log/dpkg.log.1", "corpus/text/app.log",
    "formats/csv/iab.csv", "formats/csv/zip_sums.csv", "corpus/table/sales.csv", "formats/csv/zipcodes.csv", "formats/csv/birdstrikes.csv",
    "formats/csv/weather.csv", "formats/csv/seattle-weather.csv", "formats/csv/flights-airport.csv", "formats/csv/co2-concentration.csv",
    "formats/csv/stocks.csv",
    "formats/json/api.schema.json", "formats/json/iso_639-3.json", "formats/json/nls.metadata.json", "formats/json/protocol.json",
    "formats/json/telemetry-core.json", "corpus/text/tokenizer.json", "formats/json/earthquakes.json", "formats/json/flights-20k.json",
    "formats/json/cars.json", "formats/json/gapminder.json",
    "formats/xml/Locations.xml", "formats/xml/appstream.xml", "formats/xml/freedesktop.org.xml", "formats/xml/launchpad-wadl.xml",
    "corpus/text/enwik_1M.txt", "corpus/text/python_src.py", "corpus/text/rust_src.rs", "corpus/text/whitepaper.md",
]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


LEVEL = arg("--level", "5")


def head(rel):
    """The file, or its first megabyte (kept in bench/cache/numdev)."""
    p = os.path.join(HERE, rel)
    if os.path.getsize(p) <= CAP:
        return p
    os.makedirs(HEADS, exist_ok=True)
    q = os.path.join(HEADS, rel.replace("/", "_"))
    if not os.path.exists(q):
        with open(p, "rb") as f, open(q, "wb") as g:
            g.write(f.read(CAP))
    return q


def dense(rel):
    """Is the file text with many numbers (6% digits or more: what the model is switched on for)?"""
    d = open(head(rel), "rb").read()
    return sum(1 for c in d if 48 <= c <= 57) * 100 >= 6 * len(d)


def size(rel, mask=None, off=False):
    cmd = [NSC, "cm-test", head(rel), "--level", LEVEL, "--seg", "text", "--no-decode"]
    if off:
        cmd.append("--no-num")
    if mask is not None:
        cmd += ["--num", f"{mask:x}"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"-> (\d+) B", out)
    return int(m.group(1)) if m else None


def total(ex, mask=None, off=False):
    return dict(zip(FILES, ex.map(lambda f: size(f, mask, off), FILES)))


def main():
    global FILES
    if "--dense" in sys.argv:
        FILES = [f for f in FILES if dense(f)]
    ex = concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "3")))
    mask = int(arg("--num"), 16) if arg("--num") else None
    results = os.path.join(HERE, "results")
    ref_name = arg("--against", "off")
    ref_path = os.path.join(results, f"num_dev_{ref_name}.json")
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
    score = lambda res: 1000 * sum(res[f] / ref[f] for f in FILES) / len(FILES)
    if "--round" in sys.argv:
        n = int(re.search(r"pub const CONTEXTS: u32 = (\d+);", open(os.path.join(ROOT, "src", "cm", "num.rs")).read()).group(1))
        keep = mask if mask is not None else (1 << n) - 1 | 3 << 62
        base = score(total(ex, keep))
        print(f"{keep:x}: {base:.2f} (mean size, per mille of '{ref_name}')", flush=True)
        tried = []
        for i in [i for i in list(range(n)) + [62, 63] if keep >> i & 1]:
            s = score(total(ex, keep & ~(1 << i)))
            tried.append((s, i))
            print(f"  without {i:>2}: {s:.2f} ({s - base:+.2f})", flush=True)
        s, i = min(tried)
        tol = float(arg("--tolerance", "0.0003")) * 1000
        print(f"next: --num {keep & ~(1 << i):x} (without {i}: {s:.2f})" if s <= base + tol else "stop: every context left earns its place")
        return
    if "--grow" in sys.argv:
        # the other direction: what each context left out would add
        n = int(re.search(r"pub const CONTEXTS: u32 = (\d+);", open(os.path.join(ROOT, "src", "cm", "num.rs")).read()).group(1))
        keep = mask if mask is not None else 3 << 62
        base = score(total(ex, keep))
        print(f"{keep:x}: {base:.2f} (mean size, per mille of '{ref_name}')", flush=True)
        tried = []
        for i in [i for i in range(n) if not keep >> i & 1]:
            s = score(total(ex, keep | 1 << i))
            tried.append((s, i))
            print(f"  with {i:>2}: {s:.2f} ({s - base:+.2f})", flush=True)
        s, i = min(tried)
        tol = float(arg("--tolerance", "0.0003")) * 1000
        print(f"next: --num {keep | 1 << i:x} (with {i}: {s:.2f})" if s < base - tol else "stop: no context left out is worth adding")
        return
    res = total(ex, mask, "--no-num" in sys.argv)
    if arg("--save"):
        json.dump(res, open(os.path.join(results, f"num_dev_{arg('--save')}.json"), "w"), indent=1)
    tot = sum(res.values())
    for f in FILES:
        line = f"{f:<40} {os.path.getsize(head(f)):>8} -> {res[f]:>7}"
        if ref:
            line += f"  {100 * (res[f] / ref[f] - 1):+6.2f}%"
        print(line)
    if ref:
        tb = sum(ref[f] for f in FILES)
        print(f"{'TOTAL':<40} {'':>8}    {tot:>7}  {100 * (tot / tb - 1):+6.2f}% vs {ref_name};  mean size {score(res):.2f} per mille")
    else:
        print(f"TOTAL {tot}")


if __name__ == "__main__":
    main()
