#!/usr/bin/env python3
"""Development loop for the executable models: compress a fixed set of executables and compare
with a saved baseline.

    python3 bench/exe_dev.py [--save base] [--against base] [--level 5] [--jobs 2] [--nsc path] [--big]

`--big` adds Silesia's mozilla and samba (slow).
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


NSC = arg("--nsc", os.path.join(ROOT, "target", "release", "nsc"))
LEVEL = arg("--level", "5")
FILES = [f"formats/elf/{n}" for n in ["VBoxClient", "Xephyr", "Xvfb", "Xwayland", "libLerc.a"]] + \
        [f"corpus/exe/{n}" for n in ["git.elf", "libc.so.6", "python3.12.elf", "zlib1_x64.dll", "zlib1_x86.dll"]] + \
        [f"formats/pe/{n}" for n in sorted(os.listdir(os.path.join(HERE, "formats", "pe")))] + ["corpus/silesia/ooffice"]
if "--big" in sys.argv:
    FILES += ["corpus/silesia/mozilla", "corpus/silesia/samba"]


def size(rel):
    """Compressed size, or None when the file does not come back exactly (always checked: a size
    measured without decoding it is not a result)."""
    path = os.path.join(HERE, rel)
    with tempfile.NamedTemporaryFile(suffix=".nsc") as t, tempfile.NamedTemporaryFile(suffix=".out") as back:
        r = subprocess.run([NSC, "compress", path, t.name, f"-{LEVEL}", "--force", "--jobs", "1", "-q"], capture_output=True, text=True)
        if r.returncode != 0:
            return rel, None
        d = subprocess.run([NSC, "decompress", t.name, back.name, "--force", "--jobs", "1", "-q"], capture_output=True, text=True)
        if d.returncode != 0 or subprocess.run(["cmp", "-s", path, back.name]).returncode != 0:
            return rel, None
        return rel, os.path.getsize(t.name)


def main():
    with concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "2"))) as ex:
        res = dict(ex.map(size, FILES))
    base = {}
    name = arg("--against", "base")
    path = os.path.join(HERE, "results", f"exe_dev_{name}.json")
    if os.path.exists(path):
        base = json.load(open(path))
    tot = tot_b = tot_raw = 0
    for rel in FILES:
        raw = os.path.getsize(os.path.join(HERE, rel))
        n, b = res[rel], base.get(rel)
        if n is None:
            print(f"{rel:<40} ROUNDTRIP FAILED")
            continue
        d = f"{100 * (n / b - 1):+6.2f}%" if b else "      -"
        print(f"{rel:<40} {raw:>9} -> {n:>9}  ({100 * (1 - n / raw):5.2f}% saved)  {d}")
        tot += n
        tot_raw += raw
        tot_b += b or n
    print(f"{'TOTAL':<40} {tot_raw:>9} -> {tot:>9}  ({100 * (1 - tot / tot_raw):5.2f}% saved)  {100 * (tot / tot_b - 1):+6.2f}% vs {name}")
    if "--save" in sys.argv:
        json.dump(res, open(os.path.join(HERE, "results", f"exe_dev_{arg('--save')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
