#!/usr/bin/env python3
"""Development loop for the generic binary model: code each file of bench/sections (data sections
of executables, heads of table-like corpus files) as one generic segment and compare with a saved
baseline and with paq8px (bench/results/sections_ref.json).

    python3 bench/bin_dev.py [--save name] [--against name] [--jobs 3] [--level 5] [--bin HEX] [--no-bin]
    python3 bench/bin_dev.py --ablate [--greedy] [--tolerance 0.0005]

`--bin HEX` keeps only those contexts of the binary model (ablation); `--no-bin` leaves it off.
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
DIR = os.path.join(HERE, "sections")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


LEVEL = arg("--level", "5")
FILES = sorted(f for f in os.listdir(DIR) if f.endswith(".bin"))


def size(f, mask=None):
    cmd = [NSC, "cm-test", os.path.join(DIR, f), "--level", LEVEL, "--seg", "generic", "--no-decode"]
    if "--no-bin" in sys.argv:
        cmd.append("--no-bin")
    if mask is not None:
        cmd += ["--bin", f"{mask:x}"]

    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"-> (\d+) B", out)
    return int(m.group(1)) if m else None


def total(ex, mask=None):
    return dict(zip(FILES, ex.map(lambda f: size(f, mask), FILES)))


def main():
    ex = concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "3")))
    if "--ablate" in sys.argv:
        n = int(re.search(r"pub const BIN2: u32 = (\d+);", open(os.path.join(ROOT, "src", "cm", "predictor.rs")).read()).group(1))
        keep = int(arg("--bin", "0"), 16) or (1 << n) - 1
        # the measure: every file counts the same (its size against the saved baseline), so the
        # megabyte files do not drown the small sections
        ref = json.load(open(os.path.join(HERE, "results", f"bin_dev_{arg('--against', 'base')}.json")))
        score = lambda res: 1000 * sum(res[f] / ref[f] for f in FILES) / len(FILES)
        base = score(total(ex, keep))
        print(f"all: {base:.2f} (mean size, per mille of the baseline)", flush=True)
        while True:
            ids = [i for i in range(n) if keep >> i & 1]
            sizes = [score(total(ex, keep & ~(1 << i))) for i in ids]
            for i, s in zip(ids, sizes):
                print(f"  without {i:>2}: {s:.2f} ({100 * (s / base - 1):+.3f}%)", flush=True)
            best = min(zip(sizes, ids))
            if "--greedy" not in sys.argv or best[0] > base * (1 + float(arg("--tolerance", "0.0005"))) or len(ids) <= 1:
                break
            keep &= ~(1 << best[1])
            base = best[0]
            print(f"dropped {best[1]}: {base:.2f}, mask {keep:x}", flush=True)
        print(f"mask {keep:x}")
        return
    mask = int(arg("--bin"), 16) if arg("--bin") else None
    res = total(ex, mask)
    name = arg("--against", "base")
    path = os.path.join(HERE, "results", f"bin_dev_{name}.json")
    base = json.load(open(path)) if os.path.exists(path) else {}
    ref = json.load(open(os.path.join(HERE, "results", "sections_ref.json")))
    tot = tb = tr = tn = 0
    for f in FILES:
        n, b, r = res[f], base.get(f), ref.get(f)
        raw = os.path.getsize(os.path.join(DIR, f))
        d = f"{100 * (n / b - 1):+6.2f}%" if b else "      -"
        q = f"paq8px {r:>7} ({100 * (n / r - 1):+5.1f}%)" if r else ""
        print(f"{f:<34} {raw:>8} -> {n:>7}  {d}  {q}")
        tot += n
        tb += b or n
        if r:
            tr += r
            tn += n
    print(f"{'TOTAL':<34} {'':>8}    {tot:>7}  {100 * (tot / tb - 1):+6.2f}% vs {name};  on the files with a reference: {tn} vs paq8px {tr} ({100 * (tn / tr - 1):+.1f}%)")
    if "--save" in sys.argv:
        json.dump(res, open(os.path.join(HERE, "results", f"bin_dev_{arg('--save')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
