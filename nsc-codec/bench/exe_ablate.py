#!/usr/bin/env python3
"""Which of the x86 model's contexts earn their place: code-only size with each one dropped
(and optionally greedy elimination).

    python3 bench/exe_ablate.py files... [--jobs 2] [--greedy] [--level 5]
"""
import concurrent.futures
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(HERE), "target", "release", "nsc")
N = 14


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run(files, mask):
    out = subprocess.run([NSC, "exe-test"] + files + ["--x86-mask", f"{mask:x}", "--level", arg("--level", "5")], capture_output=True, text=True).stdout
    return int(re.search(r"TOTAL code \d+ -> (\d+) B", out).group(1))


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.exists(a)]
    ex = concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "2")))
    keep = (1 << N) - 1
    base = run(files, keep)
    print(f"all contexts: {base} B   none: {run(files, 0)} B", flush=True)
    while True:
        ids = [i for i in range(N) if keep >> i & 1]
        sizes = list(ex.map(lambda i: run(files, keep & ~(1 << i)), ids))
        for i, s in zip(ids, sizes):
            print(f"  without {i:>2}: {s} B ({100 * (s / base - 1):+.3f}%)", flush=True)
        best = min(zip(sizes, ids))
        if "--greedy" not in sys.argv or best[0] > base * (1 + float(arg("--tolerance", "0.0005"))) or len(ids) <= 1:
            break
        keep &= ~(1 << best[1])
        base = best[0]
        print(f"dropped {best[1]}: {base} B, mask {keep:x}", flush=True)
    print(f"mask {keep:x}")


if __name__ == "__main__":
    main()
