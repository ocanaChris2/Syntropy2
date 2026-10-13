#!/usr/bin/env python3
"""Greedy backward elimination of the H.264 model's context inputs, all families at once
(a family's cost depends only on its own inputs, so one run tries one removal in every family).

    python3 bench/h264_prune.py files... [--jobs 4] [--tolerance 0.0004] [--keep 15:13,...]

Stops removing from a family when the cheapest removal costs more than `tolerance` of the
family's own size. Prints, per family, the inputs that can go (in order) and what each cost.
"""
import concurrent.futures
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(HERE), "target", "release", "nsc")
NAMES = ["mb skip", "mb type", "sub mb type", "ref idx", "mvd", "mvd suffix", "mvd sign", "qp delta", "intra pred flag", "intra pred mode", "chroma pred",
         "cbp luma", "cbp chroma", "transform 8x8", "coded block flag", "significance", "last", "level", "level suffix", "sign", "end of slice"]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run(files, drops):
    cmd = [NSC, "h264-test"] + files + (["--drop", ",".join(f"{f}:{i}" for f, i in drops)] if drops else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    tot = {}
    for line in out.split("\n"):
        m = re.match(r"  (\S.*?)\s+[\d.]+% of bits\s+(\d+) ->\s+(\d+) B", line)
        if m:
            tot[NAMES.index(m.group(1))] = int(m.group(3))
    return tot


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.exists(a)]
    tol = float(arg("--tolerance", "0.0004"))
    nin = [int(x) for x in re.search(r"pub const NIN: \[usize; FAMILIES\] = \[([^\]]+)\]", open(os.path.join(os.path.dirname(HERE), "src", "cm", "h264.rs")).read()).group(1).split(",")]
    left = {f: list(range(nin[f])) for f in range(len(NAMES))}
    gone = {f: [] for f in left}
    done = set()
    base = run(files, [])
    start = dict(base)
    print(f"full model: {sum(base.values())} B", flush=True)
    ex = concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "4")))
    while len(done) < len(left):
        removed = [(f, i) for f in gone for i, _ in gone[f]]
        # trial k: in every live family, also drop its k-th remaining input
        width = max(len(left[f]) for f in left if f not in done)
        trials = [[(f, left[f][k]) for f in left if f not in done and k < len(left[f])] for k in range(width)]
        res = list(ex.map(lambda t: run(files, removed + t), trials))
        for f in list(left):
            if f in done:
                continue
            costs = [(res[k][f], left[f][k]) for k in range(len(left[f])) if f in res[k]]
            if not costs or len(left[f]) <= 1:
                done.add(f)
                continue
            best, i = min(costs)
            if best - base[f] > tol * start[f]:
                done.add(f)
                continue
            gone[f].append((i, best - base[f]))
            left[f].remove(i)
            base[f] = best
        print("  step:", {NAMES[f]: [i for i, _ in gone[f]] for f in gone if gone[f]}, flush=True)
    print(f"pruned model: {sum(base.values())} B ({100 * (sum(base.values()) / sum(start.values()) - 1):+.3f}%)")
    for f in left:
        print(f"{NAMES[f]:<18} keep {left[f]}  drop {[(i, d) for i, d in gone[f]]}  {start[f]} -> {base[f]} B")


if __name__ == "__main__":
    main()
