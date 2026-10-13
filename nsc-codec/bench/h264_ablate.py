#!/usr/bin/env python3
"""Ablation of the H.264 model's context inputs: for each (family, input), the model's size with
that input blanked, against the full model.

    python3 bench/h264_ablate.py file.mp4 [--jobs 4] [--families 15,16]
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


def run(files, drop=None):
    cmd = [NSC, "h264-test"] + files + (["--drop", drop] if drop else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    tot = {}
    for line in out.split("\n"):
        m = re.match(r"  (\S.*?)\s+[\d.]+% of bits\s+(\d+) ->\s+(\d+) B", line)
        if m:
            tot[m.group(1)] = int(m.group(3))
    return tot


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.exists(a)]
    nin = [int(x) for x in re.search(r"pub const NIN: \[usize; FAMILIES\] = \[([^\]]+)\]", open(os.path.join(os.path.dirname(HERE), "src", "cm", "h264.rs")).read()).group(1).split(",")]
    fams = [int(x) for x in arg("--families", ",".join(str(i) for i in range(len(NAMES)))).split(",")]
    base = run(files)
    total = sum(base.values())
    print(f"full model: {total} B")
    jobs = [(f, i) for f in fams for i in range(nin[f])]
    with concurrent.futures.ThreadPoolExecutor(int(arg("--jobs", "4"))) as ex:
        res = list(ex.map(lambda fi: run(files, f"{fi[0]}:{fi[1]}"), jobs))
    for (f, i), r in zip(jobs, res):
        name = NAMES[f]
        d = r.get(name, 0) - base.get(name, 0)
        print(f"{name:<18} input {i:>2}: {d:+8d} B  ({100 * d / max(base.get(name, 1), 1):+6.2f}% of the element, {100 * d / total:+6.3f}% of all)", flush=True)


if __name__ == "__main__":
    main()
