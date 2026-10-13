#!/usr/bin/env python3
"""Run the H.264 parser over the conformance bitstreams (see fetch.py).

    python3 bench/conformance/run.py [--nsc target/release/nsc]

A stream passes when every slice the parser takes re-encodes exactly; streams it does not cover
(interlaced, CAVLC, 4:2:2 / 4:4:4) must be refused whole, with the reason. Exit status 1 on any
slice that is parsed but not reproduced.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "target", "release", "nsc")
if "--nsc" in sys.argv:
    NSC = sys.argv[sys.argv.index("--nsc") + 1]


def main():
    d = os.path.join(HERE, "h264")
    files = sorted(os.path.join(d, f) for f in os.listdir(d))
    out = subprocess.run([NSC, "h264-test"] + files, capture_output=True, text=True).stdout.split("\n")
    bad = taken = refused = 0
    reasons = {}
    for i, line in enumerate(out):
        m = re.match(r"(\S+)\s+slices\s+(\d+) \((\d+) B\): parsed (\d+), exact (\d+), unsupported (\d+)\s+mbs (\d+) \((\d+) PCM\)", line)
        if not m:
            continue
        name, slices, _, parsed, exact, unsup, _, pcm = m.groups()
        slices, parsed, exact, unsup = int(slices), int(parsed), int(exact), int(unsup)
        why = []
        for l in out[i + 1:]:
            if not l.startswith("    "):
                break
            if re.match(r"\s+\d+ x ", l):
                why.append(l.strip())
        if parsed != exact or parsed + unsup != slices:
            bad += 1
            print(f"FAIL  {name}: {slices} slices, {parsed} parsed, {exact} exact, {unsup} unsupported  {why}")
        elif parsed:
            taken += 1
            print(f"ok    {name}: {exact} slices exact{', ' + pcm + ' PCM macroblocks' if pcm != '0' else ''}{'  (' + '; '.join(why) + ')' if why else ''}")
        else:
            refused += 1
            for w in why:
                reasons[w.split("unsupported: ")[-1]] = reasons.get(w.split("unsupported: ")[-1], 0) + 1
            print(f"skip  {name}: {'; '.join(why)}")
    print(f"\n{taken} streams exact, {refused} refused ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))}), {bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
