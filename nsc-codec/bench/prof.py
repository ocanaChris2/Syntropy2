#!/usr/bin/env python3
"""Symbolise an NSC_PROFILE sample file: python3 bench/prof.py <binary-with-debug-info> <samples> [N]

Prints the hottest inlined functions and source lines. Build the binary with full debug info in a
separate target dir so the normal build is untouched:
    CARGO_PROFILE_RELEASE_DEBUG=2 cargo build --release --target-dir target-prof
"""
import collections
import subprocess
import sys

binary, samples = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
offs = [l.strip() for l in open(samples) if l.strip()]
uniq = sorted(set(offs))
res = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", binary] + ["0x" + o for o in uniq],
                     capture_output=True, text=True).stdout.split("\n")
# addr2line -i prints (function, file:line) pairs; inline chains add extra pairs. We need the chain
# per address, so re-run per address in batches with a sentinel.
func_of, line_of = {}, {}
batch = 400
for i in range(0, len(uniq), batch):
    chunk = uniq[i:i + batch]
    args = []
    for o in chunk:
        args += ["0x" + o, "0x0"]  # 0x0 resolves to ??, used as a separator
    out = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", binary] + args, capture_output=True, text=True).stdout.split("\n")
    j = 0
    for o in chunk:
        chain = []
        while j + 1 < len(out):
            fn, loc = out[j], out[j + 1]
            j += 2
            if fn == "??" and loc.startswith("??"):
                break
            chain.append((fn, loc))
        if chain:
            func_of[o] = chain[0][0]
            line_of[o] = chain[0][1].split(" (")[0]
cnt = collections.Counter(offs)
total = len(offs)
by_fn = collections.Counter()
by_line = collections.Counter()
for o, c in cnt.items():
    f = func_of.get(o, "??")
    f = f.split("<")[0][-70:] if len(f) > 70 else f
    by_fn[f] += c
    by_line[line_of.get(o, "??").split("/src/")[-1]] += c
print(f"{total} samples\n--- by (innermost) function")
for f, c in by_fn.most_common(top):
    print(f"{100 * c / total:6.2f}%  {f}")
print("--- by source line")
for l, c in by_line.most_common(top):
    print(f"{100 * c / total:6.2f}%  {l}")
