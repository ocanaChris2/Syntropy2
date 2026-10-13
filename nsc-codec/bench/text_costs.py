#!/usr/bin/env python3
"""Where the bits of a delimited text file go: cost per field (column) and per kind of character.

    python3 bench/text_costs.py <file> [--sep ,] [--level 5] [--lines 12] [--at LINE]

Runs `nsc cm-test --costs` and sums what each byte cost by the field of its line it is in (fields
split at `--sep`; default: the commonest of , ; tab | in the first lines, else space), then prints
a few lines with the bits each field cost.
"""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(HERE), "target", "release", "nsc")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    path = sys.argv[1]
    data = open(path, "rb").read()
    with tempfile.NamedTemporaryFile(suffix=".costs") as t:
        cmd = [NSC, "cm-test", path, "--level", arg("--level", "5"), "--costs", t.name] + (["--seg", arg("--seg")] if arg("--seg") else [])
        print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
        raw = open(t.name, "rb").read()
    costs = struct.unpack(f"<{len(raw) // 4}f", raw)
    head = data[:20000]
    sep = arg("--sep")
    if sep is None:
        best = max(b",;\t|", key=lambda c: head.count(bytes([c])))
        sep = chr(best) if head.count(bytes([best])) > head.count(b"\n") else " "
    sep = sep.encode().decode("unicode_escape").encode()
    fields = {}
    kinds = {}
    lines = []
    f, start, line = 0, 0, []
    for i, c in enumerate(data):
        k = "digit" if 48 <= c <= 57 else "letter" if chr(c).isalpha() else "newline" if c == 10 else "separator" if bytes([c]) == sep else "other"
        a = kinds.setdefault(k, [0, 0.0])
        a[0] += 1
        a[1] += costs[i]
        a = fields.setdefault(min(f, 40), [0, 0.0])
        a[0] += 1
        a[1] += costs[i]
        if bytes([c]) == sep or c == 10:
            line.append((data[start:i + 1], sum(costs[start:i + 1])))
            start = i + 1
            f += 1
        if c == 10:
            lines.append(line)
            f, line = 0, []
    total = sum(costs)
    print(f"{len(data)} bytes, {total / 8:.0f} bytes coded, {len(lines)} lines, {total / max(1, len(lines)):.1f} bits a line, separator {sep!r}")
    print("field   bytes   bits/line  share")
    for f in sorted(fields):
        n, b = fields[f]
        print(f"{f:>5} {n:>8} {b / max(1, len(lines)):>9.2f} {100 * b / total:>6.1f}%")
    print("kind        bytes  bits/byte  share")
    for k, (n, b) in sorted(kinds.items(), key=lambda x: -x[1][1]):
        print(f"{k:<10} {n:>7} {b / n:>9.3f} {100 * b / total:>6.1f}%")
    # digits by their place in a number (runs of 8 digits or more), and the byte that ends one
    place, after, i = {}, [0, 0.0], 0
    while i < len(data):
        if 48 <= data[i] <= 57:
            j = i
            while j < len(data) and 48 <= data[j] <= 57:
                j += 1
            if j - i >= 8:
                for k in range(i, j):
                    a = place.setdefault(min(k - i, 24), [0, 0.0])
                    a[0] += 1
                    a[1] += costs[k]
                if j < len(data):
                    after[0] += 1
                    after[1] += costs[j]
            i = j
        else:
            i += 1
    if place:
        print("long numbers: bits per digit by place:", " ".join(f"{k}:{b / n:.2f}" for k, (n, b) in sorted(place.items())),
              f"| the byte after: {after[1] / max(1, after[0]):.2f} bits ({after[0]} numbers)")
    at = int(arg("--at", str(len(lines) // 2)))
    for line in lines[at:at + int(arg("--lines", "12"))]:
        print(" ".join(f"{t.decode(errors='replace').rstrip()}[{b:.1f}]" for t, b in line))


if __name__ == "__main__":
    main()
