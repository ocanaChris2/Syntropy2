#!/usr/bin/env python3
"""Where the bits of a SQLite database go: cost per kind of byte (page headers, cell pointers,
cell headers, record headers, values by serial type, free space, index pages, overflow).

    python3 bench/sqlite_costs.py <file.sqlite> [--level 5] [--no-bin]

Runs `nsc cm-test --seg generic --costs` on the file and walks its pages as SQLite lays them out.
"""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NSC = os.path.join(os.path.dirname(HERE), "target", "release", "nsc")


def varint(b, o):
    v = 0
    for k in range(9):
        x = b[o + k]
        if k == 8:
            return (v << 8) | x, 9
        v = (v << 7) | (x & 0x7F)
        if not x & 0x80:
            return v, k + 1


def serial_len(t):
    return [0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0][t] if t < 12 else (t - 12) // 2


def kind_of(t):
    return "null" if t == 0 else f"int{serial_len(t)}" if t <= 6 else "real" if t == 7 else "const" if t <= 9 else "blob" if t % 2 == 0 else "text"


def main():
    path = sys.argv[1]
    data = open(path, "rb").read()
    with tempfile.NamedTemporaryFile(suffix=".costs") as t:
        extra = [a for a in sys.argv[2:] if a.startswith("--")]
        out = subprocess.run([NSC, "cm-test", path, "--seg", "generic", "--costs", t.name] + extra, capture_output=True, text=True).stdout
        print(out.strip())
        raw = open(t.name, "rb").read()
    cost = struct.unpack(f"<{len(raw) // 4}f", raw)
    page = struct.unpack(">H", data[16:18])[0]
    page = 65536 if page == 1 else page
    u = page - data[20]
    tag = [None] * len(data)

    def mark(a, b, name):
        for i in range(a, min(b, len(data))):
            if tag[i] is None:
                tag[i] = name

    for p in range(len(data) // page):
        at = p * page
        h = at + (100 if p == 0 else 0)
        if p == 0:
            mark(0, 100, "file header")
        ty = data[h]
        if ty not in (0x0D, 0x05, 0x0A, 0x02):
            mark(at, at + page, "overflow / free / other pages")
            continue
        head = 12 if ty in (0x05, 0x02) else 8
        cells = struct.unpack(">H", data[h + 3:h + 5])[0]
        mark(h, h + head, "page header")
        mark(h + head, h + head + 2 * cells, "cell pointers")
        index = ty in (0x0A, 0x02)
        most = (u - 12) * 64 // 255 - 23 if index else u - 35
        least = (u - 12) * 32 // 255 - 23
        name = {0x0D: "table", 0x05: "table interior", 0x0A: "index", 0x02: "index interior"}[ty]
        for c in range(cells):
            ptr = struct.unpack(">H", data[h + head + 2 * c:h + head + 2 * c + 2])[0]
            o = at + ptr
            try:
                if ty in (0x05, 0x02):
                    mark(o, o + 4, f"{name}: child page")
                    o += 4
                if ty == 0x05:
                    _, n = varint(data, o)
                    mark(o, o + n, f"{name}: key")
                    continue
                size, n = varint(data, o)
                mark(o, o + n, f"{name}: payload size")
                o += n
                if ty == 0x0D:
                    _, n = varint(data, o)
                    mark(o, o + n, f"{name}: row id")
                    o += n
                local = size
                if size > most:
                    k = least + (size - least) % (u - 4)
                    local = k if k <= most else least
                end = o + local
                hl, n = varint(data, o)
                hend = o + hl
                mark(o, o + n, f"{name}: record header size")
                q = o + n
                types = []
                while q < hend and q < end:
                    t, n = varint(data, q)
                    mark(q, q + n, f"{name}: serial types")
                    types.append(t)
                    q += n
                q = hend
                for col, t in enumerate(types):
                    l = serial_len(t)
                    mark(q, min(q + l, end), f"{name}: {kind_of(t)} values")
                    q += l
                if size > most:
                    mark(end, end + 4, f"{name}: overflow pointer")
            except (IndexError, struct.error, TypeError):
                pass
        mark(at, at + page, "free space in b-tree pages")
    tot = {}
    for i, name in enumerate(tag):
        a = tot.setdefault(name or "?", [0, 0.0])
        a[0] += 1
        a[1] += cost[i]
    total = sum(cost)
    print(f"{len(data)} bytes in {len(data) // page} pages of {page}; {total / 8:.0f} bytes coded")
    print(f"{'kind of byte':<34} {'bytes':>9} {'coded':>9} {'bits/byte':>9} {'share':>6}")
    for name, (n, b) in sorted(tot.items(), key=lambda x: -x[1][1]):
        print(f"{name:<34} {n:>9} {b / 8:>9.0f} {b / n:>9.3f} {100 * b / total:>5.1f}%")


if __name__ == "__main__":
    main()
