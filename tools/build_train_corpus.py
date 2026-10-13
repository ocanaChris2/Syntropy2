#!/usr/bin/env python3
"""Build the round-2 mixed-format TRAINING corpus under data/corpus_r2/ — deterministic.

    python3 tools/build_train_corpus.py [--no-network] [--out DIR]

Output:
  corpus_r2.bin    one file, physically ordered [train][val][test]; inside each split the
                   domains appear as code, docs, structured, prose, wiki (wiki last so the
                   enwik8 pieces start after a newline and end at the split end)
  corpus_r2.json   {"splits": {split: [offset, length]}, "domains": {split: [[name, offset, length]]}}
  MANIFEST.tsv     source path, bytes, crc32, domain, split

Rules:
  * wiki = enwik8 with its historical split: train [0, 90M), test [95M, 100M) (so the test
    bpb stays comparable with the base tier); val is the first 1.5 MB of [90M, 95M).
  * every other domain is split by WHOLE FILES (seeded shuffle): val/test never share a file
    with train.
  * nothing from the nsc benchmark corpus (nsc-codec/bench/corpus) is used: files are dropped
    by crc32 against its MANIFEST.tsv, /usr/share/doc/**/*.gz (bench deflate set) is never
    read, and the generated logs/tables use other schemas and seeds than the bench generators.
  * system files only + public-domain books (Project Gutenberg); no user-owned files.
"""
import glob
import gzip
import json
import os
import random
import sys
import time
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "data", "corpus_r2")
if "--out" in sys.argv:
    OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
CACHE = os.path.join(OUT, "cache")
NET = "--no-network" not in sys.argv
BENCH_MANIFEST = os.path.join(REPO, "nsc-codec", "bench", "corpus", "MANIFEST.tsv")
ENWIK8 = os.path.join(REPO, "data", "enwik8")
UA = {"User-Agent": "syntropy2-corpus/0.1 (local compression research)"}

MB = 1 << 20
MIN_FILE, MAX_FILE = 512, 4 * MB
VAL_BYTES, TEST_BYTES = 768 << 10, 1536 << 10        # per non-wiki domain
WIKI_TRAIN, WIKI_VAL, WIKI_TEST = (0, 90_000_000), (90_000_000, 91_500_000), (95_000_000, 100_000_000)
SEED = 20261003

# (domain, source label, quota bytes, glob patterns)
RUST_SRC = os.path.expanduser("~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src")
SOURCES = [
    ("code", "python", 26 * MB, ["/usr/lib/python3.12/**/*.py", "/usr/lib/python3/dist-packages/**/*.py"]),
    ("code", "c_headers", 24 * MB, ["/usr/include/**/*.h", "/usr/include/**/*.hpp"]),
    ("code", "rust", 20 * MB, [RUST_SRC + "/**/*.rs"]),
    ("code", "javascript", 10 * MB, ["/usr/share/javascript/**/*.js"]),
    ("docs", "doc_text", 22 * MB, ["/usr/share/doc/**/*.html", "/usr/share/doc/**/*.txt", "/usr/share/doc/**/*.md",
                                   "/usr/share/doc/**/README*", "/usr/share/doc/**/*.rst"]),
    ("docs", "man", 28 * MB, ["/usr/share/man/man*/*.gz"]),
    ("structured", "xml", 14 * MB, ["/usr/share/**/*.xml"]),
    ("structured", "json", 8 * MB, ["/usr/share/**/*.json"]),
    ("structured", "config", 8 * MB, ["/usr/share/**/*.desktop", "/usr/share/**/*.css", "/usr/share/**/*.yaml",
                                      "/usr/share/**/*.yml", "/usr/share/**/*.conf", "/usr/share/**/*.ini"]),
]
GENERATED_BYTES = 10 * MB        # structured: synthetic logs / JSON lines / CSV
PROSE_BYTES = 30 * MB

# Project Gutenberg plain-text ids. No Dickens (Silesia "dickens" is in the benchmark corpus),
# no dictionary (Silesia "webster").
GUTENBERG = [1342, 2701, 84, 11, 1661, 2600, 145, 1399, 135, 28054, 4300, 996, 2000, 1184, 100, 10,
             3207, 4280, 1497, 6130, 1727, 345, 174, 76, 74, 5200, 2554, 1260, 768, 158, 161, 2591,
             1232, 3600, 8800, 4363, 7370, 3300, 1228, 25344, 514, 205, 16328, 2852, 244, 120, 43,
             35, 36, 829, 1998, 2680, 3296, 17489, 2413]


def log(msg):
    print(msg, flush=True)


def bench_crcs():
    crcs = set()
    if os.path.exists(BENCH_MANIFEST):
        for line in open(BENCH_MANIFEST).read().splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 3:
                crcs.add((int(f[1]), f[2]))
    return crcs


def looks_text(data):
    if b"\0" in data:
        return False
    ctrl = sum(1 for b in data[:65536] if b < 9 or (13 < b < 32 and b != 27))
    return ctrl * 200 <= min(len(data), 65536)        # < 0.5 % stray control bytes


def read_source(p):
    """File bytes (gunzipped for man pages), or None if unusable."""
    try:
        if os.path.islink(p) or not os.path.isfile(p):
            return None
        sz = os.path.getsize(p)
        if sz < 64 or sz > MAX_FILE:
            return None
        raw = open(p, "rb").read()
        if p.endswith(".gz"):
            raw = gzip.decompress(raw)
    except Exception:
        return None
    if not (MIN_FILE <= len(raw) <= MAX_FILE) or not looks_text(raw):
        return None
    return raw if raw.endswith(b"\n") else raw + b"\n"


def collect(label, quota, patterns, seen, excluded):
    """Seeded-shuffled files of one source up to `quota` bytes -> [(path, data, crc)]."""
    paths = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    # the bench deflate set is built from /usr/share/doc/**/*.gz — never train on those
    paths = [p for p in paths if not (p.startswith("/usr/share/doc/") and p.endswith(".gz"))]
    if label == "javascript":
        paths = [p for p in paths if ".min." not in os.path.basename(p)]
    random.Random(f"{SEED}:{label}").shuffle(paths)
    out, total = [], 0
    for p in paths:
        if total >= quota:
            break
        data = read_source(p)
        if data is None:
            continue
        crc = "%08x" % zlib.crc32(data)
        key = (len(data), crc)
        if key in seen or key in excluded:
            continue
        seen.add(key)
        out.append((p, data, crc))
        total += len(data)
    log(f"  {label:12s} {len(out):6d} files  {total / MB:7.1f} MB")
    return out


def generated():
    """Synthetic structured text. Schemas/seeds differ from bench/build_corpus.py (app.log, sales.csv)."""
    rng = random.Random(SEED)
    per = GENERATED_BYTES // 4
    files = []

    # 1. web-server access log
    paths = ["/", "/index.html", "/api/v2/items", "/api/v2/items/{}", "/static/app.{}.js", "/login",
             "/search?q={}", "/img/{}.png", "/feed.xml", "/health"]
    agents = ["Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/131.0", "curl/8.5.0",
              "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 Safari/605.1.15",
              "python-requests/2.32.3", "Googlebot/2.1 (+http://www.google.com/bot.html)"]
    lines, size, t = [], 0, 1_750_000_000
    while size < per:
        t += rng.randint(0, 7)
        tm = time.gmtime(t)
        ln = (f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)} - - "
              f"[{time.strftime('%d/%b/%Y:%H:%M:%S +0000', tm)}] \"{rng.choice(['GET'] * 8 + ['POST', 'PUT'])} "
              f"{rng.choice(paths).format(rng.randint(1, 99999))} HTTP/1.1\" "
              f"{rng.choice([200] * 12 + [301, 304, 304, 403, 404, 404, 500, 502])} {rng.randint(120, 480000)} "
              f"\"-\" \"{rng.choice(agents)}\"")
        lines.append(ln)
        size += len(ln) + 1
    files.append(("generated/access.log", ("\n".join(lines) + "\n").encode()))

    # 2. JSON-lines telemetry events
    kinds = ["page_view", "click", "purchase", "signup", "error", "heartbeat"]
    lines, size = [], 0
    while size < per:
        ev = {"ts": 1_750_000_000_000 + len(lines) * 137 + rng.randint(0, 99), "event": rng.choice(kinds),
              "user": f"u{rng.randint(1, 50000):05d}", "session": "%016x" % rng.getrandbits(64),
              "props": {"dur_ms": rng.randint(1, 30000), "ok": rng.random() > 0.07,
                        "region": rng.choice(["eu-west", "us-east", "ap-south", "sa-east"]),
                        "value": round(rng.random() * 500, 2)}}
        ln = json.dumps(ev, separators=(",", ":"))
        lines.append(ln)
        size += len(ln) + 1
    files.append(("generated/events.jsonl", ("\n".join(lines) + "\n").encode()))

    # 3. syslog-style daemon log
    procs = ["sshd", "systemd", "kernel", "cron", "NetworkManager", "dockerd", "postfix/smtpd"]
    tmpl = ["Accepted publickey for user{} from 10.0.{}.{} port {}", "Started Session {} of User user{}.",
            "eth0: link up, {} Mbps, full duplex", "(root) CMD (/usr/local/bin/job-{} --quiet)",
            "connection from unknown[{}.{}.{}.{}] lost", "container {} exited with status {}",
            "Out of memory: Killed process {} (worker-{})", "device state change: {} -> {}"]
    lines, size, t = [], 0, 1_750_000_000
    while size < per:
        t += rng.randint(0, 40)
        m = rng.choice(tmpl).format(*[rng.randint(0, 65535) for _ in range(4)])
        ln = (f"{time.strftime('%b %d %H:%M:%S', time.gmtime(t))} host{rng.randint(1, 12):02d} "
              f"{rng.choice(procs)}[{rng.randint(100, 99999)}]: {m}")
        lines.append(ln)
        size += len(ln) + 1
    files.append(("generated/syslog.log", ("\n".join(lines) + "\n").encode()))

    # 4. CSV sensor readings
    rows, size = ["station,timestamp,temp_c,humidity,pressure_hpa,wind_ms,status"], 0
    stations = [f"ST{n:03d}" for n in range(40)]
    t = 1_750_000_000
    while size < per:
        t += 60
        ln = (f"{rng.choice(stations)},{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))},"
              f"{rng.gauss(14, 9):.1f},{rng.randint(20, 100)},{rng.gauss(1013, 8):.1f},"
              f"{abs(rng.gauss(4, 3)):.1f},{rng.choice(['ok'] * 20 + ['cal', 'fault'])}")
        rows.append(ln)
        size += len(ln) + 1
    files.append(("generated/sensors.csv", ("\n".join(rows) + "\n").encode()))

    # cut each generated stream into 256 KB line-aligned pieces so val/test get whole "files"
    out = []
    for name, data in files:
        i, k = 0, 0
        while i < len(data):
            j = min(len(data), i + (256 << 10))
            if j < len(data):
                j = data.rfind(b"\n", i, j) + 1
            out.append((f"{name}#{k:03d}", data[i:j], "%08x" % zlib.crc32(data[i:j])))
            i, k = j, k + 1
    log(f"  {'generated':12s} {len(out):6d} files  {sum(len(d) for _, d, _ in out) / MB:7.1f} MB")
    return out


def gutenberg():
    os.makedirs(CACHE, exist_ok=True)
    out, total = [], 0
    for gid in GUTENBERG:
        if total >= PROSE_BYTES:
            break
        p = os.path.join(CACHE, f"pg{gid}.txt")
        if not os.path.exists(p) and NET:
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                with open(p + ".part", "wb") as f:
                    f.write(data)
                os.replace(p + ".part", p)
                time.sleep(1.0)
            except Exception as e:
                log(f"  [prose] pg{gid}: download failed ({e}); skipped")
                continue
        if not os.path.exists(p):
            continue
        data = open(p, "rb").read().replace(b"\r\n", b"\n")
        if len(data) < 20_000 or not looks_text(data):
            continue
        # books are large: cut into 512 KB line-aligned pieces (whole pieces go to val/test)
        i, k = 0, 0
        while i < len(data):
            j = min(len(data), i + (512 << 10))
            if j < len(data):
                j = data.rfind(b"\n", i, j) + 1
            piece = data[i:j] if data[i:j].endswith(b"\n") else data[i:j] + b"\n"
            out.append((f"gutenberg/pg{gid}.txt#{k:02d}", piece, "%08x" % zlib.crc32(piece)))
            total += len(piece)
            i, k = j, k + 1
    log(f"  {'gutenberg':12s} {len(out):6d} files  {total / MB:7.1f} MB")
    return out


def split_files(domain, files):
    """Whole-file seeded split -> {"train": [...], "val": [...], "test": [...]}."""
    files = list(files)
    random.Random(f"{SEED}:split:{domain}").shuffle(files)
    parts = {"train": [], "val": [], "test": []}
    nv = nt = 0
    for f in files:
        n = len(f[1])
        if nt < TEST_BYTES and n <= 512 << 10:
            parts["test"].append(f); nt += n
        elif nv < VAL_BYTES and n <= 512 << 10:
            parts["val"].append(f); nv += n
        else:
            parts["train"].append(f)
    return parts


def main():
    if not os.path.exists(ENWIK8) or os.path.getsize(ENWIK8) != 100_000_000:
        sys.exit(f"missing {ENWIK8} (100 MB)")
    os.makedirs(OUT, exist_ok=True)
    excluded, seen = bench_crcs(), set()
    log(f"[corpus_r2] collecting (bench exclusions: {len(excluded)} files)")
    by_domain = {"code": [], "docs": [], "structured": [], "prose": []}
    for domain, label, quota, patterns in SOURCES:
        by_domain[domain] += collect(label, quota, patterns, seen, excluded)
    by_domain["structured"] += generated()
    by_domain["prose"] += gutenberg()

    parts = {d: split_files(d, fs) for d, fs in by_domain.items()}
    with open(ENWIK8, "rb") as f:
        wiki = f.read()
    wiki_parts = {"train": wiki[WIKI_TRAIN[0]:WIKI_TRAIN[1]], "val": wiki[WIKI_VAL[0]:WIKI_VAL[1]],
                  "test": wiki[WIKI_TEST[0]:WIKI_TEST[1]]}
    del wiki

    layout = {"name": "corpus_r2", "splits": {}, "domains": {}}
    manifest = ["path\tbytes\tcrc32\tdomain\tsplit"]
    binp = os.path.join(OUT, "corpus_r2.bin")
    off = 0
    with open(binp + ".part", "wb") as out:
        for split in ("train", "val", "test"):
            s0, doms = off, []
            for domain in ("code", "docs", "structured", "prose"):
                d0 = off
                for p, data, crc in parts[domain][split]:
                    out.write(data)
                    off += len(data)
                    manifest.append(f"{p}\t{len(data)}\t{crc}\t{domain}\t{split}")
                if off > d0:
                    doms.append([domain, d0, off - d0])
            w = wiki_parts[split]
            out.write(w)
            doms.append(["wiki", off, len(w)])
            manifest.append(f"enwik8[{split}]\t{len(w)}\t{zlib.crc32(w):08x}\twiki\t{split}")
            off += len(w)
            layout["splits"][split] = [s0, off - s0]
            layout["domains"][split] = doms
    os.replace(binp + ".part", binp)
    layout["bytes"] = off
    with open(os.path.join(OUT, "corpus_r2.json"), "w") as f:
        json.dump(layout, f, indent=1)
    with open(os.path.join(OUT, "MANIFEST.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    log(f"[corpus_r2] wrote {binp}: {off / MB:.1f} MB")
    for split in ("train", "val", "test"):
        log(f"  {split:5s} {layout['splits'][split][1] / MB:8.1f} MB  " +
            "  ".join(f"{n}={ln / MB:.1f}" for n, _, ln in layout["domains"][split]))


if __name__ == "__main__":
    main()
