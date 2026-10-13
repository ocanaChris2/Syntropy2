#!/usr/bin/env python3
"""Build the nsc benchmark corpus under bench/corpus/ — deterministic and idempotent.

    python3 bench/build_corpus.py [--no-network] [--force]

Every generated file is a pure function of the system files it is derived from (fixed seeds,
fixed tar/zip timestamps, sorted inputs), so re-running reproduces identical bytes. A manifest
(corpus/MANIFEST.tsv: path, bytes, crc32, category) records what was built. Network steps
(Silesia corpus, freely licensed Wikimedia music) are skipped with --no-network or when offline.
No user-owned files are used.
"""
import glob
import io
import os
import random
import shutil
import sqlite3
import struct
import subprocess
import sys
import tarfile
import urllib.request
import wave
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(CRATE)
OUT = os.path.join(HERE, "corpus")
CACHE = os.path.join(HERE, "cache")
NET = "--no-network" not in sys.argv
FORCE = "--force" in sys.argv
UA = {"User-Agent": "nsc-bench/0.1 (local compression benchmark)"}

SILESIA_URL = "https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip"
WIKI = "https://upload.wikimedia.org/wikipedia/commons/"
MUSIC = [  # (output name, url, seconds to keep or None) — freely licensed Wikimedia Commons uploads
    ("debussy_gymnopedie.wav", WIKI + "0/05/Debussy_Gymnopedie_1%2C_arrangement_of_Satie%E2%80%99s_Gymnopedie_3.wav", None),
    ("satie_gymnopedie2.wav", WIKI + "3/3d/Satie_Gymnopedie_No_2_performed_by_Michael_Laucke.flac", 90),
    ("chopin_nocturne21.wav", WIKI + "0/0b/Chopin_-_Nocturne_No._21_in_C_minor%2C_B._108_%28Diana_Hughes%29.flac", 60),
]


def log(msg):
    print(msg, flush=True)


def path(cat, name):
    d = os.path.join(OUT, cat)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def need(p):
    """True if p must be (re)built."""
    return FORCE or not os.path.exists(p) or os.path.getsize(p) == 0


def write_bytes(p, data):
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)


def copy(src, cat, name=None):
    p = path(cat, name or os.path.basename(src))
    if os.path.exists(src) and need(p):
        shutil.copyfile(src, p)
    return p


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    if not NET:
        return False
    import time
    for attempt in range(5):
        try:
            log(f"  downloading {url.split('/')[-1][:70]} ...")
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dst + ".tmp", dst)
            time.sleep(2)  # be polite to shared hosts (Wikimedia rate-limits bursts with HTTP 429)
            return True
        except Exception as e:  # offline, moved, or rate-limited: retry with backoff, then give up
            if "429" in str(e) and attempt < 4:
                log(f"  rate-limited; retrying in {10 * (attempt + 1)} s")
                time.sleep(10 * (attempt + 1))
                continue
            log(f"  ! download failed ({e}); skipping")
            return False
    return False


def det_tar(p, members):
    """Deterministic uncompressed tar: members = [(arcname, filepath)] (sorted by caller)."""
    if not need(p):
        return
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for arc, fp in members:
            ti = tarfile.TarInfo(arc)
            ti.size = os.path.getsize(fp)
            ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = 1700000000, 0o644, 1000, 1000, "user", "user"
            with open(fp, "rb") as f:
                t.addfile(ti, f)
    write_bytes(p, buf.getvalue())


def det_zip(p, entries):
    """Deterministic zip (zlib deflate level 6): entries = [(name, bytes)]."""
    if not need(p):
        return
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in entries:
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data, compresslevel=6)
    write_bytes(p, buf.getvalue())


def enwik(n, off=0):
    with open(os.path.join(REPO, "data", "enwik8"), "rb") as f:
        f.seek(off)
        return f.read(n)


# ----------------------------------------------------------------------------------- sections
def silesia():
    log("[silesia]")
    os.makedirs(CACHE, exist_ok=True)
    z = os.path.join(CACHE, "silesia.zip")
    if not fetch(SILESIA_URL, z):
        return
    with zipfile.ZipFile(z) as zf:
        for name in sorted(zf.namelist()):
            base = os.path.basename(name)
            if not base:
                continue
            p = path("silesia", base)
            if need(p):
                write_bytes(p, zf.read(name))


def text():
    log("[text]")
    for n, name in [(1_000_000, "enwik_1M.txt"), (10_000_000, "enwik_10M.txt")]:
        p = path("text", name)
        if need(p):
            write_bytes(p, enwik(n))
    p = path("text", "rust_src.rs")
    if need(p):
        srcs = sorted(glob.glob(os.path.join(CRATE, "src", "*.rs")) + glob.glob(os.path.join(CRATE, "gui", "src", "*.rs")))
        write_bytes(p, b"".join(open(s, "rb").read() for s in srcs))
    copy(os.path.join(REPO, "neuro_symbolic_compressor.py"), "text", "python_src.py")
    copy(os.path.join(REPO, "WHITEPAPER.md"), "text", "whitepaper.md")
    copy(os.path.join(REPO, "results", "export", "tokenizer.json"), "text", "tokenizer.json")
    p = path("text", "app.log")
    if need(p):
        rng = random.Random(1)
        lvls = ["INFO"] * 12 + ["DEBUG"] * 6 + ["WARN"] * 2 + ["ERROR"]
        comps = ["http", "db", "cache", "auth", "worker", "scheduler", "api"]
        msgs = ["request completed in {}ms status={}", "cache miss key=user:{} ttl={}",
                "query took {}ms rows={}", "token refreshed for uid={} exp={}",
                "job {} finished with code {}", "connection pool size={} waiting={}"]
        t, lines = 1_700_000_000_000, []
        for i in range(60_000):
            t += rng.randint(1, 900)
            m = rng.choice(msgs).format(rng.randint(1, 99999), rng.choice([200, 200, 200, 201, 304, 404, 500]))
            lines.append(f"2026-09-{1 + (t // 86_400_000) % 28:02d}T{(t // 3_600_000) % 24:02d}:{(t // 60_000) % 60:02d}:"
                         f"{(t // 1000) % 60:02d}.{t % 1000:03d}Z [{rng.choice(lvls):5}] {rng.choice(comps)}: {m}")
        write_bytes(p, ("\n".join(lines) + "\n").encode())


def exe():
    log("[exe]")
    copy("/usr/bin/git", "exe", "git.elf")
    copy("/usr/bin/python3.12", "exe", "python3.12.elf")
    copy("/usr/lib/x86_64-linux-gnu/libc.so.6", "exe", "libc.so.6")
    copy("/usr/x86_64-w64-mingw32/lib/zlib1.dll", "exe", "zlib1_x64.dll")
    copy("/usr/i686-w64-mingw32/lib/zlib1.dll", "exe", "zlib1_x86.dll")


def image():
    log("[image]")
    from PIL import Image
    jpgs = sorted(glob.glob("/usr/share/backgrounds/linuxmint-wallpapers/*.jpg"))
    if len(jpgs) < 6:
        jpgs = sorted(glob.glob("/usr/share/backgrounds/**/*.jpg", recursive=True))

    def src(i, mode="RGB", size=(1920, 1080)):
        im = Image.open(jpgs[i % len(jpgs)]).convert(mode)
        return im.resize(size, Image.LANCZOS) if im.size != size else im

    jobs = [
        ("photo_rgb24.bmp", lambda: src(0), "BMP"),
        ("photo.ppm", lambda: src(2), "PPM"),
        ("photo_gray.pgm", lambda: src(3, "L"), "PPM"),
        ("photo.tga", lambda: src(4), "TGA"),
        ("photo.tif", lambda: src(5), "TIFF"),
    ]
    for name, mk, fmt in jobs:
        p = path("image", name)
        if need(p):
            mk().save(p, fmt)
    p = path("image", "photo_rgba32.bmp")
    if need(p):
        im = src(1).convert("RGBA")
        w, h = im.size
        alpha = Image.linear_gradient("L").resize((w, h))
        im.putalpha(alpha)
        im.save(p, "BMP")
    p = path("image", "photo_gray16.pgm")
    if need(p):  # 10 bits of real precision (sum of 2x2 pixels) in a 16-bit big-endian container
        im = src(3, "L")
        w, h = im.size
        px = im.tobytes()
        out = bytearray()
        for y in range(0, h - 1, 2):
            r0, r1 = y * w, (y + 1) * w
            for x in range(0, w - 1, 2):
                s = px[r0 + x] + px[r0 + x + 1] + px[r1 + x] + px[r1 + x + 1]
                out += struct.pack(">H", s << 6)
        hdr = f"P5\n{w // 2} {h // 2}\n65535\n".encode()
        write_bytes(p, hdr + bytes(out))
    for srcp, name in [("/usr/share/help/C/drawing/figures/screenshot_menu.png", "screenshot.bmp"),
                       ("/usr/share/backgrounds/linuxmint/ktee_linuxmint.png", "flat_art.bmp")]:
        p = path("image", name)
        if os.path.exists(srcp) and need(p):
            Image.open(srcp).convert("RGB").save(p, "BMP")


def gst_to_wav(src, dst):
    cmd = ["gst-launch-1.0", "-q", "filesrc", f"location={src}", "!", "decodebin", "!", "audioconvert", "!",
           "audio/x-raw,format=S16LE", "!", "wavenc", "!", "filesink", f"location={dst}"]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(dst)


def truncate_wav(p, seconds):
    with wave.open(p, "rb") as w:
        params = w.getparams()
        frames = w.readframes(min(w.getnframes(), int(seconds * w.getframerate())))
    with wave.open(p + ".cut", "wb") as o:
        o.setparams(params)
        o.writeframes(frames)
    os.replace(p + ".cut", p)


def audio():
    log("[audio]")
    for f in sorted(glob.glob("/usr/share/sounds/*.wav")):
        copy(f, "audio")
    for name in ["bell.oga", "complete.oga", "dialog-warning.oga", "audio-test-signal.oga"]:
        s = os.path.join("/usr/share/sounds/freedesktop/stereo", name)
        p = path("audio", "fd_" + name.replace(".oga", ".wav"))
        if os.path.exists(s) and need(p):
            gst_to_wav(s, p)
    os.makedirs(CACHE, exist_ok=True)
    for name, url, secs in MUSIC:
        p = path("audio", name)
        if not need(p):
            continue
        raw = os.path.join(CACHE, url.split("/")[-1].split("?")[0])
        if not fetch(url, raw):
            continue
        if raw.endswith(".wav"):
            shutil.copyfile(raw, p)
        elif not gst_to_wav(raw, p):
            log(f"  ! could not decode {raw}")
            continue
        if secs:
            truncate_wav(p, secs)


def table():
    log("[table]")
    rng = random.Random(7)
    p = path("table", "sales.csv")
    if need(p):
        cities = ["Madrid", "Lisboa", "Paris", "Berlin", "Roma", "Wien", "Praha", "Oslo", "Dublin", "Bern"]
        prods = ["widget", "gadget", "doohickey", "gizmo", "sprocket", "flange", "bracket"]
        rows = ["order_id,date,city,product,qty,unit_price,discount,total"]
        for i in range(80_000):
            q, up, d = rng.randint(1, 40), rng.randint(199, 99999) / 100, rng.choice([0, 0, 0, 5, 10, 15])
            rows.append(f"{100000 + i},2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d},{rng.choice(cities)},"
                        f"{rng.choice(prods)},{q},{up:.2f},{d},{q * up * (100 - d) / 100:.2f}")
        write_bytes(p, ("\n".join(rows) + "\n").encode())
    p = path("table", "records.bin")
    if need(p):  # fixed 48-byte little-endian records (id, ts, price, qty, flags, name[16], crc)
        out = bytearray()
        ts = 1_700_000_000
        words = enwik(400_000, 5_000_000).split()
        for i in range(120_000):
            ts += rng.randint(0, 120)
            name = words[i % len(words)][:16].ljust(16, b"\0")
            rec = struct.pack("<IqdhH16s", i, ts, rng.randint(100, 10_000_000) / 100.0, rng.randint(-5, 500),
                              rng.choice([0, 1, 1, 3, 0x8001]), name)
            out += rec + struct.pack("<I", zlib.crc32(rec))
        write_bytes(p, bytes(out))
    p = path("table", "inventory.sqlite")
    if need(p):
        tmp = p + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = sqlite3.connect(tmp)
        con.execute("PRAGMA page_size=4096")
        con.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, sku TEXT, title TEXT, qty INTEGER, price REAL, updated INTEGER)")
        words = [w.decode("latin-1") for w in enwik(2_000_000, 20_000_000).split() if w.isalpha()]
        for i in range(60_000):
            title = " ".join(words[(i * 7 + k) % len(words)] for k in range(rng.randint(2, 8)))
            con.execute("INSERT INTO items VALUES(?,?,?,?,?,?)",
                        (i, f"SKU-{rng.randint(0, 999999):06d}", title, rng.randint(0, 500), rng.randint(99, 999999) / 100,
                         1_700_000_000 + i * 37))
        con.execute("CREATE INDEX ix_sku ON items(sku)")
        con.commit()
        con.close()
        os.replace(tmp, p)


def deflate():
    log("[deflate]")
    gz = sorted(f for f in glob.glob("/usr/share/doc/**/*.gz", recursive=True) if os.path.isfile(f))
    det_tar(path("deflate", "gz_changelogs.tar"), [(os.path.relpath(f, "/usr/share/doc"), f) for f in gz[:300]])
    for f in sorted(gz, key=lambda x: (-os.path.getsize(x), x))[:3]:
        copy(f, "deflate", "gnu_" + f.split("/")[-2] + "_" + os.path.basename(f))
    copy("/usr/lib/ipscan/ipscan-linux64-3.9.3.jar", "deflate", "ipscan.jar")
    copy("/usr/lib/portmaster/portmaster.zip", "deflate", "portmaster.zip")
    for f in ["default-testpage.pdf", "form_english.pdf", "form_russian.pdf"]:
        copy(os.path.join("/usr/share/cups/data", f), "deflate", f)
    for f, name in [("/usr/share/backgrounds/linuxmint/ktee_linuxmint.png", "art.png"),
                    ("/usr/share/ldm/themes/mint/bg.png", "background.png"),
                    ("/usr/share/backgrounds/linuxmint/edesigner_linuxmint.png", "design.png"),
                    ("/usr/share/help/C/drawing/figures/screenshot_menu.png", "screenshot.png")]:
        copy(f, "deflate", name)
    paras = enwik(3_000_000, 40_000_000).decode("latin-1").split("\n")
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{esc(x)}</w:t></w:r></w:p>' for x in paras if x.strip())
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document xmlns:w="http://schemas.'
           'openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + "</w:body></w:document>")
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/'
          'content-types"><Default Extension="xml" ContentType="application/xml"/></Types>')
    det_zip(path("deflate", "report.docx"), [("[Content_Types].xml", ct.encode()), ("word/document.xml", doc.encode("utf-8"))])
    rng = random.Random(11)
    cells = "".join(f'<row r="{r}">' + "".join(f'<c r="{chr(65 + c)}{r}"><v>{rng.randint(0, 10 ** (c + 2))}</v></c>'
                                             for c in range(8)) + "</row>" for r in range(1, 40_001))
    sheet = ('<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/'
             'spreadsheetml/2006/main"><sheetData>' + cells + "</sheetData></worksheet>")
    det_zip(path("deflate", "sheet.xlsx"), [("[Content_Types].xml", ct.encode()), ("xl/worksheets/sheet1.xml", sheet.encode())])


def jpeg_kind(p):
    b = open(p, "rb").read(1 << 16)
    i = 2
    while i + 4 <= len(b):
        if b[i] != 0xFF:
            return None
        m = b[i + 1]
        if m in (0xC0, 0xC1):
            return "baseline"
        if m == 0xC2:
            return "progressive"
        if m in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return None
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    return None


def jpeg():
    log("[jpeg]")
    cands = sorted(p for p in glob.glob("/usr/share/**/*.jp*g", recursive=True)
                   if 100_000 <= os.path.getsize(p) <= 2_500_000)
    picked = {"baseline": [], "progressive": []}
    for p in cands:
        k = jpeg_kind(p)
        if k and len(picked[k]) < 6:
            picked[k].append(p)
    for k, ps in picked.items():
        for i, p in enumerate(ps):
            copy(p, "jpeg", f"{k}_{i}_{os.path.basename(p)}")


def mixed():
    log("[mixed]")
    skip = {"target", "artifacts", "bench", ".git"}
    files = []
    for root, dirs, fs in os.walk(CRATE):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for f in sorted(fs):
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and not os.path.islink(fp) and os.path.getsize(fp) < 8_000_000:
                files.append((os.path.relpath(fp, os.path.dirname(CRATE)), fp))
    det_tar(path("mixed", "nsc_src.tar"), sorted(files))
    parts = [path("image", "photo_rgb24.bmp"), path("audio", "linuxmint-login.wav"), path("text", "enwik_1M.txt"),
             path("exe", "git.elf"), path("deflate", "art.png")]
    jp = sorted(glob.glob(os.path.join(OUT, "jpeg", "*.jpg")))
    parts += jp[:1]
    det_tar(path("mixed", "mixed.tar"), sorted((os.path.basename(p), p) for p in parts if os.path.exists(p)))


def manifest():
    rows = []
    for root, _, fs in os.walk(OUT):
        for f in fs:
            fp = os.path.join(root, f)
            if f == "MANIFEST.tsv" or f.endswith(".tmp"):
                continue
            crc = 0
            with open(fp, "rb") as h:
                for chunk in iter(lambda: h.read(1 << 20), b""):
                    crc = zlib.crc32(chunk, crc)
            rel = os.path.relpath(fp, OUT)
            rows.append(f"{rel}\t{os.path.getsize(fp)}\t{crc:08x}\t{rel.split(os.sep)[0]}")
    rows.sort()
    write_bytes(os.path.join(OUT, "MANIFEST.tsv"), ("path\tbytes\tcrc32\tcategory\n" + "\n".join(rows) + "\n").encode())
    total = sum(int(r.split("\t")[1]) for r in rows)
    log(f"manifest: {len(rows)} files, {total / 1e6:.1f} MB -> {os.path.join(OUT, 'MANIFEST.tsv')}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for step in (silesia, text, exe, image, audio, table, deflate, jpeg, mixed):
        step()
    manifest()
