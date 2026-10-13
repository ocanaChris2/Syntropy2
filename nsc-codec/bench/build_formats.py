#!/usr/bin/env python3
"""Build the format-specialist corpus under bench/formats/ (corpus v2) — idempotent.

    python3 bench/build_formats.py [--no-network] [--force]

One directory per format (mp3, m4a, mp4, mkv, webm, avi, ts, flac, ogg, gif, webp, tiff, pe, elf,
dex, wasm, class, pyc, font, pak, pdf, iso, sqlite, csv, json, xml, log, bin, xcursor). Three origins, recorded per file in
formats/MANIFEST.tsv (path, bytes, crc32, format, origin, source, licence, split):

  local      application/system files (/usr, /opt, the Android SDK, Steam's UI assets…);
             never user-created content
  download   freely licensed public files (LibriVox, Wikimedia Commons, Blender Foundation)
  generated  encoded here from the corpus' own public sources with the static ffmpeg / flac /
             PIL (bench/oracles), single-threaded and bitexact so re-runs reproduce the bytes

`split` puts about one file in five (by crc32) in a holdout set that tuning never looks at.
"""
import glob
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "formats")
CACHE = os.path.join(HERE, "cache")
ORACLES = os.path.join(HERE, "oracles", "bin")
FFMPEG = os.path.join(ORACLES, "ffmpeg")
FLAC = os.path.join(ORACLES, "flac")
NET = "--no-network" not in sys.argv
FORCE = "--force" in sys.argv
HOME = os.path.expanduser("~")
UA = {"User-Agent": "nsc-bench/0.2 (local compression benchmark)"}
WIKI = "https://upload.wikimedia.org/wikipedia/commons/"
ROWS = []  # (relpath, fmt, origin, source, licence)


def log(msg):
    print(msg, flush=True)


def dst(fmt, name):
    d = os.path.join(OUT, fmt)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def need(p):
    return FORCE or not os.path.exists(p) or os.path.getsize(p) == 0


def record(p, fmt, origin, source, licence):
    if os.path.exists(p) and os.path.getsize(p) > 0:
        ROWS.append((os.path.relpath(p, OUT), fmt, origin, source, licence))


def fetch(url, name=None):
    """Download `url` into the cache (once); returns the cached path or None."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name or os.path.basename(urllib.parse.urlparse(url).path))
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    if not NET:
        return None
    try:
        log(f"  download {url}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(p + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(p + ".part", p)
        return p
    except Exception as e:  # offline, moved, rate-limited: skip this file
        log(f"  skip {url}: {e}")
        return None


LOCAL_ROOTS = ["/usr/share", "/usr/lib", "/usr/bin", "/opt", f"{HOME}/.local/share/Steam/steamui", f"{HOME}/.local/share/Steam/steamapps/common",
               f"{HOME}/Android/Sdk", f"{HOME}/.local/share/JetBrains/Toolbox/apps"]
_INDEX = None


def local_index():
    """Every regular file under the application/system roots, found once: {extension: sorted paths}."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        roots = [r for r in LOCAL_ROOTS if os.path.isdir(r)]
        r = subprocess.run(["find"] + roots + ["-xdev", "-type", "f", "-size", "+3k"], capture_output=True)
        for line in r.stdout.decode(errors="surrogateescape").split("\n"):
            base = os.path.basename(line)
            ext = base.rsplit(".", 1)[1].lower() if "." in base else ""
            _INDEX.setdefault(ext, []).append(line)
        for v in _INDEX.values():
            v.sort()
    return _INDEX


def take_local(fmt, pattern_globs, n, lo=4 << 10, hi=64 << 20, licence="system/application file"):
    """The `n` files matching the globs (sorted paths, sizes lo..hi, distinct contents). A glob is
    matched against the indexed roots: its extension selects the candidates, the text before its
    first wildcard is the directory they must be under."""
    seen, picked = set(), []
    paths = set()
    for g in pattern_globs:
        ext = g.rsplit(".", 1)[1].lower() if "." in os.path.basename(g) else ""
        prefix = g.split("*", 1)[0]
        tail_dir = None
        if "**" not in g:  # a plain directory glob: only files directly matching it
            paths.update(glob.glob(g))
            continue
        paths.update(p for p in local_index().get(ext, []) if p.startswith(prefix) and (tail_dir is None or tail_dir in p))
    paths = sorted(paths)
    for p in paths:
        if len(picked) >= n:
            break
        try:
            if os.path.islink(p) or not os.path.isfile(p) or not (lo <= os.path.getsize(p) <= hi):
                continue
            with open(p, "rb") as f:
                crc = zlib.crc32(f.read())
        except OSError:
            continue
        if crc in seen:
            continue
        seen.add(crc)
        picked.append(p)
    for p in picked:
        q = dst(fmt, os.path.basename(p))
        if need(q):
            shutil.copyfile(p, q)
        record(q, fmt, "local", p, licence)


def ff(args, out):
    """Run the static ffmpeg (single-threaded, bitexact) to produce `out`."""
    if not need(out):
        return True
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-fflags", "+bitexact"] + args + [out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(f"  ffmpeg failed for {os.path.basename(out)}: {r.stderr.decode(errors='replace')[-300:]}")
        if os.path.exists(out):
            os.remove(out)
        return False
    return True


BITEXACT = ["-flags:v", "+bitexact", "-flags:a", "+bitexact"]


# ------------------------------------------------------------------ sources
def wav_sources():
    """Public music WAV/FLAC already used by the P0 corpus (Wikimedia Commons)."""
    out = {}
    for name, url in [
        ("debussy", WIKI + "0/05/Debussy_Gymnopedie_1%2C_arrangement_of_Satie%E2%80%99s_Gymnopedie_3.wav"),
        ("satie", WIKI + "3/3d/Satie_Gymnopedie_No_2_performed_by_Michael_Laucke.flac"),
        ("chopin", WIKI + "0/0b/Chopin_-_Nocturne_No._21_in_C_minor%2C_B._108_%28Diana_Hughes%29.flac"),
    ]:
        p = fetch(url)
        if p:
            out[name] = p
    return out


def video_source():
    return fetch("https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4")


# ------------------------------------------------------------------ formats
def mp3(src):
    log("mp3")
    take_local("mp3", ["/usr/share/code/resources/app/out/**/*.mp3", "/usr/share/javascript/**/*.mp3"], 4, lo=20 << 10)
    for ident, files in [("art_of_war_librivox", ["art_of_war_01-02_sun_tzu.mp3", "art_of_war_03-04_sun_tzu_64kb.mp3"]),
                         ("pride_and_prejudice_librivox", ["prideandprejudice_06_austen_64kb.mp3"])]:
        for fn in files:
            p = fetch(f"https://archive.org/download/{ident}/{fn}")
            if p:
                q = dst("mp3", fn)
                if need(q):
                    shutil.copyfile(p, q)
                record(q, "mp3", "download", f"archive.org/{ident}", "public domain (LibriVox)")
    cover = os.path.join(HERE, "corpus", "jpeg", "baseline_0_jvasek_xmas_bokeh.jpg")
    for name, s in src.items():
        clip = ["-t", "45", "-i", s, "-map", "0:a:0"]
        variants = [
            (f"{name}_cbr128.mp3", clip + ["-c:a", "libmp3lame", "-b:a", "128k"]),
            (f"{name}_cbr320.mp3", clip + ["-c:a", "libmp3lame", "-b:a", "320k"]),
            (f"{name}_vbr2.mp3", clip + ["-c:a", "libmp3lame", "-q:a", "2"]),
            (f"{name}_mono64_22k.mp3", clip + ["-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar", "22050"]),
            (f"{name}_cbr32_8k.mp3", clip + ["-c:a", "libmp3lame", "-b:a", "32k", "-ac", "1", "-ar", "8000"]),
        ]
        if name == "debussy" and os.path.exists(cover):
            variants.append((f"{name}_tagged_cover.mp3", ["-t", "45", "-i", s, "-i", cover, "-map", "0:a:0", "-map", "1:v", "-c:a", "libmp3lame", "-b:a", "192k",
                             "-c:v", "copy", "-id3v2_version", "3", "-metadata", "title=Gymnopedie 1", "-metadata", "artist=Debussy/Satie",
                             "-metadata:s:v", "comment=Cover (front)"]))
        for fn, args in variants:
            q = dst("mp3", fn)
            if ff(args + BITEXACT, q):
                record(q, "mp3", "generated", f"{os.path.basename(s)} via ffmpeg/libmp3lame", "CC (Wikimedia Commons source)")


def aac(src):
    log("m4a / aac")
    take_local("m4a", [f"{HOME}/.local/share/Steam/steamui/sounds/*.m4a"], 3)
    for name, s in src.items():
        clip = ["-t", "45", "-i", s, "-map", "0:a:0"]
        for fn, args in [(f"{name}_aac128.m4a", clip + ["-c:a", "aac", "-b:a", "128k"]),
                         (f"{name}_aac256.m4a", clip + ["-c:a", "aac", "-b:a", "256k"]),
                         (f"{name}_aac64_mono.m4a", clip + ["-c:a", "aac", "-b:a", "64k", "-ac", "1"])]:
            q = dst("m4a", fn)
            if ff(args + BITEXACT, q):
                record(q, "m4a", "generated", f"{os.path.basename(s)} via ffmpeg aac", "CC (Wikimedia Commons source)")
        q = dst("aac", f"{name}_adts128.aac")
        if ff(clip + ["-c:a", "aac", "-b:a", "128k", "-f", "adts"] + BITEXACT, q):
            record(q, "aac", "generated", f"{os.path.basename(s)} via ffmpeg aac (ADTS)", "CC (Wikimedia Commons source)")


def flac(src):
    log("flac")
    for name in ("satie", "chopin"):
        s = src.get(name)
        if s and s.endswith(".flac"):
            q = dst("flac", f"{name}_wikimedia.flac")
            if need(q):
                shutil.copyfile(s, q)
            record(q, "flac", "download", os.path.basename(s), "CC (Wikimedia Commons)")
    s = src.get("debussy")
    if s and os.path.exists(FLAC):
        for lvl in ("5", "8"):
            q = dst("flac", f"debussy_flac{lvl}.flac")
            if need(q):
                subprocess.run([FLAC, "-s", "-f", f"-{lvl}", "-o", q, s], check=False)
            record(q, "flac", "generated", "Debussy WAV via libFLAC", "CC (Wikimedia Commons source)")


def ogg(src):
    log("ogg / opus")
    take_local("ogg", ["/usr/share/sounds/**/*.oga", "/usr/share/sounds/**/*.ogg", "/usr/share/mint-artwork/sounds/*.ogg"], 4, lo=20 << 10)
    for name, s in src.items():
        clip = ["-t", "45", "-i", s, "-map", "0:a:0"]
        for fn, args in [(f"{name}_vorbis_q5.ogg", clip + ["-c:a", "libvorbis", "-q:a", "5"]),
                         (f"{name}_opus96.opus", clip + ["-c:a", "libopus", "-b:a", "96k"])]:
            q = dst("ogg", fn)
            if ff(args + BITEXACT, q):
                record(q, "ogg", "generated", f"{os.path.basename(s)} via ffmpeg", "CC (Wikimedia Commons source)")


def video(src_audio):
    log("video (mp4 / mov / mkv / webm / avi / ts)")
    take_local("mp4", [f"{HOME}/Android/Sdk/emulator/resources/**/*.mp4"], 5)
    take_local("webm", [f"{HOME}/.local/share/Steam/steamui/movies/*.webm"], 3, lo=200 << 10, hi=4 << 20)
    take_local("mkv", [f"{HOME}/.local/share/Steam/steamapps/common/Proton*/files/share/media/*.mkv"], 1)
    s = video_source()
    if not s:
        return
    q = dst("mp4", "sintel_trailer-480p.mp4")
    if need(q):
        shutil.copyfile(s, q)
    record(q, "mp4", "download", "download.blender.org/durian/trailer", "CC BY 3.0 (Blender Foundation)")
    clip = ["-ss", "4", "-t", "12", "-i", s]
    lic = "CC BY 3.0 (Blender Foundation source)"
    for fmt, fn, args in [
        ("mp4", "sintel_x264_baseline_cavlc.mp4", ["-c:v", "libx264", "-profile:v", "baseline", "-crf", "23", "-c:a", "aac", "-b:a", "128k"]),
        ("mp4", "sintel_x264_high_cabac.mp4", ["-c:v", "libx264", "-profile:v", "high", "-crf", "23", "-c:a", "aac", "-b:a", "128k"]),
        ("mp4", "sintel_x264_high_faststart.mp4", ["-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]),
        ("mp4", "sintel_x264_fragmented.mp4", ["-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov"]),
        ("mp4", "sintel_x265_hevc.mp4", ["-c:v", "libx265", "-crf", "26", "-tag:v", "hvc1", "-x265-params", "pools=1:frame-threads=1:log-level=error", "-c:a", "aac"]),
        ("mov", "sintel_x264.mov", ["-c:v", "libx264", "-crf", "23", "-c:a", "aac"]),
        ("mkv", "sintel_x264_aac.mkv", ["-c:v", "libx264", "-crf", "23", "-c:a", "aac"]),
        ("mkv", "sintel_x264_cavlc_mp3.mkv", ["-c:v", "libx264", "-profile:v", "baseline", "-crf", "24", "-c:a", "libmp3lame", "-b:a", "128k"]),
        ("mkv", "sintel_x264_live.mkv", ["-c:v", "libx264", "-profile:v", "main", "-crf", "24", "-c:a", "aac", "-b:a", "96k", "-live", "1"]),
        ("webm", "sintel_vp9_opus.webm", ["-c:v", "libvpx-vp9", "-b:v", "800k", "-row-mt", "0", "-c:a", "libopus"]),
        ("webm", "sintel_vp8_vorbis.webm", ["-c:v", "libvpx", "-b:v", "800k", "-c:a", "libvorbis"]),
        ("mp4", "sintel_av1.mp4", ["-c:v", "libaom-av1", "-crf", "35", "-cpu-used", "8", "-row-mt", "0", "-c:a", "aac"]),
        ("avi", "sintel_mpeg4asp_mp3.avi", ["-c:v", "mpeg4", "-q:v", "4", "-vtag", "XVID", "-c:a", "libmp3lame", "-b:a", "128k"]),
        ("avi", "sintel_mjpeg_pcm.avi", ["-t", "4", "-c:v", "mjpeg", "-q:v", "5", "-c:a", "pcm_s16le"]),
        ("ts", "sintel_x264_aac.ts", ["-c:v", "libx264", "-crf", "23", "-c:a", "aac", "-f", "mpegts"]),
    ]:
        q = dst(fmt, fn)
        if ff(clip + args + BITEXACT, q):
            record(q, fmt, "generated", "sintel trailer via ffmpeg", lic)


def images():
    log("gif / webp / tiff / ico")
    take_local("gif", ["/usr/share/**/*.gif", "/usr/lib/**/*.gif"], 6, lo=8 << 10)
    for url in [WIKI + "2/2c/Rotating_earth_%28large%29.gif", WIKI + "d/d3/Newtons_cradle_animation_book_2.gif"]:
        p = fetch(url)
        if p:
            q = dst("gif", urllib.parse.unquote(os.path.basename(url)).replace(" ", "_"))
            if need(q):
                shutil.copyfile(p, q)
            record(q, "gif", "download", url, "Wikimedia Commons (see file page)")
    s = video_source()
    if s:
        q = dst("gif", "sintel_anim.gif")
        if ff(["-ss", "30", "-t", "3", "-i", s, "-vf", "fps=10,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"], q):
            record(q, "gif", "generated", "sintel trailer via ffmpeg palettegen", "CC BY 3.0 (Blender Foundation source)")
    take_local("webp", [f"{HOME}/Android/Sdk/skins/**/*.webp"], 6, lo=16 << 10)
    corpus = os.path.join(HERE, "corpus")
    for src, stem in [(os.path.join(corpus, "deflate", "art.png"), "art"), (os.path.join(corpus, "deflate", "screenshot.png"), "screenshot"),
                      (os.path.join(corpus, "jpeg", "progressive_2_jpanchal_cubes.jpg"), "cubes")]:
        if not os.path.exists(src):
            continue
        for fn, args in [(f"{stem}_lossy75.webp", ["-i", src, "-c:v", "libwebp", "-quality", "75"]),
                         (f"{stem}_lossless.webp", ["-i", src, "-c:v", "libwebp", "-lossless", "1"])]:
            q = dst("webp", fn)
            if ff(args, q):
                record(q, "webp", "generated", f"{os.path.basename(src)} via ffmpeg/libwebp", "derived from corpus image")
    try:
        from PIL import Image
        img = os.path.join(corpus, "image", "photo.ppm")
        if os.path.exists(img):
            im = Image.open(img)
            for comp in ("tiff_lzw", "tiff_deflate", "packbits"):
                q = dst("tiff", f"photo_{comp}.tif")
                if need(q):
                    im.save(q, compression=comp)
                record(q, "tiff", "generated", f"photo.ppm via PIL ({comp})", "derived from corpus image")
    except ImportError:
        log("  PIL missing: no TIFF variants")
    take_local("tiff", ["/usr/share/**/*.tif", "/usr/share/**/*.tiff"], 3, lo=16 << 10)
    take_local("ico", ["/usr/share/**/*.ico", "/opt/**/*.ico"], 4, lo=8 << 10)


def executables():
    log("pe / elf / dex / class / wasm / pyc")
    take_local("pe", ["/usr/lib/x86_64-linux-gnu/wine/x86_64-windows/*.dll", "/usr/lib/x86_64-linux-gnu/wine/i386-windows/*.dll"], 6, lo=200 << 10, hi=8 << 20)
    take_local("pe", [f"{HOME}/.local/share/Steam/steamapps/common/Proton*/files/lib/wine/x86_64-windows/*.exe"], 2, lo=200 << 10, hi=8 << 20)
    take_local("elf", ["/usr/bin/*"], 4, lo=500 << 10, hi=8 << 20)
    take_local("elf", ["/usr/lib/x86_64-linux-gnu/*.a", "/usr/lib/gcc/x86_64-linux-gnu/*/*.o"], 3, lo=16 << 10, hi=8 << 20)
    take_local("dex", [f"{HOME}/.local/share/JetBrains/Toolbox/apps/**/*.dex", f"{HOME}/Android/Sdk/**/*.dex"], 3, lo=64 << 10, hi=16 << 20)
    take_local("apk", [f"{HOME}/Android/Sdk/system-images/**/*.apk"], 2, lo=64 << 10, hi=16 << 20)
    take_local("class", ["/usr/share/**/*.class", f"{HOME}/.local/share/JetBrains/Toolbox/apps/**/*.class"], 6, lo=4 << 10)
    take_local("wasm", ["/usr/share/code/resources/app/**/*.wasm"], 3, lo=64 << 10, hi=16 << 20)
    take_local("pyc", ["/usr/lib/python3*/__pycache__/*.pyc"], 6, lo=16 << 10)


def documents():
    log("pdf / fonts / pak / iso")
    take_local("pdf", ["/usr/share/doc/**/*.pdf", "/usr/share/cups/**/*.pdf"], 5, lo=16 << 10, hi=16 << 20)
    take_local("font", ["/usr/share/fonts/**/*.ttf"], 4, lo=64 << 10, hi=16 << 20)
    take_local("font", ["/usr/share/fonts/**/*.otf"], 2, lo=64 << 10, hi=16 << 20)
    take_local("font", [f"{HOME}/.local/share/JetBrains/Toolbox/apps/**/*.woff", "/usr/share/**/*.woff"], 2, lo=16 << 10)
    take_local("font", [f"{HOME}/.local/share/JetBrains/Toolbox/apps/**/*.woff2", "/usr/share/**/*.woff2"], 2, lo=16 << 10)
    take_local("pak", ["/opt/**/*.pak", "/usr/share/code/*.pak", "/usr/share/code/locales/*.pak"], 3, lo=64 << 10, hi=40 << 20)
    take_local("iso", ["/usr/share/virtualbox/*.iso"], 1, lo=1 << 20, hi=200 << 20)


def take_files(fmt, paths, cap=None, licence="system/application file"):
    """The listed files (globs; those that exist), or the first `cap` bytes of the larger ones."""
    seen = set()
    for g in paths:
        for p in sorted(glob.glob(g))[:1]:
            try:
                if os.path.islink(p) or not os.path.isfile(p) or os.path.getsize(p) < 4 << 10:
                    continue
                with open(p, "rb") as f:
                    data = f.read(cap) if cap else f.read()
                    cut = bool(f.read(1))
            except OSError:
                continue
            crc = zlib.crc32(data)
            if crc in seen:
                continue
            seen.add(crc)
            q = dst(fmt, os.path.basename(p))
            if need(q):
                with open(q, "wb") as f:
                    f.write(data)
            record(q, fmt, "local", f"first {cap >> 20} MB of {p}" if cut else p, licence)


def data():
    log("sqlite / csv / json / xml / log / bin / xcursor")
    take_files("sqlite", ["/var/lib/command-not-found/commands.db", "/usr/share/webapp-manager/firefox/profile/places.sqlite",
                          "/usr/share/doc/virtualbox-*/UserManual.qch", "/usr/share/doc/virtualbox-*/UserManual.qhc", "/var/cache/fwupd/quirks.db",
                          "/var/lib/PackageKit/transactions.db", "/var/lib/fwupd/pending.db"])
    take_files("csv", ["/usr/share/ieee-data/oui.csv", "/usr/share/ieee-data/mam.csv", "/usr/share/ieee-data/oui36.csv", "/usr/share/ieee-data/iab.csv",
                       "/usr/share/go-*/src/cmd/go/internal/modfetch/zip_sum_test/testdata/zip_sums.csv"])
    take_files("json", ["/usr/share/code/resources/app/out/nls.metadata.json", "/usr/share/code/resources/app/node_modules.asar.unpacked/@github/copilot-sdk-linux-x64/schemas/api.schema.json",
                        "/usr/share/antigravity/resources/app/node_modules/chrome-remote-interface/lib/protocol.json", "/usr/share/iso-codes/json/iso_639-3.json",
                        "/usr/share/code/resources/app/telemetry-core.json", "/usr/share/code/resources/app/extensions/cpp/syntaxes/platform.tmLanguage.json"])
    take_files("xml", ["/usr/share/mime/packages/freedesktop.org.xml", "/usr/share/libgweather-4/Locations.xml", "/usr/lib/python3/dist-packages/launchpadlib/testing/launchpad-wadl.xml",
                       "/usr/share/liblangtag/language-subtag-registry.xml", "/var/lib/flatpak/appstream/flathub/x86_64/*/appstream.xml"], cap=6 << 20)
    take_files("log", ["/var/log/dpkg.log.1"])
    for name in ("HDFS", "Apache", "OpenSSH", "Linux", "Android", "Spark", "Zookeeper", "BGL"):
        s = fetch(f"https://raw.githubusercontent.com/logpai/loghub/master/{name}/{name}_2k.log")
        if s:
            q = dst("log", f"{name}_2k.log")
            if need(q):
                shutil.copyfile(s, q)
            record(q, "log", "download", "github.com/logpai/loghub", "loghub sample (free for research use)")
    # tables and JSON of numbers: public sample data (vega-datasets; each file's own source is public)
    vega = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/"
    for fmt, name in (("csv", "zipcodes.csv"), ("csv", "birdstrikes.csv"), ("csv", "seattle-weather.csv"), ("csv", "sp500.csv"), ("csv", "stocks.csv"),
                      ("csv", "flights-airport.csv"), ("csv", "weather.csv"), ("csv", "co2-concentration.csv"), ("csv", "gapminder-health-income.csv"),
                      ("json", "us-10m.json"), ("json", "earthquakes.json"), ("json", "movies.json"), ("json", "cars.json"), ("json", "flights-20k.json"),
                      ("json", "gapminder.json"), ("json", "weather.json")):
        s = fetch(vega + name, "vega_" + name)
        if s:
            q = dst(fmt, name)
            if need(q):
                shutil.copyfile(s, q)
            record(q, fmt, "download", "github.com/vega/vega-datasets", "sample data (BSD-3-Clause repository; public sources)")
    take_files("bin", ["/usr/lib/file/magic.mgc", "/usr/lib/udev/hwdb.bin", "/usr/lib/locale/locale-archive", "/usr/share/code/icudtl.dat",
                       "/var/lib/portmaster/intel/geoipv4.mmdb", "/var/cache/apt/pkgcache.bin", "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/classes.jsa",
                       "/var/cache/fwupd/metadata.xmlb", "/usr/share/GeoIP/GeoIPv6.dat", "/usr/share/spotify/v8_context_snapshot.bin",
                       "/var/cache/swcatalog/cache/C-os-catalog.xb"], cap=6 << 20)
    take_files("xcursor", ["/usr/share/icons/Adwaita/cursors/wait", "/usr/share/icons/Yaru/cursors/left_ptr_watch", "/usr/share/icons/XCursor-Pro-Dark/cursors/wait",
                           "/usr/share/icons/Adwaita/cursors/left_ptr"])


def manifest():
    rows = []
    for rel, fmt, origin, source, lic in sorted(set(ROWS)):
        p = os.path.join(OUT, rel)
        with open(p, "rb") as f:
            data = f.read()
        crc = zlib.crc32(data)
        split = "holdout" if crc % 5 == 0 else "dev"
        rows.append(f"{rel}\t{len(data)}\t{crc:08x}\t{fmt}\t{origin}\t{source}\t{lic}\t{split}")
    with open(os.path.join(OUT, "MANIFEST.tsv"), "w") as f:
        f.write("path\tbytes\tcrc32\tformat\torigin\tsource\tlicence\tsplit\n" + "\n".join(rows) + "\n")
    total = sum(int(r.split("\t")[1]) for r in rows)
    log(f"manifest: {len(rows)} files, {total / 1e6:.1f} MB")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(FFMPEG):
        log("bench/oracles/bin/ffmpeg missing: run bash bench/oracles/build.sh first (generated files are skipped)")
    src = wav_sources()
    for step in (lambda: mp3(src), lambda: aac(src), lambda: flac(src), lambda: ogg(src), lambda: video(src),
                 images, executables, documents, data):
        step()
    manifest()


if __name__ == "__main__":
    main()
