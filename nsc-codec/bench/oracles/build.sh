#!/bin/bash
# Reference tools ("oracles") for the format-specialist benchmarks — built locally from source,
# used only by bench scripts, never linked into or shipped with nsc.
#   bash bench/oracles/build.sh            # builds what is missing into bench/oracles/bin
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"; BIN="$HERE/bin"; mkdir -p "$SRC" "$BIN"
J=$(nproc)
log() { echo "[oracles] $*"; }

# static ffmpeg/ffprobe (libmp3lame, libx264/x265, libvpx, libopus/vorbis, libwebp, libaom…)
if [ ! -x "$BIN/ffmpeg" ]; then
  log "ffmpeg static"
  (cd "$SRC" && curl -sSL -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar xf ffmpeg.tar.xz && d=$(ls -d ffmpeg-*-static | head -1) && cp "$d/ffmpeg" "$d/ffprobe" "$BIN/") || log "ffmpeg FAILED"
fi

# paq8px — strongest open context-mixing compressor (exe/image/audio/jpeg/text models)
if [ ! -x "$BIN/paq8px" ]; then
  log "paq8px"
  (cd "$SRC" && rm -rf paq8px && git clone -q --depth 1 https://github.com/hxim/paq8px.git && cd paq8px \
    && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DNATIVECPU=ON .. >/dev/null && make -j"$J" >/dev/null \
    && cp paq8px "$BIN/") || log "paq8px FAILED"
fi

# packMP3 / packJPG — Matthias Stirner's lossless MP3 / JPEG recompressors
for p in packMP3 packJPG; do
  if [ ! -x "$BIN/$p" ]; then
    log "$p"
    (cd "$SRC" && rm -rf $p && git clone -q --depth 1 https://github.com/packjpg/$p.git && cd $p \
      && s=$(dirname "$(find . -name Makefile | head -1)") && cd "$s" && make -j"$J" >/dev/null 2>&1; \
      b=$(find "$SRC/$p" -maxdepth 3 -type f -perm -u+x -iname "$p*" ! -name '*.*' | head -1); [ -n "$b" ] && cp "$b" "$BIN/$p") || log "$p FAILED"
  fi
done

# flac (libFLAC reference encoder; FLAC files in the wild are mostly libFLAC-made)
if [ ! -x "$BIN/flac" ]; then
  log "flac"
  (cd "$SRC" && rm -rf flac && git clone -q --depth 1 https://github.com/xiph/flac.git && cd flac && mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OGG=OFF -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DINSTALL_MANPAGES=OFF .. >/dev/null \
    && make -j"$J" flacapp >/dev/null 2>&1; f=$(find . -type f -name flac -perm -u+x | head -1); [ -n "$f" ] && cp "$f" "$BIN/flac") || log "flac FAILED"
fi
ls -la "$BIN"
