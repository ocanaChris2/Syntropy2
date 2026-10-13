# nsc — datatype-aware lossless compressor

`nsc` is a lossless compressor and archiver in pure Rust (zero third-party crates) that picks the
strongest model for every region of every file:

- a **context-mixing engine** (paq lineage, integer-only, AVX2 == scalar) with models for text,
  executables (x86/ARM64 branch conversion), images (8–16-bit rasters), audio (an integer NLMS
  cascade over PCM samples), JPEG coefficients, and raw binary;
- **recompression of already-compressed data**: deflate streams (gzip, zip/jar/docx/xlsx, PDF,
  PNG) are expanded and rebuilt bit-exactly from a tiny record; JPEGs (baseline *and* progressive)
  are re-coded at the DCT-coefficient level and rebuilt bit-exactly;
- the **Syntropy2 neural text model** (int8 forward, native tokenizer, portable math) for text when
  the size policy picks it — no Python needed;
- **archives** (`nsc pack` / `unpack`): one solid, deduplicated, type-ordered stream.

Every result is verified: each block is decoded back during compression (and stored raw if
anything differs), every block and file carries a CRC-32, and corrupt input fails cleanly. The
output depends only on the input and the options — never on thread count, memory or CPU.

```
 any file ──nsc compress──▶ file.nsc ──nsc decompress──▶ identical bytes
 folders  ──nsc pack──────▶ x.nsca   ──nsc unpack──────▶ identical files (+ modes, times, links)
```

---

## Results

`bench/corpus` (383 MB, 73 files: Silesia, text, executables, images, audio, deflate-packed files,
JPEGs, tables, mixed tars), level 5, every file round-tripped byte-exact
(`bench/results/p10_L5.txt`). "best classical" is the smallest of gzip -9, bzip2 -9, zstd -19,
xz -9e and 7z PPMd for each file.

| Data | Original | nsc -5 | bits/byte | vs best classical |
|---|---:|---:|---:|---:|
| Audio (WAV) | 33.7 MB | 9.81 MB | 2.329 | **-43.8 %** |
| Deflate-packed (PNG, PDF, docx, xlsx, jar, zip, gz) | 20.4 MB | 8.03 MB | 3.146 | **-56.4 %** |
| Executables | 14.5 MB | 3.43 MB | 1.893 | **-24.6 %** |
| Images (BMP/PNM/TGA/TIFF) | 44.2 MB | 7.16 MB | 1.296 | **-36.7 %** |
| JPEG | 8.7 MB | 5.65 MB | 5.188 | **-28.6 %** |
| Mixed tars | 18.4 MB | 4.19 MB | 1.822 | **-37.4 %** |
| Silesia | 211.9 MB | 35.84 MB | 1.353 | **-19.6 %** |
| Tables (CSV, SQLite, records) | 14.4 MB | 3.24 MB | 1.793 | **-27.0 %** |
| Text | 16.4 MB | 2.64 MB | 1.292 | **-16.7 %** |
| **Total** | **382.7 MB** | **79.99 MB** | **1.672** | **-32.5 %** |

Highlights: PDFs and Office files shrink 3–10× below their original size (their deflate streams
are expanded and modelled), PNGs by 43–63 %, JPEGs by 35 % (bit-exact), music WAVs to under a third
of their PCM size.

---

## Quick start

```bash
cd nsc-codec
./install.sh                         # build + install `nsc` (+ seed the neural model registry)

nsc compress   photo.png  photo.nsc  # any file
nsc decompress photo.nsc  photo.png  # bit-exact
nsc test       photo.nsc             # verify checksums without writing
nsc info       photo.nsc             # format, models used, ratio
nsc analyze    photo.png --exact     # how nsc sees a file: typed regions and what each costs

nsc pack   backup.nsca  ~/project docs/notes.md   # archive files and folders
nsc list   backup.nsca
nsc unpack backup.nsca  restore/                  # --only <path> for part, --force to replace
```

Levels `-1` … `-9` (default 5) trade speed and memory for strength; `--jobs N` sets parallelism
(blocks, lanes and files run in parallel); `--mem <MiB>` caps memory (it only lowers parallelism,
never changes the output); `--block <MiB>` sets the independent block size. See [Speed](#speed).

---

## Speed

Compression uses every core it can without changing a byte of the output:

- **Verification runs alongside the coding.** Every block is decoded back before it is written: a
  second model decodes each lane while its bytes leave the encoder, and recompressed streams
  (deflate, PNG, JPEG) are rebuilt and checked at the same time, so the check costs CPU rather
  than time. `--no-verify` skips it.
- **Neural text is parallel, and only tried when it can win.** For small text files a sample of the
  neural model's windows first estimates its size; the full neural pass runs only when it can come
  close to the CM result (code, logs and markup usually skip it). Its 1024-token windows are coded
  and verified in parallel.
- **Blocks run in parallel.** Within a block the model is sequential, so the block size decides how
  many cores one file can use — and each block boundary restarts the model, which costs ratio. The
  fast levels use small blocks; from `-3` up, blocks are large (16–128 MiB) and ratio comes first.

| level | block | dickens (10 MB text) | photo.ppm (6.2 MB image) | art.png (1.6 MB PNG) | mozilla (51 MB binaries) |
|---|---:|---|---|---|---|
| `-1` | 2 MiB | 2.37 MB · 8.7 s / 3.8 s | 889 KB · 7.8 s / 3.0 s | not recompressed at `-1` | 12.9 MB · 26 s / 12 s |
| `-2` | 4 MiB | 2.24 MB · 12 s / 8.6 s | 834 KB · 9.9 s / 7.5 s | 897 KB · 18 s / 14 s | 10.6 MB · 36 s / 18 s |
| `-5` | 32 MiB | 2.09 MB · 33 s / 30 s | 767 KB · 18 s / 16 s | 864 KB · 30 s / 26 s | 9.62 MB · 154 s / 110 s |

Size · compress / decompress wall time with verification on, on an 8-core/16-thread i7-11800H with
2–3 GB of free memory (memory caps how many blocks run at once). `--block <MiB>` sets the block size
at any level, e.g. `nsc compress -5 --block 4` codes dickens in 21 s for 4 % more output.

At `-5`, 0.3.1 writes byte-identical files to 0.3.0, compressing 2–46× faster on this machine:
`python_src.py` 138 s → 3 s, `whitepaper.md` 40 s → 5 s, `art.png` 67 s → 30 s, `photo.ppm`
40 s → 18 s, `report.docx` 28 s → 13 s, `sales.csv` 27 s → 12 s.

Decoding neural text is sequential within a block (about 700 tokens/s per core); neural text is
coded in 128 KB blocks that decode in parallel, but one small in-domain text file still takes about
30 s per 100 KB to decompress. `--text cm` avoids the neural model entirely.

---

## What nsc recognises

| Data | Detected from | Treatment |
|---|---|---|
| Text, code, markup, CSV, logs | 4 KB window classifier | text CM model (word, column, field, XML, indentation contexts); neural model for text up to a level-dependent size. Text that is numbers to a fair part — logs, tables, JSON of measurements — also gets a number model: each number is predicted from the number last seen in the same place of a line (the line with its numbers masked) and from what was added to it then, so a time that passes or a counter that steps costs almost nothing (levels 3–9; logs 5–18% smaller) |
| Executables | ELF / PE / Mach-O code sections; the code of ELF object files (also inside static libraries) | x86 and x86-64 code is read instruction by instruction: call targets and 64-bit `[rip + disp]` operands become absolute, and every byte is predicted as what it is — prefix, opcode, ModRM, SIB, displacement or immediate — from the instruction so far and the ones before (levels 3–9; about 10–15% smaller than without). ARM64: BL branch conversion + binary contexts |
| Images | BMP, PNM/PAM, TGA; pixels inside PNGs and TIFFs | image model (neighbours, gradient/Paeth/MED predictors, cross-channel, SSE) |
| TIFF | classic TIFFs (either byte order, directories anywhere in the file, several pictures, strips or tiles, samples together or in planes) whose pixel data is LZW, PackBits, zlib/Deflate or not compressed; whole in a block | every strip or tile is decoded to pixels — the horizontal differences undone that writers take before compressing — and each picture goes to the image model as one piece with its geometry. The strips are written again bit for bit: the usual LZW and PackBits encoders (libtiff's) are simulated and only what a file's encoder did otherwise is noted, zlib strips rebuild from the deflate record. A picture with a strip that does not come back exactly stays as bytes; so do JPEG-, fax- and other compressions (levels 3–9; an LZW photo 8% → 76% saved, Deflate 55% → 67%, PackBits 71% → 79%) |
| GIF | GIF87a / GIF89a pictures and animations, wherever they start | every image is decoded from its LZW codes to colour indices, one byte a pixel (interlaced rows put back in order), and the frames go to a model of indexed pictures that works by colour, not by index: it asks whether the pixel is the one to its left, the one above, the transparent index, or the colour the frame before left there, then for the palette colours nearest to what it expects, and only then for the index itself. Colour tables are coded against the table before. The file is written again bit for bit — the usual LZW encoder is simulated and only where a file's encoder did otherwise (a clear code, a shorter phrase, the sub-block cuts) is that noted (levels 3–9; 23–70% smaller, animations most) |
| Audio | WAV (incl. EXTENSIBLE), AIFF, AU — 8/16/24-bit, one to eight channels | samples, not bytes: a cascade of integer adaptive filters per channel (a fine-stepped short one that also sees the other channel, then filters of up to 1024 taps), a stereo pair as mid/side or left/right — whichever is shorter on a sample of the file — and a context-mixed residual coder. 9–15% smaller than `flac -8` on CD audio |
| FLAC | runs of FLAC frames (16- and 24-bit, any channel layout), wherever they start | decoded to samples for the audio model; of each frame only what is not sample values is kept (its header, each channel's predictor and Rice parameters), and the frame is written again from the samples, bit for bit, with its checksums (levels 3–9; 16-bit files 10–15% smaller, 24-bit about 7%). Cover pictures in the metadata are recompressed as the JPEGs or PNGs they are |
| Deflate | gzip members, zip entries, PDF Flate streams, PNG IDAT, zlib streams on their own (compressed rows of a database, object stores, the colour profile or text of a PNG), nested (a docx in a zip …) | inflated, content re-detected, rebuilt bit-exactly from a reconstruction record. For streams zlib wrote the record is zlib's own parse emulated, with the few exceptions listed. For any other encoder (miniz, zopfli, 7-Zip, libdeflate, Go — optimised PNGs, zips from other tools) each token is noted as a choice among the matches that exist at its place, and what is expected there is the first step of the cheapest way to code the rest of the block by the block's own Huffman tables — which is what such encoders write nine times in ten, so the notes come to a few percent of the stream (levels 3–9; such PNGs go from 0–5% saved to about 40%) |
| JPEG | baseline + progressive Huffman JPEGs (also inside PDFs, zips, tars) | coefficients modelled (lepton-style edge and DC prediction), re-encoded bit-exactly |
| MP3 | runs of MPEG-1/2/2.5 Layer III frames (CBR or VBR, any tags around them); the MP3 tracks of Matroska files | frames parsed down to scale factors and Huffman-decoded spectral values, modelled against the previous granule and the other channel, re-encoded bit-exactly (levels 3–9) |
| AAC | ADTS streams; the AAC-LC audio tracks of MP4 / M4A / MOV and Matroska files, fragmented or not (HE-AAC: the AAC-LC core) | frames parsed down to section code books, scale factors and spectral values, modelled against the previous frames and the other channel of a pair, re-encoded bit-exactly (levels 3–9) |
| H.264 video | the H.264 tracks of MP4 / M4A / MOV and Matroska files: progressive, 4:2:0 or monochrome with samples of 8 to 14 bits, in either entropy coding — CABAC (Main and High profile: x264, phones, cameras) or CAVLC (Baseline: video calls, messaging apps, older devices) | every slice is decoded to the binary decisions of its macroblock syntax, which are modelled with what the codec's own coder cannot see (neighbouring blocks' coefficients and signs, the pictures before, predicted motion vectors) and re-encoded bit-exactly: 8–13% off CABAC video, 20–30% off CAVLC video (levels 3–9) |
| MP4 / M4A / MOV | the box structure; the `moov` index is read first wherever it sits | index and metadata boxes always modelled; each track's samples are gathered through the index into one stream for the AAC or H.264 model, and put back in file order on decoding |
| Matroska (MKV / MKA) | the element tree: tracks, clusters, blocks (one frame each or several laced together), clusters and segments of unknown size (live recordings) | as for MP4: the frames of H.264, AAC and MP3 tracks are gathered into one stream per track for that codec's model; headers, cues and tags always modelled. Tracks stored compressed, encrypted or with stripped headers stay as bytes |
| AVI | the RIFF chunk structure: stream headers, the interleaved `movi` chunks, the index | the audio chunks of MP3 and PCM streams are gathered into one stream per track — MP3 for its model, PCM as samples for the audio model — and put back between the video frames on decoding; Motion-JPEG frames are found as the JPEGs they are (levels 3–9) |
| MPEG-TS | transport streams of 188-byte packets: the program tables say which packets carry what | the payloads of the H.264 and audio (AAC in ADTS, MPEG audio) streams are gathered per stream, past the packet and PES headers, and put back packet by packet on decoding. H.264 there is a byte stream — units behind start codes — which is turned into the length-prefixed form the H.264 recoder reads, the start codes noted (levels 3–9; a sample goes from 12% to 20% saved). A bare H.264 byte stream (a `.h264` / `.264` file) is taken the same way: 9% off CABAC video, 27% off CAVLC |
| Type 1 fonts | font programs starting `%!PS-AdobeFont` / `%!FontType1`, on their own or as the font streams of a PDF (what LaTeX embeds) | the private dictionary behind `eexec` and every glyph's charstring are enciphered with fixed keys, which makes two thirds of the font look like noise; both ciphers are undone, the clear font modelled, and the bytes enciphered again on decoding — exact, with nothing kept beside (levels 3–9; fonts 67% smaller, a LaTeX-made PDF 27% → 66% saved) |
| Camera RAW | lossless JPEG streams (Canon CR2, compressed DNG: `FF D8` … `FF C3`); DNG and other TIFF-based raw files whose sensor data is not compressed, 8, 12 (packed) or 16 bits | a lossless JPEG is decoded to its samples for the image model and written again from its own Huffman tables — a sample has one code, so only the last byte's padding is noted; packed 12-bit samples are unpacked to two bytes each (levels 3–9; a CR2 12% → 21% saved, uncompressed DNG about 55%). The makers' own compressions (Nikon NEF, Sony ARW, Pentax PEF, Olympus ORF, Panasonic RW2) are modelled as bytes: 3–20% |
| Photoshop (PSD) | documents whose layers and merged picture are run-length packed (what Photoshop writes by default), whole in a block | every channel's rows are unpacked and the channels of a layer put side by side as pixels for the image model; the rows are packed again exactly as Photoshop packs them — 128 bytes at a time, up to the edges of its tiles — so the record is a hundred bytes for a whole document, and a row packed otherwise keeps its packets' counts (levels 3–9) |
| ZIP archives | zip, jar, apk, docx/xlsx/odt and the like, from the first entry to the end of the central directory | beside recompressing each entry's deflate stream (and modelling stored entries as what they are): every entry's CRC-32 and sizes stand in its local header, again in the central directory and sometimes in a data descriptor — bytes no model predicts. They are blanked wherever they are what the content gives, and filled in again on decoding (levels 3–9; 1–2.5% on archives of many small files) |
| SQLite | databases (any page size), whole in a block | taken apart by what their bytes are: page layout and column types; numbers, column by column; text, each value ended instead of counted; blobs; and the rows too long for a page, gathered whole from their overflow chains — a compressed document or a picture stored in a table is then found and recompressed like any other. Cell pointers and sizes are derived, not stored. Pages that do not come back exactly are kept as bytes (levels 3–9; 18–46% smaller than as bytes, far more where the rows hold compressed content) |
| Already-compressed media | other video (H.264 that is interlaced or 4:2:2 / 4:4:4; HEVC, VP8, VP9, AV1), WebM, ogg, 7z, xz, zstd, bzip2, webp | stored (unless content repeats) |
| Everything else | — | generic CM model (orders 1–24, sparse, record, indirect, match) |

Large inputs are split into independent blocks coded in parallel; a block whose expanded content is
very large is further split into lanes coded in parallel. A recompressible stream that crosses a
block boundary extends the block so it is not cut.

An MP4's samples have no headers of their own: where they are is written in its index, often at the
end of the file. nsc reads that index before it starts (a piped input is spooled to a temporary file
for this, so the result is the same as for the file), and the index is only ever a hint — every
stream taken apart is rebuilt and compared before it is accepted. A Matroska file has no such table:
its frames sit in blocks spread through the file, each behind a few bytes of header, so nsc walks
the element headers (never the frames) to the same end. A file cut short is used up to the cut.
`nsc analyze` lists the tracks an index gives.

Video is where an MP4's bytes are. H.264 usually ends in an adaptive arithmetic coder (CABAC), so
there is no slack in its bytes; the gain comes from predicting each of its binary decisions better
than its 460 small adaptive contexts do. nsc walks the complete macroblock syntax to recover those
decisions (no pictures are reconstructed), codes them with a context-mixing model, and on decoding
runs CABAC forward again. Slices coded with the older variable-length codes (CAVLC) hold the same
syntax: they are read, expressed as the decisions CABAC would have coded, modelled the same way,
and written back as the same variable-length codes — which is where the larger gain comes from.
A slice is only taken when re-encoding its decisions gives back its bytes exactly; any other NAL
unit, and any stream the parser does not cover, stays as bytes. The parser is checked against the
ITU-T conformance bitstreams (`bench/conformance/`). The model costs about 2 s per megabyte of
video on one core, in both directions; large files are coded in parallel lanes. Levels 3–4 use a
lean set of contexts (about 0.7% larger, a little faster).

---

## Archives

`nsc pack <archive.nsca> <paths…>` stores files, directories and symbolic links with their
permissions and modification times. Identical files are stored once; contents are ordered so that
similar files (by extension group, extension and name) sit next to each other in one solid stream,
and every datatype model above applies inside it.

`nsc unpack` is defensive: paths must be relative with no `..`; nothing is ever written through a
symbolic link (links are created last); existing files are kept unless `--force`; set-id bits are
dropped. `--only <path>` extracts one file or directory and decodes only the blocks it needs.

---

## Compatibility and safety

- **Format.** `NSC3` format 1 is frozen as of 0.3.0: every later nsc decodes it, and the golden
  files in `tests/golden/` check both that they decode and that re-encoding reproduces them byte
  for byte. New models arrive as new segment kinds or versions; a file using one names the decoder
  level it needs, and an older nsc refuses it with a clear message instead of guessing. Files from
  development builds (experimental format 0) are refused. Legacy `NSC2`/`NSCZ`/`NSCR` files remain
  readable.
- **Decoder levels.** Level 1 is nsc 0.3. Level 2 (nsc 0.4) adds the format recoders (MP3, AAC,
  MP4, Matroska, AVI and transport-stream tracks, H.264 video, SQLite, FLAC, GIF, TIFF), the second version of the audio
  model, the models for machine code, binary tables and numbers in text, and block headers whose
  segment table is itself coded; files written at compression levels 3–9 need it. `--compat 1` writes a file any 0.3
  build reads (no recoders), and `tests/golden/level1/` holds the 0.3 golden files: they must keep
  decoding, and `--compat 1` must keep producing exactly them.
- **Never worse, never corrupt.** A block that does not shrink, or that fails its decode-back check,
  is stored raw (a few bytes of header). Within a block (levels 3–9), a stream whose parts code
  larger than the stream itself is left as the bytes it is, and bytes that code longer than they
  are get stored: the block is then coded a second time without them. Lengths are bounded before allocation (decompression
  bombs), and corrupt data yields an error, not a crash.
- **Determinism.** The output is a pure function of the input bytes and options (level, block size,
  `--text`): independent of `--jobs`, `--mem`, stdin vs file, and SIMD level.
- **Building for other machines.** `.cargo/config.toml` builds with `target-cpu=native`; for a binary
  that runs on any recent x86-64 CPU use `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`.
  Output is identical either way.

---

## Desktop app — nsc-gui

`gui/` holds `nsc-gui`, a native desktop window (Rust, egui/eframe): drag files in, pick a model,
and Compress / Decompress / Verify with per-file ratio, bits/byte, timing and a lossless check. It
is a standalone crate (the codec crate stays dependency-free) that drives the installed `nsc`
binary over its CLI, so it produces exactly what the command line does.

```bash
make install-gui                     # or: INSTALL_GUI=1 ./install.sh
cd gui && cargo run --release        # dev run (finds ../target/release/nsc)
gui/install-desktop.sh               # application-menu launcher
```

See `gui/README.md`.

---

## Install / deploy

```bash
./install.sh                         # PREFIX=~/.local/bin  NSC_HOME=~/.nsc
PREFIX=/usr/local/bin ./install.sh   # system-wide binary
NSC_HOME=/opt/nsc     ./install.sh   # shared registry (export NSC_HOME for clients)
# or:  make install
```

`install.sh` (re-runnable) builds the release binary, installs it as `nsc`, copies the tokenizer
shim to `$NSC_HOME/tok.py`, and imports the current `artifacts/` as the default model `base-v1`.

**Runtime requirements:** none beyond the binary. The neural model registry is optional (it is
used for text when installed; everything else needs no model files). Python is only needed to
export new model artifacts and for the `tok-test` parity check.

**Environment:** `NSC_HOME` (registry root, default `~/.nsc`) · `NSC_SIMD=scalar` (force the scalar
kernels; output is identical) · `NSC_PROFILE=<file>` (developer sampling profiler).

---

## Model upgrading

This is the point of the registry: **you can ship a better model without breaking existing files.**

Each model version lives in its own directory and carries its own weights **and** its own
tokenizer:

```
~/.nsc/
  default                       # id of the default model
  tok.py
  models/
    base-v1/  model.nsc  o2.nsc  tokenizer.json  manifest.json
    base-v2/  ...
```

Every compressed file records the **id + content fingerprint** of the model that produced it, so:

- `nsc decompress` **auto-selects the exact model** the file was made with — even if the default
  has since changed. Old files keep decompressing after an upgrade.
- If the model behind that id has been altered on disk (registry drift), the fingerprint mismatch
  is caught and decompression **refuses** rather than silently corrupting the output.

Upgrade workflow — after training/exporting a new checkpoint:

```bash
# 1. export the artifacts for the new checkpoint (see below)
python3 export_int8.py            # -> artifacts/model.nsc
python3 export_o2.py              # -> artifacts/o2.nsc

# 2. register it as a new version and make it the default for NEW compressions
nsc models add base-v2 \
    --model-nsc artifacts/model.nsc --o2 artifacts/o2.nsc \
    --tokenizer ../results/export/tokenizer.json \
    --tier base --bpc 1.28 --created "$(date +%F)" --default

nsc models                        # base-v2 is now the default (*)
# existing *.nsc files still decompress via base-v1 automatically
```

Registry management: `nsc models list` · `nsc models add <id> …` · `nsc models default <id>` ·
`nsc models remove <id>`.

---

## Scalability (batch, parallel)

Model load (~273 MB) is the dominant fixed cost, so batch mode loads each needed model **once** and
compresses/decompresses many files across CPU cores:

```bash
nsc batch compress   *.txt        --out-dir out  --jobs 8
nsc batch decompress out/*.nsc    --out-dir back --jobs 8
```

`--jobs` defaults to the CPU count. Each file is independent (its own coder state); the loaded
model is shared read-only across worker threads (`std::thread::scope`, still zero-dependency).
For decompression, files referencing different model versions are handled in the same run — each
distinct model is loaded once and reused.

---

## Command reference

| Command | Purpose |
|---|---|
| `nsc compress <in\|-> <out\|->` | Compress any file (best of neural text / CM per policy; never expands). |
| `nsc decompress <in\|-> <out\|->` | Decompress; picks the engine/model recorded in the file. |
| `nsc test <file.nsc\|archive.nsca>` | Verify a file or archive decodes (all checksums), writing nothing. |
| `nsc pack <archive.nsca> <paths…>` | Archive files and folders (`--no-dedup` to disable whole-file dedup). |
| `nsc unpack <archive.nsca> [dir]` | Extract (`--only <path>`, `--force`). |
| `nsc list <archive.nsca>` | List an archive's entries. |
| `nsc analyze <file> [--exact] [--json]` | Typed segments nsc detects and (with `--exact`) their cost. |
| `nsc info <file>` | Format, engine and kinds used, ratio, blocks, memory per job. |
| `nsc bench <files\|dirs…>` | nsc levels vs gzip/bzip2/zstd/xz/7z: ratio, MB/s, peak RSS, roundtrip check. |
| `nsc batch <compress\|decompress> <files…>` | Many files in parallel; `--out-dir`. |
| `nsc compress-ids` / `decompress-ids` | Pre-tokenized `.ids` files (neural only). |
| `nsc models [list\|add\|default\|remove]` | Manage the neural model registry. |
| `nsc doctor [--verify]` | Check the registry and model fingerprints. |
| `nsc selftest` | Neural parity harness vs `artifacts/`. |

Flags: `-1`…`-9` · `--jobs <n>` · `--mem <MiB>` · `--block <MiB>` · `--text auto|neural|cm` ·
`--no-verify` · `--store` · `--model <id>` · `--home <path>` · `-q`.

Developer harnesses: `nsc cm-test`, `nsc deflate-scan`, `nsc jpeg-test [--coef-model]`,
`nsc audio-test`, `nsc gif-test`, `nsc tok-test`.

---

## Exporting model artifacts

The registry is seeded from artifacts regenerated from a trained checkpoint
(`../results/export/best.pt`); they are large and not committed:

```bash
python3 export_int8.py   # -> artifacts/model.nsc  (+ artifacts/ref.nsc for tests)
python3 export_o2.py     # -> artifacts/o2.nsc     (FastO2 order-2 table)
```

Optional verification fixtures: `python3 test_vectors/gen.py` (Stage A) and
`python3 dump_intermediates.py` (Stage C).

| File | Size | Required? | Purpose |
|---|---:|---|---|
| `model.nsc` | ~273 MiB | registry | int8 weights + config + `embed`/`order0`/**`order1` (exact f32)** + gate (106 tensors) |
| `o2.nsc` | ~30 MiB | registry | FastO2 order-2 table + `K` + `cap` for the online feature model |
| `tokenizer.json` | ~527 KiB | registry | byte ↔ token-id bijection (per model version) |
| `ref.nsc` / `inter.nsc` / `vec.bin` | — | tests only | Stage A/C/D/E parity fixtures |

`model.nsc` is large because it embeds `order1` as **exact f32** (`V×V×4 B` ≈ 256 MB); the f16
export is lossy and would break the bit-exact coder tables.

---

## Verify

```bash
cd nsc-codec
cargo test --release        # 50+ tests: engine, formats, transforms, models, golden files, archives
nsc selftest                # neural Stages A–E vs Python (reads artifacts/)
UPDATE_GOLDEN=1 cargo test --release golden   # only after an intended, versioned format change
```

---

## Formats & internals

- **`NSC3` (format 1)** — `header | block* | trailer`. Each block holds a segment *tree* (pre-order):
  leaves are coded by a model (CM kinds, JPEG coefficients, audio samples, neural text) or stored,
  and containers rebuild bytes from their children (deflate stream, PNG IDAT framing, PNG row
  filters, JPEG). Leaves are coded in lanes (independent models) so big blocks decode in parallel.
  Varint-compact headers, CRC-32 per block header, per block and per file. Layout:
  `src/nsc3/format.rs`; tree and lanes: `src/nsc3/block.rs`. A block of many small streams (an
  archive of small files, a PDF of small objects) has a long segment table — 1–4% of its output —
  so a writer for decoders of level 2 codes the table with the byte model at a fixed
  configuration (`src/nsc3/table.rs`): it comes out five to nine times smaller (zip/jar/apk and
  PDFs 1–3% smaller overall).
- **Verification** — every block is decoded back before it is written, without paying for a second
  sequential pass: each lane's coded bytes are streamed to a verifier thread that runs the decoder's
  own lane loop while the encoder is still coding, container rebuilds (deflate/PNG/JPEG) are checked
  concurrently, and neural leaves are proven by recomputing the model's windows in parallel and
  replaying the range decoder over them (equivalent to a full decode, by induction over tokens).
  The final check reuses those results only after confirming they match the block exactly.
- **Archives** — NSC3 with header flag 1; the trailer's file table (paths, kinds, modes, times,
  sizes, content offsets, CRCs, link targets) is CM-compressed. `src/archive.rs`.

- **`.nsc` container v2 (`NSC2`)** — `magic "NSC2" | fmt u16 | id_len u16 | model_id | fingerprint
  u64 | vocab u32 | seg u32 | n_tokens u32 | first_token u32 | n_bits u32 | payload`. The first
  token is raw; tokens `1..` are entropy-coded. Legacy v1 (`NSCZ`, no model binding) is still read.
  Loader: `src/container.rs`.
- **`model.nsc`** — `magic "NSC1" | 9×u32 cfg | eps f32 | n_tensors u32 |` per tensor
  `name_len u16, name, dtype u8 (0=f32,1=i8,2=f64), ndim u8, dims…, data`. Linears store
  `{.code i8 [out,in], .scale f64 [out], .bias f32 [out]}` (per-output-channel int8 via `qrow`).
  Loader: `src/artifacts.rs`.
- **`o2.nsc`** — `magic "NO21" | V u32 | K u32 | cap f32 | n u32 | ctx[i64] | top[i32] | conf[f32]`.
  Loader + online feature model: `src/features.rs`.
- **`.ids`** — `magic "NIDS" | n u32 | n×token-id u32` (shared by `tok.py` and the codec).
- **`manifest.json`** — flat JSON per model: `id, tier, vocab_size, seg_len, d_model, bpc, created,
  fingerprint (FNV-1a-64 over model.nsc+o2.nsc), format, notes`. Loader: `src/registry.rs`.
- **Determinism knobs (must not change):** per-row int8 (`round_ties_even`, clamp `[-128,127]`,
  scale floor `1e-30`), i32 exact accumulation + f64 dequant, `T = 1.0`, `QUANT_BITS = 22`, neural
  window reset every `seg_len` (1024), continuous feature model.

## Source map

| File | Role |
|---|---|
| `src/main.rs` | `nsc` CLI: subcommands, routing, parallel batch, arg parsing |
| `src/cm/` | context-mixing engine: coder, state tables, maps, mixer, match model, predictor, level presets; the predictor's side models (`guess.rs`: bytes computable from the record above; `num.rs`: numbers in text); the dedicated models (JPEG, PCM audio, MP3, AAC, H.264) and the one place leaves are dispatched to them (`models.rs`) |
| `src/recode/` | format recoders: MP3 and AAC frame parsers and the H.264 slice parser (CABAC engine, CAVLC reader/writer, macroblock syntax, motion prediction) with exact rebuild, the MP4 and Matroska indexes, the container that interleaves a file's tracks again, the SQLite page parser (`sqlite.rs`), the FLAC frame decoder and encoder (`flac.rs`), the GIF block parser (`gif.rs`; its LZW is `src/transform/lzw.rs`, its model `src/cm/indexed.rs`), the TIFF directory reader and strip decoder (`tiff.rs`; PackBits is `src/transform/packbits.rs`), code tables generated from public-domain and BSD decoders (`*_tables.gen.py`) |
| `src/x86.rs` | x86 / x86-64 instruction decoder (lengths and fields, one byte at a time) behind the executable model's contexts and the code transforms |
| `src/bits.rs` | bit readers/writers and prefix-code decoding shared by the recoders |
| `src/nsc3/` | NSC3 container: format, block planner, segment trees + lanes, parallel scheduler |
| `src/detect/` | datatype detection (format parsers, window classifier) |
| `src/transform/` | BCJ, deflate (inflate, writer, zlib emulation, side records, scanner), PNG filters, JPEG |
| `src/archive.rs` | `nsc pack/unpack/list/test` |
| `src/tokenizer.rs` / `src/dmath.rs` | native BPE tokenizer; portable deterministic math for neural NSC3 |
| `src/golden.rs` / `tests/golden/` | golden files of the current decoder level, and of level 1 (`level1/`, written by nsc 0.3) |
| `bench/build_formats.py` / `bench/formats.py` | the format corpus and its per-format scorecard (against nsc 0.3.1, xz, zstd and dedicated tools); `bench/show.py <tag>` lists a run's files |
| `bench/conformance/` | `fetch.py` downloads ITU-T H.264.1 conformance bitstreams (test material, kept out of the tree), `run.py` checks that every slice the parser takes re-encodes exactly |
| `bench/h264_prune.py` | which of the H.264 model's context inputs earn their time (greedy elimination; `nsc h264-test --drop`) |
| `bench/exe_dev.py` / `bench/exe_ablate.py` | the executable models' development loop: a fixed set of executables, each compressed, decompressed and compared; and which instruction contexts earn their place (`nsc exe-test`) |
| `bench/bin_dev.py` | the same for the generic binary model, on data sections of executables and table-like files, against paq8px reference sizes (`nsc cm-test`) |
| `bench/num_dev.py` / `bench/text_costs.py` | the number model's development loop (logs, tables, JSON, XML and plain text; greedy selection of its contexts), and where the bits of a delimited text file go, field by field (`nsc cm-test --costs`) |
| `bench/sqlite_costs.py` | where the bits of a SQLite database go: page headers, cell pointers, sizes, values by type |
| `src/bench.rs` | `nsc bench` harness |
| `src/registry.rs` | versioned model registry (list/add/default/resolve, fingerprint, manifest) |
| `src/container.rs` | `.nsc` container v2 (model-bound) + legacy reader |
| `src/codec.rs` | shared core: quantizer, `encode_tokens`/`decode_tokens`, `.ids` I/O |
| `src/range_coder.rs` | 32-bit Witten–Neal–Cleary range coder |
| `src/artifacts.rs` | `model.nsc` loader; precomputes per-token order-1 `max`/entropy |
| `src/forward.rs` | int8 forward: GRU×2, RoPE transformer + int8 KV-cache, gate, fusion |
| `src/features.rs` | `o2.nsc` loader + online order-2/match feature model |
| `src/selftest.rs` | Stage A–E parity harness |
| `export_int8.py` / `export_o2.py` | export `model.nsc` / `o2.nsc` |
| `tok.py` | byte ↔ token-id shim (`enc`/`dec`, per-model tokenizer) |
| `install.sh` / `Makefile` | build + deploy + registry seeding |
