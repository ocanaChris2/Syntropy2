# Syntropy2 Compressor — native desktop GUI

A native desktop window (Rust + [egui]/[eframe]) for compressing files with the `nsc` neural
codec. Drag files in, pick a model, and **Compress / Decompress / Verify** — with live per-file
ratio, bits/byte, token count, timing, and a lossless check.

```
┌─ Syntropy2 Compressor ──────────────────────────┐
│ Syntropy2 Compressor        [➕ Add files…] [Clear]│
│ Model: base-v1 · base · 1.290 bpc      ● codec ready│
├──────────────────────────────────────────────────┤
│  README.md            12.5 KB   [Verify][Compress] │
│  ✔ compressed  12.5 KB → 5.34 KB · 2.34× · 3.418…  │
│                                                    │
│  notes.txt.nsc         5.3 KB        [Decompress]  │
├──────────────────────────────────────────────────┤
│ ⏵ Compress all  ⏴ Decompress all  ✔ Verify all     │
└──────────────────────────────────────────────────┘
```

## Design

This is a **standalone crate** (its own `[workspace]`), separate from the zero-dependency `nsc`
codec crate one directory up. The GUI never links the codec — it **drives the installed `nsc`
binary over its CLI** (compress / decompress / models). So every
byte the GUI produces goes through the same tokenizer, container, and range-coder paths as the
command line, and the in-app **Verify** (round-trip + byte-compare) is losslessness of the real
product.

## Build & run

```bash
# from nsc-codec/ : build the codec the GUI drives
cargo build --release            # produces target/release/nsc

# from nsc-codec/gui/ : build & launch the window
cargo run --release
```

The GUI finds the `nsc` binary automatically, in this order:

1. `$NSC_BIN` (if it points at an executable),
2. a `nsc` sitting next to the GUI binary,
3. `../target/release/nsc` (the dev layout),
4. anything named `nsc` on your `PATH`.

If none is found the window still opens but shows **“● codec missing”** and the action buttons are
disabled.

## Install a menu launcher

```bash
./install-desktop.sh
```

Builds both binaries if needed and drops a `.desktop` entry into
`~/.local/share/applications`, so **“Syntropy2 Compressor”** appears in your application menu.

## Notes

- **Compress** writes `<file>.nsc` next to the original.
- **Decompress** restores a `.nsc` file and never overwrites — it picks a free
  `…`/`….restored` name if the target exists.
- **Verify** round-trips into temp files, byte-compares against the original, then cleans up.
- Every operation runs on its own background thread; the window stays responsive and shows a
  spinner per in-flight file.

[egui]: https://github.com/emilk/egui
[eframe]: https://github.com/emilk/egui/tree/master/crates/eframe
