//! Thin backend that drives the installed `nsc` command-line codec.
//!
//! The GUI never links the codec library; every operation shells out to the `nsc` binary so the
//! exact tokenizer / container / range-coder paths the CLI uses are the ones exercised here. That
//! also means losslessness verified in the GUI is losslessness of the real product.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// One model as reported by `nsc models list`.
#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub tier: String,
    pub bpc: Option<f64>,
    pub is_default: bool,
}

/// Result of a successful compress.
#[derive(Clone, Debug)]
pub struct CompressOutcome {
    pub out: PathBuf,
    pub in_bytes: u64,
    pub out_bytes: u64,
    pub tokens: u64,
    pub ratio: f64,
    pub bpb: f64,
    pub elapsed: Duration,
}

/// Result of a successful decompress.
#[derive(Clone, Debug)]
pub struct DecompressOutcome {
    pub out: PathBuf,
    pub out_bytes: u64,
    pub elapsed: Duration,
}

/// Result of a round-trip verify (compress → decompress → byte-compare against the original).
#[derive(Clone, Debug)]
pub struct VerifyOutcome {
    pub lossless: bool,
    pub in_bytes: u64,
    pub nsc_bytes: u64,
    pub ratio: f64,
    pub bpb: f64,
    pub elapsed: Duration,
}

// ------------------------------------------------------------------ binary discovery

/// Find the `nsc` executable: `$NSC_BIN`, then next to this GUI binary, then the sibling
/// `../target/release/nsc` from the dev tree, then anything named `nsc` on `PATH`.
pub fn find_nsc() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NSC_BIN") {
        let pb = PathBuf::from(p);
        if is_exec_file(&pb) {
            return Some(pb);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join("nsc");
            if is_exec_file(&cand) {
                return Some(cand);
            }
        }
    }
    // Dev layout: this crate lives at nsc-codec/gui, the codec builds nsc-codec/target/release/nsc.
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/release/nsc");
    if is_exec_file(&dev) {
        return Some(dev);
    }
    which("nsc")
}

fn is_exec_file(p: &Path) -> bool {
    p.is_file()
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|c| is_exec_file(c))
}

// ------------------------------------------------------------------ model listing

/// Parse `nsc models list`. Its table rows look like:
///   `  base-v1          *   base   8192    1.290    2026-07-11 d0cd3457`
/// where the `*` (DEF column) is present only for the default model.
pub fn list_models(nsc: &Path) -> Result<Vec<Model>, String> {
    let out = Command::new(nsc)
        .args(["models", "list"])
        .output()
        .map_err(|e| format!("launch `nsc models list`: {e}"))?;
    if !out.status.success() {
        return Err(stderr_or(&out, "`nsc models list` failed"));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut models = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        // Skip the "registry:" line and the "ID DEF TIER ..." header.
        if t.is_empty() || t[0] == "registry:" || t[0] == "ID" {
            continue;
        }
        let id = t[0].to_string();
        let is_default = t.get(1).map(|s| *s == "*").unwrap_or(false);
        // Columns shift by one depending on whether the DEF star is present.
        let (tier_i, bpc_i) = if is_default { (2, 4) } else { (1, 3) };
        let tier = t.get(tier_i).unwrap_or(&"").to_string();
        let bpc = t.get(bpc_i).and_then(|s| s.parse::<f64>().ok());
        models.push(Model { id, tier, bpc, is_default });
    }
    Ok(models)
}

// ------------------------------------------------------------------ operations

/// Compress `input` to `<input>.nsc`.
pub fn compress(nsc: &Path, model: Option<&str>, input: &Path) -> Result<CompressOutcome, String> {
    let out = append_ext(input, "nsc");
    let t0 = Instant::now();
    let mut c = Command::new(nsc);
    c.arg("compress").arg(input).arg(&out);
    if let Some(m) = model {
        c.arg("--model").arg(m);
    }
    let o = c.output().map_err(|e| format!("launch nsc: {e}"))?;
    let elapsed = t0.elapsed();
    if !o.status.success() {
        return Err(stderr_or(&o, "compress failed"));
    }
    let in_bytes = len_of(input);
    let out_bytes = len_of(&out);
    let tokens = parse_tokens(&String::from_utf8_lossy(&o.stdout));
    let ratio = if out_bytes > 0 { in_bytes as f64 / out_bytes as f64 } else { 0.0 };
    let bpb = if in_bytes > 0 { out_bytes as f64 * 8.0 / in_bytes as f64 } else { 0.0 };
    Ok(CompressOutcome { out, in_bytes, out_bytes, tokens, ratio, bpb, elapsed })
}

/// Decompress a `.nsc` file. The model is chosen automatically from the file's header, so no model
/// argument is passed. The output never overwrites an existing file.
pub fn decompress(nsc: &Path, input: &Path) -> Result<DecompressOutcome, String> {
    let out = restored_path(input);
    let t0 = Instant::now();
    let o = Command::new(nsc)
        .arg("decompress")
        .arg(input)
        .arg(&out)
        .output()
        .map_err(|e| format!("launch nsc: {e}"))?;
    let elapsed = t0.elapsed();
    if !o.status.success() {
        return Err(stderr_or(&o, "decompress failed"));
    }
    Ok(DecompressOutcome { out: out.clone(), out_bytes: len_of(&out), elapsed })
}

/// Round-trip a file through the codec into temp files and byte-compare the result against the
/// original. Temp files are always cleaned up.
pub fn verify(nsc: &Path, model: Option<&str>, input: &Path) -> Result<VerifyOutcome, String> {
    let nscf = temp_path("nsc");
    let outf = temp_path("out");
    let t0 = Instant::now();
    let r = (|| {
        // compress input -> nscf
        let mut c = Command::new(nsc);
        c.arg("compress").arg(input).arg(&nscf);
        if let Some(m) = model {
            c.arg("--model").arg(m);
        }
        let o = c.output().map_err(|e| format!("launch nsc: {e}"))?;
        if !o.status.success() {
            return Err(stderr_or(&o, "verify: compress failed"));
        }
        // decompress nscf -> outf
        let o = Command::new(nsc)
            .arg("decompress")
            .arg(&nscf)
            .arg(&outf)
            .output()
            .map_err(|e| format!("launch nsc: {e}"))?;
        if !o.status.success() {
            return Err(stderr_or(&o, "verify: decompress failed"));
        }
        let lossless = files_equal(input, &outf)?;
        let in_bytes = len_of(input);
        let nsc_bytes = len_of(&nscf);
        let ratio = if nsc_bytes > 0 { in_bytes as f64 / nsc_bytes as f64 } else { 0.0 };
        let bpb = if in_bytes > 0 { nsc_bytes as f64 * 8.0 / in_bytes as f64 } else { 0.0 };
        Ok(VerifyOutcome { lossless, in_bytes, nsc_bytes, ratio, bpb, elapsed: t0.elapsed() })
    })();
    let _ = std::fs::remove_file(&nscf);
    let _ = std::fs::remove_file(&outf);
    r
}

// ------------------------------------------------------------------ helpers

fn len_of(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn stderr_or(out: &std::process::Output, fallback: &str) -> String {
    let e = String::from_utf8_lossy(&out.stderr);
    let e = e.trim();
    if e.is_empty() {
        let o = String::from_utf8_lossy(&out.stdout);
        let o = o.trim();
        if o.is_empty() { fallback.to_string() } else { o.to_string() }
    } else {
        e.to_string()
    }
}

/// Pull the token count out of the compress summary line: `... -> 12345 tok -> ...`.
fn parse_tokens(stdout: &str) -> u64 {
    let t: Vec<&str> = stdout.split_whitespace().collect();
    for (i, w) in t.iter().enumerate() {
        if *w == "tok" && i > 0 {
            if let Ok(n) = t[i - 1].replace(',', "").parse::<u64>() {
                return n;
            }
        }
    }
    0
}

/// `foo.txt` + `nsc` -> `foo.txt.nsc`.
fn append_ext(p: &Path, ext: &str) -> PathBuf {
    let mut s = p.as_os_str().to_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Choose a non-clobbering output path for a decompressed file. Strips a trailing `.nsc`; if that
/// name is taken (or absent), inserts `.restored` before the extension and, if needed, a counter.
fn restored_path(input: &Path) -> PathBuf {
    let base: PathBuf = if input.extension().and_then(|e| e.to_str()) == Some("nsc") {
        input.with_extension("")
    } else {
        append_ext(input, "out")
    };
    if !base.exists() {
        return base;
    }
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("file").to_string();
    let ext = base.extension().and_then(|s| s.to_str()).map(|s| s.to_string());
    let dir = base.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    for n in 1..10_000 {
        let name = match &ext {
            Some(e) if n == 1 => format!("{stem}.restored.{e}"),
            Some(e) => format!("{stem}.restored-{n}.{e}"),
            None if n == 1 => format!("{stem}.restored"),
            None => format!("{stem}.restored-{n}"),
        };
        let cand = dir.join(name);
        if !cand.exists() {
            return cand;
        }
    }
    base
}

static CTR: AtomicU64 = AtomicU64::new(0);

fn temp_path(ext: &str) -> PathBuf {
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    std::env::temp_dir().join(format!("nsc-gui-{}-{}-{}.{}", std::process::id(), nanos, n, ext))
}

/// Stream-compare two files chunk by chunk; true iff identical.
fn files_equal(a: &Path, b: &Path) -> Result<bool, String> {
    if len_of(a) != len_of(b) {
        return Ok(false);
    }
    let mut fa = File::open(a).map_err(|e| format!("open {}: {e}", a.display()))?;
    let mut fb = File::open(b).map_err(|e| format!("open {}: {e}", b.display()))?;
    let mut ba = [0u8; 64 * 1024];
    let mut bb = [0u8; 64 * 1024];
    loop {
        let na = fa.read(&mut ba).map_err(|e| format!("read: {e}"))?;
        let nb = fb.read(&mut bb).map_err(|e| format!("read: {e}"))?;
        if na != nb {
            return Ok(false);
        }
        if na == 0 {
            return Ok(true);
        }
        if ba[..na] != bb[..nb] {
            return Ok(false);
        }
    }
}

/// Human-readable byte size, used by the UI.
pub fn fmt_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}
//! Syntropy2 Compressor — a native desktop GUI for the `nsc` neural file codec.
//!
//! Drag files onto the window (or click "Add files…"), pick a model, and Compress / Decompress /
//! Verify. Each job runs on its own background thread and reports back over a channel, so the
//! window never blocks. All codec work is delegated to the installed `nsc` CLI (see `backend`).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use backend::{fmt_bytes, CompressOutcome, DecompressOutcome, Model, VerifyOutcome};
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Duration;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 560.0])
            .with_min_inner_size([560.0, 360.0])
            .with_title("Syntropy2 Compressor"),
        ..Default::default()
    };
    eframe::run_native(
        "Syntropy2 Compressor",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

// ------------------------------------------------------------------ operations & messages

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Compress,
    Decompress,
    Verify,
}

/// A finished job, sent from a worker thread back to the UI.
enum JobResult {
    Compress(Result<CompressOutcome, String>),
    Decompress(Result<DecompressOutcome, String>),
    Verify(Result<VerifyOutcome, String>),
}

struct Msg {
    id: u64,
    result: JobResult,
}

// ------------------------------------------------------------------ per-file state

enum Detail {
    None,
    Compressed(CompressOutcome),
    Decompressed(DecompressOutcome),
    Verified(VerifyOutcome),
    Error(String),
}

struct Entry {
    id: u64,
    path: PathBuf,
    size: u64,
    busy: bool,
    detail: Detail,
}

impl Entry {
    fn name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string())
    }
    fn is_nsc(&self) -> bool {
        self.path.extension().and_then(|e| e.to_str()) == Some("nsc")
    }
}

// ------------------------------------------------------------------ app

struct App {
    nsc: Option<PathBuf>,
    models: Vec<Model>,
    selected_model: Option<String>,
    entries: Vec<Entry>,
    next_id: u64,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    status: String,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let nsc = backend::find_nsc();
        let (models, selected_model, status) = match &nsc {
            Some(path) => match backend::list_models(path) {
                Ok(ms) => {
                    let def = ms.iter().find(|m| m.is_default).or_else(|| ms.first());
                    let sel = def.map(|m| m.id.clone());
                    let status = format!("nsc: {}", path.display());
                    (ms, sel, status)
                }
                Err(e) => (Vec::new(), None, format!("nsc found but `models list` failed: {e}")),
            },
            None => (
                Vec::new(),
                None,
                "nsc binary not found — set $NSC_BIN or build it with `cargo build --release` in nsc-codec".to_string(),
            ),
        };
        Self {
            nsc,
            models,
            selected_model,
            entries: Vec::new(),
            next_id: 1,
            tx,
            rx,
            status,
        }
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if !p.is_file() {
                continue;
            }
            if self.entries.iter().any(|e| e.path == p) {
                continue; // dedupe
            }
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            self.entries.push(Entry {
                id: self.next_id,
                path: p,
                size,
                busy: false,
                detail: Detail::None,
            });
            self.next_id += 1;
        }
    }

    /// Kick off `op` on the entry with the given id (on a worker thread).
    fn spawn(&mut self, ctx: &egui::Context, id: u64, op: Op) {
        let Some(nsc) = self.nsc.clone() else { return };
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else { return };
        if entry.busy {
            return;
        }
        entry.busy = true;
        entry.detail = Detail::None;
        let path = entry.path.clone();
        let model = self.selected_model.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = match op {
                Op::Compress => JobResult::Compress(backend::compress(&nsc, model.as_deref(), &path)),
                Op::Decompress => JobResult::Decompress(backend::decompress(&nsc, &path)),
                Op::Verify => JobResult::Verify(backend::verify(&nsc, model.as_deref(), &path)),
            };
            let _ = tx.send(Msg { id, result });
            ctx.request_repaint();
        });
    }

    fn drain_results(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            if let Some(e) = self.entries.iter_mut().find(|e| e.id == msg.id) {
                e.busy = false;
                e.detail = match msg.result {
                    JobResult::Compress(Ok(o)) => Detail::Compressed(o),
                    JobResult::Decompress(Ok(o)) => Detail::Decompressed(o),
                    JobResult::Verify(Ok(o)) => Detail::Verified(o),
                    JobResult::Compress(Err(e))
                    | JobResult::Decompress(Err(e))
                    | JobResult::Verify(Err(e)) => Detail::Error(e),
                };
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_results();
        let ctx = ui.ctx().clone();

        // Accept files dropped anywhere on the window.
        let dropped: Vec<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.add_paths(dropped);
        }

        // Deferred actions collected during rendering (can't mutate self while borrowing entries).
        let mut to_run: Vec<(u64, Op)> = Vec::new();
        let mut to_remove: Vec<u64> = Vec::new();
        let mut clear_all = false;
        let mut add_dialog = false;

        // Panels are children of this root Ui; the central panel MUST be shown last.
        self.top_panel(ui, &mut add_dialog, &mut clear_all);
        self.bottom_panel(ui, &mut to_run);
        self.central_panel(ui, &mut to_run, &mut to_remove);
        draw_drag_overlay(&ctx);

        // Apply deferred actions.
        if add_dialog {
            if let Some(files) = rfd::FileDialog::new().pick_files() {
                self.add_paths(files);
            }
        }
        if clear_all {
            self.entries.retain(|e| e.busy);
        }
        for id in to_remove {
            self.entries.retain(|e| e.id != id || e.busy);
        }
        for (id, op) in to_run {
            self.spawn(&ctx, id, op);
        }

        // Keep animating spinners while any job is in flight.
        if self.entries.iter().any(|e| e.busy) {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

// ------------------------------------------------------------------ panels

impl App {
    fn top_panel(&mut self, ui: &mut egui::Ui, add_dialog: &mut bool, clear_all: &mut bool) {
        egui::Panel::top(egui::Id::new("top")).show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("Syntropy2 Compressor");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Clear").on_hover_text("Remove all idle files from the list").clicked() {
                        *clear_all = true;
                    }
                    if ui.button("➕ Add files…").clicked() {
                        *add_dialog = true;
                    }
                });
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("Model:");
                let enabled = !self.models.is_empty();
                ui.add_enabled_ui(enabled, |ui| {
                    let current = self.selected_model.clone().unwrap_or_else(|| "—".into());
                    egui::ComboBox::from_id_salt("model")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for m in &self.models {
                                let label = match m.bpc {
                                    Some(b) => format!("{}  ·  {}  ·  {:.3} bpc", m.id, m.tier, b),
                                    None => format!("{}  ·  {}", m.id, m.tier),
                                };
                                ui.selectable_value(&mut self.selected_model, Some(m.id.clone()), label);
                            }
                        });
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (col, txt) = if self.nsc.is_some() {
                        (egui::Color32::from_rgb(80, 190, 120), "● codec ready")
                    } else {
                        (egui::Color32::from_rgb(220, 100, 100), "● codec missing")
                    };
                    ui.colored_label(col, txt);
                });
            });
            ui.add_space(6.0);
        });
    }

    fn bottom_panel(&mut self, ui: &mut egui::Ui, to_run: &mut Vec<(u64, Op)>) {
        egui::Panel::bottom(egui::Id::new("bottom")).show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let any = !self.entries.is_empty() && self.nsc.is_some();
                ui.add_enabled_ui(any, |ui| {
                    // Compress everything that isn't already a .nsc file.
                    if ui.button("⏵  Compress all").clicked() {
                        for e in &self.entries {
                            if !e.busy && !e.is_nsc() {
                                to_run.push((e.id, Op::Compress));
                            }
                        }
                    }
                    // Decompress every .nsc file.
                    if ui.button("⏴  Decompress all").clicked() {
                        for e in &self.entries {
                            if !e.busy && e.is_nsc() {
                                to_run.push((e.id, Op::Decompress));
                            }
                        }
                    }
                    if ui
                        .button("✔  Verify all")
                        .on_hover_text("Round-trip each non-.nsc file and byte-compare")
                        .clicked()
                    {
                        for e in &self.entries {
                            if !e.busy && !e.is_nsc() {
                                to_run.push((e.id, Op::Verify));
                            }
                        }
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.status).weak().small());
                });
            });
            ui.add_space(6.0);
        });
    }

    fn central_panel(
        &mut self,
        ui: &mut egui::Ui,
        to_run: &mut Vec<(u64, Op)>,
        to_remove: &mut Vec<u64>,
    ) {
        egui::CentralPanel::default().show(ui, |ui| {
            if self.entries.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Drop files here\n\nor use “Add files…”")
                            .size(20.0)
                            .weak(),
                    );
                });
                return;
            }
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                // Snapshot the fields we need per row so we don't hold an immutable borrow of
                // self.entries while pushing into the deferred action buffers.
                let rows: Vec<(u64, String, u64, bool, bool)> = self
                    .entries
                    .iter()
                    .map(|e| (e.id, e.name(), e.size, e.busy, e.is_nsc()))
                    .collect();
                for (id, name, size, busy, is_nsc) in rows {
                    let detail_idx = self.entries.iter().position(|e| e.id == id).unwrap();
                    row_frame(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Left: name + size.
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(&name).strong());
                                ui.label(egui::RichText::new(fmt_bytes(size)).small().weak());
                            });
                            // Right: actions + status.
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if busy {
                                    ui.add(egui::Spinner::new());
                                } else {
                                    if ui.small_button("✖").on_hover_text("Remove from list").clicked() {
                                        to_remove.push(id);
                                    }
                                    let can_codec = self.nsc.is_some();
                                    ui.add_enabled_ui(can_codec, |ui| {
                                        if is_nsc {
                                            if ui.button("Decompress").clicked() {
                                                to_run.push((id, Op::Decompress));
                                            }
                                        } else {
                                            if ui.button("Verify").clicked() {
                                                to_run.push((id, Op::Verify));
                                            }
                                            if ui.button("Compress").clicked() {
                                                to_run.push((id, Op::Compress));
                                            }
                                        }
                                    });
                                }
                            });
                        });
                        // Result line beneath the row.
                        draw_detail(ui, &self.entries[detail_idx].detail);
                    });
                    ui.add_space(6.0);
                }
            });
        });
    }
}

// ------------------------------------------------------------------ small drawing helpers

/// A translucent hint while files are being dragged over the window.
fn draw_drag_overlay(ctx: &egui::Context) {
    let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
    if !hovering {
        return;
    }
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("drop_overlay"),
    ));
    painter.rect_filled(screen, egui::CornerRadius::ZERO, egui::Color32::from_black_alpha(160));
    painter.text(
        screen.center(),
        egui::Align2::CENTER_CENTER,
        "＋  Drop to add files",
        egui::FontId::proportional(28.0),
        egui::Color32::WHITE,
    );
}

fn row_frame(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

fn draw_detail(ui: &mut egui::Ui, detail: &Detail) {
    let green = egui::Color32::from_rgb(80, 190, 120);
    let red = egui::Color32::from_rgb(220, 100, 100);
    match detail {
        Detail::None => {}
        Detail::Compressed(o) => {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(green, "✔ compressed");
                ui.weak(fmt_bytes(o.in_bytes));
                ui.label("→");
                ui.strong(fmt_bytes(o.out_bytes));
                ui.weak(format!(
                    "· {:.2}× · {:.3} bits/byte · {} tok · {}",
                    o.ratio,
                    o.bpb,
                    o.tokens,
                    fmt_dur(o.elapsed)
                ));
            });
            ui.label(egui::RichText::new(o.out.display().to_string()).small().weak());
        }
        Detail::Decompressed(o) => {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(green, "✔ decompressed");
                ui.label("→");
                ui.strong(fmt_bytes(o.out_bytes));
                ui.weak(format!("· {}", fmt_dur(o.elapsed)));
            });
            ui.label(egui::RichText::new(o.out.display().to_string()).small().weak());
        }
        Detail::Verified(o) => {
            ui.horizontal_wrapped(|ui| {
                if o.lossless {
                    ui.colored_label(green, "✔ lossless");
                } else {
                    ui.colored_label(red, "✘ MISMATCH — not lossless");
                }
                ui.weak(format!(
                    "· {:.2}× · {:.3} bits/byte · {} → {} · {}",
                    o.ratio,
                    o.bpb,
                    fmt_bytes(o.in_bytes),
                    fmt_bytes(o.nsc_bytes),
                    fmt_dur(o.elapsed)
                ));
            });
        }
        Detail::Error(e) => {
            ui.colored_label(red, format!("✘ {e}"));
        }
    }
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1.0 {
        format!("{:.0} ms", s * 1000.0)
    } else {
        format!("{s:.1} s")
    }
}
//! Loads the int8 artifacts exported by export_int8.py (model.nsc) plus the classical
//! order0/order1 tables. Dependency-free binary parsing.
use std::collections::HashMap;
use std::fs;

#[derive(Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub gru_hidden: usize,
    pub gru_layers: usize,
    pub seg_len: usize,
    pub gate_dim: usize,
    pub conf_dim: usize,
    pub eps: f32,
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub f32: Vec<f32>, // dequant/float tensors (embed, order0, scales-as-f32 unused, LN, bias)
    pub i8: Vec<i8>,   // int8 weight codes
    pub f64: Vec<f64>, // per-channel weight scales
}
impl Tensor {
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}

pub struct Model {
    pub cfg: Config,
    pub t: HashMap<String, Tensor>,
    pub o1_max: Vec<f32>, // per-token max of order1 row (conf_state, precomputed)
    pub o1_ent: Vec<f32>, // per-token normalized entropy of order1 row
}

fn ru16(b: &[u8], o: &mut usize) -> u16 { let v = u16::from_le_bytes(b[*o..*o + 2].try_into().unwrap()); *o += 2; v }
fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

impl Model {
    pub fn load(path: &str) -> Model {
        let b = fs::read(path).expect("model.nsc missing (run export_int8.py)");
        let mut o = 0usize;
        assert_eq!(&b[0..4], b"NSC1", "bad magic");
        o = 4;
        let g: Vec<usize> = (0..9).map(|_| ru32(&b, &mut o) as usize).collect();
        let cfg = Config {
            vocab_size: g[0], d_model: g[1], n_layers: g[2], n_heads: g[3], gru_hidden: g[4],
            gru_layers: g[5], seg_len: g[6], gate_dim: g[7], conf_dim: g[8], eps: rf32(&b, &mut o),
        };
        let n = ru32(&b, &mut o) as usize;
        let mut t = HashMap::new();
        for _ in 0..n {
            let nl = ru16(&b, &mut o) as usize;
            let name = String::from_utf8(b[o..o + nl].to_vec()).unwrap(); o += nl;
            let dt = b[o]; let nd = b[o + 1] as usize; o += 2;
            let shape: Vec<usize> = (0..nd).map(|_| ru32(&b, &mut o) as usize).collect();
            let ne: usize = shape.iter().product();
            let mut ten = Tensor { shape, f32: vec![], i8: vec![], f64: vec![] };
            match dt {
                0 => { ten.f32 = (0..ne).map(|i| f32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect(); o += ne * 4; }
                1 => { ten.i8 = (0..ne).map(|i| b[o + i] as i8).collect(); o += ne; }
                2 => { ten.f64 = (0..ne).map(|i| f64::from_le_bytes(b[o + i * 8..o + i * 8 + 8].try_into().unwrap())).collect(); o += ne * 8; }
                _ => panic!("bad dtype"),
            }
            t.insert(name, ten);
        }
        // precompute per-token order1 stats (conf_state depends only on the token)
        let v = cfg.vocab_size;
        let o1 = &t.get("order1").expect("order1").f32;
        let logv = (v as f32).ln();
        let mut o1_max = vec![0f32; v];
        let mut o1_ent = vec![0f32; v];
        for i in 0..v {
            let row = &o1[i * v..(i + 1) * v];
            o1_max[i] = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut ent = 0f32;
            for &p in row { ent += p * (p + cfg.eps).ln(); }
            o1_ent[i] = -ent / logv;
        }
        Model { cfg, t, o1_max, o1_ent }
    }

    pub fn get(&self, name: &str) -> &Tensor { self.t.get(name).unwrap_or_else(|| panic!("no tensor {name}")) }

    /// order1 row p(next | prev=i) as an f32 slice of length V (stored f32 in model.nsc).
    pub fn order1_row(&self, i: usize) -> &[f32] {
        let v = self.cfg.vocab_size;
        &self.get("order1").f32[i * v..(i + 1) * v]
    }

    /// A quantized Linear: (int8 codes [out,in], per-channel scale [out], bias [out], out, in).
    pub fn lin(&self, name: &str) -> (&[i8], &[f64], &[f32], usize, usize) {
        let c = self.get(&format!("{name}.code"));
        let s = self.get(&format!("{name}.scale"));
        let b = self.get(&format!("{name}.bias"));
        (&c.i8, &s.f64, &b.f32, c.shape[0], c.shape[1])
    }
}
//! Benchmark support: classical-compressor helpers shared with `nsc studio`, and the `nsc bench`
//! command — ratio, encode/decode speed and peak memory of nsc vs gzip/bzip2/xz/zstd/7z on real
//! files, with a byte-exact roundtrip check on every measurement.
//!
//! Peak RSS comes from `wait4(2)` on each child. `nsc bench` itself stays small (it streams files and
//! never holds a model), so the watermark a vfork'd child inherits from its parent is negligible.
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static SEQ: AtomicU64 = AtomicU64::new(0);

// ------------------------------------------------------------------ helpers shared with studio
/// Classical codecs available on this machine, by studio display name.
pub fn detect_classical() -> Vec<String> {
    let mut out = vec![];
    for (name, probe) in [("gzip -9", "gzip"), ("xz -9", "xz"), ("zstd -19", "zstd"), ("7z-PPMd", "7z")] {
        if tool_exists(probe) { out.push(name.to_string()); }
    }
    out
}

fn tool_exists(probe: &str) -> bool {
    Command::new(probe).arg("--help").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
}

/// Run `cmd args`, capture stdout, return (stdout bytes, seconds).
pub fn bench_pipe(cmd: &str, args: &[&str]) -> Option<(u64, f64)> {
    let t0 = Instant::now();
    let out = Command::new(cmd).args(args).output().ok()?;
    let dt = t0.elapsed().as_secs_f64();
    if !out.status.success() { return None; }
    Some((out.stdout.len() as u64, dt))
}

/// 7z-PPMd archive size and time for `path` (studio head-to-head).
pub fn bench_7z(path: &Path) -> Option<(u64, f64)> {
    let dir = scratch_dir("7z");
    fs::create_dir_all(&dir).ok()?;
    let arch = dir.join("a.7z");
    let t0 = Instant::now();
    let st = Command::new("7z").args(["a", "-m0=PPMd", "-bso0", "-bsp0"]).arg(&arch).arg(path)
        .stdout(Stdio::null()).stderr(Stdio::null()).status().ok()?;
    let dt = t0.elapsed().as_secs_f64();
    let sz = if st.success() { fs::metadata(&arch).map(|m| m.len()).unwrap_or(0) } else { 0 };
    let _ = fs::remove_dir_all(&dir);
    if sz == 0 { None } else { Some((sz, dt)) }
}

/// Chunked byte comparison of two files (never loads either file whole).
pub fn files_equal(a: &Path, b: &Path) -> bool {
    let (Ok(ma), Ok(mb)) = (fs::metadata(a), fs::metadata(b)) else { return false };
    if ma.len() != mb.len() { return false; }
    let (Ok(mut fa), Ok(mut fb)) = (File::open(a), File::open(b)) else { return false };
    let mut ba = vec![0u8; 1 << 20];
    let mut bb = vec![0u8; 1 << 20];
    loop {
        let na = match read_full(&mut fa, &mut ba) { Ok(n) => n, Err(_) => return false };
        let nb = match read_full(&mut fb, &mut bb) { Ok(n) => n, Err(_) => return false };
        if na != nb || ba[..na] != bb[..nb] { return false; }
        if na == 0 { return true; }
    }
}

/// Fill `buf` as far as possible (handles short reads); returns bytes read (0 = EOF).
pub fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

fn scratch_dir(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nsc-bench-{tag}-{}-{n}", std::process::id()))
}

// ------------------------------------------------------------------ measured child processes
#[repr(C)]
#[derive(Default)]
struct Timeval { tv_sec: i64, tv_usec: i64 }

#[repr(C)]
#[derive(Default)]
struct Rusage {
    ru_utime: Timeval, ru_stime: Timeval,
    ru_maxrss: i64, ru_ixrss: i64, ru_idrss: i64, ru_isrss: i64, ru_minflt: i64, ru_majflt: i64,
    ru_nswap: i64, ru_inblock: i64, ru_oublock: i64, ru_msgsnd: i64, ru_msgrcv: i64,
    ru_nsignals: i64, ru_nvcsw: i64, ru_nivcsw: i64,
}

unsafe extern "C" {
    fn wait4(pid: i32, status: *mut i32, options: i32, rusage: *mut Rusage) -> i32;
}

/// One measured child run.
pub struct Run { pub ok: bool, pub wall_s: f64, pub cpu_s: f64, pub max_rss_kb: u64 }

/// Spawn `cmd args` (stdin/stdout optionally redirected to files) and reap it with `wait4` to get
/// its exit status, wall time, CPU time and peak resident set size.
pub fn run_measured(cmd: &Path, args: &[String], stdin: Option<&Path>, stdout: Option<&Path>) -> Result<Run, String> {
    let mut c = Command::new(cmd);
    c.args(args).stderr(Stdio::null());
    match stdin { Some(p) => { c.stdin(File::open(p).map_err(|e| format!("open {}: {e}", p.display()))?); } None => { c.stdin(Stdio::null()); } }
    match stdout { Some(p) => { c.stdout(File::create(p).map_err(|e| format!("create {}: {e}", p.display()))?); } None => { c.stdout(Stdio::null()); } }
    let t0 = Instant::now();
    let child = c.spawn().map_err(|e| format!("spawn {}: {e}", cmd.display()))?;
    let pid = child.id() as i32;
    let mut status: i32 = 0;
    let mut ru = Rusage::default();
    // SAFETY: plain libc call on our own child's pid with valid out-pointers. We reap the child here
    // and never call `child.wait()`; dropping a `Child` neither waits nor kills.
    let r = loop {
        let r = unsafe { wait4(pid, &mut status, 0, &mut ru) };
        if r == -1 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted { continue; }
        break r;
    };
    let wall_s = t0.elapsed().as_secs_f64();
    drop(child);
    if r != pid { return Err(format!("wait4 failed for {}", cmd.display())); }
    let exited_ok = (status & 0x7f) == 0 && ((status >> 8) & 0xff) == 0;
    let cpu_s = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6
        + ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 * 1e-6;
    Ok(Run { ok: exited_ok, wall_s, cpu_s, max_rss_kb: ru.ru_maxrss.max(0) as u64 })
}

// ------------------------------------------------------------------ codec table
#[derive(Clone)]
enum Kind {
    /// `<tool> <cargs> -c IN > OUT`, `<tool> <dargs> -c OUT > RT`
    Pipe { tool: &'static str, cargs: &'static [&'static str], dargs: &'static [&'static str] },
    /// 7z archive with the given method arguments.
    SevenZip { margs: &'static [&'static str] },
    /// this nsc binary at a level.
    Nsc { level: u32 },
}

#[derive(Clone)]
struct Codec { name: String, kind: Kind }

fn classical_codecs() -> Vec<Codec> {
    let mk = |name: &str, kind: Kind| Codec { name: name.into(), kind };
    let mut v = vec![];
    if tool_exists("gzip") { v.push(mk("gzip-9", Kind::Pipe { tool: "gzip", cargs: &["-9", "-c"], dargs: &["-d", "-c"] })); }
    if tool_exists("bzip2") { v.push(mk("bzip2-9", Kind::Pipe { tool: "bzip2", cargs: &["-9", "-c"], dargs: &["-d", "-c"] })); }
    if tool_exists("zstd") { v.push(mk("zstd-19", Kind::Pipe { tool: "zstd", cargs: &["-19", "-q", "--single-thread", "-c"], dargs: &["-d", "-q", "-c"] })); }
    if tool_exists("xz") { v.push(mk("xz-9e", Kind::Pipe { tool: "xz", cargs: &["-9e", "-T1", "-c"], dargs: &["-d", "-T1", "-c"] })); }
    if tool_exists("7z") {
        v.push(mk("7z-lzma2", Kind::SevenZip { margs: &["-t7z", "-m0=lzma2", "-mx=9", "-mmt=1"] }));
        v.push(mk("7z-ppmd", Kind::SevenZip { margs: &["-t7z", "-m0=PPMd", "-mx=9", "-mmt=1"] }));
    }
    v
}

struct Measure {
    codec: String,
    bytes: u64,
    enc: Run,
    dec: Option<Run>,
    roundtrip: Option<bool>,
    error: Option<String>,
}

fn s(p: &Path) -> String { p.to_string_lossy().into_owned() }

fn measure(codec: &Codec, input: &Path, dir: &Path, nsc_exe: &Path, nsc_jobs: usize, verify: bool) -> Measure {
    let comp = dir.join("c.bin");
    let rt = dir.join("rt.bin");
    let _ = fs::remove_file(&comp);
    let _ = fs::remove_file(&rt);
    let fail = |codec: &str, e: String| Measure {
        codec: codec.into(), bytes: 0, enc: Run { ok: false, wall_s: 0.0, cpu_s: 0.0, max_rss_kb: 0 },
        dec: None, roundtrip: None, error: Some(e),
    };
    let (enc, comp_file) = match &codec.kind {
        Kind::Pipe { tool, cargs, .. } => {
            let mut a: Vec<String> = cargs.iter().map(|x| x.to_string()).collect();
            a.push(s(input));
            (run_measured(Path::new(tool), &a, None, Some(&comp)), comp.clone())
        }
        Kind::SevenZip { margs } => {
            let arch = dir.join("a.7z");
            let _ = fs::remove_file(&arch);
            let mut a: Vec<String> = vec!["a".into()];
            a.extend(margs.iter().map(|x| x.to_string()));
            a.extend(["-bso0".into(), "-bsp0".into(), s(&arch), s(input)]);
            (run_measured(Path::new("7z"), &a, None, None), arch)
        }
        Kind::Nsc { level } => {
            let a = vec!["compress".into(), format!("-{level}"), "--jobs".into(), nsc_jobs.to_string(), "-q".into(), s(input), s(&comp)];
            (run_measured(nsc_exe, &a, None, None), comp.clone())
        }
    };
    let enc = match enc { Ok(r) if r.ok => r, Ok(_) => return fail(&codec.name, "compressor exited with an error".into()), Err(e) => return fail(&codec.name, e) };
    let bytes = fs::metadata(&comp_file).map(|m| m.len()).unwrap_or(0);
    if !verify {
        return Measure { codec: codec.name.clone(), bytes, enc, dec: None, roundtrip: None, error: None };
    }
    let dec = match &codec.kind {
        Kind::Pipe { tool, dargs, .. } => {
            let mut a: Vec<String> = dargs.iter().map(|x| x.to_string()).collect();
            a.push(s(&comp_file));
            run_measured(Path::new(tool), &a, None, Some(&rt))
        }
        Kind::SevenZip { .. } => run_measured(Path::new("7z"), &["e".into(), "-so".into(), s(&comp_file)], None, Some(&rt)),
        Kind::Nsc { .. } => {
            let a = vec!["decompress".into(), "--jobs".into(), nsc_jobs.to_string(), "-q".into(), s(&comp_file), s(&rt)];
            run_measured(nsc_exe, &a, None, None)
        }
    };
    let (dec, roundtrip, error) = match dec {
        Ok(r) if r.ok => { let eq = files_equal(input, &rt); (Some(r), Some(eq), if eq { None } else { Some("ROUNDTRIP MISMATCH".into()) }) }
        Ok(r) => (Some(r), Some(false), Some("decompressor exited with an error".into())),
        Err(e) => (None, Some(false), Some(e)),
    };
    let _ = fs::remove_file(&comp_file);
    let _ = fs::remove_file(&rt);
    Measure { codec: codec.name.clone(), bytes, enc, dec, roundtrip, error }
}

// ------------------------------------------------------------------ the command
pub struct BenchOpts {
    pub paths: Vec<String>,
    pub levels: Vec<u32>,
    pub classical: bool,
    pub only: Option<Vec<String>>,
    pub json: Option<String>,
    pub verify: bool,
    pub nsc_jobs: usize,
    pub max_bytes: Option<u64>,
}

fn collect_files(paths: &[String], max_bytes: Option<u64>) -> Result<Vec<(PathBuf, String)>, String> {
    let mut out = vec![];
    for p in paths {
        let root = PathBuf::from(p);
        if root.is_file() {
            let cat = root.parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            out.push((root, cat));
            continue;
        }
        if !root.is_dir() { return Err(format!("bench: not a file or directory: {p}")); }
        let mut stack = vec![root.clone()];
        let mut found: Vec<PathBuf> = vec![];
        while let Some(d) = stack.pop() {
            let mut ents: Vec<PathBuf> = fs::read_dir(&d).map_err(|e| format!("read_dir {}: {e}", d.display()))?
                .filter_map(|e| e.ok().map(|e| e.path())).collect();
            ents.sort();
            for e in ents {
                let name = e.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if name.starts_with('.') || name == "MANIFEST.tsv" || name.ends_with(".tmp") { continue; }
                if e.is_dir() { stack.push(e); } else if e.is_file() { found.push(e); }
            }
        }
        found.sort();
        for f in found {
            let rel = f.strip_prefix(&root).unwrap_or(&f).to_path_buf();
            let cat = if rel.components().count() > 1 {
                rel.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned()).unwrap_or_default()
            } else {
                root.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            };
            out.push((f, cat));
        }
    }
    if let Some(m) = max_bytes {
        out.retain(|(f, _)| fs::metadata(f).map(|md| md.len() <= m).unwrap_or(false));
    }
    Ok(out)
}

fn qs(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn human(n: u64) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1000.0 && i < 3 { v /= 1000.0; i += 1; }
    if i == 0 { format!("{n} B") } else { format!("{v:.2} {}", U[i]) }
}

pub fn run(opts: &BenchOpts) -> Result<(), String> {
    let files = collect_files(&opts.paths, opts.max_bytes)?;
    if files.is_empty() { return Err("bench: no input files".into()); }
    let nsc_exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let mut codecs: Vec<Codec> = if opts.classical { classical_codecs() } else { vec![] };
    for &l in &opts.levels { codecs.push(Codec { name: format!("nsc-{l}"), kind: Kind::Nsc { level: l } }); }
    if let Some(only) = &opts.only { codecs.retain(|c| only.iter().any(|o| c.name.starts_with(o.as_str()))); }
    if codecs.is_empty() { return Err("bench: no codecs selected (use --classical and/or --levels)".into()); }
    let dir = scratch_dir("run");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;

    let total_in: u64 = files.iter().map(|(f, _)| fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum();
    eprintln!("bench: {} files, {} | codecs: {}", files.len(), human(total_in),
              codecs.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", "));
    let mut rows: Vec<(PathBuf, String, u64, Vec<Measure>)> = vec![];
    for (i, (f, cat)) in files.iter().enumerate() {
        let size = fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        let mut ms = vec![];
        for c in &codecs {
            let m = measure(c, f, &dir, &nsc_exe, opts.nsc_jobs, opts.verify);
            if let Some(e) = &m.error { eprintln!("  ! {} on {}: {e}", m.codec, f.display()); }
            ms.push(m);
        }
        let best = ms.iter().filter(|m| m.error.is_none()).min_by_key(|m| m.bytes).map(|m| m.codec.clone()).unwrap_or_default();
        eprintln!("[{}/{}] {} ({}) best: {}", i + 1, files.len(), f.display(), human(size), best);
        rows.push((f.clone(), cat.clone(), size, ms));
    }
    let _ = fs::remove_dir_all(&dir);
    print_tables(&codecs, &rows);
    if let Some(j) = &opts.json { write_json(j, &rows)?; eprintln!("bench: wrote {j}"); }
    let bad = rows.iter().flat_map(|r| r.3.iter()).filter(|m| m.roundtrip == Some(false)).count();
    if bad > 0 { return Err(format!("{bad} measurement(s) failed the roundtrip check")); }
    Ok(())
}

fn print_tables(codecs: &[Codec], rows: &[(PathBuf, String, u64, Vec<Measure>)]) {
    let name_w = rows.iter().map(|r| display_name(&r.0, &r.1).len()).max().unwrap_or(10).clamp(12, 44);
    println!("\nbits per byte (lower is better; * = best, ! = failed)");
    print!("{:<name_w$} {:>10}", "file", "size");
    for c in codecs { print!(" {:>9}", c.name); }
    println!();
    // category totals preserve first-seen order (rows are already sorted by path)
    let mut cats: Vec<String> = vec![];
    for r in rows { if !cats.contains(&r.1) { cats.push(r.1.clone()); } }
    let mut grand_in = 0u64;
    let mut grand: Vec<(u64, f64, f64, bool)> = vec![(0, 0.0, 0.0, true); codecs.len()];
    for cat in &cats {
        let mut cin = 0u64;
        let mut tot: Vec<(u64, f64, f64, bool)> = vec![(0, 0.0, 0.0, true); codecs.len()];
        for r in rows.iter().filter(|r| &r.1 == cat) {
            cin += r.2;
            let best = r.3.iter().filter(|m| m.error.is_none()).map(|m| m.bytes).min();
            print!("{:<name_w$} {:>10}", trunc(&display_name(&r.0, &r.1), name_w), human(r.2));
            for (k, m) in r.3.iter().enumerate() {
                if m.error.is_some() { print!(" {:>9}", "!"); tot[k].3 = false; continue; }
                let bpb = if r.2 > 0 { m.bytes as f64 * 8.0 / r.2 as f64 } else { 0.0 };
                let mark = if Some(m.bytes) == best { "*" } else { "" };
                print!(" {:>9}", format!("{bpb:.3}{mark}"));
                tot[k].0 += m.bytes;
                tot[k].1 += m.enc.wall_s;
                tot[k].2 += m.dec.as_ref().map(|d| d.wall_s).unwrap_or(0.0);
            }
            println!();
        }
        print!("{:<name_w$} {:>10}", format!("= {cat}"), human(cin));
        for t in &tot {
            if !t.3 { print!(" {:>9}", "!"); continue; }
            print!(" {:>9}", format!("{:.3}", if cin > 0 { t.0 as f64 * 8.0 / cin as f64 } else { 0.0 }));
        }
        println!("\n");
        grand_in += cin;
        for (g, t) in grand.iter_mut().zip(&tot) { g.0 += t.0; g.1 += t.1; g.2 += t.2; g.3 &= t.3; }
    }
    print!("{:<name_w$} {:>10}", "TOTAL", human(grand_in));
    for g in &grand { print!(" {:>9}", if g.3 { format!("{:.3}", g.0 as f64 * 8.0 / grand_in.max(1) as f64) } else { "!".into() }); }
    println!();
    print!("{:<name_w$} {:>10}", "  compressed", "");
    for g in &grand { print!(" {:>9}", if g.3 { human(g.0) } else { "!".into() }); }
    println!();
    print!("{:<name_w$} {:>10}", "  enc MB/s", "");
    for g in &grand { print!(" {:>9}", if g.1 > 0.0 { format!("{:.2}", grand_in as f64 / 1e6 / g.1) } else { "-".into() }); }
    println!();
    print!("{:<name_w$} {:>10}", "  dec MB/s", "");
    for g in &grand { print!(" {:>9}", if g.2 > 0.0 { format!("{:.2}", grand_in as f64 / 1e6 / g.2) } else { "-".into() }); }
    println!();
    let mut peak: Vec<u64> = vec![0; codecs.len()];
    for r in rows { for (k, m) in r.3.iter().enumerate() {
        peak[k] = peak[k].max(m.enc.max_rss_kb).max(m.dec.as_ref().map(|d| d.max_rss_kb).unwrap_or(0));
    } }
    print!("{:<name_w$} {:>10}", "  peak RSS", "");
    for p in &peak { print!(" {:>9}", format!("{}MB", p / 1024)); }
    println!();
}

fn display_name(p: &Path, cat: &str) -> String {
    let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if cat.is_empty() { name } else { format!("{cat}/{name}") }
}

fn trunc(s: &str, w: usize) -> String {
    if s.chars().count() <= w { s.to_string() } else { format!("{}…", s.chars().take(w - 1).collect::<String>()) }
}

fn write_json(path: &str, rows: &[(PathBuf, String, u64, Vec<Measure>)]) -> Result<(), String> {
    let mut files = vec![];
    for (p, cat, size, ms) in rows {
        let res: Vec<String> = ms.iter().map(|m| {
            let dec = m.dec.as_ref();
            format!("{{\"codec\":{},\"bytes\":{},\"enc_s\":{:.4},\"enc_cpu_s\":{:.4},\"enc_rss_kb\":{},\"dec_s\":{},\"dec_rss_kb\":{},\"roundtrip\":{},\"error\":{}}}",
                qs(&m.codec), m.bytes, m.enc.wall_s, m.enc.cpu_s, m.enc.max_rss_kb,
                dec.map(|d| format!("{:.4}", d.wall_s)).unwrap_or_else(|| "null".into()),
                dec.map(|d| d.max_rss_kb.to_string()).unwrap_or_else(|| "null".into()),
                m.roundtrip.map(|b| b.to_string()).unwrap_or_else(|| "null".into()),
                m.error.as_deref().map(qs).unwrap_or_else(|| "null".into()))
        }).collect();
        files.push(format!("{{\"path\":{},\"category\":{},\"bytes\":{},\"results\":[{}]}}",
            qs(&p.to_string_lossy()), qs(cat), size, res.join(",")));
    }
    let body = format!("{{\"nsc_version\":{},\"files\":[\n{}\n]}}\n", qs(env!("CARGO_PKG_VERSION")), files.join(",\n"));
    fs::write(path, body).map_err(|e| format!("write {path}: {e}"))
}
//! Shared codec core: little-endian byte readers, probability quantizer, and the
//! `encode_tokens`/`decode_tokens` drivers used by both the CLI (`main.rs`) and the parity
//! harness (`selftest.rs`). Byte-exact by construction: the identical int8 forward + feature
//! model drive encode and decode, so their coder tables match bit-for-bit.
use crate::artifacts::Model;
use crate::features::{FeatureModel, O2Table};
use crate::forward;
use crate::range_coder::{RangeDecoder, RangeEncoder};
use std::fs;

pub const QUANT_BITS: u32 = 22;

// ---- little-endian readers over an in-memory buffer (o = running offset) ----
pub fn ru16(b: &[u8], o: &mut usize) -> u16 { let v = u16::from_le_bytes(b[*o..*o + 2].try_into().unwrap()); *o += 2; v }
pub fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
pub fn ru64(b: &[u8], o: &mut usize) -> u64 { let v = u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v }
pub fn ri32(b: &[u8], o: &mut usize) -> i32 { let v = i32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
pub fn rf64(b: &[u8], o: &mut usize) -> f64 { let v = f64::from_le_bytes(b[*o..*o + 8].try_into().unwrap()); *o += 8; v }
pub fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

fn argmax(f: &[i64]) -> usize { let mut b = 0; for i in 1..f.len() { if f[i] > f[b] { b = i; } } b }

/// Port of nsc._quantize(p, total_bits): floor(p*total) clamped>=1, renormalized to `total`, cumsum.
pub fn quantize(p: &[f32], total_bits: u32) -> Vec<i64> {
    let v = p.len();
    let mut total: i64 = 1 << total_bits;
    while total < v as i64 { total <<= 1; }
    let mut f: Vec<i64> = p.iter().map(|&x| ((x as f64 * total as f64).floor() as i64).max(1)).collect();
    let diff = total - f.iter().sum::<i64>();
    if diff != 0 {
        let j = argmax(&f); f[j] = (f[j] + diff).max(1);
        let diff2 = total - f.iter().sum::<i64>();
        if diff2 != 0 { let j = argmax(&f); f[j] = (f[j] + diff2).max(1); }
    }
    let mut cum = vec![0i64; v + 1];
    for s in 0..v { cum[s + 1] = cum[s] + f[s]; }
    cum
}

// ---- .ids token-id files: magic "NIDS" | n u32 | n x token-id u32 (shared with tok.py) ----
pub fn read_ids(path: &str) -> Result<Vec<usize>, String> {
    let b = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if b.len() < 8 || &b[0..4] != b"NIDS" { return Err(format!("bad .ids magic in {path}")); }
    let mut o = 4usize;
    let n = ru32(&b, &mut o) as usize;
    if b.len() < 8 + n * 4 { return Err(format!("truncated .ids in {path}")); }
    Ok((0..n).map(|_| ru32(&b, &mut o) as usize).collect())
}

pub fn write_ids(path: &str, ids: &[usize]) -> Result<(), String> {
    let mut out = Vec::with_capacity(8 + ids.len() * 4);
    out.extend_from_slice(b"NIDS");
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &t in ids { out.extend_from_slice(&(t as u32).to_le_bytes()); }
    fs::write(path, out).map_err(|e| format!("write {path}: {e}"))
}

/// Drive the int8 forward + range coder over a token stream. The neural window resets every
/// `seg_len` tokens (matching Python's `if i % seg == 0: reset_window()`); the feature model runs
/// continuously. Returns (payload, n_bits). The first token is stored raw by the caller; only
/// tokens[1..] are entropy-coded.
///
/// The expensive neural forward (~96% of the cost) is window-local — the [`forward::Decoder`] is
/// reset every `seg_len` tokens, so no window depends on any earlier window's neural state. Because
/// encoding also knows the whole token stream up front, every window's contribution to the range
/// coder (the `(cum[next], cum[next+1], total)` triple per position) can be computed independently
/// and in parallel with `jobs` worker threads, then the (cheap) range coding is replayed
/// sequentially. This is byte-exact: identical arithmetic in identical order into the coder. Pass
/// `jobs == 1` to force the plain single-threaded loop (e.g. when the caller already parallelises
/// across files, as `batch` does).
pub fn encode_tokens(model: &Model, o2: &O2Table, tokens: &[usize], jobs: usize) -> (Vec<u8>, usize) {
    let (v, seg) = (model.cfg.vocab_size, model.cfg.seg_len);
    let n = tokens.len();

    // Feature tuples are stateful across the whole stream but cheap; compute them sequentially once.
    let mut feat = FeatureModel::new(o2);
    let feats: Vec<(usize, f32, usize, f32)> = tokens.iter().map(|&t| feat.push(t)).collect();

    // Per-position range-coder input: (cum[next], cum[next+1], total). The final position is never
    // encoded (it has no successor) so its triple stays the harmless (0, 0, 0).
    let mut triples: Vec<(u64, u64, u64)> = vec![(0, 0, 0); n];
    let nwin = n.div_ceil(seg);
    let njobs = jobs.max(1).min(nwin.max(1));

    let fill_window = |w: usize, out: &mut [(u64, u64, u64)]| {
        let start = w * seg;
        let mut dec = forward::Decoder::new(model);
        for (off, slot) in out.iter_mut().enumerate() {
            let i = start + off;
            let (o2t, o2c, mt, mc) = feats[i];
            let probs = dec.step(tokens[i], o2t, o2c, mt, mc);
            if i + 1 < n {
                let cum = quantize(&probs, QUANT_BITS);
                let s = tokens[i + 1];
                *slot = (cum[s] as u64, cum[s + 1] as u64, cum[v] as u64);
            }
        }
    };

    if njobs <= 1 {
        for (w, out) in triples.chunks_mut(seg).enumerate() { fill_window(w, out); }
    } else {
        // Deal the disjoint per-window mutable slices round-robin into per-thread buckets, then run
        // one scoped thread per bucket. `chunks_mut(seg)` yields exactly one window per chunk.
        let mut buckets: Vec<Vec<(usize, &mut [(u64, u64, u64)])>> = (0..njobs).map(|_| Vec::new()).collect();
        for (w, out) in triples.chunks_mut(seg).enumerate() { buckets[w % njobs].push((w, out)); }
        std::thread::scope(|scope| {
            for bucket in buckets {
                let fw = &fill_window;
                scope.spawn(move || { for (w, out) in bucket { fw(w, out); } });
            }
        });
    }

    let mut enc = RangeEncoder::new();
    for i in 0..n.saturating_sub(1) {
        let (lo, hi, tot) = triples[i];
        enc.encode(lo, hi, tot);
    }
    enc.finish()
}

/// Inverse of `encode_tokens`: the same forward drives the decoder, so the coder tables match
/// bit-for-bit and every token is recovered. Byte-exact by construction.
pub fn decode_tokens(model: &Model, o2: &O2Table, first: usize, n: usize, payload: Vec<u8>, nbits: usize) -> Vec<usize> {
    let seg = model.cfg.seg_len;
    let mut feat = FeatureModel::new(o2);
    let mut dec = forward::Decoder::new(model);
    let mut rdec = RangeDecoder::new(payload, nbits);
    let mut rec: Vec<usize> = Vec::with_capacity(n);
    rec.push(first);
    for i in 0..n.saturating_sub(1) {
        if i % seg == 0 { dec = forward::Decoder::new(model); }
        let (o2t, o2c, mt, mc) = feat.push(rec[i]);
        let probs = dec.step(rec[i], o2t, o2c, mt, mc);
        let cum: Vec<u64> = quantize(&probs, QUANT_BITS).iter().map(|&x| x as u64).collect();
        rec.push(rdec.decode(&cum));
    }
    rec
}
//! `.nsc` compressed-file container.
//!
//! v2 (`NSC2`) records which model version produced the file, so `decompress` can select the exact
//! model from the registry — a file stays decodable after the default model is upgraded, and a
//! mismatched/tampered registry is caught by the fingerprint. v1 (`NSCZ`) had no model binding and
//! is still readable (the caller supplies the model).
//!
//! v2 layout: magic "NSC2" | fmt u16 (=2) | id_len u16 | id bytes | fingerprint u64 |
//!            vocab u32 | seg u32 | n_tokens u32 | first_token u32 | n_bits u32 | payload
//! v1 layout: magic "NSCZ" | vocab u32 | seg u32 | n_tokens u32 | first_token u32 | n_bits u32 | payload
//!
//! Stored layout: magic "NSCR" | fmt u16 (=1) | orig_len u64 | raw bytes verbatim.
//! Used as a universal fallback: already-compressed / binary input (mp4, images, archives) that the
//! neural model would only expand — and might exhaust memory to tokenize — is stored as-is so
//! `compress`/`decompress` always round-trips byte-exact and never fails.
use crate::codec::{ru16, ru32, ru64};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

pub const FORMAT_VERSION: u16 = 2;
pub const STORED_MAGIC: &[u8; 4] = b"NSCR";
const STORED_HEADER_LEN: u64 = 4 + 2 + 8; // magic + fmt + orig_len

pub struct Header {
    pub model_id: Option<String>, // None for legacy NSCZ and stored files
    pub fingerprint: u64,         // model fingerprint; 0 for legacy/stored
    pub vocab: usize,
    pub seg: usize,
    pub n: usize,                 // neural: token count; stored: original byte length
    pub first: usize,
    pub n_bits: usize,
    pub stored: bool,             // true => NSCR (raw fallback), no model/tokenizer needed
}

pub struct Container {
    pub header: Header,
    pub payload: Vec<u8>,
}

/// Serialize a v2 container to `path`.
pub fn write(path: &str, model_id: &str, fingerprint: u64, vocab: usize, seg: usize,
             n: usize, first: usize, n_bits: usize, payload: &[u8]) -> Result<(), String> {
    let id = model_id.as_bytes();
    if id.len() > u16::MAX as usize { return Err("model id too long".into()); }
    let mut out = Vec::with_capacity(32 + id.len() + payload.len());
    out.extend_from_slice(b"NSC2");
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(id.len() as u16).to_le_bytes());
    out.extend_from_slice(id);
    out.extend_from_slice(&fingerprint.to_le_bytes());
    for x in [vocab as u32, seg as u32, n as u32, first as u32, n_bits as u32] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(payload);
    fs::write(path, &out).map_err(|e| format!("write {path}: {e}"))
}

/// Parse a header from a byte prefix; returns (header, offset-of-payload). Cheap: reads no payload.
pub fn parse_header(b: &[u8]) -> Result<(Header, usize), String> {
    if b.len() < 4 { return Err("file too short".into()); }
    let mut o = 4usize;
    match &b[0..4] {
        b"NSC2" => {
            if b.len() < 8 { return Err("truncated NSC2 header".into()); }
            let _fmt = ru16(b, &mut o);
            let id_len = ru16(b, &mut o) as usize;
            if b.len() < o + id_len + 8 + 20 { return Err("truncated NSC2 header".into()); }
            let model_id = String::from_utf8(b[o..o + id_len].to_vec())
                .map_err(|_| "bad model id utf8".to_string())?;
            o += id_len;
            let fingerprint = ru64(b, &mut o);
            let vocab = ru32(b, &mut o) as usize;
            let seg = ru32(b, &mut o) as usize;
            let n = ru32(b, &mut o) as usize;
            let first = ru32(b, &mut o) as usize;
            let n_bits = ru32(b, &mut o) as usize;
            Ok((Header { model_id: Some(model_id), fingerprint, vocab, seg, n, first, n_bits, stored: false }, o))
        }
        b"NSCR" => {
            if b.len() < STORED_HEADER_LEN as usize { return Err("truncated NSCR header".into()); }
            let _fmt = ru16(b, &mut o);
            let orig = ru64(b, &mut o) as usize;
            Ok((Header { model_id: None, fingerprint: 0, vocab: 0, seg: 0, n: orig, first: 0, n_bits: 0, stored: true }, o))
        }
        b"NSCZ" => {
            if b.len() < 24 { return Err("truncated NSCZ header".into()); }
            let vocab = ru32(b, &mut o) as usize;
            let seg = ru32(b, &mut o) as usize;
            let n = ru32(b, &mut o) as usize;
            let first = ru32(b, &mut o) as usize;
            let n_bits = ru32(b, &mut o) as usize;
            Ok((Header { model_id: None, fingerprint: 0, vocab, seg, n, first, n_bits, stored: false }, o))
        }
        _ => Err("not an .nsc file (bad magic)".into()),
    }
}

/// Read a full container (header + payload) from `path`.
pub fn read(path: &str) -> Result<Container, String> {
    let b = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let (header, off) = parse_header(&b)?;
    Ok(Container { header, payload: b[off..].to_vec() })
}

/// Read just the header from `path` (for `nsc info`, and to route decompress) without loading the
/// payload. Reads only a small prefix, so it stays cheap even on multi-hundred-MB stored files.
pub fn read_header(path: &str) -> Result<Header, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut buf = [0u8; 8192];
    let mut got = 0usize;
    loop {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => { got += k; if got == buf.len() { break; } }
            Err(e) => return Err(format!("read {path}: {e}")),
        }
    }
    Ok(parse_header(&buf[..got])?.0)
}

/// Write a stored (raw, no-neural) container: a small header followed by `src` copied verbatim.
/// Streams the copy, so arbitrarily large files never load fully into memory. Returns bytes written.
pub fn write_stored(dst: &str, src: &Path) -> Result<u64, String> {
    let orig_len = fs::metadata(src).map_err(|e| format!("stat {}: {e}", src.display()))?.len();
    let mut r = fs::File::open(src).map_err(|e| format!("open {}: {e}", src.display()))?;
    let mut w = std::io::BufWriter::new(fs::File::create(dst).map_err(|e| format!("create {dst}: {e}"))?);
    w.write_all(STORED_MAGIC)
        .and_then(|_| w.write_all(&1u16.to_le_bytes()))
        .and_then(|_| w.write_all(&orig_len.to_le_bytes()))
        .map_err(|e| format!("write {dst}: {e}"))?;
    let copied = std::io::copy(&mut r, &mut w).map_err(|e| format!("copy -> {dst}: {e}"))?;
    w.flush().map_err(|e| format!("flush {dst}: {e}"))?;
    if copied != orig_len { return Err(format!("stored copy short: {copied} != {orig_len} bytes")); }
    Ok(STORED_HEADER_LEN + copied)
}

/// Extract a stored (NSCR) container to `dst`, streaming and verifying the declared length. Returns
/// the number of payload bytes written.
pub fn extract_stored(src: &str, dst: &str) -> Result<u64, String> {
    let mut r = fs::File::open(src).map_err(|e| format!("open {src}: {e}"))?;
    let mut hdr = [0u8; STORED_HEADER_LEN as usize];
    r.read_exact(&mut hdr).map_err(|e| format!("read {src}: {e}"))?;
    if &hdr[0..4] != STORED_MAGIC { return Err("not a stored .nsc file".into()); }
    let orig_len = u64::from_le_bytes(hdr[6..14].try_into().unwrap());
    let mut w = std::io::BufWriter::new(fs::File::create(dst).map_err(|e| format!("create {dst}: {e}"))?);
    let copied = std::io::copy(&mut r, &mut w).map_err(|e| format!("copy -> {dst}: {e}"))?;
    w.flush().map_err(|e| format!("flush {dst}: {e}"))?;
    if copied != orig_len { return Err(format!("stored container corrupt: {copied} bytes != declared {orig_len}")); }
    Ok(copied)
}
//! Online classical feature model — a bit-exact port of `IncrementalFeatures.push`
//! (test_compression_integrity.py:266). The order-2 expert is a static lookup into the
//! exported FastO2 table (`o2.nsc`); the match expert is rebuilt online from the decoded
//! prefix. Because encoder and decoder both replay the SAME prefix through this model, they
//! compute identical `(o2_top, o2_conf, match_top, match_conf)` features — hence identical
//! coder tables and a byte-exact round-trip. This is what lets the codec process arbitrary
//! input without shipping the Python-side `ref.nsc` feature dump.
use std::collections::HashMap;
use std::fs;

/// FastO2 order-2 table: for a two-token context key `a*V + b`, the most likely next token
/// `top` and its confidence `conf = count(top)/total`. `ctx` is strictly ascending so the
/// lookup is a binary search (mirrors Python's `np.searchsorted` + exact-match check).
pub struct O2Table {
    pub v: usize,
    pub k: usize, // match order (order2_match_order)
    pub cap: f32, // match_len_cap
    pub ctx: Vec<i64>,
    pub top: Vec<i32>,
    pub conf: Vec<f32>,
}

fn ru32(b: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }
fn rf32(b: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes(b[*o..*o + 4].try_into().unwrap()); *o += 4; v }

impl O2Table {
    /// Parse `o2.nsc` (magic "NO21" | V u32 | K u32 | cap f32 | n u32 | ctx[i64] | top[i32] | conf[f32]).
    pub fn load(path: &str) -> O2Table {
        let b = fs::read(path).expect("o2.nsc missing (run: python export_o2.py)");
        assert_eq!(&b[0..4], b"NO21", "bad o2.nsc magic");
        let mut o = 4usize;
        let v = ru32(&b, &mut o) as usize;
        let k = ru32(&b, &mut o) as usize;
        let cap = rf32(&b, &mut o);
        let n = ru32(&b, &mut o) as usize;
        let ctx: Vec<i64> = (0..n).map(|i| i64::from_le_bytes(b[o + i * 8..o + i * 8 + 8].try_into().unwrap())).collect();
        o += n * 8;
        let top: Vec<i32> = (0..n).map(|i| i32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        o += n * 4;
        let conf: Vec<f32> = (0..n).map(|i| f32::from_le_bytes(b[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        O2Table { v, k, cap, ctx, top, conf }
    }
}

/// Continuous online feature model — NEVER reset at window boundaries (matches Python's
/// `compute_features`, which runs over the whole token array before windowing).
pub struct FeatureModel<'a> {
    o2: &'a O2Table,
    v: usize,
    k: usize,
    cap: f64,
    tokens: Vec<usize>,
    pos: HashMap<u64, usize>, // exact packed K-gram key -> last index
    run: i64,
    prev_mtop: usize,
}

impl<'a> FeatureModel<'a> {
    pub fn new(o2: &'a O2Table) -> Self {
        FeatureModel {
            o2, v: o2.v, k: o2.k, cap: o2.cap as f64,
            tokens: Vec::new(), pos: HashMap::new(), run: 0, prev_mtop: 0,
        }
    }

    /// Append token `t`; return `(o2_top, o2_conf, match_top, match_conf)` — the features used
    /// to predict the NEXT token. Exact port of `IncrementalFeatures.push`.
    pub fn push(&mut self, t: usize) -> (usize, f32, usize, f32) {
        let i = self.tokens.len();
        self.tokens.push(t);
        let v = self.v;

        // order-2: context (tokens[i-1], t) predicts i+1 (defined for i >= 1)
        let (mut o2t, mut o2c) = (0usize, 0f32);
        if i >= 1 {
            let key = (self.tokens[i - 1] * v + t) as i64;
            if let Ok(idx) = self.o2.ctx.binary_search(&key) {
                o2t = self.o2.top[idx] as usize;
                o2c = self.o2.conf[idx];
            }
        }

        // match: exact packed K-gram ending at i -> last occurrence -> the token that followed it
        let (mut mt, mut mc) = (0usize, 0f32);
        if i >= self.k {
            let mut h: u64 = 0;
            for j in (i - self.k + 1)..=i {
                h = h * v as u64 + self.tokens[j] as u64;
            }
            if let Some(&prev) = self.pos.get(&h) {
                mt = self.tokens[prev + 1];
                self.run = if self.prev_mtop == t { self.run + 1 } else { 1 };
                // computed f64-then-f32 to match Python's `float(np.float32(min(run/cap, 0.99)))`
                mc = (self.run as f64 / self.cap).min(0.99) as f32;
            } else {
                self.run = 0;
            }
            self.pos.insert(h, i);
        }
        self.prev_mtop = mt;
        (o2t, o2c, mt, mc)
    }
}
//! int8 neural forward — bit-exact port of int8_codec.py. Integer dot products accumulate in
//! i32 (int8·int8 summed over ≤1024 ≤ 16.5M < 2^31, exact), which vectorizes well; dequant in
//! f64 then f32 + f32 bias (matches int8_codec's `.double().mul().float()+b`). round-half-to-EVEN
//! quant (matches torch.round). erf-GELU via system libm.
use crate::artifacts::Model;

#[link(name = "m")]
unsafe extern "C" {
    fn erff(x: f32) -> f32;
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + unsafe { erff(x * std::f32::consts::FRAC_1_SQRT_2) })
}
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// per-row int8 dynamic quant: scale = max|x|/127 (>=1e-30), codes = round_ties_even(x/scale).
pub fn qrow(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s = (amax / 127.0).max(1e-30);
    let q = x.iter().map(|&v| (v / s).round_ties_even().clamp(-128.0, 127.0) as i8).collect();
    (q, s)
}

/// integer dot product of two int8 slices (exact in i32). Bit-identical to the scalar sum:
/// int8·int8 ≤ 16129 and i32 accumulation is associative, so SIMD reordering changes no bit.
#[inline]
fn idot(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { idot_avx2(a, b) };
        }
    }
    idot_scalar(a, b)
}

#[inline]
fn idot_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

/// AVX2 int8 dot: widen i8→i16 (cvtepi8_epi16) then madd_epi16 (pairwise i16·i16→i32, summed).
/// Accumulates in eight i32 lanes; over inp≤1024 a lane holds ≤128·16129 ≈ 2.06M, no overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idot_avx2(a: &[i8], b: &[i8]) -> i32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    // 32 int8 per iteration: two 16-wide widened madds.
    while i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i + 16) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0, b0));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a0, b0));
        i += 16;
    }
    // horizontal sum of the eight i32 lanes.
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let s128 = _mm_add_epi32(lo, hi);
    let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0b01_00_11_10));
    let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0b00_00_00_01));
    let mut acc_s = _mm_cvtsi128_si32(s32);
    while i < n {
        acc_s += *pa.add(i) as i32 * *pb.add(i) as i32;
        i += 1;
    }
    acc_s
}}

/// int8 Linear: y[o] = f32( (Σ xq·code[o]) · (xs·scale[o]) ) + bias[o].
pub fn mm(x: &[f32], code: &[i8], scale: &[f64], bias: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let (xq, xs) = qrow(x);
    let xs = xs as f64;
    let mut y = vec![0f32; out];
    for o in 0..out {
        let base = o * inp;
        let acc = idot(&xq, &code[base..base + inp]);
        y[o] = (acc as f64 * (xs * scale[o])) as f32 + bias[o];
    }
    y
}

pub fn conf_state(m: &Model, tok: usize, o2c: f32, mc: f32) -> [f32; 4] {
    [m.o1_max[tok], m.o1_ent[tok], o2c, mc] // precomputed per-token (depends only on the row)
}

/// FiLM: gb = film2(gelu(film0(conf))); (1+gamma)*hid + beta.
fn film(m: &Model, prefix: &str, hid: &[f32], conf: &[f32]) -> Vec<f32> {
    let (f0c, f0s, f0b, f0o, f0i) = m.lin(&format!("{prefix}.film0"));
    let t0: Vec<f32> = mm(conf, f0c, f0s, f0b, f0o, f0i).iter().map(|&v| gelu(v)).collect();
    let (f2c, f2s, f2b, f2o, f2i) = m.lin(&format!("{prefix}.film2"));
    let gb = mm(&t0, f2c, f2s, f2b, f2o, f2i);
    let n = hid.len();
    (0..n).map(|k| (1.0 + gb[k]) * hid[k] + gb[n + k]).collect()
}

/// GRU(2L)+FiLM+to_resid for a single fresh token (pos 0) — Stage-C verification.
pub fn resid_gru_pos0(m: &Model, tok: usize, o2c: f32, mc: f32) -> (Vec<f32>, Vec<f32>, [f32; 4]) {
    let cfg = m.cfg;
    let (d, h) = (cfg.d_model, cfg.gru_hidden);
    let e = m.get("embed").f32[tok * d..(tok + 1) * d].to_vec();
    let conf = conf_state(m, tok, o2c, mc);
    let mut hs: Vec<Vec<f32>> = (0..cfg.gru_layers).map(|_| vec![0f32; h]).collect();
    let mut inp = e;
    for l in 0..cfg.gru_layers {
        let (wc, ws, wb, wo, wi) = m.lin(&format!("gru.wih.l{l}"));
        let gi = mm(&inp, wc, ws, wb, wo, wi);
        let (hc, hsc, hb, ho, hi) = m.lin(&format!("gru.whh.l{l}"));
        let gg = mm(&hs[l], hc, hsc, hb, ho, hi);
        let mut hnew = vec![0f32; h];
        for k in 0..h {
            let r = sigmoid(gi[k] + gg[k]);
            let z = sigmoid(gi[h + k] + gg[h + k]);
            let n = (gi[2 * h + k] + r * gg[2 * h + k]).tanh();
            hnew[k] = (1.0 - z) * n + z * hs[l][k];
        }
        hs[l] = hnew.clone();
        inp = hnew;
    }
    let gru_h = hs[cfg.gru_layers - 1].clone();
    let ghf = film(m, "local", &gru_h, &conf);
    let (trc, trs, trb, tro, tri) = m.lin("local.to_resid");
    (mm(&ghf, trc, trs, trb, tro, tri), ghf, conf)
}

pub fn layernorm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    (0..x.len()).map(|i| (x[i] - mean) * inv * w[i] + b[i]).collect()
}

pub fn peaked(top: usize, conf: f32, v: usize) -> Vec<f32> {
    let c = conf.clamp(1e-4, 1.0 - 1e-4);
    let base = (1.0 - c) / ((v - 1) as f32);
    let mut p = vec![base; v];
    p[top.min(v - 1)] = c;
    p
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let mx = z.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = z.iter().map(|&v| (v - mx).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|&v| v / s).collect()
}

fn apply_rope(x: &mut [f32], pos: usize, hd: usize) {
    let half = hd / 2;
    let mut out = vec![0f32; hd];
    for i in 0..half {
        let inv = ((2 * i) as f32 * (-(10000f32.ln()) / hd as f32)).exp();
        let (s, c) = (pos as f32 * inv).sin_cos();
        out[i] = x[i] * c - x[i + half] * s;
        out[i + half] = x[i + half] * c + x[i] * s;
    }
    x.copy_from_slice(&out);
}

/// Stateful int8 KV-cache decoder — reproduces int8_codec.Int8Decoder (T-pad softmax).
pub struct Decoder<'a> {
    m: &'a Model,
    pos: usize,
    gru_hs: Vec<Vec<f32>>,
    kcode: Vec<Vec<Vec<i8>>>, // [block][key] -> [h*hd]
    kscale: Vec<Vec<Vec<f32>>>, // [block][key] -> [h]
    vcode: Vec<Vec<Vec<i8>>>,
    vscale: Vec<Vec<Vec<f32>>>,
}

impl<'a> Decoder<'a> {
    pub fn new(m: &'a Model) -> Self {
        let nb = m.cfg.n_layers;
        Decoder {
            m, pos: 0,
            gru_hs: (0..m.cfg.gru_layers).map(|_| vec![0f32; m.cfg.gru_hidden]).collect(),
            kcode: vec![vec![]; nb], kscale: vec![vec![]; nb],
            vcode: vec![vec![]; nb], vscale: vec![vec![]; nb],
        }
    }

    pub fn step(&mut self, tok: usize, o2t: usize, o2c: f32, mt: usize, mc: f32) -> Vec<f32> {
        let m = self.m;
        let cfg = m.cfg;
        let (d, h, hd, gh) = (cfg.d_model, cfg.n_heads, cfg.d_model / cfg.n_heads, cfg.gru_hidden);
        let pos = self.pos;
        let e = m.get("embed").f32[tok * d..(tok + 1) * d].to_vec();
        let conf = conf_state(m, tok, o2c, mc);
        let o1_max = conf[0];

        // GRU
        let mut inp = e.clone();
        for l in 0..cfg.gru_layers {
            let (wc, ws, wb, wo, wi) = m.lin(&format!("gru.wih.l{l}"));
            let gi = mm(&inp, wc, ws, wb, wo, wi);
            let (hc, hsc, hb, ho, hi) = m.lin(&format!("gru.whh.l{l}"));
            let gg = mm(&self.gru_hs[l], hc, hsc, hb, ho, hi);
            let mut hnew = vec![0f32; gh];
            for k in 0..gh {
                let r = sigmoid(gi[k] + gg[k]);
                let z = sigmoid(gi[gh + k] + gg[gh + k]);
                let n = (gi[2 * gh + k] + r * gg[2 * gh + k]).tanh();
                hnew[k] = (1.0 - z) * n + z * self.gru_hs[l][k];
            }
            self.gru_hs[l] = hnew.clone();
            inp = hnew;
        }
        let ghf = film(m, "local", &self.gru_hs[cfg.gru_layers - 1].clone(), &conf);
        let resid_gru = { let (c, s, b, o, i) = m.lin("local.to_resid"); mm(&ghf, c, s, b, o, i) };

        // transformer
        let mut x = e.clone();
        for bi in 0..cfg.n_layers {
            let xn = layernorm(&x, &m.get(&format!("blk{bi}.ln1.w")).f32, &m.get(&format!("blk{bi}.ln1.b")).f32);
            let qkv = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.qkv")); mm(&xn, c, s, b, o, i) };
            let (mut q, mut k, v) = (qkv[0..d].to_vec(), qkv[d..2 * d].to_vec(), qkv[2 * d..3 * d].to_vec());
            let (mut kc, mut ks, mut vc, mut vs) = (vec![0i8; d], vec![0f32; h], vec![0i8; d], vec![0f32; h]);
            for hh in 0..h {
                apply_rope(&mut q[hh * hd..(hh + 1) * hd], pos, hd);
                apply_rope(&mut k[hh * hd..(hh + 1) * hd], pos, hd);
                let (c, sc) = qrow(&k[hh * hd..(hh + 1) * hd]); kc[hh * hd..(hh + 1) * hd].copy_from_slice(&c); ks[hh] = sc;
                let (c, sc) = qrow(&v[hh * hd..(hh + 1) * hd]); vc[hh * hd..(hh + 1) * hd].copy_from_slice(&c); vs[hh] = sc;
            }
            self.kcode[bi].push(kc); self.kscale[bi].push(ks);
            self.vcode[bi].push(vc); self.vscale[bi].push(vs);
            let np = pos + 1;
            let mut out = vec![0f32; d];
            let sqhd = (hd as f32).sqrt();
            let mut iacc = vec![0i32; hd];
            for hh in 0..h {
                let (qc, qs) = qrow(&q[hh * hd..(hh + 1) * hd]);
                let mut sc = vec![0f32; np];
                for j in 0..np {
                    let acc = idot(&qc, &self.kcode[bi][j][hh * hd..(hh + 1) * hd]);
                    sc[j] = (acc as f64 * (qs as f64 * self.kscale[bi][j][hh] as f64)) as f32 / sqhd;
                }
                let att = softmax(&sc);
                let bfold: Vec<f32> = (0..np).map(|j| att[j] * self.vscale[bi][j][hh]).collect();
                let (bc, bs) = qrow(&bfold);
                // out[c] = Σ_j bc[j]·vcode[j][c] (exact i32). Key-outer/channel-inner so the inner
                // reads a contiguous vcode row and autovectorizes; same summation order → byte-exact.
                iacc.iter_mut().for_each(|x| *x = 0);
                for j in 0..np {
                    let bcj = bc[j] as i32;
                    let vrow = &self.vcode[bi][j][hh * hd..hh * hd + hd];
                    for c in 0..hd {
                        iacc[c] += bcj * vrow[c] as i32;
                    }
                }
                for c in 0..hd {
                    out[hh * hd + c] = (iacc[c] as f64 * bs as f64) as f32;
                }
            }
            let proj = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.proj")); mm(&out, c, s, b, o, i) };
            for t in 0..d { x[t] += proj[t]; }
            let x2 = layernorm(&x, &m.get(&format!("blk{bi}.ln2.w")).f32, &m.get(&format!("blk{bi}.ln2.b")).f32);
            let m0: Vec<f32> = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.mlp0")); mm(&x2, c, s, b, o, i) }.iter().map(|&v| gelu(v)).collect();
            let m1 = { let (c, s, b, o, i) = m.lin(&format!("blk{bi}.mlp2")); mm(&m0, c, s, b, o, i) };
            for t in 0..d { x[t] += m1[t]; }
        }
        let xf_h = film(m, "xformer", &layernorm(&x, &m.get("xformer.lnf.w").f32, &m.get("xformer.lnf.b").f32), &conf);
        let resid_xf = { let (c, s, b, o, i) = m.lin("xformer.to_resid"); mm(&xf_h, c, s, b, o, i) };

        // gate
        let mut qin = ghf.clone();
        qin.extend_from_slice(&xf_h);
        let q = { let (c, s, b, o, i) = m.lin("gate.q_proj"); mm(&qin, c, s, b, o, i) };
        let ma = |z: &[f32]| z.iter().map(|v| v.abs()).sum::<f32>() / z.len() as f32;
        let ef = [0.5f32, o1_max, o2c, mc, ma(&ghf), ma(&xf_h)];
        let (fc, fs, fb, fo, _fi) = m.lin("gate.feat_proj");
        let ee = &m.get("gate.expert_embed").f32;
        let gd = cfg.gate_dim;
        let mut scores = [0f32; 6];
        for e6 in 0..6 {
            let kfe = mm(&[ef[e6]], fc, fs, fb, fo, 1);
            let mut acc = 0f32;
            for kk in 0..gd { acc += q[kk] * (ee[e6 * gd + kk] + kfe[kk]); }
            scores[e6] = acc / (gd as f32).sqrt();
        }
        let w = softmax(&scores[0..4]);
        let (g0, g1) = (sigmoid(scores[4]), sigmoid(scores[5]));

        // mixture -> probs
        let v = cfg.vocab_size;
        let p0 = &m.get("order0").f32;
        let p1 = m.order1_row(tok);
        let p2 = peaked(o2t, o2c, v);
        let pm = peaked(mt, mc, v);
        let mut z = vec![0f32; v];
        for s in 0..v {
            let pcls = w[0] * p0[s] + w[1] * p1[s] + w[2] * p2[s] + w[3] * pm[s];
            z[s] = (pcls + cfg.eps).ln() + g0 * resid_gru[s] + g1 * resid_xf[s];
        }
        self.pos += 1;
        softmax(&z)
    }
}
//! `nsc` — neuro-symbolic lossless codec CLI.
//!
//! A byte-exact port of `int8_codec.py`: the int8 GRU + RoPE transformer + gate + fusion, the
//! classical order-0/1/2 + match feature model, and a range coder — all in dependency-free Rust.
//! Byte↔token conversion is delegated to `tok.py` (the one Python touch-point).
//!
//! This binary is built around a **versioned model registry** so models can be upgraded safely:
//! each compressed file records the id + fingerprint of the model that produced it, so a file is
//! always decompressed by the exact model it was compressed with — even after the default model is
//! upgraded. Batch mode loads the model once and compresses many files across CPU cores.
mod artifacts;
mod codec;
mod container;
mod features;
mod forward;
mod range_coder;
mod registry;
mod selftest;
mod studio;

use artifacts::Model;
use features::O2Table;
use registry::{ModelDir, Registry};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// ------------------------------------------------------------------ arg parsing
/// Flags that take a following value; everything else `--x` is a boolean flag.
const VALUE_FLAGS: &[&str] = &["model", "home", "out-dir", "jobs", "tier", "bpc", "notes",
                               "created", "model-nsc", "o2", "tokenizer", "port", "root"];

struct Cli {
    positionals: Vec<String>,
    kv: HashMap<String, String>,
    flags: HashSet<String>,
}
impl Cli {
    fn get(&self, k: &str) -> Option<&str> { self.kv.get(k).map(|s| s.as_str()) }
    fn has(&self, k: &str) -> bool { self.flags.contains(k) }
    fn quiet(&self) -> bool { self.has("quiet") || self.has("q") }
}

fn parse_args(argv: &[String]) -> Result<(String, Cli), String> {
    // The command is the first bare word; flags may appear before or after it.
    let mut cmd: Option<String> = None;
    let mut cli = Cli { positionals: vec![], kv: HashMap::new(), flags: HashSet::new() };
    let mut i = 0usize;
    while i < argv.len() {
        let a = &argv[i];
        if let Some(key) = a.strip_prefix("--") {
            if VALUE_FLAGS.contains(&key) {
                let v = argv.get(i + 1).ok_or_else(|| format!("flag --{key} needs a value"))?;
                cli.kv.insert(key.to_string(), v.clone());
                i += 2;
            } else {
                cli.flags.insert(key.to_string());
                i += 1;
            }
        } else if a == "-q" { cli.flags.insert("q".into()); i += 1; }
        else if a == "-h" { cli.flags.insert("help".into()); i += 1; }
        else {
            if cmd.is_none() { cmd = Some(a.clone()); } else { cli.positionals.push(a.clone()); }
            i += 1;
        }
    }
    Ok((cmd.unwrap_or_else(|| "help".into()), cli))
}

fn open_registry(cli: &Cli) -> Result<Registry, String> {
    match cli.get("home") {
        Some(h) => Ok(Registry::with_home(PathBuf::from(h))),
        None => Registry::open(),
    }
}

fn p2s(p: &Path) -> String { p.to_string_lossy().into_owned() }

/// Worker threads for a single-file encode: `--jobs` if given, else the CPU count. Encode windows
/// are independent, so this parallelises the neural forward (`batch` passes 1 — it already runs one
/// job per file).
fn jobs_from_cli(cli: &Cli) -> usize {
    cli.get("jobs").and_then(|s| s.parse::<usize>().ok()).filter(|&j| j > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

fn load_pair(m: &ModelDir) -> (Model, O2Table) {
    (Model::load(&p2s(&m.model_nsc())), O2Table::load(&p2s(&m.o2_nsc())))
}

// ------------------------------------------------------------------ tokenizer bridge
static TMP_CTR: AtomicU64 = AtomicU64::new(0);
fn temp_ids() -> PathBuf {
    let n = TMP_CTR.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    std::env::temp_dir().join(format!("nsc-{}-{}-{}.ids", std::process::id(), t, n))
}

/// Run the tokenizer shim: `python3 tok.py {enc|dec} <a> <b> <model-tokenizer.json>`.
fn run_tok(reg: &Registry, model: &ModelDir, mode: &str, a: &Path, b: &Path, quiet: bool) -> Result<(), String> {
    let py = std::env::var("NSC_PYTHON").unwrap_or_else(|_| "python3".into());
    let tok = reg.tok_py();
    if !tok.exists() { return Err(format!("tokenizer shim not found: {} (set NSC_TOK or reinstall)", tok.display())); }
    let mut c = std::process::Command::new(&py);
    c.arg(&tok).arg(mode).arg(a).arg(b).arg(model.tokenizer());
    if quiet { c.stdout(std::process::Stdio::null()); }
    let status = c.status().map_err(|e| format!("failed to launch {py}: {e} (is python3 on PATH?)"))?;
    if !status.success() { return Err(format!("tokenizer step '{mode}' failed")); }
    Ok(())
}

// ------------------------------------------------------------------ core encode/decode (one file)
fn encode_file(m: &Model, o2: &O2Table, model_id: &str, fp: u64, in_ids: &Path, out_nsc: &Path, jobs: usize) -> Result<(usize, usize), String> {
    let tokens = codec::read_ids(&p2s(in_ids))?;
    if tokens.is_empty() { return Err("empty token stream".into()); }
    let (payload, nbits) = codec::encode_tokens(m, o2, &tokens, jobs);
    container::write(&p2s(out_nsc), model_id, fp, m.cfg.vocab_size, m.cfg.seg_len,
                     tokens.len(), tokens[0], nbits, &payload)?;
    Ok((tokens.len(), payload.len()))
}

fn decode_file(m: &Model, o2: &O2Table, in_nsc: &Path, out_ids: &Path) -> Result<usize, String> {
    let c = container::read(&p2s(in_nsc))?;
    if c.header.vocab != m.cfg.vocab_size {
        return Err(format!("container vocab {} != model vocab {}", c.header.vocab, m.cfg.vocab_size));
    }
    let rec = codec::decode_tokens(m, o2, c.header.first, c.header.n, c.payload, c.header.n_bits);
    codec::write_ids(&p2s(out_ids), &rec)?;
    Ok(rec.len())
}

/// Pick the model to decompress with: explicit override > id recorded in the file > registry default.
/// Enforces the fingerprint recorded in the file so a swapped/tampered registry is caught.
fn model_for_decompress(reg: &Registry, cli: &Cli, header: &container::Header) -> Result<ModelDir, String> {
    let id = cli.get("model").or(header.model_id.as_deref());
    let model = reg.resolve(id)?;
    if header.fingerprint != 0 && header.fingerprint != model.manifest.fingerprint {
        return Err(format!(
            "model fingerprint mismatch for '{}': this file was produced by a different build of that \
             model than the one installed. Install the exact model version to decompress it.",
            model.manifest.id));
    }
    Ok(model)
}

// ------------------------------------------------------------------ commands: compress / decompress
/// Decide whether the neural (text/code) path applies. Already-compressed or binary input (mp4,
/// images, archives) would only be *expanded* by the neural model — and tokenizing a large binary
/// can exhaust memory — so such files are stored verbatim instead. Samples a prefix: a NUL byte or
/// a high density of control characters marks the file as non-textual (git's binary-detection rule).
fn looks_textual(path: &Path) -> bool {
    let mut f = match std::fs::File::open(path) { Ok(f) => f, Err(_) => return true }; // let the normal path surface the error
    let mut buf = [0u8; 8192];
    let n = std::io::Read::read(&mut f, &mut buf).unwrap_or(0);
    if n == 0 { return true; } // empty -> let the normal path report "empty token stream"
    let sample = &buf[..n];
    if sample.contains(&0) { return false; } // NUL => binary
    let ctrl = sample.iter().filter(|&&b| (b < 0x20 && b != 9 && b != 10 && b != 13) || b == 0x7f).count();
    (ctrl as f64) / (n as f64) < 0.05
}

/// Store a file verbatim (raw fallback) and print a summary. Guarantees byte-exact round-trip for
/// any input; `reason` explains why the neural path was skipped.
fn store_and_report(input: &str, output: &str, reason: &str, quiet: bool) -> Result<(), String> {
    let in_bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let out_bytes = container::write_stored(output, Path::new(input))?;
    if !quiet {
        let ratio = if out_bytes > 0 { in_bytes as f64 / out_bytes as f64 } else { 0.0 };
        println!("compress: {} -> {}  [stored: {}]  {} B -> {} B  ({:.2}x, neural skipped)",
                 input, output, reason, in_bytes, out_bytes, ratio);
    }
    Ok(())
}

fn cmd_compress(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "compress", "<input> <output.nsc>")?;
    // Universal fallback: binary/incompressible input (mp4, images, archives) or `--store` is stored
    // verbatim rather than fed to the neural model. `--neural` forces the model path regardless.
    if !cli.has("neural") && (cli.has("store") || !looks_textual(Path::new(&input))) {
        let reason = if cli.has("store") { "forced" } else { "binary" };
        return store_and_report(&input, &output, reason, cli.quiet());
    }
    let model = reg.resolve(cli.get("model"))?;
    let in_bytes = std::fs::metadata(&input).map(|md| md.len()).unwrap_or(0);
    let tmp = temp_ids();
    let neural = (|| -> Result<(usize, usize), String> {
        run_tok(reg, &model, "enc", Path::new(&input), &tmp, cli.quiet())?;
        let (m, o2) = load_pair(&model);
        encode_file(&m, &o2, &model.manifest.id, model.manifest.fingerprint, &tmp, Path::new(&output), jobs_from_cli(cli))
    })();
    if !cli.has("keep-ids") { let _ = std::fs::remove_file(&tmp); }
    match neural {
        // Neural won: keep it only if it actually beat storing the bytes raw.
        Ok((ntok, pbytes)) if in_bytes == 0 || (pbytes as u64) < in_bytes => {
            if !cli.quiet() {
                let bpb = if in_bytes > 0 { pbytes as f64 * 8.0 / in_bytes as f64 } else { 0.0 };
                let ratio = if pbytes > 0 { in_bytes as f64 / pbytes as f64 } else { 0.0 };
                println!("compress: {} -> {}  [model {}]  {} B -> {} tok -> {} B  ({:.3} bits/byte, {:.2}x)",
                         input, output, model.manifest.id, in_bytes, ntok, pbytes, bpb, ratio);
            }
            Ok(())
        }
        // Neural expanded the file: store raw instead so the ratio is never worse than ~1.0.
        Ok((_ntok, pbytes)) => {
            if !cli.quiet() { eprintln!("note: neural output {pbytes} B >= input {in_bytes} B; storing raw instead"); }
            store_and_report(&input, &output, "incompressible", cli.quiet())
        }
        // Neural path failed (e.g. tokenizer OOM on a huge file): never error — store raw instead.
        Err(e) => {
            if !cli.quiet() { eprintln!("note: neural compression unavailable ({e}); storing raw instead"); }
            store_and_report(&input, &output, "fallback", cli.quiet())
        }
    }
}

fn cmd_decompress(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "decompress", "<input.nsc> <output>")?;
    let header = container::read_header(&input)?;
    if header.stored {
        let n = container::extract_stored(&input, &output)?;
        if !cli.quiet() { println!("decompress: {} -> {}  [stored]  {} bytes", input, output, n); }
        return Ok(());
    }
    let model = model_for_decompress(reg, cli, &header)?;
    let tmp = temp_ids();
    let r = (|| {
        let (m, o2) = load_pair(&model);
        let ntok = decode_file(&m, &o2, Path::new(&input), &tmp)?;
        run_tok(reg, &model, "dec", &tmp, Path::new(&output), cli.quiet())?;
        if !cli.quiet() {
            println!("decompress: {} -> {}  [model {}]  {} tokens", input, output, model.manifest.id, ntok);
        }
        Ok(())
    })();
    if !cli.has("keep-ids") { let _ = std::fs::remove_file(&tmp); }
    r
}

// ------------------------------------------------------------------ commands: *-ids (no tokenizer)
fn cmd_compress_ids(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "compress-ids", "<in.ids> <out.nsc>")?;
    let model = reg.resolve(cli.get("model"))?;
    let (m, o2) = load_pair(&model);
    let (ntok, pbytes) = encode_file(&m, &o2, &model.manifest.id, model.manifest.fingerprint, Path::new(&input), Path::new(&output), jobs_from_cli(cli))?;
    if !cli.quiet() {
        let bpt = if ntok > 1 { pbytes as f64 * 8.0 / (ntok - 1) as f64 } else { 0.0 };
        println!("compress-ids: {} tok -> {} ({} B, {:.4} bits/token)  [model {}]", ntok, output, pbytes, bpt, model.manifest.id);
    }
    Ok(())
}

fn cmd_decompress_ids(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "decompress-ids", "<in.nsc> <out.ids>")?;
    let header = container::read_header(&input)?;
    let model = model_for_decompress(reg, cli, &header)?;
    let (m, o2) = load_pair(&model);
    let ntok = decode_file(&m, &o2, Path::new(&input), Path::new(&output))?;
    if !cli.quiet() { println!("decompress-ids: {} -> {} ({} tokens)  [model {}]", input, output, ntok, model.manifest.id); }
    Ok(())
}

// ------------------------------------------------------------------ command: batch (scalability)
fn out_path(input: &str, out_dir: Option<&str>, add: bool) -> PathBuf {
    let inp = Path::new(input);
    if add {
        // compress: <name>.nsc
        let name = format!("{}.nsc", inp.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into()));
        out_dir.map(|d| Path::new(d).join(&name)).unwrap_or_else(|| PathBuf::from(format!("{input}.nsc")))
    } else {
        // decompress: strip .nsc (else append .out)
        let base: PathBuf = if input.ends_with(".nsc") { PathBuf::from(&input[..input.len() - 4]) } else { PathBuf::from(format!("{input}.out")) };
        match out_dir {
            Some(d) => Path::new(d).join(base.file_name().unwrap_or_default()),
            None => base,
        }
    }
}

fn cmd_batch(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let sub = cli.positionals.first().map(|s| s.as_str()).ok_or("usage: nsc batch <compress|decompress> <files...>")?;
    let files: Vec<String> = cli.positionals[1..].to_vec();
    if files.is_empty() { return Err("batch: no input files given".into()); }
    let compress = match sub {
        "compress" => true,
        "decompress" => false,
        other => return Err(format!("batch: unknown sub-command '{other}' (use compress|decompress)")),
    };
    if let Some(d) = cli.get("out-dir") { std::fs::create_dir_all(d).map_err(|e| format!("mkdir {d}: {e}"))?; }
    let jobs = cli.get("jobs").and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        .max(1);

    // For decompress, every file may reference a different model; group is simplest if we resolve
    // per file. For compress, all files share one model — load it once. Both share ONE loaded model
    // per distinct id. Here we load lazily and cache by id.
    let out_dir = cli.get("out-dir");
    let keep_ids = cli.has("keep-ids");
    let quiet = cli.quiet();

    // Resolve+load each needed model once (compress: the chosen model; decompress: per-file id).
    // Cache: id -> (ModelDir, Model, O2Table). Built up-front single-threaded (loads are 273 MB).
    // model_id is None for files handled by the raw-stored fallback (binary/`--store` on compress,
    // or NSCR files on decompress): those need no model, so we don't resolve or load one for them.
    let force_store = cli.has("store");
    let force_neural = cli.has("neural");
    let mut cache: HashMap<String, (ModelDir, Model, O2Table)> = HashMap::new();
    let mut plan: Vec<(Option<String>, String, PathBuf)> = Vec::with_capacity(files.len());
    for f in &files {
        let (id, out) = if compress {
            let out = out_path(f, out_dir, true);
            if !force_neural && (force_store || !looks_textual(Path::new(f))) {
                (None, out) // stored fallback — no model
            } else {
                let md = reg.resolve(cli.get("model"))?;
                let id = md.manifest.id.clone();
                cache.entry(id.clone()).or_insert_with(|| { let (m, o2) = load_pair(&md); (md, m, o2) });
                (Some(id), out)
            }
        } else {
            let out = out_path(f, out_dir, false);
            let header = container::read_header(f)?;
            if header.stored { (None, out) } else {
                let md = model_for_decompress(reg, cli, &header)?;
                let id = md.manifest.id.clone();
                cache.entry(id.clone()).or_insert_with(|| { let (m, o2) = load_pair(&md); (md, m, o2) });
                (Some(id), out)
            }
        };
        plan.push((id, f.clone(), out));
    }

    println!("batch {}: {} files, {} job(s), {} model(s) loaded", sub, files.len(), jobs, cache.len());
    let results: Mutex<Vec<(String, Result<String, String>)>> = Mutex::new(Vec::new());
    let next = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed) as usize;
                    if idx >= plan.len() { break; }
                    let (id, input, output) = &plan[idx];
                    let model = id.as_ref().map(|i| { let (md, m, o2) = &cache[i]; (md, m, o2) });
                    let r = process_one(reg, model, input, output, compress, keep_ids, quiet);
                    results.lock().unwrap().push((input.clone(), r));
                }
            });
        }
    });

    let mut done = results.into_inner().unwrap();
    done.sort_by(|a, b| a.0.cmp(&b.0));
    let mut ok = 0usize;
    for (input, r) in &done {
        match r {
            Ok(msg) => { ok += 1; if !quiet { println!("  ✓ {input}: {msg}"); } }
            Err(e) => eprintln!("  ✗ {input}: {e}"),
        }
    }
    println!("batch {}: {}/{} succeeded", sub, ok, done.len());
    if ok < done.len() { return Err(format!("{} file(s) failed", done.len() - ok)); }
    Ok(())
}

/// One batch item. `model` is None when the raw-stored fallback applies (binary/`--store` on
/// compress, NSCR on decompress); otherwise it is the resolved+loaded model. Never returns an error
/// for a normal compress: if the neural path fails or expands, it falls back to storing the bytes.
fn process_one(reg: &Registry, model: Option<(&ModelDir, &Model, &O2Table)>, input: &str, output: &Path,
               compress: bool, keep_ids: bool, quiet: bool) -> Result<String, String> {
    let _ = quiet;
    if compress {
        let (md, m, o2) = match model {
            None => { let n = container::write_stored(&p2s(output), Path::new(input))?; return Ok(format!("stored -> {} ({} B, binary)", output.display(), n)); }
            Some(t) => t,
        };
        let in_bytes = std::fs::metadata(input).map(|md| md.len()).unwrap_or(0);
        let tmp = temp_ids();
        let neural = (|| {
            run_tok(reg, md, "enc", Path::new(input), &tmp, true)?;
            // batch runs one job per file already, so keep each encode single-threaded.
            encode_file(m, o2, &md.manifest.id, md.manifest.fingerprint, &tmp, output, 1)
        })();
        if !keep_ids { let _ = std::fs::remove_file(&tmp); }
        match neural {
            Ok((ntok, pbytes)) if in_bytes == 0 || (pbytes as u64) < in_bytes =>
                Ok(format!("{} tok -> {} ({} B)", ntok, output.display(), pbytes)),
            Ok((_ntok, _pbytes)) => {
                let n = container::write_stored(&p2s(output), Path::new(input))?;
                Ok(format!("stored -> {} ({} B, incompressible)", output.display(), n))
            }
            Err(e) => {
                let n = container::write_stored(&p2s(output), Path::new(input))?;
                Ok(format!("stored -> {} ({} B, fallback: {})", output.display(), n, e))
            }
        }
    } else {
        match model {
            None => { let n = container::extract_stored(input, &p2s(output))?; Ok(format!("stored -> {} ({} bytes)", output.display(), n)) }
            Some((md, m, o2)) => {
                let tmp = temp_ids();
                let r = (|| {
                    let ntok = decode_file(m, o2, Path::new(input), &tmp)?;
                    run_tok(reg, md, "dec", &tmp, output, true)?;
                    Ok(format!("{} tok -> {}", ntok, output.display()))
                })();
                if !keep_ids { let _ = std::fs::remove_file(&tmp); }
                r
            }
        }
    }
}

// ------------------------------------------------------------------ command: info
fn cmd_info(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc info <file.nsc>")?;
    let h = container::read_header(f)?;
    let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
    if h.stored {
        println!("file:        {f}");
        println!("format:      NSCR (stored — raw fallback, no neural compression)");
        println!("original:    {} bytes", h.n);
        println!("file size:   {} bytes", size);
        return Ok(());
    }
    println!("file:        {f}");
    println!("format:      {}", if h.model_id.is_some() { "NSC2 (versioned)" } else { "NSCZ (legacy)" });
    println!("model id:    {}", h.model_id.as_deref().unwrap_or("(none — legacy file)"));
    if h.fingerprint != 0 { println!("fingerprint: {:#018x}", h.fingerprint); }
    println!("vocab:       {}", h.vocab);
    println!("seg_len:     {}", h.seg);
    println!("tokens:      {}", h.n);
    println!("file size:   {} bytes", size);
    if h.n > 1 { println!("bits/token:  {:.4}", h.n_bits as f64 / (h.n - 1) as f64); }
    Ok(())
}

// ------------------------------------------------------------------ command: models
fn short_fp(fp: u64) -> String { format!("{:08x}", (fp >> 32) as u32) }

fn cmd_models(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let sub = cli.positionals.first().map(|s| s.as_str()).unwrap_or("list");
    match sub {
        "list" => {
            let models = reg.list()?;
            let def = reg.default_id();
            println!("registry: {}", reg.home().display());
            if models.is_empty() { println!("  (no models installed — run install.sh or `nsc models add ...`)"); return Ok(()); }
            println!("  {:<16} {:<3} {:<6} {:<7} {:<8} {:<10} {}", "ID", "DEF", "TIER", "VOCAB", "BPC", "CREATED", "FINGERPRINT");
            for m in &models {
                let man = &m.manifest;
                let is_def = if def.as_deref() == Some(&man.id) { "*" } else { "" };
                println!("  {:<16} {:<3} {:<6} {:<7} {:<8.3} {:<10} {}",
                    man.id, is_def, man.tier, man.vocab_size, man.bpc, man.created, short_fp(man.fingerprint));
            }
            Ok(())
        }
        "add" => {
            let id = cli.positionals.get(1).ok_or("usage: nsc models add <id> --model-nsc <p> --o2 <p> --tokenizer <p> [--tier t --bpc x --notes s --created YYYY-MM-DD --default]")?;
            let src_model = cli.get("model-nsc").ok_or("models add: --model-nsc <path> required")?;
            let src_o2 = cli.get("o2").ok_or("models add: --o2 <path> required")?;
            let src_tok = cli.get("tokenizer").ok_or("models add: --tokenizer <path> required")?;
            let tier = cli.get("tier").unwrap_or("");
            let bpc = cli.get("bpc").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let created = cli.get("created").unwrap_or("");
            let notes = cli.get("notes").unwrap_or("");
            let man = reg.add(id, Path::new(src_model), Path::new(src_o2), Path::new(src_tok),
                              tier, bpc, created, notes, cli.has("default"))?;
            let is_def = reg.default_id().as_deref() == Some(id);
            println!("added model '{}' (vocab {}, seg {}, fp {}){}",
                     man.id, man.vocab_size, man.seg_len, short_fp(man.fingerprint),
                     if is_def { "  [now default]" } else { "" });
            Ok(())
        }
        "default" => {
            let id = cli.positionals.get(1).ok_or("usage: nsc models default <id>")?;
            reg.set_default(id)?;
            println!("default model set to '{id}'");
            Ok(())
        }
        "remove" | "rm" => {
            let id = cli.positionals.get(1).ok_or("usage: nsc models remove <id>")?;
            reg.remove(id)?;
            println!("removed model '{id}'");
            Ok(())
        }
        other => Err(format!("unknown 'models' sub-command '{other}' (use list|add|default|remove)")),
    }
}

// ------------------------------------------------------------------ command: doctor
fn cmd_doctor(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let mut problems = 0usize;
    println!("registry home: {}", reg.home().display());
    // python + tok.py
    let py = std::env::var("NSC_PYTHON").unwrap_or_else(|_| "python3".into());
    let tok = reg.tok_py();
    match std::process::Command::new(&py).arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status() {
        Ok(s) if s.success() => println!("python:        {py}  ✓"),
        _ => { println!("python:        {py}  ✗ (not runnable — end-to-end compress/decompress needs it)"); problems += 1; }
    }
    println!("tok.py:        {}  {}", tok.display(), if tok.exists() { "✓" } else { problems += 1; "✗ (missing)" });
    // models
    let models = reg.list()?;
    let def = reg.default_id();
    println!("default model: {}", def.as_deref().unwrap_or("(none set)"));
    if models.is_empty() { println!("models:        none installed"); problems += 1; }
    let verify = cli.has("verify");
    for m in &models {
        let man = &m.manifest;
        let mut notes = vec![];
        for (label, p) in [("model.nsc", m.model_nsc()), ("o2.nsc", m.o2_nsc()), ("tokenizer.json", m.tokenizer())] {
            if !p.exists() { notes.push(format!("missing {label}")); problems += 1; }
        }
        if verify && notes.is_empty() {
            match registry::fingerprint(&m.model_nsc(), &m.o2_nsc()) {
                Ok(fp) if fp == man.fingerprint => notes.push("fingerprint ✓".into()),
                Ok(_) => { notes.push("FINGERPRINT MISMATCH".into()); problems += 1; }
                Err(e) => { notes.push(format!("hash error: {e}")); problems += 1; }
            }
        }
        println!("  model {:<16} {}", man.id, if notes.is_empty() { "ok".into() } else { notes.join(", ") });
    }
    if !verify { println!("(run `nsc doctor --verify` to re-hash and check model fingerprints)"); }
    if problems == 0 { println!("\nall checks passed ✓"); Ok(()) }
    else { Err(format!("{problems} problem(s) found")) }
}

// ------------------------------------------------------------------ helpers / help
fn two_positionals(cli: &Cli, cmd: &str, usage: &str) -> Result<(String, String), String> {
    if cli.positionals.len() < 2 { return Err(format!("usage: nsc {cmd} {usage}")); }
    Ok((cli.positionals[0].clone(), cli.positionals[1].clone()))
}

fn print_help() {
    println!("\
nsc {ver} — neuro-symbolic lossless codec

USAGE
  nsc <command> [args] [flags]

COMMANDS
  compress    <input> <output.nsc>     Compress a file end-to-end (auto tokenize + code).
                                       Binary/already-compressed input (mp4, images, archives) is
                                       stored verbatim instead — always lossless, never fails.
  decompress  <input.nsc> <output>     Decompress; auto-selects the model recorded in the file.
  compress-ids   <in.ids> <out.nsc>    Compress a pre-tokenized .ids file (no tokenizer).
  decompress-ids <in.nsc> <out.ids>    Decompress to a .ids file (no tokenizer).
  batch <compress|decompress> <files...>   Process many files; model loaded once (see --jobs).
  info   <file.nsc>                    Show a compressed file's header (model, tokens, ratio).
  models [list]                        List installed models (* = default).
  models add <id> --model-nsc <p> --o2 <p> --tokenizer <p> [--tier t --bpc x --notes s --default]
  models default <id>                  Set the default model for new compressions.
  models remove <id>                   Remove a model version.
  doctor [--verify]                    Check registry, python/tok.py, and (‑‑verify) fingerprints.
  studio [--port N] [--root DIR]       Launch the local web dashboard (compress, metrics, upgrades).
                                       --root sets where the file browser opens (default: current dir).
  selftest                             Developer parity harness (Stages A–E) vs artifacts/.
  version | help

FLAGS
  --model <id>     Model to compress with (default: registry default).
  --home <path>    Registry home (default: $NSC_HOME or ~/.nsc).
  --jobs <n>       Parallel workers for `batch` (default: CPU count).
  --out-dir <dir>  Output directory for `batch`.
  --keep-ids       Keep the intermediate .ids file.
  --store          Force the raw-stored fallback (skip the neural model) for this compress.
  --neural         Force the neural model even for binary-looking input (kept only if it wins).
  -q, --quiet      Reduce output.

Model upgrading: every .nsc file records the id + fingerprint of the model that made it, so old
files keep decompressing after you `nsc models add` a newer default. See README.md.", ver = env!("CARGO_PKG_VERSION"));
}

fn dispatch(cmd: &str, cli: &Cli) -> Result<(), String> {
    match cmd {
        "compress" => cmd_compress(&open_registry(cli)?, cli),
        "decompress" => cmd_decompress(&open_registry(cli)?, cli),
        "compress-ids" => cmd_compress_ids(&open_registry(cli)?, cli),
        "decompress-ids" => cmd_decompress_ids(&open_registry(cli)?, cli),
        "batch" => cmd_batch(&open_registry(cli)?, cli),
        "info" => cmd_info(cli),
        "models" => cmd_models(&open_registry(cli)?, cli),
        "doctor" => cmd_doctor(&open_registry(cli)?, cli),
        "studio" => studio::run(open_registry(cli)?, cli),
        "selftest" => { selftest::run(); Ok(()) }
        "version" | "--version" => { println!("nsc {}", env!("CARGO_PKG_VERSION")); Ok(()) }
        "help" | "--help" | "-h" => { print_help(); Ok(()) }
        other => Err(format!("unknown command '{other}'. Run `nsc help`.")),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, cli) = match parse_args(&argv) {
        Ok(v) => v,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
    };
    if cli.has("help") && cmd != "help" { print_help(); return; }
    match dispatch(&cmd, &cli) {
        Ok(()) => {}
        Err(e) => { eprintln!("error: {e}"); std::process::exit(1); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Live-feature encode -> decode must recover the token stream exactly. Uses the dev fixtures in
    /// artifacts/ (skips on a fresh checkout where the 285 MB artifacts are absent).
    #[test]
    fn roundtrip_bit_identical() {
        for f in ["artifacts/model.nsc", "artifacts/o2.nsc", "artifacts/ref.nsc"] {
            if std::fs::metadata(f).is_err() { eprintln!("roundtrip: {f} missing — skipping"); return; }
        }
        let model = Model::load("artifacts/model.nsc");
        let o2 = O2Table::load("artifacts/o2.nsc");
        let b = std::fs::read("artifacts/ref.nsc").unwrap();
        let mut o = 0usize;
        let t = codec::ru32(&b, &mut o) as usize;
        let _v = codec::ru32(&b, &mut o) as usize;
        let tokens: Vec<usize> = (0..t).map(|_| codec::ri32(&b, &mut o) as usize).collect();
        let (payload, nbits) = codec::encode_tokens(&model, &o2, &tokens, 4);
        let rec = codec::decode_tokens(&model, &o2, tokens[0], tokens.len(), payload, nbits);
        assert_eq!(rec, tokens, "round-trip token mismatch");
    }
}
//! Witten–Neal–Cleary range coder — a bit-exact port of the Python reference
//! (test_compression_integrity.RangeEncoder / RangeDecoder). Encoder and decoder must
//! produce/consume identical bits given identical cumulative-frequency tables, so this
//! reproduces the Python integer arithmetic exactly. Values fit in u64: rng ≤ 2^32,
//! total ≤ 2^30 ⇒ rng*ch ≤ 2^62.

const MASK: u64 = (1 << 32) - 1;
const HALF: u64 = 1 << 31;
const QUARTER: u64 = 1 << 30;
const THREEQ: u64 = 3 << 30;

pub struct RangeEncoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: Vec<u8>, // one entry per bit (0/1); packed MSB-first at finish
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self { low: 0, high: MASK, pending: 0, bits: Vec::new() }
    }

    #[inline]
    fn emit(&mut self, bit: u8) {
        self.bits.push(bit);
        for _ in 0..self.pending {
            self.bits.push(1 - bit);
        }
        self.pending = 0;
    }

    pub fn encode(&mut self, cl: u64, ch: u64, total: u64) {
        let rng = self.high - self.low + 1;
        self.high = self.low + (rng * ch) / total - 1;
        self.low = self.low + (rng * cl) / total;
        loop {
            if self.high < HALF {
                self.emit(0);
            } else if self.low >= HALF {
                self.emit(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREEQ {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) | 1) & MASK;
        }
    }

    /// Returns (packed payload bytes, number of bits). Matches numpy.packbits (MSB-first).
    pub fn finish(mut self) -> (Vec<u8>, usize) {
        self.pending += 1;
        let b = if self.low < QUARTER { 0 } else { 1 };
        self.emit(b);
        let n_bits = self.bits.len();
        let mut payload = vec![0u8; (n_bits + 7) / 8];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        (payload, n_bits)
    }
}

pub struct RangeDecoder {
    payload: Vec<u8>,
    n_bits: usize,
    i: usize,
    low: u64,
    high: u64,
    value: u64,
}

impl RangeDecoder {
    pub fn new(payload: Vec<u8>, n_bits: usize) -> Self {
        let mut d = Self { payload, n_bits, i: 0, low: 0, high: MASK, value: 0 };
        for _ in 0..32 {
            let b = d.next_bit();
            d.value = ((d.value << 1) | b) & MASK;
        }
        d
    }

    #[inline]
    fn next_bit(&mut self) -> u64 {
        let i = self.i;
        self.i += 1;
        if i < self.n_bits {
            ((self.payload[i / 8] >> (7 - (i % 8))) & 1) as u64
        } else {
            0 // zero tail past the end, matching the reference
        }
    }

    /// `cum` is the cumulative table (len V+1, cum[V] == total). Returns the symbol.
    pub fn decode(&mut self, cum: &[u64]) -> usize {
        let total = cum[cum.len() - 1];
        let rng = self.high - self.low + 1;
        let scaled = ((self.value - self.low + 1) * total - 1) / rng;
        // np.searchsorted(cum, scaled, side="right") - 1, then clamp to [0, V-1]
        let sr = cum.partition_point(|&x| x <= scaled) as i64 - 1;
        let s = sr.clamp(0, cum.len() as i64 - 2) as usize;
        let cl = cum[s];
        let ch = cum[s + 1];
        self.high = self.low + (rng * ch) / total - 1;
        self.low = self.low + (rng * cl) / total;
        loop {
            if self.high < HALF {
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.value -= HALF;
            } else if self.low >= QUARTER && self.high < THREEQ {
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.value -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) | 1) & MASK;
            let b = self.next_bit();
            self.value = ((self.value << 1) | b) & MASK;
        }
        s
    }
}
//! Versioned model registry — the mechanism that makes model upgrading safe.
//!
//! Layout under the registry home (`$NSC_HOME`, else `$HOME/.nsc`):
//!
//! ```text
//!   ~/.nsc/
//!     default                 # one line: the id of the default model
//!     tok.py                  # tokenizer shim (installed once)
//!     models/
//!       v004-base/
//!         model.nsc  o2.nsc  tokenizer.json  manifest.json
//!       v005-base/ ...
//! ```
//!
//! Every model version is self-contained (its own weights AND its own tokenizer), and each
//! compressed file records the id + fingerprint of the model that produced it (see `container.rs`).
//! So you can `nsc models add` a newer, better model and make it the default for new compressions
//! while every previously produced `.nsc` file still decompresses through its original model.
//!
//! Dependency-free: manifests are a tiny flat JSON parsed by a hand-rolled scanner; the model
//! fingerprint is FNV-1a-64 over the weight + order-2 blobs.
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------- fingerprint
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
    h
}

/// FNV-1a-64 over model.nsc then o2.nsc — a fast content fingerprint that binds a compressed file
/// to the exact registry artifacts, catching a swapped/tampered model at decompress time.
pub fn fingerprint(model_nsc: &Path, o2_nsc: &Path) -> Result<u64, String> {
    let mut h = FNV_OFFSET;
    for p in [model_nsc, o2_nsc] {
        let b = fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
        h = fnv1a_update(h, &b);
    }
    Ok(h)
}

// ---------------------------------------------------------------- minimal JSON
/// Scan a flat JSON object into key -> raw-value strings (strings unquoted; numbers/bools raw).
/// Only supports one un-nested object, which is all a manifest needs — no external crate.
fn parse_flat_json(s: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let read_string = |c: &[char], i: &mut usize| -> String {
        let mut out = String::new(); // assumes opening quote already consumed
        while *i < c.len() && c[*i] != '"' {
            if c[*i] == '\\' && *i + 1 < c.len() { *i += 1; out.push(c[*i]); } else { out.push(c[*i]); }
            *i += 1;
        }
        *i += 1; // closing quote
        out
    };
    while i < c.len() {
        if c[i] == '"' {
            i += 1;
            let key = read_string(&c, &mut i);
            while i < c.len() && c[i] != ':' { i += 1; }
            i += 1; // ':'
            while i < c.len() && c[i].is_whitespace() { i += 1; }
            if i >= c.len() { break; }
            let val = if c[i] == '"' {
                i += 1; read_string(&c, &mut i)
            } else {
                let mut out = String::new();
                while i < c.len() && c[i] != ',' && c[i] != '}' { out.push(c[i]); i += 1; }
                out.trim().to_string()
            };
            m.insert(key, val);
        } else { i += 1; }
    }
    m
}

fn json_escape(s: &str) -> String { s.replace('\\', "\\\\").replace('"', "\\\"") }

// ---------------------------------------------------------------- manifest
#[derive(Clone)]
pub struct Manifest {
    pub id: String,
    pub tier: String,
    pub vocab_size: usize,
    pub seg_len: usize,
    pub d_model: usize,
    pub bpc: f64,        // reference bits/char (informational)
    pub created: String, // YYYY-MM-DD
    pub fingerprint: u64,
    pub format: u16,     // container format this model emits
    pub notes: String,
}

impl Manifest {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"id\": \"{}\",\n  \"tier\": \"{}\",\n  \"vocab_size\": {},\n  \"seg_len\": {},\n  \"d_model\": {},\n  \"bpc\": {},\n  \"created\": \"{}\",\n  \"fingerprint\": {},\n  \"format\": {},\n  \"notes\": \"{}\"\n}}\n",
            json_escape(&self.id), json_escape(&self.tier), self.vocab_size, self.seg_len,
            self.d_model, self.bpc, json_escape(&self.created), self.fingerprint, self.format,
            json_escape(&self.notes))
    }
    pub fn from_json(s: &str) -> Result<Manifest, String> {
        let m = parse_flat_json(s);
        let get = |k: &str| m.get(k).cloned().unwrap_or_default();
        let id = get("id");
        if id.is_empty() { return Err("manifest missing id".into()); }
        Ok(Manifest {
            id,
            tier: get("tier"),
            vocab_size: get("vocab_size").parse().unwrap_or(0),
            seg_len: get("seg_len").parse().unwrap_or(0),
            d_model: get("d_model").parse().unwrap_or(0),
            bpc: get("bpc").parse().unwrap_or(0.0),
            created: get("created"),
            fingerprint: get("fingerprint").parse().unwrap_or(0),
            format: get("format").parse().unwrap_or(crate::container::FORMAT_VERSION),
            notes: get("notes"),
        })
    }
}

/// A resolved model directory in the registry.
pub struct ModelDir {
    pub dir: PathBuf,
    pub manifest: Manifest,
}
impl ModelDir {
    pub fn model_nsc(&self) -> PathBuf { self.dir.join("model.nsc") }
    pub fn o2_nsc(&self) -> PathBuf { self.dir.join("o2.nsc") }
    pub fn tokenizer(&self) -> PathBuf { self.dir.join("tokenizer.json") }
}

// ---------------------------------------------------------------- registry
pub struct Registry {
    home: PathBuf,
}

impl Registry {
    /// Resolve the registry home: `$NSC_HOME`, else `$HOME/.nsc`.
    pub fn open() -> Result<Registry, String> {
        let home = if let Ok(h) = std::env::var("NSC_HOME") {
            PathBuf::from(h)
        } else if let Ok(h) = std::env::var("HOME") {
            PathBuf::from(h).join(".nsc")
        } else {
            return Err("cannot locate registry: set NSC_HOME or HOME".into());
        };
        Ok(Registry { home })
    }

    /// Open a registry rooted at an explicit home (the `--home` flag).
    pub fn with_home(home: PathBuf) -> Registry { Registry { home } }

    pub fn home(&self) -> &Path { &self.home }
    pub fn models_dir(&self) -> PathBuf { self.home.join("models") }
    fn default_file(&self) -> PathBuf { self.home.join("default") }

    /// Path to the tokenizer shim: `$NSC_TOK`, else `<home>/tok.py`.
    pub fn tok_py(&self) -> PathBuf {
        std::env::var("NSC_TOK").map(PathBuf::from).unwrap_or_else(|_| self.home.join("tok.py"))
    }

    pub fn ensure_dirs(&self) -> Result<(), String> {
        fs::create_dir_all(self.models_dir()).map_err(|e| format!("mkdir {}: {e}", self.models_dir().display()))
    }

    /// List installed models, sorted by id.
    pub fn list(&self) -> Result<Vec<ModelDir>, String> {
        let md = self.models_dir();
        if !md.exists() { return Ok(vec![]); }
        let mut out = vec![];
        for e in fs::read_dir(&md).map_err(|e| format!("read_dir {}: {e}", md.display()))? {
            let e = e.map_err(|e| e.to_string())?;
            if !e.path().is_dir() { continue; }
            let man = e.path().join("manifest.json");
            if let Ok(s) = fs::read_to_string(&man) {
                if let Ok(m) = Manifest::from_json(&s) {
                    out.push(ModelDir { dir: e.path(), manifest: m });
                }
            }
        }
        out.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
        Ok(out)
    }

    pub fn default_id(&self) -> Option<String> {
        fs::read_to_string(self.default_file()).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
    }

    pub fn set_default(&self, id: &str) -> Result<(), String> {
        if self.get(id).is_err() { return Err(format!("no such model '{id}'")); }
        fs::write(self.default_file(), format!("{id}\n")).map_err(|e| format!("write default: {e}"))
    }

    /// Load a specific model directory by id.
    pub fn get(&self, id: &str) -> Result<ModelDir, String> {
        let dir = self.models_dir().join(id);
        let man = dir.join("manifest.json");
        let s = fs::read_to_string(&man).map_err(|_| format!("no such model '{id}' (looked in {})", dir.display()))?;
        Ok(ModelDir { dir, manifest: Manifest::from_json(&s)? })
    }

    /// Resolve a model: the given id, else the default, else (if exactly one is installed) that one.
    pub fn resolve(&self, id: Option<&str>) -> Result<ModelDir, String> {
        if let Some(id) = id { return self.get(id); }
        if let Some(def) = self.default_id() { return self.get(&def); }
        let mut all = self.list()?;
        match all.len() {
            0 => Err("no models installed — run install.sh or `nsc models add ...`".into()),
            1 => Ok(all.pop().unwrap()),
            _ => Err("no default model set — pass --model <id> or `nsc models default <id>`".into()),
        }
    }

    /// Install a model version into the registry from source artifact paths. Copies model.nsc,
    /// o2.nsc and tokenizer.json into `models/<id>/`, computes the fingerprint, reads the config
    /// from model.nsc's header, and writes manifest.json.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&self, id: &str, src_model: &Path, src_o2: &Path, src_tok: &Path,
               tier: &str, bpc: f64, created: &str, notes: &str, make_default: bool) -> Result<Manifest, String> {
        if id.is_empty() || id.contains('/') || id.contains(std::path::MAIN_SEPARATOR) {
            return Err("invalid model id (no path separators)".into());
        }
        self.ensure_dirs()?;
        let dir = self.models_dir().join(id);
        if dir.exists() { return Err(format!("model '{id}' already exists at {}", dir.display())); }
        for (label, p) in [("model.nsc", src_model), ("o2.nsc", src_o2), ("tokenizer.json", src_tok)] {
            if !p.exists() { return Err(format!("missing source {label}: {}", p.display())); }
        }
        let (vocab_size, d_model, seg_len) = peek_model_cfg(src_model)?;

        fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        // copy into a fresh dir; clean up on any failure so a partial add doesn't linger
        let copy = |from: &Path, to: PathBuf| -> Result<(), String> {
            fs::copy(from, &to).map(|_| ()).map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))
        };
        let do_add = || -> Result<Manifest, String> {
            copy(src_model, dir.join("model.nsc"))?;
            copy(src_o2, dir.join("o2.nsc"))?;
            copy(src_tok, dir.join("tokenizer.json"))?;
            let fp = fingerprint(&dir.join("model.nsc"), &dir.join("o2.nsc"))?;
            let man = Manifest {
                id: id.to_string(), tier: tier.to_string(), vocab_size, seg_len, d_model, bpc,
                created: created.to_string(), fingerprint: fp,
                format: crate::container::FORMAT_VERSION, notes: notes.to_string(),
            };
            fs::write(dir.join("manifest.json"), man.to_json()).map_err(|e| format!("write manifest: {e}"))?;
            Ok(man)
        };
        match do_add() {
            Ok(man) => {
                if make_default || self.default_id().is_none() { self.set_default(id)?; }
                Ok(man)
            }
            Err(e) => { let _ = fs::remove_dir_all(&dir); Err(e) }
        }
    }

    pub fn remove(&self, id: &str) -> Result<(), String> {
        let dir = self.models_dir().join(id);
        if !dir.exists() { return Err(format!("no such model '{id}'")); }
        fs::remove_dir_all(&dir).map_err(|e| format!("remove {}: {e}", dir.display()))?;
        if self.default_id().as_deref() == Some(id) { let _ = fs::remove_file(self.default_file()); }
        Ok(())
    }
}

/// Read (vocab_size, d_model, seg_len) from a model.nsc header without loading the 273 MB body.
fn peek_model_cfg(path: &Path) -> Result<(usize, usize, usize), String> {
    use std::io::Read;
    let mut f = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hdr = [0u8; 4 + 9 * 4];
    f.read_exact(&mut hdr).map_err(|e| format!("read header {}: {e}", path.display()))?;
    if &hdr[0..4] != b"NSC1" { return Err(format!("{}: bad model magic", path.display())); }
    let g = |i: usize| u32::from_le_bytes(hdr[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize;
    // config order: vocab, d_model, n_layers, n_heads, gru_hidden, gru_layers, seg_len, ...
    Ok((g(0), g(1), g(6)))
}
//! Developer parity harness (Stages A–E) — verifies the Rust codec against the Python reference
//! fixtures in `artifacts/` (or `$NSC_ARTIFACTS`). Run from the crate dir after exporting the
//! fixtures. Not part of the product path; the shipped codec uses the model registry instead.
use crate::artifacts::Model;
use crate::codec::{quantize, rf32, rf64, ri32, ru16, ru32, ru64, QUANT_BITS};
use crate::features::{FeatureModel, O2Table};
use crate::forward;
use crate::range_coder::{RangeDecoder, RangeEncoder};
use std::fs;

fn art(name: &str) -> String {
    let base = std::env::var("NSC_ARTIFACTS").unwrap_or_else(|_| "artifacts".into());
    format!("{base}/{name}")
}

fn stage_c(m: &Model) -> bool {
    let b = match fs::read(art("inter.nsc")) { Ok(d) => d, Err(_) => { println!("Stage C: inter.nsc missing (skip)"); return true; } };
    let mut o = 0usize;
    let tok0 = ri32(&b, &mut o) as usize;
    let o2t = ri32(&b, &mut o) as usize;
    let o2c = rf64(&b, &mut o) as f32;
    let mt = ri32(&b, &mut o) as usize;
    let mc = rf64(&b, &mut o) as f32;
    let _h = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let conf_ref: Vec<f32> = (0..4).map(|_| rf32(&b, &mut o)).collect();
    let gh_ref: Vec<f32> = (0.._h).map(|_| rf32(&b, &mut o)).collect();
    let rg_ref: Vec<f32> = (0..v).map(|_| rf32(&b, &mut o)).collect();
    let probs_ref: Vec<f32> = (0..v).map(|_| rf32(&b, &mut o)).collect();

    let (rg, gh, conf) = forward::resid_gru_pos0(m, tok0, o2c, mc);
    let dc = (0..4).map(|i| (conf[i] - conf_ref[i]).abs()).fold(0.0f32, f32::max);
    let dg = (0.._h).map(|i| (gh[i] - gh_ref[i]).abs()).fold(0.0f32, f32::max);
    let dr = (0..v).map(|i| (rg[i] - rg_ref[i]).abs()).fold(0.0f32, f32::max);
    println!("Stage C GRU path (pos0): conf max|d|={:.2e}  gru_h max|d|={:.2e}  resid_gru max|d|={:.2e}", dc, dg, dr);
    let _ = (o2t, mt);

    let mut dec = forward::Decoder::new(m);
    let probs = dec.step(tok0, o2t, o2c, mt, mc);
    let dp = (0..v).map(|i| (probs[i] - probs_ref[i]).abs()).fold(0.0f32, f32::max);
    let mut fi = None;
    for i in 0..v { if (probs[i] - probs_ref[i]).abs() > 1e-7 { fi = Some((i, probs[i], probs_ref[i])); break; } }
    println!("Stage C pos0 probs: max|d|={:.2e}  first mismatch: {:?}", dp, fi);
    dr == 0.0 && dp == 0.0
}

fn stage_a() -> bool {
    let data = match fs::read("test_vectors/vec.bin") { Ok(d) => d, Err(_) => { println!("stage A: vec.bin missing (skip)"); return true; } };
    let mut o = 0usize;
    let n = ru32(&data, &mut o) as usize;
    let v = ru32(&data, &mut o) as usize;
    let ref_nbits = ru32(&data, &mut o) as usize;
    let mut cums: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut syms: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n {
        let s = ru16(&data, &mut o) as usize; // symbols are u16 in vec.bin
        let cum: Vec<u64> = (0..v + 1).map(|_| ru64(&data, &mut o)).collect();
        cums.push(cum); syms.push(s);
    }
    let plen = ru32(&data, &mut o) as usize;
    let ref_payload = data[o..o + plen].to_vec();
    let mut enc = RangeEncoder::new();
    for i in 0..n { enc.encode(cums[i][syms[i]], cums[i][syms[i] + 1], cums[i][v]); }
    let (payload, nbits) = enc.finish();
    let enc_ok = payload == ref_payload && nbits == ref_nbits;
    let mut dec = RangeDecoder::new(ref_payload.clone(), ref_nbits);
    let dec_ok = (0..n).all(|i| dec.decode(&cums[i]) == syms[i]);
    println!("Stage A range coder: encode {}  decode {}",
        if enc_ok { "bit-identical ✓" } else { "MISMATCH ✗" },
        if dec_ok { "symbols ok ✓" } else { "MISMATCH ✗" });
    enc_ok && dec_ok
}

fn stage_b(m: &Model) -> bool {
    let c = m.cfg;
    println!("Stage B config: V={} d={} L={} H={} gru_h={} gru_L={} seg={} gate={} conf={} eps={:.1e}",
        c.vocab_size, c.d_model, c.n_layers, c.n_heads, c.gru_hidden, c.gru_layers,
        c.seg_len, c.gate_dim, c.conf_dim, c.eps);
    println!("           tensors loaded: {}", m.t.len());
    let qkv = m.get("blk0.qkv.code");
    println!("           blk0.qkv.code shape {:?}, scale[0]={:.6e}", qkv.shape, m.get("blk0.qkv.scale").f64[0]);
    let rb = fs::read(art("ref.nsc")).expect("ref.nsc");
    let mut o = 0usize; let _t = ru32(&rb, &mut o); let _v = ru32(&rb, &mut o);
    let tok0 = i32::from_le_bytes(rb[o..o + 4].try_into().unwrap()) as usize;
    let row = m.order1_row(tok0);
    let o1max = row.iter().cloned().fold(f32::MIN, f32::max);
    let embed = m.get("embed"); let order0 = m.get("order0");
    println!("           tok0={} | order0[0]={:.8} order1[tok0][tok0+1]={:.8} o1_max={:.8} embed[tok0][0]={:.8}",
        tok0, order0.f32[0], row[tok0 + 1], o1max, embed.f32[tok0 * c.d_model]);
    m.t.contains_key("order1") && m.t.contains_key("blk3.qkv.code")
}

fn stage_c_full(m: &Model) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage C full: ref.nsc missing"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();
    let exp: Vec<Vec<i64>> = (0..t).map(|_| (0..v + 1).map(|_| ru64(&b, &mut o) as i64).collect()).collect();

    let mut dec = forward::Decoder::new(m);
    let mut diff = 0usize;
    let mut first = None;
    for i in 0..t {
        let f = feats[i];
        let probs = dec.step(tokens[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum = quantize(&probs, QUANT_BITS);
        if cum != exp[i] { diff += 1; if first.is_none() { first = Some(i); } }
    }
    println!("Stage C full forward: {}/{} coder tables differ (first mismatch @ {:?})", diff, t, first);
    diff == 0
}

fn stage_d(m: &Model) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage D: ref.nsc missing"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();

    let mut ed = forward::Decoder::new(m);
    let mut enc = RangeEncoder::new();
    for i in 0..t {
        let f = feats[i];
        let probs = ed.step(tokens[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum = quantize(&probs, QUANT_BITS);
        if i + 1 < t {
            let s = tokens[i + 1];
            enc.encode(cum[s] as u64, cum[s + 1] as u64, cum[v] as u64);
        }
    }
    let (payload, nbits) = enc.finish();

    let mut dd = forward::Decoder::new(m);
    let mut dec = RangeDecoder::new(payload.clone(), nbits);
    let mut rec = vec![tokens[0]];
    for i in 0..t - 1 {
        let f = feats[i];
        let probs = dd.step(rec[i], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        let cum: Vec<u64> = quantize(&probs, QUANT_BITS).iter().map(|&x| x as u64).collect();
        rec.push(dec.decode(&cum));
    }
    let ok = rec == tokens;
    println!("Stage D Rust roundtrip: {} tokens, payload {}B, tokens_match={} {}",
        t, payload.len(), ok, if ok { "✓ BYTE-EXACT" } else { "✗" });

    let iters = 2000usize;
    let mut sd = forward::Decoder::new(m);
    let t0 = std::time::Instant::now();
    let mut sink = 0f32;
    for i in 0..iters {
        let j = i % t;
        let f = feats[j];
        let p = sd.step(tokens[j], f[0] as usize, f[1] as f32, f[2] as usize, f[3] as f32);
        sink += p[0];
        if (i + 1) % t == 0 { sd = forward::Decoder::new(m); }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("Stage D forward speed: {:.0} tok/s  ({:.3} ms/token)  [sink {:.3}]",
        iters as f64 / secs, secs / iters as f64 * 1e3, sink);
    ok
}

fn stage_e(m: &Model, o2: &O2Table) -> bool {
    let b = match fs::read(art("ref.nsc")) { Ok(d) => d, Err(_) => { println!("Stage E: ref.nsc missing (skip)"); return true; } };
    let mut o = 0usize;
    let t = ru32(&b, &mut o) as usize;
    let _v = ru32(&b, &mut o) as usize;
    let tokens: Vec<usize> = (0..t).map(|_| ri32(&b, &mut o) as usize).collect();
    let feats: Vec<[f64; 4]> = (0..t).map(|_| [rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o), rf64(&b, &mut o)]).collect();
    assert_eq!(o2.v, m.cfg.vocab_size, "o2.nsc vocab mismatch");
    let mut fm = FeatureModel::new(o2);
    let mut bad = 0usize;
    for i in 0..t {
        let (o2t, o2c, mt, mc) = fm.push(tokens[i]);
        let e = feats[i];
        if o2t != e[0] as usize || mt != e[2] as usize || o2c as f64 != e[1] || mc as f64 != e[3] { bad += 1; }
    }
    println!("Stage E feature model vs Python: {}/{} positions differ", bad, t);
    bad == 0
}

/// Run the full Stage A–E parity harness against the fixtures in `$NSC_ARTIFACTS` (default `artifacts/`).
pub fn run() {
    let a = stage_a();
    let m = Model::load(&art("model.nsc"));
    let b = stage_b(&m);
    let e = match fs::read(art("o2.nsc")) {
        Ok(_) => stage_e(&m, &O2Table::load(&art("o2.nsc"))),
        Err(_) => { println!("Stage E: o2.nsc missing (skip; run python export_o2.py)"); true }
    };
    let c = stage_c(&m);
    let cf = stage_c_full(&m);
    let dd = stage_d(&m);
    println!("\n=> A coder:{} | B load:{} | E feats:{} | C GRU:{} | C fidelity-vs-python:{} | D Rust roundtrip:{}",
        if a { "PASS" } else { "FAIL" },
        if b { "OK" } else { "FAIL" },
        if e { "exact" } else { "DIFF" },
        if c { "exact" } else { "diff" },
        if cf { "exact" } else { "~5e-7 (fine)" },
        if dd { "BYTE-EXACT ✓" } else { "FAIL" });
}
//! `nsc studio` — a local control panel for the codec, served as a tiny web app.
//!
//! Dependency-free, std-only: a `TcpListener` HTTP/1.1 server (thread per connection) exposes a
//! small JSON API and serves an embedded single-page dashboard. It lets you compress/decompress
//! real local files, watch live metrics (ratio, bits-per-byte, speed), verify losslessness, race
//! the codec against classical compressors (gzip/xz/zstd/7z-PPMd), deploy model upgrades through
//! the versioned registry, and track behaviour over time via a persisted run history.
//!
//! The heavy lifting reuses the same binary: model management reads the [`Registry`] directly, and
//! compress/decompress/info shell out to this very executable's CLI subcommands (so tokenization,
//! the fingerprint container, and every codec path are byte-for-byte the ones the CLI uses).
use crate::bench::{bench_7z, bench_pipe, detect_classical, files_equal};
use crate::registry::Registry;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const INDEX_HTML: &str = include_str!("studio.html");
static SEQ: AtomicU64 = AtomicU64::new(0);

struct Ctx {
    self_exe: PathBuf,
    home: PathBuf,
    browse_root: PathBuf,
    classical: Vec<String>,
    hist_path: PathBuf,
    hist_lock: Mutex<()>,
    quiet: bool,
}

pub fn run(reg: Registry, cli: &crate::Cli) -> Result<(), String> {
    let port: u16 = cli.get("port").and_then(|s| s.parse().ok()).unwrap_or(8765);
    let self_exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let home = reg.home().to_path_buf();
    let classical = detect_classical();
    // Where the file browser opens by default: --root if given, else the launch dir. This makes the
    // Browse buttons surface the files you're actually working with (e.g. results/) instead of $HOME.
    let browse_root = match cli.get("root") {
        Some(r) => expand_home(r),
        None => std::env::current_dir().unwrap_or_else(|_| home.clone()),
    };
    let browse_root = fs::canonicalize(&browse_root).unwrap_or(browse_root);
    let ctx = Arc::new(Ctx {
        self_exe,
        home: home.clone(),
        browse_root,
        classical: classical.clone(),
        hist_path: home.join("studio_history.json"),
        hist_lock: Mutex::new(()),
        quiet: cli.quiet(),
    });

    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    let url = format!("http://{addr}");
    println!("Syntropy2 Studio  ->  {url}");
    println!("  nsc binary : {}", ctx.self_exe.display());
    println!("  registry   : {}", ctx.home.display());
    println!("  browse root: {}", ctx.browse_root.display());
    println!("  classical  : {}", if classical.is_empty() { "none detected".into() } else { classical.join(", ") });
    println!("  Ctrl-C to stop.");
    // best-effort: open the browser
    let _ = Command::new("xdg-open").arg(&url).stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null()).spawn();

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let ctx = Arc::clone(&ctx);
                std::thread::spawn(move || { let _ = serve(&ctx, s); });
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- HTTP plumbing
struct Req {
    method: String,
    path: String,
    query: String,
    body: String,
}

fn read_request(stream: &TcpStream) -> Option<Req> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 { return None; }
    let mut it = line.split_whitespace();
    let method = it.next()?.to_string();
    let target = it.next()?.to_string();
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 { break; }
        let t = h.trim_end();
        if t.is_empty() { break; }
        if let Some((k, v)) = t.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 { reader.read_exact(&mut body).ok()?; }
    Some(Req { method, path, query, body: String::from_utf8_lossy(&body).into_owned() })
}

fn respond(stream: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) {
    let status = match code { 200 => "OK", 404 => "Not Found", 500 => "Internal Server Error", _ => "OK" };
    let head = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len());
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn json(stream: &mut TcpStream, code: u16, body: &str) {
    respond(stream, code, "application/json", body.as_bytes());
}

fn serve(ctx: &Ctx, mut stream: TcpStream) -> std::io::Result<()> {
    let req = match read_request(&stream) { Some(r) => r, None => return Ok(()) };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => respond(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/api/status") => json(&mut stream, 200, &api_status(ctx)),
        ("GET", "/api/models") => json(&mut stream, 200, &api_models(ctx)),
        ("GET", "/api/history") => json(&mut stream, 200, &read_history(ctx)),
        ("GET", "/api/browse") => json(&mut stream, 200, &api_browse(&query_get(&req.query, "path"), &ctx.browse_root)),
        ("POST", "/api/compress") => stream_compress(ctx, &req.body, &mut stream),
        ("POST", "/api/decompress") => json(&mut stream, 200, &api_decompress(ctx, &req.body)),
        ("POST", "/api/models/add") => json(&mut stream, 200, &api_model_add(ctx, &req.body)),
        ("POST", "/api/models/default") => json(&mut stream, 200, &api_model_default(ctx, &req.body)),
        ("POST", "/api/models/remove") => json(&mut stream, 200, &api_model_remove(ctx, &req.body)),
        ("POST", "/api/history/clear") => { let _ = fs::remove_file(&ctx.hist_path); json(&mut stream, 200, "{\"ok\":true}"); }
        _ => json(&mut stream, 404, "{\"error\":\"not found\"}"),
    }
    Ok(())
}

// ---------------------------------------------------------------- JSON helpers
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}
fn qs(s: &str) -> String { format!("\"{}\"", esc(s)) }

/// Parse a flat JSON object body into key -> string value (bools/numbers kept as their literal).
fn parse_body(s: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let read_str = |c: &[char], i: &mut usize| -> String {
        let mut out = String::new();
        while *i < c.len() && c[*i] != '"' {
            if c[*i] == '\\' && *i + 1 < c.len() {
                *i += 1;
                out.push(match c[*i] { 'n' => '\n', 't' => '\t', 'r' => '\r', other => other });
            } else { out.push(c[*i]); }
            *i += 1;
        }
        *i += 1;
        out
    };
    while i < c.len() {
        if c[i] == '"' {
            i += 1;
            let key = read_str(&c, &mut i);
            while i < c.len() && c[i] != ':' { i += 1; }
            i += 1;
            while i < c.len() && c[i].is_whitespace() { i += 1; }
            if i >= c.len() { break; }
            let val = if c[i] == '"' { i += 1; read_str(&c, &mut i) } else {
                let mut out = String::new();
                while i < c.len() && c[i] != ',' && c[i] != '}' { out.push(c[i]); i += 1; }
                out.trim().to_string()
            };
            m.insert(key, val);
        } else { i += 1; }
    }
    m
}
fn field<'a>(m: &'a HashMap<String, String>, k: &str) -> &'a str { m.get(k).map(|s| s.as_str()).unwrap_or("") }
fn flag(m: &HashMap<String, String>, k: &str) -> bool { m.get(k).map(|s| s == "true").unwrap_or(false) }

// ---------------------------------------------------------------- status / models
fn open_reg(ctx: &Ctx) -> Registry { Registry::with_home(ctx.home.clone()) }

fn models_json(ctx: &Ctx) -> String {
    let reg = open_reg(ctx);
    let def = reg.default_id();
    let mut out = String::from("[");
    if let Ok(models) = reg.list() {
        for (i, m) in models.iter().enumerate() {
            let man = &m.manifest;
            let disk: u64 = ["model.nsc", "o2.nsc", "tokenizer.json"].iter()
                .filter_map(|f| fs::metadata(m.dir.join(f)).ok().map(|md| md.len())).sum();
            let is_def = def.as_deref() == Some(&man.id);
            if i > 0 { out.push(','); }
            out.push_str(&format!(
                "{{\"id\":{},\"is_default\":{},\"tier\":{},\"vocab_size\":{},\"seg_len\":{},\"d_model\":{},\"bpc\":{},\"created\":{},\"fingerprint\":{},\"notes\":{},\"disk_bytes\":{}}}",
                qs(&man.id), is_def, qs(&man.tier), man.vocab_size, man.seg_len, man.d_model,
                man.bpc, qs(&man.created), man.fingerprint, qs(&man.notes), disk));
        }
    }
    out.push(']');
    out
}

fn api_models(ctx: &Ctx) -> String {
    let reg = open_reg(ctx);
    format!("{{\"registry\":{},\"default\":{},\"models\":{}}}",
        qs(&ctx.home.display().to_string()),
        reg.default_id().map(|d| qs(&d)).unwrap_or_else(|| "null".into()),
        models_json(ctx))
}

fn api_status(ctx: &Ctx) -> String {
    let ver = run_capture(&ctx.self_exe, &["version".into()]).map(|(o, _, _)| o.trim().to_string());
    let classical = ctx.classical.iter().map(|c| qs(c)).collect::<Vec<_>>().join(",");
    let reg = open_reg(ctx);
    format!("{{\"nsc\":{},\"nsc_version\":{},\"home\":{},\"classical\":[{}],\"registry\":{},\"default\":{},\"models\":{}}}",
        qs(&ctx.self_exe.display().to_string()),
        ver.map(|v| qs(&v)).unwrap_or_else(|| "null".into()),
        qs(&ctx.home.display().to_string()), classical,
        qs(&ctx.home.display().to_string()),
        reg.default_id().map(|d| qs(&d)).unwrap_or_else(|| "null".into()),
        models_json(ctx))
}

// ---------------------------------------------------------------- process helpers
fn run_capture(exe: &Path, args: &[String]) -> Option<(String, String, bool)> {
    let out = Command::new(exe).args(args).output().ok()?;
    Some((String::from_utf8_lossy(&out.stdout).into_owned(),
          String::from_utf8_lossy(&out.stderr).into_owned(),
          out.status.success()))
}

fn nsc(ctx: &Ctx, args: &[&str]) -> Option<(String, String, bool)> {
    let mut a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    a.push("--home".into());
    a.push(ctx.home.display().to_string());
    run_capture(&ctx.self_exe, &a)
}

fn parse_info(out: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for line in out.lines() {
        if let Some((k, v)) = line.split_once(':') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    m
}

// ---------------------------------------------------------------- compress (streamed progress)
/// Newline-delimited JSON progress stream. The connection stays open and each stage flushes an
/// event as it happens (encode -> verify -> each classical codec -> final record) so the browser
/// can render a live checklist instead of freezing on one long request.
fn stream_open(s: &mut TcpStream) {
    let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
}
fn stream_line(s: &mut TcpStream, line: &str) {
    let _ = s.write_all(line.as_bytes());
    let _ = s.write_all(b"\n");
    let _ = s.flush();
}

fn stream_compress(ctx: &Ctx, body: &str, s: &mut TcpStream) {
    stream_open(s);
    let b = parse_body(body);
    let path = field(&b, "path");
    if path.is_empty() { stream_line(s, &format!("{{\"stage\":\"error\",\"message\":{}}}", qs("no file path"))); return; }
    let orig = match fs::metadata(path) { Ok(m) if m.is_file() => m.len(),
        _ => { stream_line(s, &format!("{{\"stage\":\"error\",\"message\":{}}}", qs(&format!("not a file: {path}")))); return; } };
    let model = field(&b, "model");
    let want_classical = flag(&b, "classical");
    let verify = !b.contains_key("verify") || flag(&b, "verify");
    let save = !b.contains_key("save") || flag(&b, "save");

    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("syn_studio_{}_{}", std::process::id(), seq));
    if fs::create_dir_all(&tmp).is_err() { stream_line(s, &format!("{{\"stage\":\"error\",\"message\":{}}}", qs("cannot create temp dir"))); return; }
    let base = Path::new(path).file_name().and_then(|x| x.to_str()).unwrap_or("file");
    let nscfile = tmp.join(format!("{base}.nsc"));

    // ---- encode ----
    stream_line(s, &format!("{{\"stage\":\"encode\",\"status\":\"run\",\"file\":{},\"orig_bytes\":{orig}}}", qs(base)));
    let mut args: Vec<&str> = vec!["compress", path, nscfile.to_str().unwrap()];
    if !model.is_empty() { args.push("--model"); args.push(model); }
    if ctx.quiet { args.push("-q"); }
    let t0 = Instant::now();
    let res = nsc(ctx, &args);
    let enc_s = t0.elapsed().as_secs_f64();
    match res {
        Some((_, _, true)) if nscfile.exists() => {}
        Some((o, e, _)) => { let _ = fs::remove_dir_all(&tmp); stream_line(s, &format!("{{\"stage\":\"error\",\"message\":{}}}", qs(&pick_err(&o, &e)))); return; }
        None => { let _ = fs::remove_dir_all(&tmp); stream_line(s, &format!("{{\"stage\":\"error\",\"message\":{}}}", qs("compress failed to run"))); return; }
    }
    let file_bytes = fs::metadata(&nscfile).map(|m| m.len()).unwrap_or(0);
    let info = nsc(ctx, &["info", nscfile.to_str().unwrap()]).map(|(o, _, _)| parse_info(&o)).unwrap_or_default();
    let get = |k: &str| info.get(k).cloned().unwrap_or_default();
    let tokens: u64 = get("tokens").replace(',', "").parse().unwrap_or(0);
    let bpt: Option<f64> = get("bits/token").parse().ok();
    let model_id = if get("model id").is_empty() { model.to_string() } else { get("model id") };
    let bpc = if orig > 0 { file_bytes as f64 * 8.0 / orig as f64 } else { 0.0 };
    let ratio = if file_bytes > 0 { orig as f64 / file_bytes as f64 } else { 0.0 };
    stream_line(s, &format!(
        "{{\"stage\":\"encode\",\"status\":\"done\",\"file_bytes\":{file_bytes},\"tokens\":{tokens},\"bpc\":{:.4},\"ratio\":{:.3},\"encode_s\":{:.4}}}",
        bpc, ratio, enc_s));

    let mut fields: Vec<String> = vec![
        format!("\"ts\":{}", now_secs()),
        format!("\"file\":{}", qs(base)),
        format!("\"path\":{}", qs(path)),
        format!("\"orig_bytes\":{orig}"),
        format!("\"file_bytes\":{file_bytes}"),
        format!("\"tokens\":{tokens}"),
        format!("\"bpc\":{:.4}", bpc),
        format!("\"ratio\":{:.3}", ratio),
        format!("\"model\":{}", qs(&model_id)),
        format!("\"fingerprint\":{}", qs(&get("fingerprint"))),
        format!("\"vocab\":{}", qs(&get("vocab"))),
        format!("\"seg_len\":{}", qs(&get("seg_len"))),
        format!("\"encode_s\":{:.4}", enc_s),
        format!("\"encode_mb_s\":{:.5}", if enc_s > 0.0 { orig as f64 / 1e6 / enc_s } else { 0.0 }),
    ];
    if let Some(v) = bpt { fields.push(format!("\"bits_per_token\":{:.4}", v)); }

    // ---- verify roundtrip ----
    if verify {
        stream_line(s, "{\"stage\":\"verify\",\"status\":\"run\"}");
        let outp = tmp.join("roundtrip.out");
        let t1 = Instant::now();
        let dres = nsc(ctx, &["decompress", nscfile.to_str().unwrap(), outp.to_str().unwrap()]);
        let dec_s = t1.elapsed().as_secs_f64();
        let ok = matches!(dres, Some((_, _, true))) && files_equal(Path::new(path), &outp);
        let dtok = if tokens > 0 && dec_s > 0.0 { tokens as f64 / dec_s } else { 0.0 };
        stream_line(s, &format!("{{\"stage\":\"verify\",\"status\":\"done\",\"roundtrip_ok\":{ok},\"decode_s\":{:.4},\"decode_tok_s\":{:.1}}}", dec_s, dtok));
        fields.push(format!("\"roundtrip_ok\":{ok}"));
        fields.push(format!("\"decode_s\":{:.4}", dec_s));
        if tokens > 0 && dec_s > 0.0 { fields.push(format!("\"decode_tok_s\":{:.1}", dtok)); }
    }

    // ---- classical head-to-head (one event per codec) ----
    if want_classical {
        let mut items: Vec<(String, u64, f64)> = vec![];
        for name in &ctx.classical {
            stream_line(s, &format!("{{\"stage\":\"classical\",\"name\":{},\"status\":\"run\"}}", qs(name)));
            let r = match name.as_str() {
                "gzip -9" => bench_pipe("gzip", &["-9", "-c", path]),
                "xz -9" => bench_pipe("xz", &["-9", "-c", path]),
                "zstd -19" => bench_pipe("zstd", &["-19", "-q", "-c", path]),
                "7z-PPMd" => bench_7z(Path::new(path)),
                _ => None,
            };
            match r {
                Some((bytes, dt)) if bytes > 0 => {
                    let cbpc = if orig > 0 { bytes as f64 * 8.0 / orig as f64 } else { 0.0 };
                    let cratio = if bytes > 0 { orig as f64 / bytes as f64 } else { 0.0 };
                    stream_line(s, &format!("{{\"stage\":\"classical\",\"name\":{},\"status\":\"done\",\"bytes\":{bytes},\"bpc\":{:.4},\"ratio\":{:.3},\"encode_s\":{:.4}}}", qs(name), cbpc, cratio, dt));
                    items.push((name.clone(), bytes, dt));
                }
                _ => stream_line(s, &format!("{{\"stage\":\"classical\",\"name\":{},\"status\":\"skip\"}}", qs(name))),
            }
        }
        items.sort_by_key(|r| r.1);
        let arr: Vec<String> = items.iter().map(|(name, bytes, dt)| {
            let cbpc = if orig > 0 { *bytes as f64 * 8.0 / orig as f64 } else { 0.0 };
            let cratio = if *bytes > 0 { orig as f64 / *bytes as f64 } else { 0.0 };
            let mbs = if *dt > 0.0 { orig as f64 / 1e6 / dt } else { 0.0 };
            format!("{{\"name\":{},\"bytes\":{},\"bpc\":{:.4},\"ratio\":{:.3},\"encode_s\":{:.4},\"encode_mb_s\":{:.5}}}",
                qs(name), bytes, cbpc, cratio, dt, mbs)
        }).collect();
        fields.push(format!("\"classical\":[{}]", arr.join(",")));
    }

    let _ = fs::remove_dir_all(&tmp);
    let rec = format!("{{{}}}", fields.join(","));
    if save { append_history(ctx, &rec); }
    stream_line(s, &format!("{{\"stage\":\"final\",\"record\":{rec}}}"));
}

fn pick_err(o: &str, e: &str) -> String {
    let s = if !e.trim().is_empty() { e } else { o };
    s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("compress failed").trim().to_string()
}

fn api_decompress(ctx: &Ctx, body: &str) -> String {
    let b = parse_body(body);
    let path = field(&b, "path");
    let out = field(&b, "out");
    if path.is_empty() || out.is_empty() { return err("need input .nsc and output path"); }
    if !Path::new(path).is_file() { return err(&format!("not a file: {path}")); }
    let t0 = Instant::now();
    match nsc(ctx, &["decompress", path, out]) {
        Some((_, _, true)) => {
            let n = fs::metadata(out).map(|m| m.len()).unwrap_or(0);
            format!("{{\"ok\":true,\"out\":{},\"out_bytes\":{},\"decode_s\":{:.4}}}", qs(out), n, t0.elapsed().as_secs_f64())
        }
        Some((o, e, _)) => err(&pick_err(&o, &e)),
        None => err("decompress failed to run"),
    }
}

// ---------------------------------------------------------------- model management
fn api_model_add(ctx: &Ctx, body: &str) -> String {
    let b = parse_body(body);
    let (id, m, o2, tok) = (field(&b, "id"), field(&b, "model_nsc"), field(&b, "o2"), field(&b, "tokenizer"));
    if id.is_empty() || m.is_empty() || o2.is_empty() || tok.is_empty() { return err("id + 3 files are required"); }
    for (lbl, p) in [("model.nsc", m), ("o2.nsc", o2), ("tokenizer.json", tok)] {
        if !Path::new(p).is_file() { return err(&format!("not a file ({lbl}): {p}")); }
    }
    let today = run_capture(Path::new("date"), &["+%Y-%m-%d".into()]).map(|(o, _, _)| o.trim().to_string()).unwrap_or_default();
    let mut args: Vec<String> = vec!["models".into(), "add".into(), id.into(),
        "--model-nsc".into(), m.into(), "--o2".into(), o2.into(), "--tokenizer".into(), tok.into()];
    if !today.is_empty() { args.push("--created".into()); args.push(today); }
    for (k, f) in [("tier", "tier"), ("bpc", "bpc"), ("notes", "notes")] {
        let v = field(&b, f);
        if !v.is_empty() { args.push(format!("--{k}")); args.push(v.into()); }
    }
    if flag(&b, "make_default") { args.push("--default".into()); }
    args.push("--home".into()); args.push(ctx.home.display().to_string());
    match run_capture(&ctx.self_exe, &args) {
        Some((o, _, true)) => format!("{{\"ok\":true,\"stdout\":{},\"models\":{}}}", qs(o.trim()), models_json(ctx)),
        Some((o, e, _)) => err(&pick_err(&o, &e)),
        None => err("models add failed to run"),
    }
}

fn api_model_default(ctx: &Ctx, body: &str) -> String {
    let id = field(&parse_body(body), "id").to_string();
    if id.is_empty() { return err("no id"); }
    match nsc(ctx, &["models", "default", &id]) {
        Some((_, _, true)) => format!("{{\"ok\":true,\"models\":{}}}", models_json(ctx)),
        Some((o, e, _)) => err(&pick_err(&o, &e)),
        None => err("failed to run"),
    }
}

fn api_model_remove(ctx: &Ctx, body: &str) -> String {
    let id = field(&parse_body(body), "id").to_string();
    if id.is_empty() { return err("no id"); }
    match nsc(ctx, &["models", "remove", &id]) {
        Some((_, _, true)) => format!("{{\"ok\":true,\"models\":{}}}", models_json(ctx)),
        Some((o, e, _)) => err(&pick_err(&o, &e)),
        None => err("failed to run"),
    }
}

// ---------------------------------------------------------------- history
fn read_history(ctx: &Ctx) -> String {
    fs::read_to_string(&ctx.hist_path).ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "[]".into())
}

fn append_history(ctx: &Ctx, rec: &str) {
    let _guard = ctx.hist_lock.lock().unwrap();
    let cur = fs::read_to_string(&ctx.hist_path).unwrap_or_default();
    let trimmed = cur.trim();
    let new = if trimmed.is_empty() || trimmed == "[]" {
        format!("[{rec}]")
    } else if let Some(pos) = trimmed.rfind(']') {
        format!("{},{}]", &trimmed[..pos], rec)
    } else {
        format!("[{rec}]")
    };
    let tmp = ctx.hist_path.with_extension("json.tmp");
    if fs::write(&tmp, new).is_ok() { let _ = fs::rename(&tmp, &ctx.hist_path); }
}

// ---------------------------------------------------------------- file browse
fn api_browse(path: &str, root: &Path) -> String {
    let start = if path.is_empty() {
        root.to_path_buf()
    } else {
        expand_home(path)
    };
    let dir = if start.is_dir() { start.clone() } else {
        start.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("/"))
    };
    let dir = fs::canonicalize(&dir).unwrap_or(dir);
    let mut entries: Vec<(String, PathBuf, bool, u64)> = vec![];
    match fs::read_dir(&dir) {
        Ok(rd) => {
            for e in rd.flatten() {
                let p = e.path();
                let name = e.file_name().to_string_lossy().into_owned();
                let is_dir = p.is_dir();
                let sz = if is_dir { 0 } else { fs::metadata(&p).map(|m| m.len()).unwrap_or(0) };
                entries.push((name, p, is_dir, sz));
            }
        }
        Err(_) => return format!("{{\"error\":\"cannot read dir\",\"path\":{}}}", qs(&dir.display().to_string())),
    }
    entries.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.to_lowercase().cmp(&b.0.to_lowercase())));
    let items: Vec<String> = entries.iter().map(|(name, p, is_dir, sz)| {
        format!("{{\"name\":{},\"path\":{},\"dir\":{},\"size\":{}}}",
            qs(name), qs(&p.display().to_string()), is_dir, sz)
    }).collect();
    let parent = dir.parent().map(|p| p.display().to_string()).unwrap_or_else(|| dir.display().to_string());
    format!("{{\"path\":{},\"parent\":{},\"entries\":[{}]}}",
        qs(&dir.display().to_string()), qs(&parent), items.join(","))
}

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~") {
        if let Ok(h) = std::env::var("HOME") {
            return PathBuf::from(h).join(rest.trim_start_matches('/'));
        }
    }
    PathBuf::from(p)
}

// ---------------------------------------------------------------- misc

fn err(msg: &str) -> String { format!("{{\"error\":{}}}", qs(msg)) }

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn query_get(query: &str, key: &str) -> String {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key { return percent_decode(v); }
        }
    }
    String::new()
}

fn percent_decode(s: &str) -> String {
    let b = s.replace('+', " ");
    let bytes = b.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h = hex(bytes[i + 1]).zip(hex(bytes[i + 2]));
            if let Some((hi, lo)) = h { out.push(hi * 16 + lo); i += 3; continue; }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
fn hex(b: u8) -> Option<u8> {
    match b { b'0'..=b'9' => Some(b - b'0'), b'a'..=b'f' => Some(b - b'a' + 10), b'A'..=b'F' => Some(b - b'A' + 10), _ => None }
}
