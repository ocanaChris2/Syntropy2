//! Syntropy2 Compressor — a native desktop GUI for the `nsc` codec.
//!
//! Drag files or folders onto the window (or click "Add files…"), pick a level, and Compress /
//! Decompress / Verify, pack everything into one archive, or unpack an archive. Each job runs on
//! its own background thread and reports back over a channel, so the window never blocks. All codec
//! work is delegated to the installed `nsc` CLI (see `backend`).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use backend::{fmt_bytes, CompressOutcome, DecompressOutcome, Model, PackOutcome, UnpackOutcome, VerifyOutcome};
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

#[derive(Clone, PartialEq, Eq)]
enum Op {
    Compress,
    Decompress,
    Verify,
    /// Unpack the archive into this folder.
    Unpack(PathBuf),
}

/// A finished job, sent from a worker thread back to the UI.
enum JobResult {
    Compress(Result<CompressOutcome, String>),
    Decompress(Result<DecompressOutcome, String>),
    Verify(Result<VerifyOutcome, String>),
    Pack(Result<PackOutcome, String>),
    Unpack(Result<UnpackOutcome, String>),
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
    Packed(PackOutcome),
    Unpacked(UnpackOutcome),
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
    fn is_archive(&self) -> bool {
        self.path.extension().and_then(|e| e.to_str()) == Some("nsca")
    }
    fn is_dir(&self) -> bool { self.path.is_dir() }
}

// ------------------------------------------------------------------ app

struct App {
    nsc: Option<PathBuf>,
    models: Vec<Model>,
    selected_model: Option<String>,
    level: u32,
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
            level: 5,
            entries: Vec::new(),
            next_id: 1,
            tx,
            rx,
            status,
        }
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if !p.is_file() && !p.is_dir() {
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
        let level = self.level;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = match op {
                Op::Compress => JobResult::Compress(backend::compress(&nsc, model.as_deref(), level, &path)),
                Op::Decompress => JobResult::Decompress(backend::decompress(&nsc, &path)),
                Op::Verify => JobResult::Verify(backend::verify(&nsc, model.as_deref(), level, &path)),
                Op::Unpack(dest) => JobResult::Unpack(backend::unpack(&nsc, &path, &dest)),
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
                    JobResult::Pack(Ok(o)) => { e.size = o.out_bytes; Detail::Packed(o) }
                    JobResult::Unpack(Ok(o)) => Detail::Unpacked(o),
                    JobResult::Compress(Err(e))
                    | JobResult::Decompress(Err(e))
                    | JobResult::Verify(Err(e))
                    | JobResult::Pack(Err(e))
                    | JobResult::Unpack(Err(e)) => Detail::Error(e),
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
        let mut pack_dialog = false;

        // Panels are children of this root Ui; the central panel MUST be shown last.
        self.top_panel(ui, &mut add_dialog, &mut clear_all);
        self.bottom_panel(ui, &mut to_run, &mut pack_dialog);
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
        if pack_dialog {
            self.start_pack(&ctx);
        }
        for (id, op) in to_run {
            // unpacking asks where to first
            let op = match op {
                Op::Unpack(_) => match rfd::FileDialog::new().set_title("Unpack into folder").pick_folder() {
                    Some(dir) => Op::Unpack(dir),
                    None => continue,
                },
                o => o,
            };
            self.spawn(&ctx, id, op);
        }

        // Keep animating spinners while any job is in flight.
        if self.entries.iter().any(|e| e.busy) {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

impl App {
    /// Ask for an archive name, then pack every listed file and folder (except compressed and
    /// archive files) into it on a worker thread; the archive appears in the list.
    fn start_pack(&mut self, ctx: &egui::Context) {
        let Some(nsc) = self.nsc.clone() else { return };
        let paths: Vec<PathBuf> = self.entries.iter().filter(|e| !e.busy && !e.is_nsc() && !e.is_archive()).map(|e| e.path.clone()).collect();
        if paths.is_empty() { self.status = "add files or folders to pack first".into(); return; }
        let Some(mut out) = rfd::FileDialog::new().set_title("Save archive as").set_file_name("archive.nsca").save_file() else { return };
        if out.extension().and_then(|e| e.to_str()) != Some("nsca") { out.set_extension("nsca"); }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.retain(|e| e.path != out);
        self.entries.push(Entry { id, path: out.clone(), size: 0, busy: true, detail: Detail::None });
        let (tx, ctx, level) = (self.tx.clone(), ctx.clone(), self.level);
        thread::spawn(move || {
            let result = JobResult::Pack(backend::pack(&nsc, level, &paths, &out));
            let _ = tx.send(Msg { id, result });
            ctx.request_repaint();
        });
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
                ui.add_space(12.0);
                ui.label("Level:");
                egui::ComboBox::from_id_salt("level")
                    .selected_text(level_label(self.level))
                    .show_ui(ui, |ui| {
                        for l in 1..=9u32 {
                            ui.selectable_value(&mut self.level, l, level_label(l));
                        }
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

    fn bottom_panel(&mut self, ui: &mut egui::Ui, to_run: &mut Vec<(u64, Op)>, pack_dialog: &mut bool) {
        egui::Panel::bottom(egui::Id::new("bottom")).show(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let any = !self.entries.is_empty() && self.nsc.is_some();
                ui.add_enabled_ui(any, |ui| {
                    // Compress every plain file (not .nsc, archives or folders).
                    if ui.button("⏵  Compress all").clicked() {
                        for e in &self.entries {
                            if !e.busy && !e.is_nsc() && !e.is_archive() && !e.is_dir() {
                                to_run.push((e.id, Op::Compress));
                            }
                        }
                    }
                    if ui.button("📦  Pack into archive…").on_hover_text("One solid, deduplicated archive of every listed file and folder").clicked() {
                        *pack_dialog = true;
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
                            if !e.busy && !e.is_nsc() && !e.is_archive() && !e.is_dir() {
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
                let rows: Vec<(u64, String, u64, bool, bool, bool, bool)> = self
                    .entries
                    .iter()
                    .map(|e| (e.id, e.name(), e.size, e.busy, e.is_nsc(), e.is_archive(), e.is_dir()))
                    .collect();
                for (id, name, size, busy, is_nsc, is_archive, is_dir) in rows {
                    let detail_idx = self.entries.iter().position(|e| e.id == id).unwrap();
                    row_frame(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Left: name + size.
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(if is_dir { format!("📁 {name}") } else { name.clone() }).strong());
                                ui.label(egui::RichText::new(if is_dir { "folder".to_string() } else { fmt_bytes(size) }).small().weak());
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
                                        if is_dir {
                                            // folders are packed with "Pack into archive…"
                                        } else if is_archive {
                                            if ui.button("Unpack…").clicked() {
                                                to_run.push((id, Op::Unpack(PathBuf::new())));
                                            }
                                        } else if is_nsc {
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

fn level_label(l: u32) -> String {
    match l {
        1 => "1 · fastest".into(),
        5 => "5 · default".into(),
        9 => "9 · strongest".into(),
        n => n.to_string(),
    }
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
                let tok = o.tokens.map(|t| format!(" · {t} tok")).unwrap_or_default();
                ui.weak(format!(
                    "· {:.2}× · {:.3} bits/byte{} · {}",
                    o.ratio,
                    o.bpb,
                    tok,
                    fmt_dur(o.elapsed)
                ));
            });
            if !o.engine.is_empty() {
                ui.label(egui::RichText::new(&o.engine).small());
            }
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
        Detail::Packed(o) => {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(green, "✔ packed");
                ui.strong(fmt_bytes(o.out_bytes));
                ui.weak(format!("· {} · {}", o.summary, fmt_dur(o.elapsed)));
            });
            ui.label(egui::RichText::new(o.out.display().to_string()).small().weak());
        }
        Detail::Unpacked(o) => {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(green, "✔ unpacked");
                ui.weak(format!("{} · {}", o.summary, fmt_dur(o.elapsed)));
            });
            ui.label(egui::RichText::new(o.dest.display().to_string()).small().weak());
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
