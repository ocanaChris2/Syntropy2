//! The `nsc bench` command: ratio, encode/decode speed and peak memory of nsc vs
//! gzip/bzip2/xz/zstd/7z on real files, with a byte-exact roundtrip check on every measurement.
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

// ------------------------------------------------------------------ helpers
fn tool_exists(probe: &str) -> bool {
    Command::new(probe).arg("--help").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok()
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
