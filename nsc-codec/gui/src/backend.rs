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
    pub tokens: Option<u64>,
    /// Which path nsc took, e.g. "nsc3 L5: generic; 1 block(s), 0 stored" or "model base-v1".
    pub engine: String,
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
        // Skip the "registry:" line, the "ID DEF TIER ..." header, and notes such as
        // "(no models installed …)" — those are not model rows.
        if t.is_empty() || t[0] == "registry:" || t[0] == "ID" || t[0].starts_with('(') {
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

/// Compress `input` to `<input>.nsc` at compression `level` (1..=9).
pub fn compress(nsc: &Path, model: Option<&str>, level: u32, input: &Path) -> Result<CompressOutcome, String> {
    let out = append_ext(input, "nsc");
    let t0 = Instant::now();
    let mut c = Command::new(nsc);
    c.arg("compress").arg(format!("-{level}")).arg(input).arg(&out);
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
    let stdout = String::from_utf8_lossy(&o.stdout);
    let tokens = parse_tokens(&stdout);
    let engine = parse_engine(&stdout);
    let ratio = if out_bytes > 0 { in_bytes as f64 / out_bytes as f64 } else { 0.0 };
    let bpb = if in_bytes > 0 { out_bytes as f64 * 8.0 / in_bytes as f64 } else { 0.0 };
    Ok(CompressOutcome { out, in_bytes, out_bytes, tokens, engine, ratio, bpb, elapsed })
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
pub fn verify(nsc: &Path, model: Option<&str>, level: u32, input: &Path) -> Result<VerifyOutcome, String> {
    let nscf = temp_path("nsc");
    let outf = temp_path("out");
    let t0 = Instant::now();
    let r = (|| {
        // compress input -> nscf
        let mut c = Command::new(nsc);
        c.arg("compress").arg(format!("-{level}")).arg(input).arg(&nscf);
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

/// Result of packing files and folders into an archive.
#[derive(Clone, Debug)]
pub struct PackOutcome {
    pub out: PathBuf,
    pub out_bytes: u64,
    pub summary: String,
    pub elapsed: Duration,
}

/// Result of unpacking an archive.
#[derive(Clone, Debug)]
pub struct UnpackOutcome {
    pub dest: PathBuf,
    pub summary: String,
    pub elapsed: Duration,
}

/// Deepest directory containing all of `paths` (their parents).
fn common_parent(paths: &[PathBuf]) -> PathBuf {
    let mut common = paths.first().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_default();
    for p in paths.iter().skip(1) {
        while !p.starts_with(&common) {
            match common.parent() {
                Some(up) => common = up.to_path_buf(),
                None => return PathBuf::from("/"),
            }
        }
    }
    common
}

/// Pack `paths` (files and folders) into the archive `out`; entries are named relative to their
/// common parent folder.
pub fn pack(nsc: &Path, level: u32, paths: &[PathBuf], out: &Path) -> Result<PackOutcome, String> {
    if paths.is_empty() { return Err("nothing to pack".into()); }
    let base = common_parent(paths);
    let t0 = Instant::now();
    let mut c = Command::new(nsc);
    c.current_dir(&base).arg("pack").arg(format!("-{level}")).arg(out);
    for p in paths {
        c.arg(p.strip_prefix(&base).unwrap_or(p));
    }
    let o = c.output().map_err(|e| format!("launch nsc: {e}"))?;
    if !o.status.success() {
        return Err(stderr_or(&o, "pack failed"));
    }
    let stdout = String::from_utf8_lossy(&o.stdout);
    let summary = stdout.lines().find(|l| l.starts_with("pack:")).map(|l| l.split(']').nth(1).unwrap_or(l).trim().to_string()).unwrap_or_default();
    Ok(PackOutcome { out: out.to_path_buf(), out_bytes: len_of(out), summary, elapsed: t0.elapsed() })
}

/// Unpack `archive` into `dest` (never replacing existing files).
pub fn unpack(nsc: &Path, archive: &Path, dest: &Path) -> Result<UnpackOutcome, String> {
    let t0 = Instant::now();
    let o = Command::new(nsc).arg("unpack").arg(archive).arg(dest).output().map_err(|e| format!("launch nsc: {e}"))?;
    if !o.status.success() {
        return Err(stderr_or(&o, "unpack failed"));
    }
    let stdout = String::from_utf8_lossy(&o.stdout);
    let summary = stdout.lines().find(|l| l.starts_with("unpack:")).and_then(|l| l.split_once("  ")).map(|(_, r)| r.trim().to_string()).unwrap_or_default();
    Ok(UnpackOutcome { dest: dest.to_path_buf(), summary, elapsed: t0.elapsed() })
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

/// Pull the token count out of the compress summary line: `... -> 12345 tok -> ...`. Only the
/// neural path reports tokens; the CM engine and raw storage do not.
fn parse_tokens(stdout: &str) -> Option<u64> {
    let t: Vec<&str> = stdout.split_whitespace().collect();
    for (i, w) in t.iter().enumerate() {
        if *w == "tok" && i > 0 {
            if let Ok(n) = t[i - 1].replace(',', "").parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// The bracketed path label from the compress summary line (`[nsc3 L5: …]`, `[model …]`, `[stored: …]`).
fn parse_engine(stdout: &str) -> String {
    let line = stdout.lines().find(|l| l.starts_with("compress:")).unwrap_or("");
    match (line.find('['), line.find(']')) {
        (Some(a), Some(b)) if b > a => line[a + 1..b].to_string(),
        _ => String::new(),
    }
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
    let mut ba = vec![0u8; 64 * 1024];
    let mut bb = vec![0u8; 64 * 1024];
    loop {
        // `read` may return short counts; fill both buffers fully before comparing.
        let na = read_full(&mut fa, &mut ba).map_err(|e| format!("read: {e}"))?;
        let nb = read_full(&mut fb, &mut bb).map_err(|e| format!("read: {e}"))?;
        if na != nb || ba[..na] != bb[..nb] {
            return Ok(false);
        }
        if na == 0 {
            return Ok(true);
        }
    }
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack two files from different folders (named relative to their common parent) and unpack
    /// them again. Needs an `nsc` binary (NSC_BIN or the dev build); skipped otherwise.
    #[test]
    fn pack_and_unpack_through_the_cli() {
        let Some(nsc) = find_nsc() else { eprintln!("no nsc binary — skipping"); return };
        let dir = std::env::temp_dir().join(format!("nsc-gui-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b/c")).unwrap();
        std::fs::write(dir.join("a/one.txt"), b"first file ".repeat(100)).unwrap();
        std::fs::write(dir.join("b/c/two.txt"), b"second file ".repeat(100)).unwrap();
        assert_eq!(common_parent(&[dir.join("a/one.txt"), dir.join("b/c/two.txt")]), dir);
        let out = dir.join("x.nsca");
        let p = pack(&nsc, 3, &[dir.join("a/one.txt"), dir.join("b/c")], &out).unwrap();
        assert!(p.out_bytes > 0, "{p:?}");
        let dest = dir.join("restore");
        unpack(&nsc, &out, &dest).unwrap();
        assert_eq!(std::fs::read(dest.join("a/one.txt")).unwrap(), b"first file ".repeat(100));
        assert_eq!(std::fs::read(dest.join("b/c/two.txt")).unwrap(), b"second file ".repeat(100));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
