//! `nsc` — datatype-aware lossless compressor and archiver CLI.
//!
//! A byte-exact port of `int8_codec.py`: the int8 GRU + RoPE transformer + gate + fusion, the
//! classical order-0/1/2 + match feature model, and a range coder — all in dependency-free Rust.
//! Byte↔token conversion is delegated to `tok.py` (the one Python touch-point).
//!
//! This binary is built around a **versioned model registry** so models can be upgraded safely:
//! each compressed file records the id + fingerprint of the model that produced it, so a file is
//! always decompressed by the exact model it was compressed with — even after the default model is
//! upgraded. Batch mode loads the model once and compresses many files across CPU cores.
mod archive;
mod artifacts;
mod bench;
mod bits;
mod x86;
mod cm;
mod codec;
mod container;
mod crc;
mod detect;
mod dmath;
mod features;
mod forward;
mod golden;
mod json;
mod nsc3;
mod prof;
mod range_coder;
mod recode;
mod registry;
mod selftest;
mod simd;
mod tokenizer;
mod transform;

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
                               "created", "model-nsc", "o2", "tokenizer",
                               "levels", "only", "report", "nsc-jobs", "max-mb", "level", "kind", "lr", "lr2",
                               "ctx-mask", "sel-mask", "hash-bits", "mem", "block", "text", "seg", "trace-scan", "policy", "out", "only", "order", "compat", "reset", "drop", "x86-mask", "bin", "costs", "num", "dump", "dump-coefs"];

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
        else if a.len() == 2 && a.starts_with('-') && (b'1'..=b'9').contains(&a.as_bytes()[1]) {
            cli.kv.insert("level".into(), a[1..].to_string());
            i += 1;
        }
        else if a.len() > 1 && a.starts_with('-') && !a.starts_with("--") {
            // Outputs are positional; a stray short flag (e.g. `-o out`) must not become a filename.
            return Err(format!("unknown flag '{a}' (outputs are positional: nsc compress <in> <out>; long flags use --)"));
        }
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

/// `--mem <MiB>` as bytes.
fn mem_cap_from_cli(cli: &Cli) -> Option<u64> {
    cli.get("mem").and_then(|v| v.parse::<u64>().ok()).map(|m| m << 20)
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

/// Turn a decoded `.ids` file into the original bytes with the native tokenizer (no Python); falls
/// back to `tok.py` only if the native path cannot handle this tokenizer.
fn ids_to_file(reg: &Registry, model: &ModelDir, ids: &Path, out: &Path, quiet: bool) -> Result<(), String> {
    let native = (|| -> Result<(), String> {
        let tok = tokenizer::Tokenizer::load(&model.tokenizer())?;
        let ids = codec::read_ids(&p2s(ids))?;
        let bytes = tok.decode(&ids)?;
        std::fs::write(out, bytes).map_err(|e| format!("write {}: {e}", out.display()))
    })();
    match native {
        Ok(()) => Ok(()),
        Err(e) => {
            if !quiet { eprintln!("note: native tokenizer decode failed ({e}); trying tok.py"); }
            run_tok(reg, model, "dec", ids, out, quiet)
        }
    }
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

/// Options shared by `compress` and `batch compress`.
struct CompressOpts {
    level: u32,
    jobs: usize,
    block: usize,
    mem: Option<u64>,
    verify: bool,
    text: TextPolicy,
    force_store: bool,
    force_neural: bool,
    quiet: bool,
    keep_ids: bool,
    /// `--compat 1`: write only what nsc 0.3 (decoder level 1) reads.
    compat: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum TextPolicy { Auto, Neural, Cm }

impl CompressOpts {
    fn from_cli(cli: &Cli) -> Result<CompressOpts, String> {
        let level = match cli.get("level") {
            Some(v) => v.parse::<u32>().ok().filter(|l| (1..=9).contains(l)).ok_or_else(|| format!("--level must be 1..9, got '{v}'"))?,
            None => 5,
        };
        let block = match cli.get("block") {
            Some(v) => v.parse::<usize>().ok().filter(|&b| (1..=256).contains(&b)).ok_or("--block takes 1..256 (MiB)")? << 20,
            None => cm::levels::block_size(level),
        };
        let text = match cli.get("text").unwrap_or("auto") {
            "auto" => TextPolicy::Auto, "neural" => TextPolicy::Neural, "cm" => TextPolicy::Cm,
            o => return Err(format!("--text must be auto|neural|cm, got '{o}'")),
        };
        let mem = match cli.get("mem") {
            Some(v) => Some(v.parse::<u64>().map_err(|_| "--mem takes a size in MiB")? << 20),
            None => None,
        };
        let compat = match cli.get("compat") {
            Some("1") => true,
            Some(v) if v == nsc3::format::DECODER.to_string() => false,
            Some(v) => return Err(format!("--compat takes 1 (readable by nsc 0.3) or {}, got '{v}'", nsc3::format::DECODER)),
            None => false,
        };
        Ok(CompressOpts {
            level, jobs: jobs_from_cli(cli), block, mem, verify: !cli.has("no-verify"), text,
            force_store: cli.has("store"), force_neural: cli.has("neural"), quiet: cli.quiet(), keep_ids: cli.has("keep-ids"), compat,
        })
    }
}

impl CompressOpts {
    /// Whether the neural pass may be skipped on a sample's evidence (`--text auto` only: an
    /// explicit `--text neural` or `--neural` always runs it).
    fn gate(&self) -> bool { self.text == TextPolicy::Auto && !self.force_neural }
}

/// Largest text (bytes) sent to the neural model under `--text auto`. Size-based on purpose: a time
/// budget would make the output depend on machine speed. Neural text is split into 128 KB blocks
/// that decode in parallel (each at ~2 KB/s per core), so the higher levels accept larger inputs.
fn neural_limit(level: u32) -> u64 {
    match level { 1..=3 => 0, 4 => 128 << 10, 5 => 256 << 10, 6 => 1 << 20, 7 | 8 => 8 << 20, _ => 64 << 20 }
}

enum Route { Store(&'static str), Neural, Cm }

fn route(o: &CompressOpts, input: &str, model_available: bool) -> Route {
    if o.force_store { return Route::Store("forced"); }
    if input == "-" { return Route::Cm; }
    let size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    if size == 0 { return Route::Store("empty"); }
    if o.force_neural && model_available { return Route::Neural; }
    let text = looks_textual(Path::new(input));
    let neural = model_available && text && match o.text {
        TextPolicy::Neural => true,
        TextPolicy::Cm => false,
        TextPolicy::Auto => size <= neural_limit(o.level),
    };
    if neural { Route::Neural } else { Route::Cm }
}

fn ratio_str(in_b: u64, out_b: u64) -> String {
    let bpb = if in_b > 0 { out_b as f64 * 8.0 / in_b as f64 } else { 0.0 };
    let ratio = if out_b > 0 { in_b as f64 / out_b as f64 } else { 0.0 };
    format!("{bpb:.3} bits/byte, {ratio:.2}x")
}

/// Compress with the CM engine (NSC3); falls back to raw storage if that would not shrink the file.
fn compress_cm(o: &CompressOpts, input: &str, output: &str, jobs: usize) -> Result<String, String> {
    compress_nsc3(o, input, output, jobs, None)
}

/// NSC3 compression; with `neural`, text segments are coded by that model (small blocks so they
/// decode in parallel) and everything else by the CM engine.
fn compress_nsc3(o: &CompressOpts, input: &str, output: &str, jobs: usize, neural: Option<std::sync::Arc<nsc3::NeuralCtx>>) -> Result<String, String> {
    let block = if neural.is_some() { nsc3::NEURAL_BLOCK } else { o.block };
    let mut opts = nsc3::EncOpts::new(o.level, o.verify, neural);
    opts.compat = o.compat;
    // lanes never shrink below the level's block size: a lane restarts the model, and long-range
    // context is worth more than parallelism within one block
    opts.lane = o.block.max(cm::levels::block_size(o.level));
    opts.lane_jobs = jobs;
    opts.threads = nsc3::block::Threads::new(jobs);
    let in_bytes = if input == "-" { 0 } else { std::fs::metadata(input).map(|m| m.len()).unwrap_or(0) };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| nsc3::compress_path(input, output, &opts, jobs, block, o.mem)));
    let (st, warn) = match r {
        Ok(Ok(v)) => v,
        Ok(Err(e)) if input == "-" => return Err(e),
        Ok(Err(e)) => { eprintln!("note: CM compression failed ({e}); storing raw instead"); return store_one(input, output, "fallback"); }
        Err(_) if input == "-" => return Err("internal error in the CM engine".into()),
        Err(_) => { eprintln!("note: internal error in the CM engine; storing raw instead"); return store_one(input, output, "fallback"); }
    };
    if let Some(w) = warn { if !o.quiet { eprintln!("note: {w}"); } }
    if st.verify_failures > 0 { eprintln!("warning: {} block(s) failed verification and were stored raw", st.verify_failures); }
    if input != "-" && output != "-" && st.out_bytes >= in_bytes + 14 {
        return store_one(input, output, "incompressible");
    }
    Ok(format!("[nsc3 L{}: {}; {} block(s), {} stored]  {} B -> {} B  ({})",
        o.level, nsc3::kind_names(st.kinds), st.blocks, st.stored_blocks, st.raw_bytes, st.out_bytes, ratio_str(st.raw_bytes, st.out_bytes)))
}

fn store_one(input: &str, output: &str, reason: &str) -> Result<String, String> {
    let in_bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let out_bytes = container::write_stored(output, Path::new(input))?;
    Ok(format!("[stored: {reason}]  {in_bytes} B -> {out_bytes} B  ({})", ratio_str(in_bytes, out_bytes)))
}

/// Load a registry model as a shared neural context for NSC3 (binding 0). Refuses a model whose
/// fingerprint differs from the one a file was made with (`want_fp` != 0).
fn load_neural(reg: &Registry, id: &str, want_fp: u64) -> Result<std::sync::Arc<nsc3::NeuralCtx>, String> {
    let md = reg.get(id)?;
    if want_fp != 0 && md.manifest.fingerprint != want_fp {
        return Err(format!("model fingerprint mismatch for '{id}': this file was produced by a different build of that \
                            model than the one installed. Install the exact model version to decompress it."));
    }
    let _t = prof::span(|| "load neural model".into());
    let tok = tokenizer::Tokenizer::load(&md.tokenizer())?;
    let (model, o2) = load_pair(&md);
    Ok(std::sync::Arc::new(nsc3::NeuralCtx { id: md.manifest.id.clone(), fingerprint: md.manifest.fingerprint, binding: 0, model, o2, tok }))
}

/// Decompression-side loader: resolves the model a file binds (explicit `--model` overrides the id).
fn registry_loader<'a>(reg: &'a Registry, cli: &'a Cli) -> impl Fn(&[(String, u64)]) -> Result<Option<std::sync::Arc<nsc3::NeuralCtx>>, String> + 'a {
    let cache: Mutex<Option<std::sync::Arc<nsc3::NeuralCtx>>> = Mutex::new(None);
    move |b: &[(String, u64)]| {
        let Some((id, fp)) = b.first() else { return Ok(None) };
        let mut c = cache.lock().unwrap();
        if let Some(nc) = c.as_ref().filter(|nc| &nc.id == id && nc.fingerprint == *fp) { return Ok(Some(nc.clone())); }
        let nc = load_neural(reg, cli.get("model").unwrap_or(id), *fp)?;
        *c = Some(nc.clone());
        Ok(Some(nc))
    }
}

/// Estimated neural code length of `data` in bytes, from a few of its model windows spread evenly
/// over it (each window's cost is exact; the total is extrapolated). None when the text does not
/// tokenize exactly — the neural path would leave it to the CM engine anyway.
fn neural_estimate(nc: &nsc3::NeuralCtx, data: &[u8], jobs: usize) -> Option<f64> {
    let toks = nc.tok.encode(data);
    if toks.len() < 2 || toks.iter().any(|&t| t >= nc.model.cfg.vocab_size) || nc.tok.decode(&toks).ok()? != data { return None; }
    let steps = toks.len() - 1;
    let nwin = steps.div_ceil(nc.model.cfg.seg_len);
    let k = nwin.min(jobs.clamp(3, 8));
    let wins: Vec<usize> = (0..k).map(|i| (2 * i + 1) * nwin / (2 * k)).collect();
    let covered: usize = wins.iter().map(|&w| (steps - w * nc.model.cfg.seg_len).min(nc.model.cfg.seg_len)).sum();
    let bits = codec::window_bits_m::<forward::Pure>(&nc.model, &nc.o2, &toks, &wins, jobs);
    Some(bits / covered as f64 * steps as f64 / 8.0)
}

/// Best of both for neural-eligible text: the neural model wins on in-domain prose, the CM engine
/// on code, markup and tables. Writes whichever NSC3 file is smaller to `output`. With `gate`, the
/// neural pass is skipped when a sample of its windows shows it cannot come close to the CM result
/// (it costs far more time than the CM pass).
fn compress_best_of(o: &CompressOpts, input: &str, output: &str, jobs: usize, nc: std::sync::Arc<nsc3::NeuralCtx>, gate: bool) -> Result<String, String> {
    let alt = format!("{output}.neural.partial");
    let cm = { let _t = prof::span(|| "best-of: CM pass".into()); compress_cm(o, input, output, jobs)? };
    if gate {
        let _t = prof::span(|| "best-of: neural gate".into());
        let cm_bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0) as f64;
        let est = std::fs::read(input).ok().and_then(|d| neural_estimate(&nc, &d, jobs));
        if est.is_none_or(|e| e > cm_bytes * 1.05) { return Ok(cm); }
    }
    let neural = { let _t = prof::span(|| "best-of: neural pass".into()); compress_nsc3(o, input, &alt, jobs, Some(nc)) };
    let (a, b) = (std::fs::metadata(output).map(|m| m.len()).unwrap_or(u64::MAX), std::fs::metadata(&alt).map(|m| m.len()).unwrap_or(u64::MAX));
    match neural {
        Ok(msg) if b < a => { std::fs::rename(&alt, output).map_err(|e| format!("rename {alt} -> {output}: {e}"))?; Ok(msg) }
        _ => { let _ = std::fs::remove_file(&alt); Ok(cm) }
    }
}

fn cmd_compress(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "compress", "<input|-> <output.nsc|->")?;
    let o = CompressOpts::from_cli(cli)?;
    let model = if o.force_store || input == "-" { None } else { reg.resolve(cli.get("model")).ok() };
    let msg = match route(&o, &input, model.is_some()) {
        Route::Store(why) => store_one(&input, &output, why)?,
        Route::Neural => {
            let md = model.as_ref().unwrap();
            match load_neural(reg, &md.manifest.id, 0) {
                Ok(nc) => compress_best_of(&o, &input, &output, o.jobs, nc, o.gate())?,
                Err(e) => { if !o.quiet { eprintln!("note: neural model unavailable ({e}); using the CM engine"); } compress_cm(&o, &input, &output, o.jobs)? }
            }
        }
        Route::Cm => compress_cm(&o, &input, &output, o.jobs)?,
    };
    if !o.quiet {
        let line = format!("compress: {input} -> {output}  {msg}");
        if output == "-" { eprintln!("{line}"); } else { println!("{line}"); }
    }
    Ok(())
}

fn cmd_decompress(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let (input, output) = two_positionals(cli, "decompress", "<input.nsc|-> <output|->")?;
    let quiet = cli.quiet();
    let say = |line: String| if !quiet { if output == "-" { eprintln!("{line}"); } else { println!("{line}"); } };
    let mem = mem_cap_from_cli(cli);
    if input == "-" {
        let st = nsc3::decompress_path("-", &output, jobs_from_cli(cli), mem, &registry_loader(reg, cli))?;
        say(format!("decompress: - -> {output}  [nsc3]  {} bytes", st.raw_bytes));
        return Ok(());
    }
    if archive::is_archive(&input) { return Err(format!("{input} is an archive: use `nsc unpack {input} [dir]`")); }
    let header = container::read_header(&input)?;
    match header.format {
        container::Format::Stored => {
            let n = container::extract_stored(&input, &output)?;
            say(format!("decompress: {input} -> {output}  [stored]  {n} bytes"));
            Ok(())
        }
        container::Format::Nsc3 => {
            let st = nsc3::decompress_path(&input, &output, jobs_from_cli(cli), mem, &registry_loader(reg, cli))?;
            say(format!("decompress: {input} -> {output}  [nsc3: {}]  {} bytes", nsc3::kind_names(st.kinds), st.raw_bytes));
            Ok(())
        }
        container::Format::Nsc2 | container::Format::Nscz => {
            let model = model_for_decompress(reg, cli, &header)?;
            let tmp = temp_ids();
            let r = (|| {
                let (m, o2) = load_pair(&model);
                let ntok = decode_file(&m, &o2, Path::new(&input), &tmp)?;
                ids_to_file(reg, &model, &tmp, Path::new(&output), quiet)?;
                say(format!("decompress: {input} -> {output}  [model {}]  {ntok} tokens", model.manifest.id));
                Ok(())
            })();
            if !cli.has("keep-ids") { let _ = std::fs::remove_file(&tmp); }
            r
        }
    }
}

/// Verify a compressed file decodes cleanly (checksums for NSC3), without writing the output.
fn cmd_test(reg: &Registry, cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc test <file.nsc|archive.nsca>")?;
    if archive::is_archive(f) {
        let (n, bytes) = archive::test(f, jobs_from_cli(cli), mem_cap_from_cli(cli))?;
        println!("test: {f}: OK  (archive, {n} files, {bytes} content bytes, every file checksum verified)");
        return Ok(());
    }
    let header = container::read_header(f)?;
    match header.format {
        container::Format::Nsc3 => {
            let reader = std::io::BufReader::new(std::fs::File::open(f).map_err(|e| format!("open {f}: {e}"))?);
            let st = nsc3::sched::decompress(reader, std::io::sink(), jobs_from_cli(cli), nsc3::mem_budget(mem_cap_from_cli(cli)), &registry_loader(reg, cli))?;
            println!("test: {f}: OK  ({} bytes in {} block(s), checksums verified)", st.raw_bytes, st.blocks);
        }
        container::Format::Stored => {
            let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            if size != header.n as u64 + 14 { return Err(format!("{f}: stored length mismatch (corrupt)")); }
            println!("test: {f}: OK  (stored, {} bytes)", header.n);
        }
        _ => {
            let tmp = temp_ids();
            let model = model_for_decompress(reg, cli, &header)?;
            let (m, o2) = load_pair(&model);
            let r = decode_file(&m, &o2, Path::new(f), &tmp);
            let _ = std::fs::remove_file(&tmp);
            let n = r?;
            println!("test: {f}: decoded {n} tokens (legacy NSC2 files carry no checksum)");
        }
    }
    Ok(())
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
    if header.format != container::Format::Nsc2 && header.format != container::Format::Nscz {
        return Err("decompress-ids only applies to neural (NSC2) files; use `nsc decompress`".into());
    }
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
    let out_dir = cli.get("out-dir");
    let quiet = cli.quiet();
    let copts = CompressOpts::from_cli(cli)?;
    let default_model = if compress { reg.resolve(cli.get("model")).ok() } else { None };

    // Each needed neural model is loaded once and shared read-only by all workers: compress loads
    // the default model only if some file routes to the neural path; decompress loads per binding
    // (NSC3) or per recorded model (legacy NSC2). NSC3 CM and stored files need no model.
    let mut plan: Vec<(bool, String, PathBuf)> = Vec::with_capacity(files.len());
    let mut legacy: HashMap<String, (ModelDir, Model, O2Table)> = HashMap::new();
    let mut legacy_of: HashMap<String, String> = HashMap::new();
    for f in &files {
        if compress {
            let neural = matches!(route(&copts, f, default_model.is_some()), Route::Neural);
            plan.push((neural, f.clone(), out_path(f, out_dir, true)));
        } else {
            let header = container::read_header(f)?;
            if matches!(header.format, container::Format::Nsc2 | container::Format::Nscz) {
                let md = model_for_decompress(reg, cli, &header)?;
                let id = md.manifest.id.clone();
                legacy.entry(id.clone()).or_insert_with(|| { let (m, o2) = load_pair(&md); (md, m, o2) });
                legacy_of.insert(f.clone(), id);
            }
            plan.push((false, f.clone(), out_path(f, out_dir, false)));
        }
    }
    let neural_ctx = match (&default_model, plan.iter().any(|p| p.0)) {
        (Some(md), true) => load_neural(reg, &md.manifest.id, 0).ok(),
        _ => None,
    };
    let loader = registry_loader(reg, cli);
    let per_job = nsc3::encode_mem(copts.level, copts.block);
    let (jobs, warn) = nsc3::plan_jobs(jobs, plan.len(), per_job, copts.mem);
    if let Some(w) = warn { eprintln!("note: {w}"); }

    println!("batch {}: {} files, {} job(s), {} model(s) loaded", sub, files.len(), jobs,
             legacy.len() + neural_ctx.is_some() as usize);
    let results: Mutex<Vec<(String, Result<String, String>)>> = Mutex::new(Vec::new());
    let next = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed) as usize;
                    if idx >= plan.len() { break; }
                    let (neural, input, output) = &plan[idx];
                    let out = p2s(output);
                    let r = if compress {
                        match (neural, &neural_ctx) {
                            (true, Some(nc)) => compress_best_of(&copts, input, &out, 1, nc.clone(), copts.gate()),
                            _ => match route(&copts, input, false) {
                                Route::Store(why) => store_one(input, &out, why),
                                _ => compress_cm(&copts, input, &out, 1),
                            },
                        }
                    } else {
                        let lm = legacy_of.get(input).map(|id| { let (md, m, o2) = &legacy[id]; (md, m, o2) });
                        decompress_one(reg, &copts, lm, input, output, &loader)
                    };
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

/// One batch decompression. `legacy` is the loaded model for an NSC2 file; NSC3 files resolve
/// any neural binding through `loader`.
fn decompress_one(reg: &Registry, o: &CompressOpts, legacy: Option<(&ModelDir, &Model, &O2Table)>, input: &str, output: &Path,
                  loader: nsc3::Loader) -> Result<String, String> {
    let out = p2s(output);
    let header = container::read_header(input)?;
    match (header.format, legacy) {
        (container::Format::Stored, _) => {
            let n = container::extract_stored(input, &out)?;
            Ok(format!("stored -> {} ({} bytes)", output.display(), n))
        }
        (container::Format::Nsc3, _) => {
            let st = nsc3::decompress_path(input, &out, 1, o.mem, loader)?;
            Ok(format!("nsc3 -> {} ({} bytes)", output.display(), st.raw_bytes))
        }
        (_, Some((md, m, o2))) => {
            let tmp = temp_ids();
            let r = (|| {
                let ntok = decode_file(m, o2, Path::new(input), &tmp)?;
                ids_to_file(reg, md, &tmp, output, true)?;
                Ok(format!("{} tok -> {}", ntok, output.display()))
            })();
            if !o.keep_ids { let _ = std::fs::remove_file(&tmp); }
            r
        }
        _ => Err("no model available to decode this NSC2 file".into()),
    }
}

// ------------------------------------------------------------------ command: bench
fn cmd_bench(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() {
        return Err("usage: nsc bench <files|dirs...> [--classical] [--levels 1,5,9] [--only a,b] [--report out.json] [--no-verify] [--nsc-jobs N] [--max-mb N]".into());
    }
    let parse_list = |k: &str| cli.get(k).map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>());
    let levels: Vec<u32> = match parse_list("levels") {
        Some(v) => v.iter().map(|x| x.parse::<u32>().map_err(|_| format!("bad level '{x}'"))).collect::<Result<_, _>>()?,
        None => vec![],
    };
    let opts = bench::BenchOpts {
        paths: cli.positionals.clone(),
        classical: cli.has("classical") || levels.is_empty(),
        levels,
        only: parse_list("only"),
        json: cli.get("report").map(|s| s.to_string()),
        verify: !cli.has("no-verify"),
        nsc_jobs: cli.get("nsc-jobs").and_then(|s| s.parse().ok()).filter(|&j: &usize| j > 0).unwrap_or(1),
        max_bytes: cli.get("max-mb").and_then(|s| s.parse::<u64>().ok()).map(|mb| mb * 1_000_000),
    };
    bench::run(&opts)
}

// ------------------------------------------------------------------ command: cm-test (developer)
/// Developer harness: compress one file in memory with the CM engine, verify, and report.
fn cmd_cm_test(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc cm-test <file> [--level N] [--kind generic|fast]")?;
    let data = std::fs::read(f).map_err(|e| format!("read {f}: {e}"))?;
    let level: u32 = cli.get("level").and_then(|s| s.parse().ok()).unwrap_or(5);
    let kind = match cli.get("kind").unwrap_or("generic") { "fast" => cm::levels::Kind::Fast, _ => cm::levels::Kind::Generic };
    let mut cfg = cm::levels::preset(level, kind, data.len());
    if let Some(v) = cli.get("lr").and_then(|s| s.parse().ok()) { cfg.lr = v; }
    if let Some(v) = cli.get("lr2").and_then(|s| s.parse().ok()) { cfg.lr2 = v; }
    if let Some(v) = cli.get("ctx-mask").and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()) { cfg.ctx_mask = v; }
    if let Some(v) = cli.get("sel-mask").and_then(|s| s.parse().ok()) { cfg.sel_mask = v; }
    if let Some(v) = cli.get("hash-bits").and_then(|s| s.parse().ok()) { cfg.hash_bits = v; }
    // the second binary model, as blocks for decoders of level 2 get it; `--bin HEX` keeps some of its contexts
    if !cli.has("no-bin") { cfg.ctx_mask |= cm::levels::bin2_mask(level); }
    if let Some(v) = cli.get("bin").and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()) {
        cm::predictor::BIN2_KEEP.store(v, std::sync::atomic::Ordering::Relaxed);
    }
    // the number model of text, likewise (for text with numbers enough, or with `--num`);
    // `--num HEX` keeps some of its contexts
    if !cli.has("no-num") && (cli.get("num").is_some() || cm::num::wanted(cm::num::digits(&data), data.len())) { cfg.ctx_mask |= cm::levels::num_mask(level); }
    if let Some(v) = cli.get("num").and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()) {
        cm::num::KEEP.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    prof::start();
    let seg_kind = match cli.get("seg") {
        Some("text") => nsc3::format::kind::TEXT,
        Some(_) => nsc3::format::kind::GENERIC,
        None => if looks_textual(Path::new(f)) { nsc3::format::kind::TEXT } else { nsc3::format::kind::GENERIC },
    };
    // `--costs FILE`: what each byte cost (bits, f32 little-endian), for bench/text_costs.py
    if let Some(out) = cli.get("costs") {
        struct Costs { bytes: Vec<f32>, cur: f64, n: u32 }
        impl cm::coder::Coder for Costs {
            fn code(&mut self, bit: u32, p: u32) -> u32 {
                let p1 = p.clamp(1, 65535) as f64 / 65536.0;
                self.cur -= (if bit == 1 { p1 } else { 1.0 - p1 }).log2();
                self.n += 1;
                if self.n == 8 {
                    self.bytes.push(self.cur as f32);
                    (self.cur, self.n) = (0.0, 0);
                }
                bit
            }
        }
        let mut pred = cm::predictor::Predictor::new(&cfg);
        pred.set_kind(seg_kind);
        let mut costs = Costs { bytes: Vec::with_capacity(data.len()), cur: 0.0, n: 0 };
        cm::predictor::code_bytes(&mut pred, &mut costs, &mut cm::predictor::ReadBuf(&data), 0..data.len());
        let raw: Vec<u8> = costs.bytes.iter().flat_map(|c| c.to_le_bytes()).collect();
        std::fs::write(out, raw).map_err(|e| format!("write {out}: {e}"))?;
        println!("{f}: {} -> {:.0} B by the model's probabilities", data.len(), costs.bytes.iter().map(|&c| c as f64).sum::<f64>() / 8.0);
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let enc = cm::encode_block(&data, &cfg, seg_kind);
    let te = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let dec = if cli.has("no-decode") { data.clone() } else { cm::decode_block(&enc, data.len(), &cfg, seg_kind) };
    let td = t1.elapsed().as_secs_f64();
    let ok = dec == data;
    let mb = data.len() as f64 / 1e6;
    println!("{f}: {} -> {} B  {:.4} bpb  enc {:.2} MB/s  dec {:.2} MB/s  hash_bits {}  {}",
        data.len(), enc.len(), enc.len() as f64 * 8.0 / data.len().max(1) as f64, mb / te, mb / td, cfg.hash_bits,
        if ok { "OK" } else { "ROUNDTRIP FAILED" });
    if ok { Ok(()) } else { Err("roundtrip failed".into()) }
}

// ------------------------------------------------------------------ command: exe-test (developer)
/// Developer harness: code only the machine code of executables (the segments the detector types
/// as x86 / x86-64 code), each on its own, with the executable model; sizes, and optionally the
/// effect of the instruction contexts (`--x86-mask` keeps the contexts whose bits are set).
fn cmd_exe_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc exe-test <files...> [--level N] [--x86-mask HEX] [--hash-bits N] [--plain]".into()); }
    let level: u32 = cli.get("level").and_then(|s| s.parse().ok()).unwrap_or(5);
    let keep = cli.get("x86-mask").and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()).unwrap_or(u64::MAX);
    let (mut tot, mut tot_new, mut secs) = (0usize, 0usize, 0f64);
    for f in &cli.positionals {
        let data = std::fs::read(f).map_err(|e| format!("read {f}: {e}"))?;
        let name = Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (mut off, mut code, mut coded) = (0usize, 0usize, 0usize);
        for seg in nsc3::plan::segments(&data) {
            let len = seg.len as usize;
            if seg.kind == nsc3::format::kind::EXE && matches!(seg.params.first().map(|a| a & !detect::EXE_RELOCATABLE), Some(1 | 2)) {
                let mut bytes = data[off..off + len].to_vec();
                transform::bcj::forward(transform::bcj::for_arch(seg.params[0], !cli.has("old-transform")), &mut bytes);
                let mut cfg = cm::levels::preset(level, cm::levels::Kind::Generic, len);
                if !cli.has("plain") { cfg.ctx_mask |= cm::levels::x86_mask(level); }
                x86::KEEP.store(keep, std::sync::atomic::Ordering::Relaxed);
                if let Some(v) = cli.get("hash-bits").and_then(|s| s.parse().ok()) { cfg.hash_bits = v; }
                let mut pred = cm::predictor::Predictor::new(&cfg);
                pred.set_segment(nsc3::format::kind::EXE, &seg.params, 0);
                let mut enc = cm::coder::Encoder::new();
                let t0 = std::time::Instant::now();
                cm::predictor::code_bytes(&mut pred, &mut enc, &mut cm::predictor::ReadBuf(&bytes), 0..len);
                secs += t0.elapsed().as_secs_f64();
                code += len;
                coded += enc.finish().len();
            }
            off += len;
        }
        if code > 0 { println!("{name:<28} code {code:>9} -> {coded:>9} B  ({:.2}% saved)", 100.0 * (1.0 - coded as f64 / code as f64)); }
        tot += code;
        tot_new += coded;
    }
    println!("TOTAL code {tot} -> {tot_new} B  ({:.2}% saved)  {:.2} MB/s", 100.0 * (1.0 - tot_new as f64 / tot.max(1) as f64), tot as f64 / 1e6 / secs.max(1e-9));
    Ok(())
}

// ------------------------------------------------------------------ command: deflate-scan (developer)
/// Developer harness: find recompressible deflate streams in a file and report each candidate.
fn cmd_deflate_scan(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc deflate-scan <file> [--all]")?;
    let data = std::fs::read(f).map_err(|e| format!("read {f}: {e}"))?;
    let t0 = std::time::Instant::now();
    let mut log = vec![];
    let found = transform::deflate::scan::find_logged(&data, 1 << 30, !cli.has("no-verify"), true, &[], Some(&mut log));
    let secs = t0.elapsed().as_secs_f64();
    let show = if cli.has("all") { log.len() } else { log.len().min(40) };
    for l in &log[..show] { println!("  {l}"); }
    if show < log.len() { println!("  … {} more (use --all)", log.len() - show); }
    let (comp, inner): (usize, usize) = found.iter().fold((0, 0), |a, x| (a.0 + x.len(), a.1 + x.inner_len()));
    let side: usize = found.iter().map(|x| match x {
        transform::deflate::scan::Found::Deflate(d) => d.side.len(),
        transform::deflate::scan::Found::Png(p) => p.body.side.len(),
        transform::deflate::scan::Found::Jpeg(j) => j.split.pads.len(),
        transform::deflate::scan::Found::Recode(_) => 0,
    }).sum();
    println!("{f}: {} streams, {comp} compressed bytes ({:.1}% of file) -> {inner} bytes + {side} record bytes ({:.3}%), {:.2} s ({:.1} MB/s of output)",
        found.len(), 100.0 * comp as f64 / data.len().max(1) as f64, 100.0 * side as f64 / comp.max(1) as f64, secs, inner as f64 / 1e6 / secs);
    Ok(())
}

// ------------------------------------------------------------------ command: jpeg-test (developer)
/// Developer harness: check that JPEG files rebuild exactly from their coefficients.
/// Experiment: write a database's seven recoded streams as `<prefix>.<n>`.
fn cmd_sqlite_dump(cli: &Cli) -> Result<(), String> {
    let [f, prefix] = &cli.positionals[..] else { return Err("usage: nsc sqlite-dump <db> <prefix>".into()) };
    let b = std::fs::read(f).map_err(|e| e.to_string())?;
    let geo = recode::sqlite::header(&b).ok_or("not a sqlite database")?;
    let sp = recode::sqlite::split(&b, &geo).ok_or("database does not split")?;
    for (i, st) in sp.streams.iter().enumerate() {
        std::fs::write(format!("{prefix}.{i}"), st).map_err(|e| e.to_string())?;
    }
    println!("{f}: {} bytes, streams {:?}", b.len(), sp.streams.iter().map(|s| s.len()).collect::<Vec<_>>());
    Ok(())
}

fn cmd_jpeg_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc jpeg-test <files...> [--trace-scan N --policy P --out prefix]".into()); }
    if let Some(si) = cli.get("trace-scan").and_then(|v| v.parse().ok()) {
        let b = std::fs::read(&cli.positionals[0]).map_err(|e| e.to_string())?;
        let pol = cli.get("policy").and_then(|v| v.parse().ok()).unwrap_or(0);
        let r = transform::jpeg::trace_scan(&b, si, pol, cli.get("out").unwrap_or("/tmp/jt"));
        println!("trace written ({r:?})");
        return Ok(());
    }
    if let Some(prefix) = cli.get("dump-coefs") {
        // experiment: write each file's coefficient leaf and its params next to `prefix`
        for (i, f) in cli.positionals.iter().enumerate() {
            let Ok(b) = std::fs::read(f) else { continue };
            let Ok(sp) = transform::jpeg::analyze(&b) else { println!("{f}: not splittable"); continue };
            std::fs::write(format!("{prefix}{i}.coefs"), &sp.coefs).map_err(|e| e.to_string())?;
            std::fs::write(format!("{prefix}{i}.params"), cm::jpeg::params(&sp.frame, &sp.quant)).map_err(|e| e.to_string())?;
            println!("{i} {f} {} coef bytes", sp.coefs.len());
        }
        return Ok(());
    }
    if cli.has("coef-model") {
        // coefficient model only: cost per decision family vs the original JPEG
        let (mut tot_orig, mut tot_new) = (0u64, 0u64);
        let mut fam = [0u64; cm::jpeg::FAMILIES];
        for f in &cli.positionals {
            let Ok(b) = std::fs::read(f) else { continue };
            let Ok(sp) = transform::jpeg::analyze(&b) else { println!("{f}: not splittable"); continue };
            let mut m = cm::jpeg::JpegModel::new();
            m.stats = Some([0; cm::jpeg::FAMILIES]);
            let mut enc = cm::coder::Encoder::new();
            let t0 = std::time::Instant::now();
            m.code_leaf(&mut enc, &mut cm::predictor::ReadBuf(&sp.coefs), 0..sp.coefs.len(), &cm::jpeg::params(&sp.frame, &sp.quant))?;
            let secs = t0.elapsed().as_secs_f64();
            let coded = enc.finish().len() as u64 + sp.header.len() as u64 + sp.pads.len() as u64;
            let st = m.stats.unwrap();
            for i in 0..cm::jpeg::FAMILIES { fam[i] += st[i]; }
            tot_orig += b.len() as u64;
            tot_new += coded;
            println!("{f}: {} -> {coded} ({:+.2}%)  coefs {:.1} MB/s", b.len(), 100.0 * (coded as f64 / b.len() as f64 - 1.0), b.len() as f64 / 1e6 / secs);
        }
        let all: u64 = fam.iter().sum();
        for i in 0..cm::jpeg::FAMILIES { println!("  {:<9} {:>10} bytes  {:5.1}%", cm::jpeg::FAMILY_NAMES[i], fam[i] >> 19, 100.0 * fam[i] as f64 / all.max(1) as f64); }
        println!("TOTAL {tot_orig} -> {tot_new} ({:+.2}%)", 100.0 * (tot_new as f64 / tot_orig.max(1) as f64 - 1.0));
        return Ok(());
    }
    let (mut ok, mut n) = (0, 0);
    for f in &cli.positionals {
        let Ok(b) = std::fs::read(f) else { continue };
        n += 1;
        let d = transform::jpeg::diagnose(&b);
        if d.starts_with("OK") { ok += 1; }
        if !d.starts_with("OK") || !cli.quiet() { println!("{f}: {d}"); }
    }
    println!("{ok}/{n} rebuild exactly");
    Ok(())
}

// ------------------------------------------------------------------ command: mp3-test (developer)
/// One file's result in `nsc mp3-test`.
struct Mp3Report {
    line: String,
    bytes: u64,
    coded: [u64; 3], // frames, rest, other
    fam: [u64; cm::mp3::FAMILIES],
    orig: [u64; 8],
    exact: bool,
    steps: [u64; 16],
    layouts: [(u64, u64); 3],
}

fn mp3_test_file(f: &str, level: u32, generic: bool, decode: bool) -> Result<Mp3Report, String> {
    let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    // every run of frames in the file; the bytes between them are coded as generic data
    let (mut i, mut other, mut runs) = (0usize, Vec::new(), vec![]);
    while i < b.len() {
        match (b[i] == 0xFF).then(|| recode::mp3::split(&b[i..])).flatten() {
            Some(s) => { let n = s.len; runs.push((i, s)); i += n; }
            None => { other.push(b[i]); i += 1; }
        }
    }
    let cost = |d: &[u8]| -> u64 { cm::encode_block(d, &cm::levels::preset(level, cm::levels::Kind::Generic, d.len().max(1)), 1).len() as u64 };
    let (mut frames, mut parsed, mut covered) = (0, 0, 0usize);
    let (c_other, mut c_frames, mut c_rest) = (cost(&other), 0u64, 0u64);
    let mut exact = true;
    let mut orig = [0u64; 8];
    let t0 = std::time::Instant::now();
    let mut m = cm::mp3::Mp3Model::new();
    m.enable_stats();
    for (at, s) in &runs {
        let mut out = Vec::with_capacity(s.len);
        exact &= recode::mp3::rebuild(&s.frames, &s.rest, &mut out).is_ok() && out == b[*at..*at + s.len];
        frames += s.count.0;
        parsed += s.count.1;
        covered += s.len;
        c_rest += cost(&s.rest);
        for (k, v) in recode::mp3::original_bits(&s.frames)?.iter().enumerate() { orig[k] += v; }
        if generic {
            c_frames += cost(&s.frames);
            continue;
        }
        let mut enc = cm::coder::Encoder::new();
        m.code_leaf(&mut enc, &mut cm::predictor::ReadBuf(&s.frames), 0..s.frames.len(), &[])?;
        let coded = enc.finish();
        c_frames += coded.len() as u64;
        // a fresh model decodes the file's first run (later runs continue the encoder's model)
        if decode && runs.len() == 1 {
            let mut back = vec![0u8; s.frames.len()];
            let r = cm::mp3::Mp3Model::new().code_leaf(&mut cm::coder::Decoder::new(&coded), &mut cm::predictor::WriteBuf(&mut back), 0..s.frames.len(), &[]);
            exact &= r.is_ok() && back == s.frames;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let coded = c_other + c_frames + c_rest;
    let name = std::path::Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let line = format!("{name:<40} {:>9} {parsed:>6}/{frames:<6} {:>3.0}% {}  {coded:>9} {:+7.2}%  {:.2} MB/s",
        b.len(), 100.0 * covered as f64 / b.len().max(1) as f64, if exact { "exact   " } else { "MISMATCH" },
        100.0 * (coded as f64 / b.len().max(1) as f64 - 1.0), b.len() as f64 / 1e6 / secs);
    Ok(Mp3Report { line, bytes: b.len() as u64, coded: [c_frames, c_rest, c_other], fam: m.stats(), orig, exact, steps: m.step_cost, layouts: m.layout_cost })
}

/// Developer harness: take the MP3 streams of files apart, check the rebuild and the model's
/// decode, and report what each decision family costs against the original's bits.
fn cmd_mp3_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc mp3-test <files...> [--level N] [--generic] [--no-decode] [--jobs N]".into()); }
    let level: u32 = cli.get("level").and_then(|v| v.parse().ok()).unwrap_or(5);
    let (generic, decode) = (cli.has("generic"), !cli.has("no-decode"));
    let files = &cli.positionals;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<Mp3Report, String>>>> = Mutex::new((0..files.len()).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..jobs_from_cli(cli).min(files.len()) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= files.len() { break; }
                let r = mp3_test_file(&files[k], level, generic, decode);
                results.lock().unwrap()[k] = Some(r);
            });
        }
    });
    let (mut tot, mut coded, mut bad) = (0u64, [0u64; 3], 0usize);
    let mut fam = [0u64; cm::mp3::FAMILIES];
    let mut orig = [0u64; 8];
    let (mut steps, mut layouts) = ([0u64; 16], [(0u64, 0u64); 3]);
    for r in results.into_inner().unwrap().into_iter().flatten() {
        let r = r?;
        for k in 0..16 { steps[k] += r.steps[k]; }
        for k in 0..3 { layouts[k].0 += r.layouts[k].0; layouts[k].1 += r.layouts[k].1; }
        println!("{}", r.line);
        tot += r.bytes;
        for k in 0..3 { coded[k] += r.coded[k]; }
        for k in 0..cm::mp3::FAMILIES { fam[k] += r.fam[k]; }
        for k in 0..8 { orig[k] += r.orig[k]; }
        if !r.exact { bad += 1; }
    }
    // the model's families against what the original spends on the same information
    let fb = |ks: &[usize]| -> u64 { ks.iter().map(|&k| fam[k] >> 19).sum() };
    println!("  {:<22} {:>10} {:>10} {:>8}", "category", "original", "model", "change");
    for (name, o, m) in [
        ("header", orig[0], fb(&[0])), ("side information", orig[1], fb(&[1, 2, 3, 4, 10, 12])), ("scale factors", orig[2], fb(&[5])),
        ("pair code words", orig[3], fb(&[6, 7])), ("escape bits", orig[4], fb(&[8])), ("signs", orig[5], fb(&[9])),
        ("count1 code words", orig[6], fb(&[11])),
    ] {
        println!("  {:<22} {:>10} {:>10} {:>+7.1}%", name, o / 8, m, 100.0 * (m as f64 / (o / 8).max(1) as f64 - 1.0));
    }
    let detail: Vec<String> = (0..cm::mp3::FAMILIES).map(|k| format!("{} {}", cm::mp3::FAMILY_NAMES[k], fam[k] >> 19)).collect();
    println!("  families: {}", detail.join(", "));
    println!("  steps (bytes by threshold): {}", steps.iter().map(|s| (s >> 19).to_string()).collect::<Vec<_>>().join(" "));
    println!("  spectral bytes by layout (granules): long {} ({}), mixed {} ({}), short {} ({})",
        layouts[0].0 >> 19, layouts[0].1, layouts[1].0 >> 19, layouts[1].1, layouts[2].0 >> 19, layouts[2].1);
    let all: u64 = coded.iter().sum();
    println!("TOTAL {tot} -> {all} ({:+.3}%)  [frames {} rest {} other {}], {bad} file(s) not exact",
        100.0 * (all as f64 / tot.max(1) as f64 - 1.0), coded[0], coded[1], coded[2]);
    if bad > 0 { Err("mismatch".into()) } else { Ok(()) }
}

// ------------------------------------------------------------------ command: aac-test (developer)
/// One file's result in `nsc aac-test`.
struct AacReport {
    line: String,
    bytes: u64,
    coded: u64,
    fam: [u64; cm::aac::FAMILIES],
    orig: [u64; 7],
    exact: bool,
}

fn aac_test_file(f: &str, level: u32, generic: bool, decode: bool) -> Result<AacReport, String> {
    let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
    // the AAC runs: (offset, split, ADTS?, sampling-frequency index); everything else is generic data
    let mut runs: Vec<(usize, recode::aac::Split, bool, usize)> = vec![];
    let mut other = Vec::new();
    if let Some(idx) = recode::index(&mut std::io::Cursor::new(&b), b.len() as u64) {
        let mut o = 0usize;
        for r in &idx.runs {
            let recode::mp4::Stream::Aac { sf } = r.what else { continue };
            let start = r.start as usize;
            other.extend_from_slice(&b[o..start]);
            o = start;
            if let Some(s) = recode::aac::split_run(&b[start..], &r.sizes, sf as usize) {
                o += s.len;
                runs.push((start, s, false, sf as usize));
            }
        }
        other.extend_from_slice(&b[o..]);
    } else {
        let mut i = 0usize;
        while i < b.len() {
            match (b[i] == 0xFF).then(|| recode::aac::split_adts(&b[i..])).flatten() {
                Some(s) => { let n = s.len; runs.push((i, s, true, 0)); i += n; }
                None => { other.push(b[i]); i += 1; }
            }
        }
    }
    let cost = |d: &[u8]| -> u64 { cm::encode_block(d, &cm::levels::preset(level, cm::levels::Kind::Generic, d.len().max(1)), 1).len() as u64 };
    let (mut frames, mut parsed, mut covered, mut exact) = (0, 0, 0usize, true);
    let (c_other, mut c_frames, mut c_rest) = (cost(&other), 0u64, 0u64);
    let mut orig = [0u64; 7];
    let mut m = cm::aac::AacModel::new();
    m.enable_stats();
    let t0 = std::time::Instant::now();
    for (at, s, adts, sf) in &runs {
        let params = [*adts as u8, *sf as u8];
        let mut out = Vec::with_capacity(s.len);
        exact &= recode::aac::rebuild(&s.frames, &s.rest, *adts, *sf, &mut out).is_ok() && out == b[*at..*at + s.len];
        frames += s.count.0;
        parsed += s.count.1;
        covered += s.len;
        c_rest += cost(&s.rest);
        for (k, v) in recode::aac::original_bits(&s.frames, *adts, *sf)?.iter().enumerate() { orig[k] += v; }
        if generic {
            c_frames += cost(&s.frames);
            continue;
        }
        let mut enc = cm::coder::Encoder::new();
        m.code_leaf(&mut enc, &mut cm::predictor::ReadBuf(&s.frames), 0..s.frames.len(), &params)?;
        let bytes = enc.finish();
        c_frames += bytes.len() as u64;
        if decode && runs.len() == 1 {
            let mut back = vec![0u8; s.frames.len()];
            let r = cm::aac::AacModel::new().code_leaf(&mut cm::coder::Decoder::new(&bytes), &mut cm::predictor::WriteBuf(&mut back), 0..s.frames.len(), &params);
            exact &= r.is_ok() && back == s.frames;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let coded = c_other + c_frames + c_rest;
    let name = std::path::Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let line = format!("{name:<36} {:>9} {parsed:>6}/{frames:<6} {:>3.0}% audio {}  {coded:>9} {:+7.2}%  audio {:+7.2}%  [frames {c_frames} rest {c_rest} other {c_other}]  {:.2} MB/s",
        b.len(), 100.0 * covered as f64 / b.len().max(1) as f64, if exact { "exact   " } else { "MISMATCH" },
        100.0 * (coded as f64 / b.len().max(1) as f64 - 1.0), 100.0 * ((c_frames + c_rest) as f64 / covered.max(1) as f64 - 1.0), covered as f64 / 1e6 / secs);
    Ok(AacReport { line, bytes: b.len() as u64, coded, fam: m.stats(), orig, exact })
}

/// Developer harness: take the AAC of files apart (ADTS streams, or the audio tracks of MP4
/// files), check the rebuild and the model's decode, and report what each decision family costs.
fn cmd_aac_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc aac-test <files...> [--level N] [--generic] [--no-decode] [--jobs N]".into()); }
    let level: u32 = cli.get("level").and_then(|v| v.parse().ok()).unwrap_or(5);
    let (generic, decode) = (cli.has("generic"), !cli.has("no-decode"));
    let files = &cli.positionals;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<AacReport, String>>>> = Mutex::new((0..files.len()).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..jobs_from_cli(cli).min(files.len()) {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= files.len() { break; }
                let r = aac_test_file(&files[k], level, generic, decode);
                results.lock().unwrap()[k] = Some(r);
            });
        }
    });
    let (mut tot, mut tot_new, mut bad) = (0u64, 0u64, 0usize);
    let mut fam = [0u64; cm::aac::FAMILIES];
    let mut orig = [0u64; 7];
    for r in results.into_inner().unwrap().into_iter().flatten() {
        let r = r?;
        println!("{}", r.line);
        tot += r.bytes;
        tot_new += r.coded;
        for k in 0..cm::aac::FAMILIES { fam[k] += r.fam[k]; }
        for k in 0..7 { orig[k] += r.orig[k]; }
        if !r.exact { bad += 1; }
    }
    let fb = |ks: &[usize]| -> u64 { ks.iter().map(|&k| fam[k] >> 19).sum() };
    println!("  {:<24} {:>10} {:>10} {:>8}", "category", "original", "model", "change");
    for (name, o, m) in [
        ("headers, elements", orig[0], fb(&[0])), ("layout, gain, m/s", orig[1], fb(&[1, 2])), ("section data", orig[2], fb(&[3])),
        ("scale factors", orig[3], fb(&[4])), ("spectral data", orig[4], fb(&[5, 6, 7, 8])), ("pulse, tns, fill, data", orig[5] + orig[6], fb(&[9])),
    ] {
        println!("  {:<24} {:>10} {:>10} {:>+7.1}%", name, o / 8, m, 100.0 * (m as f64 / (o / 8).max(1) as f64 - 1.0));
    }
    let detail: Vec<String> = (0..cm::aac::FAMILIES).map(|k| format!("{} {}", cm::aac::FAMILY_NAMES[k], fam[k] >> 19)).collect();
    println!("  families: {}", detail.join(", "));
    println!("TOTAL {tot} -> {tot_new} ({:+.3}%), {bad} file(s) not exact", 100.0 * (tot_new as f64 / tot.max(1) as f64 - 1.0));
    if bad > 0 { Err("mismatch".into()) } else { Ok(()) }
}

// ------------------------------------------------------------------ command: h264-test (developer)
/// Developer harness: decode the CABAC slices of the H.264 tracks of MP4 files to bins, check that
/// re-encoding them gives the slice data back, and report what the bins cost under CABAC's own
/// model and under ours, per syntax element.
fn cmd_h264_test(cli: &Cli) -> Result<(), String> {
    use recode::h264::{self, se};
    if cli.positionals.is_empty() { return Err("usage: nsc h264-test <mp4 or .264 files...> [--verbose] [--lean] [--reset KB] [--drop family:input,...]".into()); }
    let (mut tot_slices, mut tot_exact, mut tot_bytes) = (0usize, 0usize, 0u64);
    let (mut tot_std, mut tot_model) = ([0f64; se::COUNT], [0f64; se::COUNT]);
    let mut tot_vlc = (0f64, 0f64);
    for f in &cli.positionals {
        // each walk of the stream keeps its own memory of the pictures before (as in real use)
        let (mut scratch, mut scratch2, mut scratch3) = (h264::Scratch::default(), h264::Scratch::default(), h264::Scratch::default());
        let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        let name = std::path::Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        // the NAL units of the file's H.264 streams: an MP4's video tracks, or a raw byte stream
        // (start codes); a parameter-set reset marks the start of each track
        let mut units: Vec<Option<&[u8]>> = vec![];
        if b.starts_with(&[0, 0, 1]) || b.starts_with(&[0, 0, 0, 1]) {
            units.push(None);
            let mut starts = vec![];
            let mut i = 0;
            while i + 3 <= b.len() {
                if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 { starts.push(i + 3); i += 3; } else { i += 1; }
            }
            for (k, &st) in starts.iter().enumerate() {
                let mut end = starts.get(k + 1).map_or(b.len(), |&n| n - 3);
                while end > st && b[end - 1] == 0 { end -= 1; }
                if end > st { units.push(Some(&b[st..end])); }
            }
        } else {
            let Some(idx) = recode::index(&mut std::io::Cursor::new(&b), b.len() as u64) else { println!("{name}: no index"); continue };
            for cfg in 0..idx.avc.len() {
                let avc = &idx.avc[cfg];
                units.push(None);
                // the configuration's parameter sets come first (copied: they are not in the file's samples)
                for n in avc.sps.iter().chain(&avc.pps) { units.push(Some(Box::leak(n.clone().into_boxed_slice()))); }
                for run in idx.runs.iter().filter(|r| r.what == (recode::mp4::Stream::Avc { cfg: cfg as u16 })) {
                    let mut o = run.start as usize;
                    for &size in &run.sizes {
                        let sample = &b[o..o + size as usize];
                        o += size as usize;
                        let mut p = 0usize;
                        while p + avc.nal_len as usize <= sample.len() {
                            let n = sample[p..p + avc.nal_len as usize].iter().fold(0usize, |a, &x| a << 8 | x as usize);
                            p += avc.nal_len as usize;
                            let Some(nal) = sample.get(p..p + n).filter(|x| !x.is_empty()) else { break };
                            p += n;
                            units.push(Some(nal));
                        }
                    }
                }
            }
        }
        let (mut slices, mut parsed, mut exact, mut skipped) = (0usize, 0usize, 0usize, 0usize);
        let (mut slice_bytes, mut data_bytes, mut mbs, mut bins, mut pcm) = (0u64, 0u64, 0usize, 0usize, 0usize);
        let mut std_bits = [0f64; se::COUNT];
        let mut mvstat = [0u64; 7];
        let mut vlc_bytes = 0u64;
        let mut by_type = [[0f64; 2]; 3];
        let mut model = cm::h264::H264Model::new(!cli.has("lean"));
        model.enable_stats();
        // --drop <family>:<input>,...: blank context inputs (ablation)
        let drops: Vec<(usize, usize)> = cli.get("drop").unwrap_or("").split(',').filter_map(|v| v.split_once(':'))
            .filter_map(|(f, i)| Some((f.parse().ok()?, i.parse().ok()?))).collect();
        model.drop = drops.clone();
        // --reset <KB>: start a new model every so many KB of slice data (what a lane boundary costs)
        let reset = cli.get("reset").and_then(|v| v.parse::<u64>().ok()).map(|kb| kb << 10);
        let (mut since, mut carried) = (0u64, vec![0u64; cm::h264::FAMILIES]);
        let mut why: std::collections::BTreeMap<String, usize> = Default::default();
        let (mut t_parse, mut t_model) = (0f64, 0f64);
        let mut ps = h264::Params::default();
        for unit in &units {
            let Some(nal) = *unit else { ps = h264::Params::default(); continue };
            let (ty, ref_idc) = (nal[0] & 31, (nal[0] >> 5) & 3);
            match ty {
                7 | 8 => { ps.add(nal); }
                1 | 5 => {
                    slices += 1;
                    slice_bytes += nal.len() as u64;
                    let rbsp = h264::unescape(&nal[1..]);
                    let Some((h, s, pp)) = h264::parse_slice_header(&rbsp, ty, ref_idc, &|id| ps.of(id)) else {
                        skipped += 1;
                        *why.entry(format!("unsupported: {}", ps.why_not(&rbsp))).or_default() += 1;
                        continue;
                    };
                    let t0 = std::time::Instant::now();
                    let mut std = h264::StdCost::new(&h);
                    let got = h264::slice_bins_seen(&rbsp, &h, &s, &pp, &mut scratch, &mut |ctx, bin, info| {
                        let cost = std.bin(ctx, bin);
                        std_bits[info.se as usize] += cost;
                        by_type[h.slice_type as usize][if matches!(info.se, se::MVD | se::MVD_SUF | se::MVD_SIGN) { 1 } else { 0 }] += cost;
                        if info.se == se::END && info.a & 15 == h264::class::PCM as u16 { pcm += 1; }
                        // how often a motion vector difference is one of its candidates
                        if h.slice_type == h264::SLICE_P && (info.se == se::MVD_SIGN || (info.se == se::MVD && info.d == 0 && bin == 0)) {
                            let v = if info.se == se::MVD { 0 } else if bin == 1 { -(info.d as i32) } else { info.d as i32 };
                            mvstat[0] += 1;
                            if v != 0 {
                                mvstat[1] += 1;
                                for k in 0..4 { if info.x[k] as i32 == v && info.d < 255 { mvstat[2 + k] += 1; } }
                                if info.x.iter().any(|&c| c as i32 == v) { mvstat[6] += 1; }
                            }
                        }
                    });
                    match got {
                        Err(e) => {
                            if cli.has("verbose") { println!("    slice type {} first_mb {} rbsp {} B data at bit {}: {e}; last bytes {:02x?}", h.slice_type, h.first_mb, rbsp.len(), h.data, &rbsp[rbsp.len().saturating_sub(4)..]); }
                            *why.entry(e).or_default() += 1;
                        }
                        Ok(sb) => {
                            parsed += 1;
                            mbs += sb.macroblocks;
                            bins += sb.count;
                            let orig = &rbsp[h.data / 8..sb.data_end];
                            data_bytes += orig.len() as u64;
                            // the header as a record keeps it: the bits that are slice data zeroed
                            let mut header = rbsp[..h.data.div_ceil(8)].to_vec();
                            if h.data % 8 != 0 { *header.last_mut().unwrap() &= !(0xFF >> (h.data % 8)); }
                            match h264::slice_rbsp(&h, &header, &sb.bins, sb.align, rbsp.len() - sb.data_end, &sb.ref0, &s, &pp, &mut scratch2) {
                                Ok((again, used)) if again == rbsp && used == sb.bins.len() => exact += 1,
                                Ok((again, _)) => { *why.entry(format!("re-encoding differs ({} vs {} bytes)", again.len(), rbsp.len())).or_default() += 1; }
                                Err(e) => { *why.entry(e).or_default() += 1; }
                            }
                            if h264::escape(&rbsp) != nal[1..] { *why.entry("emulation prevention is not canonical".into()).or_default() += 1; }
                            if rbsp[sb.data_end..].iter().any(|&x| x != 0) { *why.entry("bytes after the slice data".into()).or_default() += 1; }
                            if !pp.cabac { vlc_bytes += orig.len() as u64; }
                            t_parse += t0.elapsed().as_secs_f64();
                            since += orig.len() as u64;
                            if reset.is_some_and(|r| since >= r) {
                                for (c, x) in carried.iter_mut().zip(model.stats()) { *c += x; }
                                model = cm::h264::H264Model::new(!cli.has("lean"));
                                model.enable_stats();
                                model.drop = drops.clone();
                                since = 0;
                            }
                            let t1 = std::time::Instant::now();
                            let mut out = crate::bits::BitWriter::new();
                            match model.slice(&mut cm::h264::Count, Some(&sb.bins), &h, &s, &pp, &mut scratch3, &mut out, 0) {
                                Ok(used) if used == sb.bins.len() => {}
                                Ok(_) => { *why.entry("model read a different number of bins".into()).or_default() += 1; }
                                Err(e) => { *why.entry(e).or_default() += 1; }
                            }
                            t_model += t1.elapsed().as_secs_f64();
                        }
                    }
                }
                _ => {}
            }
        }
        let cabac: f64 = std_bits.iter().sum();
        let cost: Vec<u64> = model.stats().iter().zip(&carried).map(|(a, b)| a + b).collect();
        let ours: f64 = cost[..se::COUNT].iter().map(|&c| c as f64 / 65536.0).sum();
        println!("{name:<36} slices {slices:>5} ({slice_bytes} B): parsed {parsed}, exact {exact}, unsupported {skipped}  mbs {mbs} ({pcm} PCM)  bins {bins}  \
                  data {data_bytes} B, CABAC model {:.0} B ({:+.2}%)  parse {:.1} MB/s, model {:.2} MB/s",
            cabac / 8.0, 100.0 * (cabac / 8.0 / data_bytes.max(1) as f64 - 1.0), slice_bytes as f64 / 1e6 / t_parse.max(1e-9), slice_bytes as f64 / 1e6 / t_model.max(1e-9));
        for (e, n) in &why { println!("    {n} x {e}"); }
        if cli.has("verbose") {
            let all: f64 = by_type.iter().flatten().sum();
            println!("    bits by slice type: P {:.1}% (motion {:.1}%), B {:.1}% (motion {:.1}%), I {:.1}%", 100.0 * (by_type[0][0] + by_type[0][1]) / all, 100.0 * by_type[0][1] / all,
                100.0 * (by_type[1][0] + by_type[1][1]) / all, 100.0 * by_type[1][1] / all, 100.0 * by_type[2][0] / all);
        }
        if cli.has("verbose") && mvstat[1] > 0 {
            let pc = |k: usize| 100.0 * mvstat[k] as f64 / mvstat[1] as f64;
            println!("    P-slice motion differences: {} ({:.1}% zero); of the others, equal to candidate A {:.1}%, B {:.1}%, C {:.1}%, last picture {:.1}%; any {:.1}%",
                mvstat[0], 100.0 - 100.0 * mvstat[1] as f64 / mvstat[0] as f64, pc(2), pc(3), pc(4), pc(5), pc(6));
        }
        if vlc_bytes > 0 {
            println!("    model: {:.0} B = {:+.2}% vs the CAVLC slice data ({vlc_bytes} B)", ours / 8.0, 100.0 * (ours / 8.0 / vlc_bytes as f64 - 1.0));
            tot_vlc.0 += vlc_bytes as f64;
            tot_vlc.1 += ours / 8.0;
        } else if cabac > 0.0 {
            println!("    model: {:.0} B = {:+.2}% vs CABAC", ours / 8.0, 100.0 * (ours / cabac - 1.0));
            if cli.has("verbose") {
                for k in 0..se::COUNT {
                    if std_bits[k] > 0.0 {
                        let m = cost[k] as f64 / 65536.0;
                        println!("      {:<18} {:>5.1}% of bits  {:>9.0} -> {:>9.0} B  {:+6.1}%", se::NAMES[k], 100.0 * std_bits[k] / cabac, std_bits[k] / 8.0, m / 8.0, 100.0 * (m / std_bits[k] - 1.0));
                    }
                }
            }
            for k in 0..se::COUNT {
                tot_std[k] += std_bits[k];
                tot_model[k] += cost[k] as f64 / 65536.0;
            }
        }
        tot_slices += slices;
        tot_exact += exact;
        tot_bytes += slice_bytes;
    }
    let (cabac, ours): (f64, f64) = (tot_std.iter().sum(), tot_model.iter().sum());
    println!("TOTAL {tot_slices} slices ({tot_bytes} bytes), {tot_exact} re-encode exactly; model {:+.2}% vs CABAC", 100.0 * (ours / cabac.max(1.0) - 1.0));
    if tot_vlc.0 > 0.0 { println!("CAVLC slices: {:.0} B of slice data -> {:.0} B ({:+.2}%)", tot_vlc.0, tot_vlc.1, 100.0 * (tot_vlc.1 / tot_vlc.0 - 1.0)); }
    for k in 0..se::COUNT {
        if tot_std[k] > 0.0 {
            println!("  {:<18} {:>5.1}% of bits  {:>9.0} -> {:>9.0} B  {:+6.1}%   ({:+.2}% of the total)", se::NAMES[k], 100.0 * tot_std[k] / cabac, tot_std[k] / 8.0,
                tot_model[k] / 8.0, 100.0 * (tot_model[k] / tot_std[k] - 1.0), 100.0 * (tot_model[k] - tot_std[k]) / cabac);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ command: audio-test (developer)
/// Developer harness: code the PCM of audio files with the audio model only; size, speed, roundtrip.
fn cmd_audio_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc audio-test <files...> [--level N] [--old] [--order N]".into()); }
    let order: Option<usize> = cli.get("order").and_then(|v| v.parse().ok());
    let level: u32 = cli.get("level").and_then(|v| v.parse().ok()).unwrap_or(5);
    let (mut tp, mut tc) = (0u64, 0u64);
    for f in &cli.positionals {
        let data = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        let segs = nsc3::plan::segments(&data);
        let mut off = 0usize;
        for sg in &segs {
            let len = sg.len as usize;
            if sg.kind == nsc3::format::kind::AUDIO {
                let pcm = &data[off..off + len];
                let mut m = cm::audio::AudioModel::new();
                let mut enc = cm::coder::Encoder::new();
                let t0 = std::time::Instant::now();
                // the setting a block of `--level` gets (`--old`: for a decoder of level 1), or
                // the first version of the model with a long filter of `--order` taps
                let mut params = sg.params.clone();
                params.push(match order {
                    Some(o) => (o / 16).trailing_zeros() as u8,
                    None => cm::audio::setting(level, !cli.has("old"), pcm, &sg.params),
                });
                m.code_leaf(&mut enc, &mut cm::predictor::ReadBuf(pcm), 0..len, &params)?;
                let te = t0.elapsed().as_secs_f64();
                let coded = enc.finish();
                let mut out = vec![0u8; len];
                let mut m2 = cm::audio::AudioModel::new();
                let mut dec = cm::coder::Decoder::new(&coded);
                let t1 = std::time::Instant::now();
                m2.code_leaf(&mut dec, &mut cm::predictor::WriteBuf(&mut out), 0..len, &params)?;
                let td = t1.elapsed().as_secs_f64();
                println!("{f}: pcm {len} -> {} ({:.2}% of PCM, setting {:#04x})  enc {:.1} MB/s dec {:.1} MB/s  {}", coded.len(), 100.0 * coded.len() as f64 / len as f64, params[params.len() - 1],
                    len as f64 / 1e6 / te, len as f64 / 1e6 / td, if out == pcm { "OK" } else { "MISMATCH" });
                tp += len as u64;
                tc += coded.len() as u64;
            }
            off += len;
        }
    }
    println!("TOTAL pcm {tp} -> {tc} ({:.2}%)", 100.0 * tc as f64 / tp.max(1) as f64);
    Ok(())
}

// ------------------------------------------------------------------ command: gif-test (developer)
/// Developer harness: take each GIF apart, code its frames with the model of indexed pictures
/// (and its record with the byte model), decode and compare; where the bits go.
fn cmd_gif_test(cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc gif-test <files...> [--no-decode]".into()); }
    let names = ["headers", "first", "second", "later", "nearest", "value"];
    let (mut total, mut coded_total, mut secs) = (0usize, 0usize, 0f64);
    let mut all = vec![0u64; names.len()];
    for f in &cli.positionals {
        let data = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        let name = Path::new(f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let Some(sp) = recode::gif::split(&data) else { println!("{name:<36} not a GIF with an image that decodes"); continue };
        // `--dump DIR`: the frames stream of each file, for a look from outside
        if let Some(dir) = cli.get("dump") { std::fs::write(Path::new(dir).join(format!("{name}.frames")), &sp.frames).map_err(|e| format!("{dir}: {e}"))?; }
        let mut m = cm::indexed::IndexedModel::new(sp.frames.len());
        m.enable_stats();
        let mut enc = cm::coder::Encoder::new();
        let t0 = std::time::Instant::now();
        m.code_leaf(&mut enc, &mut cm::predictor::ReadBuf(&sp.frames), 0..sp.frames.len(), &[])?;
        secs += t0.elapsed().as_secs_f64();
        let coded = enc.finish();
        let ok = cli.has("no-decode") || {
            let mut out = vec![0u8; sp.frames.len()];
            cm::indexed::IndexedModel::new(sp.frames.len()).code_leaf(&mut cm::coder::Decoder::new(&coded), &mut cm::predictor::WriteBuf(&mut out), 0..sp.frames.len(), &[])?;
            out == sp.frames
        };
        let record = cm::encode_block(&sp.records, &cm::levels::preset(5, cm::levels::Kind::Generic, sp.records.len()), nsc3::format::kind::GENERIC).len();
        let st = m.stats();
        if cli.has("verbose") {
            for (k, &(n, bits)) in m.detail.iter().enumerate() {
                println!("    first question: around {}, left {}, answer {}: {:>9} pixels {:>8} bytes ({:.3} bits each)", if k & 1 != 0 { "one colour" } else { "mixed     " },
                    if k & 2 != 0 { "transparent" } else { "a colour   " }, if k & 4 != 0 { "yes" } else { "no " }, n, bits >> 19, bits as f64 / 65536.0 / n.max(1) as f64);
            }
        }
        let parts: Vec<String> = names.iter().zip(&st).map(|(n, &b)| format!("{n} {}", b >> 19)).collect();
        println!("{name:<36} {:>8} -> {:>8} frames + {:>6} record = {:>8} ({:.1}% saved)  {}  {}", sp.len, coded.len(), record, coded.len() + record,
            100.0 * (1.0 - (coded.len() + record) as f64 / sp.len as f64), parts.join(", "), if ok { "OK" } else { "MISMATCH" });
        total += sp.len;
        coded_total += coded.len() + record;
        for (a, b) in all.iter_mut().zip(&st) { *a += b; }
    }
    let parts: Vec<String> = names.iter().zip(&all).map(|(n, &b)| format!("{n} {}", b >> 19)).collect();
    println!("TOTAL {total} -> {coded_total} ({:.1}% saved)  {}  {:.1} s", 100.0 * (1.0 - coded_total as f64 / total.max(1) as f64), parts.join(", "), secs);
    Ok(())
}

// ------------------------------------------------------------------ command: tok-test (developer)
/// Parity harness for the native tokenizer: `tok.py enc` each file, decode natively, compare.
fn cmd_tok_test(reg: &Registry, cli: &Cli) -> Result<(), String> {
    if cli.positionals.is_empty() { return Err("usage: nsc tok-test <files...> [--model id]".into()); }
    let model = reg.resolve(cli.get("model"))?;
    let tok = tokenizer::Tokenizer::load(&model.tokenizer())?;
    let mut bad = 0;
    for f in &cli.positionals {
        let tmp = temp_ids();
        run_tok(reg, &model, "enc", Path::new(f), &tmp, true)?;
        let ids = codec::read_ids(&p2s(&tmp))?;
        let _ = std::fs::remove_file(&tmp);
        let orig = std::fs::read(f).map_err(|e| format!("read {f}: {e}"))?;
        let dec_ok = tok.decode(&ids).map(|d| d == orig).unwrap_or(false);
        let t0 = std::time::Instant::now();
        let mine = tok.encode(&orig);
        let enc_s = t0.elapsed().as_secs_f64();
        let enc_ok = mine == ids;
        let first_diff = if enc_ok { String::new() } else {
            let i = mine.iter().zip(&ids).position(|(a, b)| a != b).unwrap_or(mine.len().min(ids.len()));
            format!(" (first difference at token {i} of {}/{})", mine.len(), ids.len())
        };
        if !dec_ok || !enc_ok { bad += 1; }
        println!("{f}: {} bytes, {} tokens, native decode {}, native encode {}{} [{:.1} MB/s]", orig.len(), ids.len(),
            if dec_ok { "EXACT" } else { "MISMATCH" }, if enc_ok { "IDENTICAL" } else { "DIFFERS" }, first_diff, orig.len() as f64 / 1e6 / enc_s);
    }
    if bad > 0 { Err(format!("{bad} file(s) failed")) } else { Ok(()) }
}

// ------------------------------------------------------------------ command: analyze
/// Show how nsc sees a file: the typed segments the detector finds, and (with --exact) what each
/// costs after compression.
fn cmd_analyze(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc analyze <file> [--level N] [--exact] [--all]")?;
    let level: u32 = cli.get("level").and_then(|v| v.parse().ok()).unwrap_or(5);
    let block = cm::levels::block_size(level);
    let input = std::io::BufReader::new(std::fs::File::open(f).map_err(|e| format!("open {f}: {e}"))?);
    let mut planner = nsc3::plan::Planner::new(input, block);
    planner.extend = nsc3::extend_cap(level, block);
    planner.recode = nsc3::block::specialists(level, false);
    // as `compress` does: the tracks of a media container are found through its index
    let mut tracks: Vec<(&str, usize, u64)> = vec![];
    if planner.recode {
        if let Some(idx) = recode::index_path(std::path::Path::new(f)) {
            for r in &idx.runs {
                let codec = match r.what {
                    recode::mp4::Stream::Aac { .. } => "aac",
                    recode::mp4::Stream::Avc { .. } => "h264",
                    recode::mp4::Stream::Mp3 => "mp3",
                    recode::mp4::Stream::Pcm { .. } => "pcm",
                    recode::mp4::Stream::AnnexB { .. } => "h264",
                    recode::mp4::Stream::Adts { .. } => "aac",
                };
                match tracks.iter_mut().find(|t| t.0 == codec) {
                    Some(t) => { t.1 += r.sizes.len(); t.2 += r.len(); }
                    None => tracks.push((codec, r.sizes.len(), r.len())),
                }
            }
            planner.add_index(0, idx);
        }
    }
    planner.log = Some(vec![]);
    let exact = cli.has("exact");
    let mut costs: Vec<(u8, u32, u64)> = vec![];
    while let Some(pb) = planner.next_block()? {
        if exact { costs.extend(nsc3::block::segment_costs(&pb, level)); }
    }
    let log = planner.log.take().unwrap_or_default();
    // merge consecutive pieces of the same segment (split only by block boundaries)
    let mut merged: Vec<(u64, u8, u64, &str)> = vec![];
    for (off, k, len, label) in log {
        match merged.last_mut() {
            Some(m) if m.1 == k && m.3 == label && m.0 + m.2 == off => m.2 += len,
            _ => merged.push((off, k, len, label)),
        }
    }
    let total: u64 = merged.iter().map(|m| m.2).sum();
    if cli.has("json") {
        let segs: Vec<String> = merged.iter().map(|m| format!("{{\"offset\":{},\"length\":{},\"kind\":\"{}\",\"label\":\"{}\"}}",
            m.0, m.2, nsc3::format::kind::name(m.1), m.3)).collect();
        let mut kinds: std::collections::BTreeMap<u8, (u64, u64, u64)> = std::collections::BTreeMap::new();
        for m in &merged { kinds.entry(m.1).or_default().0 += m.2; }
        for (k, raw, cost) in &costs { let e = kinds.entry(*k).or_default(); e.1 += *raw as u64; e.2 += cost; }
        let ks: Vec<String> = kinds.iter().map(|(k, (raw, craw, cost))| format!("{{\"kind\":\"{}\",\"bytes\":{},\"coded_bytes\":{},\"compressed\":{}}}",
            nsc3::format::kind::name(*k), raw, craw, cost)).collect();
        let ts: Vec<String> = tracks.iter().map(|t| format!("{{\"codec\":\"{}\",\"samples\":{},\"bytes\":{}}}", t.0, t.1, t.2)).collect();
        println!("{{\"file\":{:?},\"bytes\":{},\"level\":{},\"exact\":{},\"segments\":[{}],\"kinds\":[{}],\"tracks\":[{}]}}",
            f, total, level, exact, segs.join(","), ks.join(","), ts.join(","));
        return Ok(());
    }
    println!("{f}: {} bytes, {} segment(s)", total, merged.len());
    let show = if cli.has("all") { merged.len() } else { merged.len().min(60) };
    println!("  {:>12}  {:>12}  {:<8} {}", "offset", "length", "kind", "label");
    for m in &merged[..show] {
        println!("  {:>12}  {:>12}  {:<8} {}", m.0, m.2, nsc3::format::kind::name(m.1), m.3);
    }
    if show < merged.len() { println!("  … {} more (use --all)", merged.len() - show); }
    let mut by_kind: std::collections::BTreeMap<u8, (u64, u64, u64)> = std::collections::BTreeMap::new();
    for m in &merged { by_kind.entry(m.1).or_default().0 += m.2; }
    for (k, raw, cost) in &costs { let e = by_kind.entry(*k).or_default(); e.1 += *raw as u64; e.2 += cost; }
    println!("\n  {:<8} {:>12} {:>7}{}", "kind", "bytes", "share", if exact { "   compressed   bits/byte" } else { "" });
    for (k, (raw, craw, cost)) in &by_kind {
        // with --exact: the bytes each kind coded (streams found inside other data count as
        // what they are), not what the detector called them
        if exact && *craw > 0 {
            let share = 100.0 * *craw as f64 / total.max(1) as f64;
            println!("  {:<8} {:>12} {:>6.1}%  {:>11}   {:.3}", nsc3::format::kind::name(*k), craw, share, cost, *cost as f64 * 8.0 / *craw as f64);
        } else if !exact {
            println!("  {:<8} {:>12} {:>6.1}%", nsc3::format::kind::name(*k), raw, 100.0 * *raw as f64 / total.max(1) as f64);
        }
    }
    if exact {
        let c: u64 = costs.iter().map(|x| x.2).sum();
        println!("  {:<8} {:>12} {:>7}  {:>11}   {:.3}", "total", total, "", c, c as f64 * 8.0 / total.max(1) as f64);
    }
    if !tracks.is_empty() {
        println!("\n  tracks listed by the container's index (each is proven when it is compressed)");
        println!("  {:<8} {:>12} {:>7} {:>9}", "codec", "bytes", "share", "samples");
        for t in &tracks { println!("  {:<8} {:>12} {:>6.1}% {:>9}", t.0, t.2, 100.0 * t.2 as f64 / total.max(1) as f64, t.1); }
    }
    Ok(())
}

// ------------------------------------------------------------------ commands: pack / unpack / list
/// Pack files and directories into one archive (solid, deduplicated, datatype-aware).
fn cmd_pack(cli: &Cli) -> Result<(), String> {
    let usage = "usage: nsc pack <archive.nsca> <files|dirs...> [-1..-9] [--no-dedup] [--jobs n] [--mem MiB]";
    let (out, inputs) = cli.positionals.split_first().ok_or(usage)?;
    if inputs.is_empty() { return Err(usage.into()); }
    let o = CompressOpts::from_cli(cli)?;
    let mut opts = nsc3::EncOpts::new(o.level, o.verify, None);
    opts.compat = o.compat;
    opts.lane = o.block.max(cm::levels::block_size(o.level));
    opts.lane_jobs = o.jobs;
    opts.threads = nsc3::block::Threads::new(o.jobs);
    let t0 = std::time::Instant::now();
    let st = archive::pack(out, &archive::inputs_as_given(inputs)?, &opts, o.jobs, o.block, o.mem, !cli.has("no-dedup"))?;
    for w in &st.warnings { eprintln!("note: {w}"); }
    if !o.quiet {
        let secs = t0.elapsed().as_secs_f64();
        println!("pack: {out}  [nsc3 L{}: {}]  {} files ({} distinct), {} dirs, {} links: {} B -> {} B ({:.3} bits/byte, {:.2}x)  {:.1}s",
            o.level, nsc3::kind_names(st.kinds), st.files, st.unique, st.dirs, st.links, st.bytes, st.out_bytes,
            st.out_bytes as f64 * 8.0 / st.bytes.max(1) as f64, st.bytes as f64 / st.out_bytes.max(1) as f64, secs);
    }
    Ok(())
}

/// Unpack an archive (or, with --only, one path or directory of it).
fn cmd_unpack(cli: &Cli) -> Result<(), String> {
    let usage = "usage: nsc unpack <archive.nsca> [dest-dir] [--only path] [--force]";
    let archive_path = cli.positionals.first().ok_or(usage)?;
    let dest = cli.positionals.get(1).map(String::as_str).unwrap_or(".");
    let t0 = std::time::Instant::now();
    let st = archive::unpack(archive_path, dest, cli.get("only"), cli.has("force"), jobs_from_cli(cli), mem_cap_from_cli(cli))?;
    if !cli.quiet() {
        println!("unpack: {archive_path} -> {dest}  {} files, {} dirs, {} links, {} bytes  {:.1}s", st.files, st.dirs, st.links, st.bytes, t0.elapsed().as_secs_f64());
    }
    Ok(())
}

/// List an archive's entries.
fn cmd_list(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc list <archive.nsca>")?;
    let (entries, t) = archive::read_table(f)?;
    for e in &entries {
        let extra = match &e.kind { archive::Kind::Link(t) => format!(" -> {t}"), _ => String::new() };
        println!("{} {:>12}  {}  {}{extra}", archive::mode_string(e), e.size, archive::date(e.mtime), e.path);
    }
    let files = entries.iter().filter(|e| e.kind == archive::Kind::File).count();
    let total: u64 = entries.iter().map(|e| e.size).sum();
    let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
    println!("{} entries, {files} files, {total} bytes ({} distinct), archive {size} bytes ({:.3} bits/byte)",
        entries.len(), t.orig_len, size as f64 * 8.0 / total.max(1) as f64);
    Ok(())
}

// ------------------------------------------------------------------ command: info
fn cmd_info(cli: &Cli) -> Result<(), String> {
    let f = cli.positionals.first().ok_or("usage: nsc info <file.nsc>")?;
    let h = container::read_header(f)?;
    let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
    if h.format == container::Format::Nsc3 {
        let i = nsc3::info(f)?;
        println!("file:        {f}");
        if archive::is_archive(f) {
            let (entries, _) = archive::read_table(f)?;
            let files = entries.iter().filter(|e| e.kind == archive::Kind::File).count();
            let total: u64 = entries.iter().map(|e| e.size).sum();
            println!("archive:     {} entries ({files} files), {total} bytes of files, {} stored once", entries.len(), i.orig_len);
        }
        println!("format:      NSC3 v{} (writer revision {})", nsc3::format::FORMAT, nsc3::format::REV);
        println!("engine:      context mixing ({})", nsc3::kind_names(i.kinds));
        println!("original:    {} bytes", i.orig_len);
        println!("file size:   {} bytes", i.file_len);
        if i.orig_len > 0 { println!("bits/byte:   {:.4}", i.file_len as f64 * 8.0 / i.orig_len as f64); }
        println!("blocks:      {} ({} stored raw)", i.blocks, i.stored_blocks);
        println!("mem/job:     ~{} MB", i.mem_per_job >> 20);
        if cli.has("segments") || cli.has("blocks") {
            for (k, (off, raw, comp, mode, kind)) in i.block_list.iter().enumerate() {
                println!("  block {k:>4}  @{off:<12} {raw:>10} B -> {comp:>10} B  {}  {}",
                    if *mode == nsc3::format::MODE_STORED { "stored" } else { "cm    " }, nsc3::kind_names(nsc3::format::kind::bit(*kind)));
            }
        }
        return Ok(());
    }
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
    // Python is optional: tokenization is native; tok.py is only a fallback for unusual tokenizers.
    match std::process::Command::new(&py).arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status() {
        Ok(s) if s.success() => println!("python:        {py}  ✓ (optional)"),
        _ => println!("python:        {py}  – not found (optional: the tokenizer is native)"),
    }
    println!("tok.py:        {}  {}", tok.display(), if tok.exists() { "✓ (optional fallback)" } else { "– absent (optional)" });
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
nsc {ver} — datatype-aware lossless compressor and archiver

USAGE
  nsc <command> [args] [flags]

COMMANDS
  compress    <input|-> <output|->     Compress any file: datatype models (text, executables, images,
                                       audio, JPEG, MP3, AAC), deflate/PNG/JPEG/MP3/AAC (also inside
                                       MP4/M4A/MOV) recompression, neural text when small enough for
                                       the level (--text); parallel blocks.
                                       Never expands: incompressible input is stored raw.
  decompress  <input|-> <output|->     Decompress; picks the engine/model recorded in the file.
  test        <file.nsc|archive.nsca>  Verify a file or archive decodes (all checksums), writing nothing.
  pack   <archive.nsca> <paths...>     Archive files and folders: one solid, deduplicated,
                                       datatype-aware stream (similar files grouped together).
  unpack <archive.nsca> [dir]          Extract (safely: no absolute/.. paths, never through links;
                                       --force replaces existing files, --only <path> extracts part).
  list   <archive.nsca>                List an archive's files.
  analyze     <file> [--exact]         Show the typed segments nsc detects (images, audio, code,
                                       text, compressed data…) and, with --exact, their cost.
  compress-ids   <in.ids> <out.nsc>    Compress a pre-tokenized .ids file (no tokenizer).
  decompress-ids <in.nsc> <out.ids>    Decompress to a .ids file (no tokenizer).
  batch <compress|decompress> <files...>   Process many files; model loaded once (see --jobs).
  info   <file.nsc>                    Show a compressed file's header (model, tokens, ratio).
  models [list]                        List installed models (* = default).
  models add <id> --model-nsc <p> --o2 <p> --tokenizer <p> [--tier t --bpc x --notes s --default]
  models default <id>                  Set the default model for new compressions.
  models remove <id>                   Remove a model version.
  doctor [--verify]                    Check registry, python/tok.py, and (‑‑verify) fingerprints.
  bench <files|dirs...>                Benchmark nsc levels vs gzip/bzip2/zstd/xz/7z: ratio, MB/s, peak
                                       RSS, roundtrip check. [--classical] [--levels 1,5,9] [--only a,b]
                                       [--report out.json] [--no-verify] [--nsc-jobs N] [--max-mb N]
  selftest                             Developer parity harness (Stages A–E) vs artifacts/.
  version | help

FLAGS
  -1 .. -9         Compression level (default 5): higher = stronger and slower, more memory.
                   -1/-2 cut the input into 2/4 MiB blocks coded on all cores (fastest, a few %
                   larger); from -3 up blocks are large (16-128 MiB) and ratio comes first.
  --text <p>       Text policy: auto (default; neural up to a size set by the level, run only when
                   a quick sample says it can beat the CM engine), neural, cm.
  --block <MiB>    Independent block size (default per level: 2, 4, 16, 16, 32, 32, 64, 64, 128);
                   smaller = more cores used on one file, slightly larger output.
  --mem <MiB>      Memory budget; lowers the worker count, never changes the output.
  --no-verify      Skip the per-block decode check during compression. When memory allows, the
                   check runs alongside the coding, so it costs CPU more than time.
  --compat 1       Write a file that nsc 0.3 can read: without the format recoders (MP3, AAC, …)
                   that levels 3-9 use otherwise.
  --model <id>     Model to compress with (default: registry default).
  --home <path>    Registry home (default: $NSC_HOME or ~/.nsc).
  --jobs <n>       Parallel workers (blocks, or files for `batch`; default: CPU count).
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
        "test" => cmd_test(&open_registry(cli)?, cli),
        "analyze" => cmd_analyze(cli),
        "pack" => cmd_pack(cli),
        "unpack" => cmd_unpack(cli),
        "list" => cmd_list(cli),
        "bench" => cmd_bench(cli),
        "cm-test" => cmd_cm_test(cli),
        "exe-test" => cmd_exe_test(cli),
        "deflate-scan" => cmd_deflate_scan(cli),
        "jpeg-test" => cmd_jpeg_test(cli),
        "sqlite-dump" => cmd_sqlite_dump(cli),
        "mp3-test" => cmd_mp3_test(cli),
        "aac-test" => cmd_aac_test(cli),
        "h264-test" => cmd_h264_test(cli),
        "audio-test" => cmd_audio_test(cli),
        "gif-test" => cmd_gif_test(cli),
        "tok-test" => cmd_tok_test(&open_registry(cli)?, cli),
        "models" => cmd_models(&open_registry(cli)?, cli),
        "doctor" => cmd_doctor(&open_registry(cli)?, cli),
        "selftest" => { selftest::run(); Ok(()) }
        "version" | "--version" => { println!("nsc {}", env!("CARGO_PKG_VERSION")); Ok(()) }
        "help" | "--help" | "-h" => { print_help(); Ok(()) }
        other => Err(format!("unknown command '{other}'. Run `nsc help`.")),
    }
}

unsafe extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}

fn main() {
    // Die quietly on a closed pipe (`nsc info f | head`) like other CLI tools, instead of panicking.
    // SAFETY: resets SIGPIPE to its default disposition before any threads exist.
    unsafe { signal(13, 0); }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, cli) = match parse_args(&argv) {
        Ok(v) => v,
        Err(e) => { eprintln!("error: {e}"); std::process::exit(2); }
    };
    if cli.has("help") && cmd != "help" { print_help(); return; }
    prof::start(); // developer sampling profiler; inert unless NSC_PROFILE is set
    let r = dispatch(&cmd, &cli);
    prof::stop();
    match r {
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
