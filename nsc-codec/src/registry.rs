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
