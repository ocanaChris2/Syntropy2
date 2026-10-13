//! Multi-file archives: `nsc pack`, `unpack`, `list` and `test`.
//!
//! An archive is an NSC3 stream (header flag [`FLAG_ARCHIVE`]) whose content is the bytes of the
//! archived files, each distinct content once, concatenated in an order that groups similar files
//! (by kind of extension, extension, name, then path) so they share model context. The file table
//! (paths, kinds, permissions, modification times, sizes, content offsets, CRC-32s, link targets)
//! is compressed into the trailer. Everything that affects the bytes is sorted, so an archive
//! depends on the files alone, not on directory read order.
//!
//! Unpacking is defensive: paths must be relative and free of `.`/`..` components; nothing is
//! written through a symbolic link (links are created last, and existing links on the way are
//! refused); existing files are not replaced without `--force`; set-id and sticky bits are dropped.
use crate::cm;
use crate::crc::{crc32, Crc32};
use crate::nsc3::{self, format};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub const FLAG_ARCHIVE: u32 = 1;
const TABLE_VERSION: u8 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    File,
    Dir,
    Link(String),
}

#[derive(Clone, Debug)]
pub struct Entry {
    /// Relative path, '/'-separated.
    pub path: String,
    pub kind: Kind,
    pub mode: u32,
    pub mtime: i64,
    pub size: u64,
    /// Offset of the file's bytes in the content stream.
    pub offset: u64,
    pub crc: u32,
}

// ------------------------------------------------------------------ file table

fn put_var(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 { o.push(v as u8 | 0x80); v >>= 7; }
    o.push(v as u8);
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl Rd<'_> {
    fn u8(&mut self) -> Result<u8, String> { let v = *self.b.get(self.o).ok_or("archive table truncated")?; self.o += 1; Ok(v) }
    fn var(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for sh in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 { return Ok(v); }
        }
        Err("archive table: bad number".into())
    }
    fn bytes(&mut self, n: usize) -> Result<&[u8], String> {
        if self.b.len() - self.o < n { return Err("archive table truncated".into()); }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.var()? as usize;
        String::from_utf8(self.bytes(n)?.to_vec()).map_err(|_| "archive table: bad path".to_string())
    }
}

fn table_bytes(entries: &[Entry]) -> Vec<u8> {
    let mut o = vec![TABLE_VERSION];
    put_var(&mut o, entries.len() as u64);
    for e in entries {
        o.push(match e.kind { Kind::File => 0, Kind::Dir => 1, Kind::Link(_) => 2 });
        put_var(&mut o, e.mode as u64);
        put_var(&mut o, ((e.mtime << 1) ^ (e.mtime >> 63)) as u64);
        put_var(&mut o, e.path.len() as u64);
        o.extend_from_slice(e.path.as_bytes());
        match &e.kind {
            Kind::File => {
                put_var(&mut o, e.size);
                put_var(&mut o, e.offset);
                o.extend_from_slice(&e.crc.to_le_bytes());
            }
            Kind::Link(t) => { put_var(&mut o, t.len() as u64); o.extend_from_slice(t.as_bytes()); }
            Kind::Dir => {}
        }
    }
    o
}

fn parse_table(b: &[u8]) -> Result<Vec<Entry>, String> {
    let mut r = Rd { b, o: 0 };
    if r.u8()? != TABLE_VERSION { return Err("archive table: unknown version".into()); }
    let n = r.var()? as usize;
    if n > b.len() { return Err("archive table: bad entry count".into()); }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let k = r.u8()?;
        let mode = r.var()? as u32;
        let z = r.var()?;
        let mtime = (z >> 1) as i64 ^ -((z & 1) as i64);
        let path = r.string()?;
        let (kind, size, offset, crc) = match k {
            0 => {
                let (size, offset) = (r.var()?, r.var()?);
                let crc = u32::from_le_bytes(r.bytes(4)?.try_into().unwrap());
                (Kind::File, size, offset, crc)
            }
            1 => (Kind::Dir, 0, 0, 0),
            2 => (Kind::Link(r.string()?), 0, 0, 0),
            _ => return Err("archive table: bad entry kind".into()),
        };
        out.push(Entry { path, kind, mode, mtime, size, offset, crc });
    }
    if r.o != b.len() { return Err("archive table: trailing bytes".into()); }
    Ok(out)
}

/// Compressed table: raw length u32, CRC-32 of the raw table, model config, coded bytes.
fn compress_table(raw: &[u8]) -> Vec<u8> {
    let cfg = cm::levels::preset(5, cm::levels::Kind::Generic, raw.len().max(1));
    let mut o = (raw.len() as u32).to_le_bytes().to_vec();
    o.extend_from_slice(&crc32(raw).to_le_bytes());
    o.extend_from_slice(&cfg.to_bytes());
    o.extend_from_slice(&cm::encode_block(raw, &cfg, nsc3::format::kind::TEXT));
    o
}

fn decompress_table(t: &[u8]) -> Result<Vec<u8>, String> {
    let hdr = 8 + cm::levels::CFG_BYTES;
    if t.len() < hdr { return Err("not an nsc archive (no file table)".into()); }
    let len = u32::from_le_bytes(t[..4].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(t[4..8].try_into().unwrap());
    if len > 1 << 30 { return Err("archive table too large".into()); }
    let cfg = cm::levels::ModelCfg::from_bytes(&t[8..hdr])?;
    let raw = cm::decode_block(&t[hdr..], len, &cfg, nsc3::format::kind::TEXT);
    if crc32(&raw) != crc { return Err("archive table checksum mismatch (corrupt archive)".into()); }
    Ok(raw)
}

/// The entries of an archive (header and trailer only; no content is decoded).
pub fn read_table(archive: &str) -> Result<(Vec<Entry>, format::Trailer), String> {
    let (h, t) = nsc3::read_ends(archive)?;
    if h.flags & FLAG_ARCHIVE == 0 { return Err(format!("{archive} is a compressed file, not an archive (use `nsc decompress`)")); }
    let entries = parse_table(&decompress_table(&t.file_table)?)?;
    Ok((entries, t))
}

pub fn is_archive(path: &str) -> bool {
    File::open(path).ok().and_then(|mut f| nsc3::sched::read_header(&mut f).ok()).is_some_and(|(h, _)| h.flags & FLAG_ARCHIVE != 0)
}

// ------------------------------------------------------------------ packing

/// Archive path for an input argument: '/'-separated, without leading '/', '.' components or a
/// trailing '/'. `..` is refused.
fn archive_path(input: &str) -> Result<String, String> {
    let mut parts = vec![];
    for c in input.split('/') {
        match c {
            "" | "." => {}
            ".." => return Err(format!("{input}: paths with '..' cannot be archived (run nsc from a parent directory)")),
            c => parts.push(c),
        }
    }
    Ok(parts.join("/"))
}

struct Item {
    entry: Entry,
    src: Option<PathBuf>,
}

fn walk(fs_path: &Path, arc: &str, out: &mut Vec<Item>, warn: &mut Vec<String>) -> Result<(), String> {
    let m = fs::symlink_metadata(fs_path).map_err(|e| format!("{}: {e}", fs_path.display()))?;
    let mode = m.mode() & 0o7777;
    let mtime = m.mtime();
    let ft = m.file_type();
    if ft.is_symlink() {
        let t = fs::read_link(fs_path).map_err(|e| format!("{}: {e}", fs_path.display()))?;
        let t = t.to_str().ok_or_else(|| format!("{}: link target is not UTF-8", fs_path.display()))?.to_string();
        if !arc.is_empty() { out.push(Item { entry: Entry { path: arc.into(), kind: Kind::Link(t), mode, mtime, size: 0, offset: 0, crc: 0 }, src: None }); }
    } else if ft.is_dir() {
        if !arc.is_empty() { out.push(Item { entry: Entry { path: arc.into(), kind: Kind::Dir, mode, mtime, size: 0, offset: 0, crc: 0 }, src: None }); }
        let mut names = vec![];
        for d in fs::read_dir(fs_path).map_err(|e| format!("{}: {e}", fs_path.display()))? {
            let d = d.map_err(|e| format!("{}: {e}", fs_path.display()))?;
            let name = d.file_name().into_string().map_err(|n| format!("{}: name is not UTF-8: {n:?}", fs_path.display()))?;
            names.push(name);
        }
        names.sort();
        for name in names {
            let child = if arc.is_empty() { name.clone() } else { format!("{arc}/{name}") };
            walk(&fs_path.join(&name), &child, out, warn)?;
        }
    } else if ft.is_file() {
        let path = if arc.is_empty() { fs_path.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string() } else { arc.into() };
        out.push(Item { entry: Entry { path, kind: Kind::File, mode, mtime, size: m.len(), offset: 0, crc: 0 }, src: Some(fs_path.to_path_buf()) });
    } else {
        warn.push(format!("{}: special file skipped", fs_path.display()));
    }
    Ok(())
}

/// CRC-32 and length of a file's content.
fn hash_file(p: &Path) -> Result<(u32, u64), String> {
    let mut f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut c = Crc32::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf).map_err(|e| format!("{}: {e}", p.display()))?;
        if k == 0 { break; }
        c.update(&buf[..k]);
        n += k as u64;
    }
    Ok((c.finish(), n))
}

fn same_content(a: &Path, b: &Path) -> Result<bool, String> {
    let (mut fa, mut fb) = (File::open(a).map_err(|e| e.to_string())?, File::open(b).map_err(|e| e.to_string())?);
    let (mut ba, mut bb) = (vec![0u8; 1 << 16], vec![0u8; 1 << 16]);
    loop {
        let na = crate::bench::read_full(&mut fa, &mut ba).map_err(|e| e.to_string())?;
        let nb = crate::bench::read_full(&mut fb, &mut bb).map_err(|e| e.to_string())?;
        if na != nb || ba[..na] != bb[..nb] { return Ok(false); }
        if na == 0 { return Ok(true); }
    }
}

/// Content order: similar files next to each other.
fn order_key(path: &str) -> (u8, String, String, String) {
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_string()).unwrap_or_default();
    let group = match ext.as_str() {
        "txt" | "md" | "rst" | "c" | "h" | "cc" | "cpp" | "hpp" | "rs" | "py" | "js" | "ts" | "java" | "go" | "rb" | "sh" | "pl"
        | "php" | "html" | "htm" | "css" | "xml" | "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "csv" | "tsv" | "log"
        | "tex" | "svg" | "sql" => 0,
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp" | "ico" | "ppm" | "pgm" | "pnm" | "tga" => 2,
        "wav" | "aif" | "aiff" | "au" | "flac" | "mp3" | "ogg" | "opus" | "m4a" | "mp4" | "mkv" | "avi" | "mov" | "webm" => 3,
        "zip" | "gz" | "tgz" | "bz2" | "xz" | "7z" | "zst" | "jar" | "docx" | "xlsx" | "pptx" | "odt" | "ods" | "apk" | "pdf" => 4,
        _ => 1,
    };
    (group, ext, name, path.to_string())
}

/// Reads the unique files back to back, checking that none changed since it was hashed.
struct Chain {
    files: Vec<(PathBuf, u64, u32)>,
    i: usize,
    cur: Option<File>,
    got: u64,
    crc: Crc32,
}

impl Read for Chain {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if self.cur.is_none() {
                if self.i == self.files.len() { return Ok(0); }
                self.cur = Some(File::open(&self.files[self.i].0)?);
                self.got = 0;
                self.crc = Crc32::new();
            }
            let (_, size, crc) = &self.files[self.i];
            let room = (*size - self.got).min(buf.len() as u64) as usize;
            let n = if room == 0 { 0 } else { self.cur.as_mut().unwrap().read(&mut buf[..room])? };
            if n == 0 {
                let mut probe = [0u8; 1];
                let extra = self.cur.as_mut().unwrap().read(&mut probe)?;
                let c = std::mem::replace(&mut self.crc, Crc32::new()).finish();
                if self.got != *size || extra != 0 || c != *crc {
                    return Err(std::io::Error::other(format!("{} changed while it was being archived", self.files[self.i].0.display())));
                }
                self.cur = None;
                self.i += 1;
                continue;
            }
            self.crc.update(&buf[..n]);
            self.got += n as u64;
            return Ok(n);
        }
    }
}

pub struct PackStats {
    pub files: usize,
    pub dirs: usize,
    pub links: usize,
    pub unique: usize,
    pub bytes: u64,
    pub content: u64,
    pub out_bytes: u64,
    pub kinds: u128,
    pub warnings: Vec<String>,
}

/// Archive path for each command-line input (as given, normalized).
pub fn inputs_as_given(inputs: &[String]) -> Result<Vec<(PathBuf, String)>, String> {
    inputs.iter().map(|i| Ok((PathBuf::from(i), archive_path(i)?))).collect()
}

/// Pack `inputs` — (filesystem path, path inside the archive) of files and directories — into
/// the archive `out`.
#[allow(clippy::too_many_arguments)]
pub fn pack(out: &str, inputs: &[(PathBuf, String)], opts: &nsc3::EncOpts, jobs: usize, block: usize, mem: Option<u64>, dedup: bool) -> Result<PackStats, String> {
    let mut items = vec![];
    let mut warnings = vec![];
    for (fs_path, arc) in inputs {
        walk(fs_path, arc, &mut items, &mut warnings)?;
    }
    items.sort_by(|a, b| a.entry.path.cmp(&b.entry.path));
    items.dedup_by(|a, b| a.entry.path == b.entry.path);
    // content hashes, then whole-file dedup (equal size and CRC, confirmed byte by byte)
    for it in items.iter_mut() {
        if let Some(src) = &it.src {
            let (crc, size) = hash_file(src)?;
            it.entry.crc = crc;
            it.entry.size = size;
        }
    }
    let mut by_hash: BTreeMap<(u64, u32), Vec<usize>> = BTreeMap::new();
    let mut original: Vec<Option<usize>> = vec![None; items.len()];
    let mut unique = vec![];
    for i in 0..items.len() {
        if items[i].entry.kind != Kind::File { continue; }
        let key = (items[i].entry.size, items[i].entry.crc);
        let mut dup = None;
        if dedup && items[i].entry.size > 0 {
            for &j in by_hash.get(&key).map(|v| v.as_slice()).unwrap_or(&[]) {
                if same_content(items[j].src.as_ref().unwrap(), items[i].src.as_ref().unwrap())? { dup = Some(j); break; }
            }
        }
        match dup {
            Some(j) => original[i] = Some(j),
            None => { by_hash.entry(key).or_default().push(i); unique.push(i); }
        }
    }
    unique.sort_by_key(|&i| order_key(&items[i].entry.path));
    let mut off = 0u64;
    for &i in &unique {
        items[i].entry.offset = off;
        off += items[i].entry.size;
    }
    for i in 0..items.len() {
        if let Some(j) = original[i] { items[i].entry.offset = items[j].entry.offset; }
    }
    let entries: Vec<Entry> = items.iter().map(|it| it.entry.clone()).collect();
    let table = compress_table(&table_bytes(&entries));
    let chain = Chain {
        files: unique.iter().map(|&i| (items[i].src.clone().unwrap(), items[i].entry.size, items[i].entry.crc)).collect(),
        i: 0, cur: None, got: 0, crc: Crc32::new(),
    };
    // the indexes of the container files (MP4…) in the stream, for the format recoders
    let indexes = if opts.specialists() {
        unique.iter().filter_map(|&i| crate::recode::index_path(items[i].src.as_ref()?).map(|x| (items[i].entry.offset, x))).collect()
    } else {
        vec![]
    };
    let meta = nsc3::sched::Meta { flags: FLAG_ARCHIVE, file_table: table, indexes };
    let (st, warn) = nsc3::compress_reader(chain, Some(off), out, opts, jobs, block, mem, meta)?;
    if let Some(w) = warn { warnings.push(w); }
    let count = |k: fn(&Kind) -> bool| entries.iter().filter(|e| k(&e.kind)).count();
    Ok(PackStats {
        files: count(|k| *k == Kind::File), dirs: count(|k| *k == Kind::Dir), links: count(|k| matches!(k, Kind::Link(_))),
        unique: unique.len(), bytes: entries.iter().map(|e| e.size).sum(), content: off,
        out_bytes: st.out_bytes, kinds: st.kinds, warnings,
    })
}

// ------------------------------------------------------------------ unpacking

/// A relative path with no empty, `.` or `..` components and no NUL.
fn safe_path(p: &str) -> Result<(), String> {
    if p.is_empty() || p.starts_with('/') || p.contains('\0') || p.split('/').any(|c| c.is_empty() || c == "." || c == "..") {
        return Err(format!("unsafe path in archive: {p:?}"));
    }
    Ok(())
}

/// Create the directories of `rel` under `root`, refusing anything on the way that is not a real
/// directory (a symbolic link, a file).
fn ensure_dirs(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let mut cur = root.to_path_buf();
    if rel.is_empty() { return Ok(cur); }
    for c in rel.split('/') {
        cur.push(c);
        match fs::symlink_metadata(&cur) {
            Ok(m) if m.file_type().is_dir() => {}
            Ok(_) => return Err(format!("{}: exists and is not a directory", cur.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&cur).map_err(|e| format!("{}: {e}", cur.display()))?;
            }
            Err(e) => return Err(format!("{}: {e}", cur.display())),
        }
    }
    Ok(cur)
}

fn parent_of(p: &str) -> &str { p.rsplit_once('/').map(|(d, _)| d).unwrap_or("") }

/// Make `rel` free to create: its parents are real directories; an existing file there is an
/// error unless `force` (then it is removed, never followed).
fn prepare_target(root: &Path, rel: &str, force: bool) -> Result<PathBuf, String> {
    ensure_dirs(root, parent_of(rel))?;
    let p = root.join(rel);
    if let Ok(m) = fs::symlink_metadata(&p) {
        if !force { return Err(format!("{}: already exists (use --force to replace)", p.display())); }
        if m.file_type().is_dir() { return Err(format!("{}: is a directory", p.display())); }
        fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(p)
}

fn system_time(t: i64) -> std::time::SystemTime {
    use std::time::{Duration, UNIX_EPOCH};
    if t >= 0 { UNIX_EPOCH + Duration::from_secs(t as u64) } else { UNIX_EPOCH - Duration::from_secs(t.unsigned_abs()) }
}

struct Target {
    path: PathBuf,
    mode: u32,
    mtime: i64,
}

/// One distinct content and the files it becomes.
struct Job {
    off: u64,
    size: u64,
    crc: u32,
    targets: Vec<Target>,
}

/// Receives the content stream (in order, with gaps allowed) and writes or checks the files.
struct Sink {
    jobs: Vec<Job>,
    j: usize,
    done: u64,
    open: Vec<File>,
    crc: Crc32,
    check_only: bool,
    files_done: usize,
}

impl Sink {
    fn feed(&mut self, mut off: u64, mut data: &[u8]) -> Result<(), String> {
        while self.j < self.jobs.len() && !data.is_empty() {
            let job = &self.jobs[self.j];
            let at = job.off + self.done;
            if off + data.len() as u64 <= at { return Ok(()); }
            if off > at { return Err("archive content missing (corrupt archive)".into()); }
            let skip = (at - off) as usize;
            data = &data[skip..];
            off = at;
            if self.done == 0 && !self.check_only {
                self.open.clear();
                for t in &job.targets {
                    self.open.push(OpenOptions::new().write(true).create_new(true).open(&t.path).map_err(|e| format!("{}: {e}", t.path.display()))?);
                }
            }
            let take = ((job.size - self.done) as usize).min(data.len());
            let chunk = &data[..take];
            self.crc.update(chunk);
            for f in &mut self.open { f.write_all(chunk).map_err(|e| format!("write: {e}"))?; }
            self.done += take as u64;
            data = &data[take..];
            off += take as u64;
            if self.done == job.size { self.finish_job()?; }
        }
        Ok(())
    }

    fn finish_job(&mut self) -> Result<(), String> {
        let job = &self.jobs[self.j];
        let c = std::mem::replace(&mut self.crc, Crc32::new()).finish();
        if c != job.crc { return Err(format!("{}: checksum mismatch (corrupt archive)", job.targets.first().map(|t| t.path.display().to_string()).unwrap_or_default())); }
        if !self.check_only {
            for (f, t) in self.open.drain(..).zip(&job.targets) {
                let _ = f.set_modified(system_time(t.mtime));
                drop(f);
                fs::set_permissions(&t.path, fs::Permissions::from_mode(t.mode & 0o777)).map_err(|e| format!("{}: {e}", t.path.display()))?;
            }
        }
        self.files_done += job.targets.len().max(1);
        self.j += 1;
        self.done = 0;
        Ok(())
    }
}

/// Adapts the sink to the sequential decompressor's `Write`.
struct SinkWriter<'a> {
    sink: &'a mut Sink,
    off: u64,
}

impl Write for SinkWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.sink.feed(self.off, buf).map_err(std::io::Error::other)?;
        self.off += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub struct UnpackStats {
    pub files: usize,
    pub dirs: usize,
    pub links: usize,
    pub bytes: u64,
}

/// Decode the content ranges the sink needs: everything sequentially (verifying the whole-stream
/// checksum) when most of it is needed, else only the blocks that overlap the needed ranges.
fn decode_into(archive: &str, t: &format::Trailer, sink: &mut Sink, jobs: usize, mem: Option<u64>) -> Result<(), String> {
    let need: u64 = sink.jobs.iter().map(|j| j.size).sum();
    let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<nsc3::NeuralCtx>>, String> {
        if b.is_empty() { Ok(None) } else { Err("archives with neural segments are not supported by this build".into()) }
    };
    if need * 2 >= t.orig_len {
        let reader = std::io::BufReader::with_capacity(1 << 20, File::open(archive).map_err(|e| format!("open {archive}: {e}"))?);
        let mut w = SinkWriter { sink, off: 0 };
        nsc3::sched::decompress(reader, &mut w, jobs, nsc3::mem_budget(mem), &no_model)?;
        return Ok(());
    }
    // blocks overlapping a needed range, by the trailer index
    let n = t.index.len();
    let raw_end = |i: usize| if i + 1 < n { t.index[i + 1].1 } else { t.orig_len };
    let ranges: Vec<(u64, u64)> = sink.jobs.iter().map(|j| (j.off, j.off + j.size)).collect();
    let wanted: Vec<usize> = (0..n).filter(|&i| ranges.iter().any(|&(a, b)| a < raw_end(i) && b > t.index[i].1)).collect();
    let mut f = File::open(archive).map_err(|e| format!("open {archive}: {e}"))?;
    for chunk in wanted.chunks(jobs.max(1)) {
        let mut units = vec![];
        for &i in chunk {
            f.seek(SeekFrom::Start(t.index[i].0)).map_err(|e| e.to_string())?;
            match nsc3::sched::read_unit(&mut std::io::BufReader::new(&mut f))? {
                nsc3::sched::Unit::Block(bh, payload) => units.push((i, bh, payload)),
                _ => return Err("archive index points at no block (corrupt archive)".into()),
            }
        }
        let decoded: Vec<Result<Vec<u8>, String>> = std::thread::scope(|s| {
            let hs: Vec<_> = units.iter().map(|(_, bh, p)| s.spawn(move || nsc3::block::decode(bh, p, None, 1))).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("decoder panicked".into()))).collect()
        });
        for ((i, _, _), d) in units.iter().zip(decoded) {
            let d = d?;
            if d.len() as u64 != raw_end(*i) - t.index[*i].1 { return Err("block length mismatch (corrupt archive)".into()); }
            sink.feed(t.index[*i].1, &d)?;
        }
    }
    Ok(())
}

/// Entries selected by `only` (a path or a directory prefix), or all.
fn selected(e: &Entry, only: Option<&str>) -> bool {
    match only {
        None => true,
        Some(p) => { let p = p.trim_end_matches('/'); e.path == p || e.path.starts_with(&format!("{p}/")) }
    }
}

/// Unpack `archive` into `dest` (created if needed); with `only`, just that path or directory.
pub fn unpack(archive: &str, dest: &str, only: Option<&str>, force: bool, jobs: usize, mem: Option<u64>) -> Result<UnpackStats, String> {
    let (entries, t) = read_table(archive)?;
    let mut seen = std::collections::BTreeSet::new();
    for e in &entries {
        safe_path(&e.path)?;
        if !seen.insert(e.path.as_str()) { return Err(format!("duplicate path in archive: {}", e.path)); }
        if e.kind == Kind::File && e.offset.checked_add(e.size).is_none_or(|end| end > t.orig_len) { return Err(format!("{}: content out of range (corrupt archive)", e.path)); }
    }
    let sel: Vec<&Entry> = entries.iter().filter(|e| selected(e, only)).collect();
    if sel.is_empty() { return Err(format!("no entries match {:?}", only.unwrap_or(""))); }
    let root = PathBuf::from(dest);
    fs::create_dir_all(&root).map_err(|e| format!("{dest}: {e}"))?;
    let mut st = UnpackStats { files: 0, dirs: 0, links: 0, bytes: 0 };
    // directories (writable while extracting; final modes at the end)
    for e in sel.iter().filter(|e| e.kind == Kind::Dir) {
        ensure_dirs(&root, &e.path)?;
        st.dirs += 1;
    }
    // files: one job per distinct content; empty files directly
    let mut jobs_by_off: BTreeMap<(u64, u64, u32), Vec<Target>> = BTreeMap::new();
    for e in sel.iter().filter(|e| e.kind == Kind::File) {
        let p = prepare_target(&root, &e.path, force)?;
        st.files += 1;
        st.bytes += e.size;
        if e.size == 0 {
            if crc32(&[]) != e.crc { return Err(format!("{}: checksum mismatch (corrupt archive)", e.path)); }
            let f = OpenOptions::new().write(true).create_new(true).open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let _ = f.set_modified(system_time(e.mtime));
            fs::set_permissions(&p, fs::Permissions::from_mode(e.mode & 0o777)).map_err(|e| format!("{}: {e}", p.display()))?;
            continue;
        }
        jobs_by_off.entry((e.offset, e.size, e.crc)).or_default().push(Target { path: p, mode: e.mode, mtime: e.mtime });
    }
    let mut sink = Sink {
        jobs: jobs_by_off.into_iter().map(|((off, size, crc), targets)| Job { off, size, crc, targets }).collect(),
        j: 0, done: 0, open: vec![], crc: Crc32::new(), check_only: false, files_done: 0,
    };
    // overlapping distinct contents would mean a corrupt table
    for w in sink.jobs.windows(2) { if w[0].off + w[0].size > w[1].off { return Err("overlapping contents in archive table (corrupt archive)".into()); } }
    if !sink.jobs.is_empty() { decode_into(archive, &t, &mut sink, jobs, mem)?; }
    if sink.j != sink.jobs.len() { return Err("archive content ended early (corrupt archive)".into()); }
    // links last, so nothing is ever written through one
    for e in sel.iter() {
        if let Kind::Link(target) = &e.kind {
            let p = prepare_target(&root, &e.path, force)?;
            std::os::unix::fs::symlink(target, &p).map_err(|e| format!("{}: {e}", p.display()))?;
            st.links += 1;
        }
    }
    // directory modes and times, deepest first
    let mut dirs: Vec<&&Entry> = sel.iter().filter(|e| e.kind == Kind::Dir).collect();
    dirs.sort_by_key(|e| std::cmp::Reverse(e.path.matches('/').count()));
    for e in dirs {
        let p = root.join(&e.path);
        if let Ok(f) = File::open(&p) { let _ = f.set_modified(system_time(e.mtime)); }
        fs::set_permissions(&p, fs::Permissions::from_mode(e.mode & 0o777)).map_err(|err| format!("{}: {err}", p.display()))?;
    }
    Ok(st)
}

/// Decode everything and check every file's checksum, writing nothing.
pub fn test(archive: &str, jobs: usize, mem: Option<u64>) -> Result<(usize, u64), String> {
    let (entries, t) = read_table(archive)?;
    let mut jobs_by_off: BTreeMap<(u64, u64, u32), usize> = BTreeMap::new();
    let mut n = 0;
    for e in entries.iter().filter(|e| e.kind == Kind::File) {
        safe_path(&e.path)?;
        n += 1;
        if e.size > 0 { *jobs_by_off.entry((e.offset, e.size, e.crc)).or_default() += 1; }
    }
    let mut sink = Sink {
        jobs: jobs_by_off.into_keys().map(|(off, size, crc)| Job { off, size, crc, targets: vec![] }).collect(),
        j: 0, done: 0, open: vec![], crc: Crc32::new(), check_only: true, files_done: 0,
    };
    for w in sink.jobs.windows(2) { if w[0].off + w[0].size > w[1].off { return Err("overlapping contents in archive table (corrupt archive)".into()); } }
    let reader = std::io::BufReader::with_capacity(1 << 20, File::open(archive).map_err(|e| format!("open {archive}: {e}"))?);
    let no_model = |b: &[(String, u64)]| -> Result<Option<std::sync::Arc<nsc3::NeuralCtx>>, String> {
        if b.is_empty() { Ok(None) } else { Err("archives with neural segments are not supported by this build".into()) }
    };
    nsc3::sched::decompress(reader, &mut SinkWriter { sink: &mut sink, off: 0 }, jobs, nsc3::mem_budget(mem), &no_model)?;
    if sink.j != sink.jobs.len() { return Err("archive content ended early (corrupt archive)".into()); }
    Ok((n, t.orig_len))
}

// ------------------------------------------------------------------ listing

/// "YYYY-MM-DD HH:MM" (UTC) for a Unix time.
pub fn date(t: i64) -> String {
    let (days, secs) = (t.div_euclid(86400), t.rem_euclid(86400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", secs / 3600, secs % 3600 / 60)
}

pub fn mode_string(e: &Entry) -> String {
    let t = match e.kind { Kind::File => '-', Kind::Dir => 'd', Kind::Link(_) => 'l' };
    let mut s = String::from(t);
    for sh in [6, 3, 0] {
        let m = e.mode >> sh;
        s.push(if m & 4 != 0 { 'r' } else { '-' });
        s.push(if m & 2 != 0 { 'w' } else { '-' });
        s.push(if m & 1 != 0 { 'x' } else { '-' });
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nsc-archive-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn table_roundtrip_and_dates() {
        let e = vec![
            Entry { path: "a/b.txt".into(), kind: Kind::File, mode: 0o644, mtime: 1_700_000_000, size: 12, offset: 3, crc: 7 },
            Entry { path: "a".into(), kind: Kind::Dir, mode: 0o755, mtime: -5, size: 0, offset: 0, crc: 0 },
            Entry { path: "l".into(), kind: Kind::Link("a/b.txt".into()), mode: 0o777, mtime: 0, size: 0, offset: 0, crc: 0 },
        ];
        let t = parse_table(&decompress_table(&compress_table(&table_bytes(&e))).unwrap()).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!((t[0].path.as_str(), t[0].size, t[0].offset, t[0].crc, t[1].mtime), ("a/b.txt", 12, 3, 7, -5));
        assert_eq!(t[2].kind, Kind::Link("a/b.txt".into()));
        assert_eq!(date(0), "1970-01-01 00:00");
        assert_eq!(date(1_700_000_000), "2023-11-14 22:13");
        for bad in ["/etc/passwd", "a/../b", "./a", "a//b", ""] { assert!(safe_path(bad).is_err(), "{bad} accepted"); }
    }

    #[test]
    fn pack_unpack_roundtrip_with_dedup_links_and_safety() {
        let src = tmpdir("src");
        let tree = src.join("tree");
        fs::create_dir_all(tree.join("sub/deeper")).unwrap();
        fs::write(tree.join("a.txt"), b"hello archive ".repeat(500)).unwrap();
        fs::write(tree.join("sub/copy.txt"), b"hello archive ".repeat(500)).unwrap(); // duplicate content
        fs::write(tree.join("sub/deeper/data.bin"), (0..20_000u32).map(|i| (i * 7 % 251) as u8).collect::<Vec<_>>()).unwrap();
        fs::write(tree.join("empty"), b"").unwrap();
        fs::set_permissions(tree.join("sub/deeper/data.bin"), fs::Permissions::from_mode(0o751)).unwrap();
        std::os::unix::fs::symlink("a.txt", tree.join("link")).unwrap();
        let arc = src.join("t.nsca");
        let arc_s = arc.to_str().unwrap().to_string();
        let opts = nsc3::EncOpts::new(2, true, None);
        let st = pack(&arc_s, &[(tree.clone(), "tree".to_string())], &opts, 2, 1 << 20, None, true).unwrap();
        assert_eq!((st.files, st.unique, st.links), (4, 3, 1), "dedup or walk wrong");
        assert!(is_archive(&arc_s));
        let out = src.join("out");
        unpack(&arc_s, out.to_str().unwrap(), None, false, 2, None).unwrap();
        for f in ["a.txt", "sub/copy.txt", "sub/deeper/data.bin", "empty"] {
            assert_eq!(fs::read(out.join("tree").join(f)).unwrap(), fs::read(tree.join(f)).unwrap(), "{f}");
        }
        assert_eq!(fs::read_link(out.join("tree/link")).unwrap(), PathBuf::from("a.txt"));
        assert_eq!(fs::metadata(out.join("tree/sub/deeper/data.bin")).unwrap().mode() & 0o777, 0o751);
        // no overwrite without --force; --force works; partial extraction
        assert!(unpack(&arc_s, out.to_str().unwrap(), None, false, 2, None).is_err());
        unpack(&arc_s, out.to_str().unwrap(), None, true, 2, None).unwrap();
        let part = src.join("part");
        unpack(&arc_s, part.to_str().unwrap(), Some("tree/sub"), false, 2, None).unwrap();
        assert!(part.join("tree/sub/copy.txt").exists() && !part.join("tree/a.txt").exists());
        assert_eq!(test(&arc_s, 2, None).unwrap().0, 4);
        // a destination path through a planted symlink is refused
        let evil = src.join("evil");
        fs::create_dir_all(&evil).unwrap();
        std::os::unix::fs::symlink("/tmp", evil.join("tree")).unwrap();
        assert!(unpack(&arc_s, evil.to_str().unwrap(), None, true, 2, None).is_err(), "wrote through a symlink");
        let _ = fs::remove_dir_all(&src);
    }

    /// A crafted archive whose table names `../escape` (or an absolute path) is refused before
    /// anything is written.
    #[test]
    fn hostile_paths_are_refused() {
        let dir = tmpdir("hostile");
        for bad in ["../escape", "/abs/path", "ok/../../x"] {
            let content = b"payload".to_vec();
            let e = vec![Entry { path: bad.into(), kind: Kind::File, mode: 0o644, mtime: 0, size: 7, offset: 0, crc: crc32(&content) }];
            let meta = nsc3::sched::Meta { flags: FLAG_ARCHIVE, file_table: compress_table(&table_bytes(&e)), ..Default::default() };
            let arc = dir.join("h.nsca");
            let mut out = vec![];
            nsc3::sched::compress_meta(&content[..], &mut out, &nsc3::EncOpts::new(1, true, None), 1, 1 << 20, u64::MAX, meta).unwrap();
            fs::write(&arc, &out).unwrap();
            let dest = dir.join("dest");
            let r = unpack(arc.to_str().unwrap(), dest.to_str().unwrap(), None, true, 1, None);
            assert!(r.is_err() && r.err().unwrap().contains("unsafe path"), "{bad} not refused");
            assert!(!dir.join("escape").exists());
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
