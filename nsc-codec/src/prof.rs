//! Developer-only sampling profiler: SIGPROF every millisecond of CPU time records the interrupted
//! instruction pointer from the signal context. Enabled by `NSC_PROFILE=<out file>` (Linux x86_64).
//! Symbolise the written offsets with `bench/prof.py` (addr2line on a debug-info build).
#![allow(dead_code)]
use std::sync::atomic::{AtomicUsize, Ordering};

const N: usize = 1 << 20;
static mut SAMPLES: [u64; N] = [0; N];
static COUNT: AtomicUsize = AtomicUsize::new(0);

#[repr(C)]
struct Timeval { sec: i64, usec: i64 }
#[repr(C)]
struct ITimerVal { interval: Timeval, value: Timeval }
#[repr(C)]
struct SigAction { sa_sigaction: usize, sa_mask: [u64; 16], sa_flags: i32, sa_restorer: usize }

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
    fn setitimer(which: i32, new: *const ITimerVal, old: *mut ITimerVal) -> i32;
}

const SIGPROF: i32 = 27;
const SA_SIGINFO: i32 = 4;
const SA_RESTART: i32 = 0x1000_0000;
const ITIMER_PROF: i32 = 2;
/// Byte offset of RIP in x86_64 `ucontext_t`: flags 8 + link 8 + stack_t 24 + 16 gregs × 8.
const RIP_OFFSET: usize = 168;

extern "C" fn on_prof(_sig: i32, _info: *mut u8, ctx: *mut u8) {
    // SAFETY: the kernel passes a valid ucontext_t for SA_SIGINFO handlers; the sample slot index is
    // unique per signal (atomic counter) and bounded by N.
    unsafe {
        let rip = *(ctx.add(RIP_OFFSET) as *const u64);
        let i = COUNT.fetch_add(1, Ordering::Relaxed);
        if i < N { (&raw mut SAMPLES as *mut u64).add(i).write(rip); }
    }
}

fn timer(usec: i64) {
    let t = ITimerVal { interval: Timeval { sec: 0, usec }, value: Timeval { sec: 0, usec } };
    // SAFETY: plain libc call with a valid struct.
    unsafe { setitimer(ITIMER_PROF, &t, std::ptr::null_mut()); }
}

/// Start sampling if `NSC_PROFILE` is set.
pub fn start() {
    if std::env::var_os("NSC_PROFILE").is_none() { return; }
    let act = SigAction { sa_sigaction: on_prof as usize, sa_mask: [0; 16], sa_flags: SA_SIGINFO | SA_RESTART, sa_restorer: 0 };
    // SAFETY: installs a handler that only touches atomics and a static buffer.
    unsafe { sigaction(SIGPROF, &act, std::ptr::null_mut()); }
    timer(1000);
}

/// Stop sampling and write one hex offset (relative to the executable's load base) per line.
pub fn stop() {
    let Some(path) = std::env::var_os("NSC_PROFILE") else { return };
    timer(0);
    let exe = std::env::current_exe().ok().and_then(|p| std::fs::canonicalize(p).ok());
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let mut base = 0u64;
    if let Some(exe) = &exe {
        for l in maps.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.len() >= 6 && f[5] == exe.to_string_lossy() && f[2] == "00000000" {
                base = u64::from_str_radix(f[0].split('-').next().unwrap_or("0"), 16).unwrap_or(0);
                break;
            }
        }
    }
    let n = COUNT.load(Ordering::Relaxed).min(N);
    let mut out = String::with_capacity(n * 10);
    for i in 0..n {
        // SAFETY: indices < n were written by the handler, which is now disabled.
        let rip = unsafe { (&raw const SAMPLES as *const u64).add(i).read() };
        out.push_str(&format!("{:x}\n", rip.wrapping_sub(base)));
    }
    let _ = std::fs::write(path, out);
    eprintln!("profile: {n} samples");
}

/// Developer stage timer: prints `label` and its wall time to stderr when dropped, if
/// `NSC_TIMING` is set (inert otherwise).
pub struct Span(Option<(String, std::time::Instant)>);

pub fn span(label: impl FnOnce() -> String) -> Span {
    Span(std::env::var_os("NSC_TIMING").map(|_| (label(), std::time::Instant::now())))
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((l, t)) = self.0.take() { eprintln!("[timing] {l}: {:.3} s", t.elapsed().as_secs_f64()); }
    }
}
