//! Runtime SIMD selection. Every SIMD kernel in the codec has a scalar twin that computes the
//! *identical* integers, so the choice never changes an output byte — it only changes speed.
//!
//! Because the crate builds with `target-cpu=native`, `is_x86_feature_detected!("avx2")` folds to a
//! compile-time `true` on this machine and the scalar paths would otherwise never run. Setting
//! `NSC_SIMD=scalar` forces them, so whole archives can be compared across both paths, and the unit
//! tests call both kernels directly.
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Scalar,
    Avx2,
}

pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        if std::env::var("NSC_SIMD").map(|v| v.eq_ignore_ascii_case("scalar")).unwrap_or(false) {
            return Level::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Level::Avx2;
            }
        }
        Level::Scalar
    })
}