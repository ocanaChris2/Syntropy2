"""Export the FastO2 order-2 table as a dependency-free binary (o2.nsc) for the Rust feature
model (src/features.rs). Uses the SAME `Artifacts(...).o2` that generated ref.nsc's features, so
the Rust order-2 lookup matches Python bit-for-bit.

o2.nsc layout:
  magic 'NO21' (4 bytes)
  V u32, K u32 (order2_match_order), cap f32 (match_len_cap), n u32
  ctx  (n x i64, strictly ascending a*V+b context keys)
  top  (n x i32, most-likely next token)
  conf (n x f32, confidence = count(top)/total)
"""
import os, sys, struct, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--export-dir", default="results/export", help="dir with best.pt + tokenizer.json")
ap.add_argument("--data", default="data/enwik8", help="training data file (enwik8 or <corpus>.bin)")
ap.add_argument("--out", default="nsc-codec/artifacts", help="output dir for o2.nsc")
args = ap.parse_args()
EXPORT, DATA, OUT = (os.path.abspath(p) for p in (args.export_dir, args.data, args.out))
sys.argv = sys.argv[:1]
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "python"))
import torch; torch.set_grad_enabled(False)
from test_compression_integrity import Artifacts

art = Artifacts(EXPORT, DATA, quiet=True)
cfg, o2 = art.cfg, art.o2
os.makedirs(OUT, exist_ok=True)

ctx = np.ascontiguousarray(o2.ctx, dtype="<i8")
top = np.ascontiguousarray(o2.top, dtype="<i4")
conf = np.ascontiguousarray(o2.conf, dtype="<f4")
assert ctx.shape == top.shape == conf.shape, "o2 arrays misaligned"
assert ctx.ndim == 1 and len(ctx) > 0, "empty o2 table"
assert np.all(ctx[1:] > ctx[:-1]), "ctx must be strictly ascending (binary search precondition)"
assert int(np.asarray(o2.top).max()) < cfg.vocab_size, "top token id out of range"
n = len(ctx)

path = os.path.join(OUT, "o2.nsc")
with open(path, "wb") as f:
    f.write(b"NO21")
    f.write(struct.pack("<IIfI", int(cfg.vocab_size), int(cfg.order2_match_order),
                        float(cfg.match_len_cap), n))
    f.write(ctx.tobytes()); f.write(top.tobytes()); f.write(conf.tobytes())
print(f"wrote {path}: {n:,} order-2 contexts, {os.path.getsize(path) / 1e6:.1f} MB "
      f"(V={cfg.vocab_size}, K={cfg.order2_match_order}, cap={cfg.match_len_cap})")
