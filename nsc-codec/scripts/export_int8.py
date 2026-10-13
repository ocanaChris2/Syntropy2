"""Export the model as int8 artifacts for the Rust codec, quantized EXACTLY as int8_codec.py
(per-output-channel int8 weights via qrow). Dependency-free binary the Rust loader parses.

model.nsc layout:
  magic 'NSC1' (4 bytes)
  config: vocab_size,d_model,n_layers,n_heads,gru_hidden,gru_layers,seg_len,gate_dim,conf_dim (9 x u32)
  eps f32
  n_tensors u32
  per tensor: name_len u16, name utf8, dtype u8 (0=f32,1=i8,2=f64), ndim u8, dims (ndim x u32), raw data
ref.nsc: T u32, V u32, tokens (T i32), feats (T*4 f64), then per pos a cum table (V+1 i64) [expected]
"""
import os, sys, struct, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--export-dir", default="results/export", help="dir with best.pt + tokenizer.json")
ap.add_argument("--data", default="data/enwik8", help="training data file (enwik8 or <corpus>.bin)")
ap.add_argument("--out", default="nsc-codec/artifacts", help="output dir for model.nsc / ref.nsc")
args = ap.parse_args()
EXPORT, DATA, OUT = (os.path.abspath(p) for p in (args.export_dir, args.data, args.out))
sys.argv = sys.argv[:1]
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "python"))
torch.set_grad_enabled(False)
import neuro_symbolic_compressor as nsc
from neuro_symbolic_compressor import _quantize
from test_compression_integrity import Artifacts, IncrementalFeatures, QUANT_BITS
import int8_codec as C

art = Artifacts(EXPORT, DATA, quiet=True)
m = art.model.eval(); cfg = art.cfg
os.makedirs(OUT, exist_ok=True)

tensors = []  # (name, dtype, np.array)
def add(name, arr, dtype):
    tensors.append((name, dtype, np.ascontiguousarray(arr)))

def add_lin(name, W, b):
    wq, ws = C.qrow(W.detach())                    # per-output-channel int8 (matches int8_codec)
    add(f"{name}.code", wq.numpy().astype(np.int8), 1)
    add(f"{name}.scale", ws.squeeze(-1).double().numpy(), 2)
    add(f"{name}.bias", b.detach().float().numpy(), 0)

def add_ln(name, ln):
    add(f"{name}.w", ln.weight.detach().float().numpy(), 0)
    add(f"{name}.b", ln.bias.detach().float().numpy(), 0)

add("embed", m.embed.weight.detach().float().numpy(), 0)
add("order0", m.order0.float().numpy(), 0)
# order1 is float32 in memory (the .f16 export is lossy); int8_codec uses m.order1.float(),
# so export the exact f32 values for bit-exact coder tables (256 MB).
add("order1", m.order1.float().numpy(), 0)
# GRU
for l in range(cfg.gru_layers):
    gw = dict(m.local.gru.named_parameters())
    add_lin(f"gru.wih.l{l}", gw[f"weight_ih_l{l}"], gw[f"bias_ih_l{l}"])
    add_lin(f"gru.whh.l{l}", gw[f"weight_hh_l{l}"], gw[f"bias_hh_l{l}"])
# local FiLM + to_resid
add_lin("local.film0", m.local.film.to_gb[0].weight, m.local.film.to_gb[0].bias)
add_lin("local.film2", m.local.film.to_gb[2].weight, m.local.film.to_gb[2].bias)
add_lin("local.to_resid", m.local.to_resid.weight, m.local.to_resid.bias)
# transformer blocks
for bi, blk in enumerate(m.xformer.blocks):
    add_ln(f"blk{bi}.ln1", blk.ln1); add_ln(f"blk{bi}.ln2", blk.ln2)
    add_lin(f"blk{bi}.qkv", blk.attn.qkv.weight, blk.attn.qkv.bias)
    add_lin(f"blk{bi}.proj", blk.attn.proj.weight, blk.attn.proj.bias)
    add_lin(f"blk{bi}.mlp0", blk.mlp[0].weight, blk.mlp[0].bias)
    add_lin(f"blk{bi}.mlp2", blk.mlp[2].weight, blk.mlp[2].bias)
add_ln("xformer.lnf", m.xformer.lnf)
add_lin("xformer.film0", m.xformer.film.to_gb[0].weight, m.xformer.film.to_gb[0].bias)
add_lin("xformer.film2", m.xformer.film.to_gb[2].weight, m.xformer.film.to_gb[2].bias)
add_lin("xformer.to_resid", m.xformer.to_resid.weight, m.xformer.to_resid.bias)
# gate
add_lin("gate.q_proj", m.gate.q_proj.weight, m.gate.q_proj.bias)
add_lin("gate.feat_proj", m.gate.feat_proj.weight, m.gate.feat_proj.bias)
add("gate.expert_embed", m.gate.expert_embed.detach().float().numpy(), 0)

DT = {0: ("<f4", 4), 1: ("i1", 1), 2: ("<f8", 8)}
with open(os.path.join(OUT, "model.nsc"), "wb") as f:
    f.write(b"NSC1")
    f.write(struct.pack("<9I", cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads,
                        cfg.gru_hidden, cfg.gru_layers, cfg.seg_len, cfg.gate_dim, cfg.conf_dim))
    f.write(struct.pack("<f", float(m.eps)))
    f.write(struct.pack("<I", len(tensors)))
    for name, dt, arr in tensors:
        nb = name.encode()
        f.write(struct.pack("<H", len(nb))); f.write(nb)
        f.write(struct.pack("<BB", dt, arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
        f.write(arr.astype(DT[dt][0]).tobytes())
print(f"wrote model.nsc: {len(tensors)} tensors, "
      f"{os.path.getsize(os.path.join(OUT, 'model.nsc'))/1e6:.1f} MB")

# reference window (for end-to-end Stage C verification)
a, _ = art.split_range("test")
data = art.read_slice(a, 300)
tokens = nsc.encode_with(art.tok, data)[:64]
T = len(tokens) - 1
inc = IncrementalFeatures(art.o2, cfg.order2_match_order, cfg.match_len_cap)
feats = np.array([inc.push(int(t)) for t in tokens], dtype=np.float64)
probs = C.encode_window(m, cfg, tokens[:T], feats[:T])
with open(os.path.join(OUT, "ref.nsc"), "wb") as f:
    f.write(struct.pack("<II", T, cfg.vocab_size))
    f.write(np.asarray(tokens[:T], dtype="<i4").tobytes())
    f.write(np.asarray(feats[:T], dtype="<f8").tobytes())
    for t in range(T):
        f.write(np.asarray(_quantize(probs[t], total_bits=QUANT_BITS), dtype="<i8").tobytes())
print(f"wrote ref.nsc: T={T} window with expected coder tables")
