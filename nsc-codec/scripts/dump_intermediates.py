"""Dump int8_codec intermediates for the reference window's FIRST token (pos 0, fresh GRU),
so the Rust forward can be verified piece-by-piece. Writes inter.nsc:
  tok0 i32, o2t i32, o2c f64, mt i32, mc f64,
  H u32, V u32, conf(4 f32), gru_h(H f32), resid_gru(V f32)
"""
import os, sys, struct
import numpy as np
import torch

ROOT = "/home/christian/Escritorio/compression-algorithms/Syntropy2"
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "python"))
torch.set_grad_enabled(False)
import neuro_symbolic_compressor as nsc
from test_compression_integrity import Artifacts, IncrementalFeatures
import int8_codec as C

art = Artifacts("results/export", "data/enwik8", quiet=True)
m = art.model.eval(); cfg = art.cfg
tokens = nsc.encode_with(art.tok, art.read_slice(art.split_range("test")[0], 300))[:64]
inc = IncrementalFeatures(art.o2, cfg.order2_match_order, cfg.match_len_cap)
feats = np.array([inc.push(int(t)) for t in tokens], dtype=np.float64)

tok0 = int(tokens[0]); o2t, o2c_f, mt, mc_f = feats[0]
H, V = cfg.gru_hidden, cfg.vocab_size

# mirror int8_codec.Int8Decoder.step for pos 0 (GRU hidden = 0)
idt = torch.tensor([tok0], dtype=torch.long); E = m.embed(idt)
o2c = torch.tensor([o2c_f], dtype=torch.float32); mc = torch.tensor([mc_f], dtype=torch.float32)
p1, o1_max, o1_ent, conf = C._classical(m, cfg, idt, o2c, mc)
hs = [torch.zeros(1, H) for _ in range(cfg.gru_layers)]
hs = C._gru_step(m, cfg, E, hs)
gh = C._film(m.local.film.to_gb, hs[-1], conf)
resid_gru = C.MM(gh, m.local.to_resid.weight, m.local.to_resid.bias)

# full pos-0 forward via the batched encoder (T=1) -> probs, and its resid_xf/w/g via a stepper
probs0 = C.encode_window(m, cfg, tokens[:1], feats[:1])[0]     # [V]
dec = C.Int8Decoder(m, cfg, cfg.seg_len)
probs0b = dec.step(tok0, int(o2t), float(o2c_f), int(mt), float(mc_f))  # decoder pos-0 (T-pad)

with open("nsc-codec/artifacts/inter.nsc", "wb") as f:
    f.write(struct.pack("<i i d i d", tok0, int(o2t), float(o2c_f), int(mt), float(mc_f)))
    f.write(struct.pack("<II", H, V))
    f.write(conf[0].numpy().astype("<f4").tobytes())          # [4]
    f.write(gh[0].numpy().astype("<f4").tobytes())            # [H]
    f.write(resid_gru[0].numpy().astype("<f4").tobytes())     # [V]
    f.write(np.asarray(probs0b, dtype="<f4").tobytes())       # [V] decoder pos-0 probs
print(f"wrote inter.nsc: tok0={tok0} conf={conf[0].tolist()}")
print(f"  encode_window(T=1) vs decoder pos0 probs max|d|={np.abs(probs0-probs0b).max():.2e}")
print(f"  probs0[:5]={probs0b[:5].tolist()}")
