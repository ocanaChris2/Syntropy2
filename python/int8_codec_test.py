"""Validate the int8 codec: (1) batched encoder tables == stateful decoder tables (bit-exact,
teacher-forced) — the losslessness gate; (2) full byte-exact roundtrip through the range coder."""
import os, sys, time
import numpy as np
import torch

ROOT = "/home/christian/Escritorio/compression-algorithms/Syntropy2"
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "python"))
torch.set_grad_enabled(False); torch.set_num_threads(4)

import neuro_symbolic_compressor as nsc
from neuro_symbolic_compressor import _quantize
from test_compression_integrity import (Artifacts, IncrementalFeatures, RangeEncoder,
                                         RangeDecoder, QUANT_BITS)
import int8_codec as C

art = Artifacts("results/export", "data/enwik8", quiet=True)
m = art.model.eval(); cfg = art.cfg
a, _ = art.split_range("test")
data = art.read_slice(a, 700)
tokens = nsc.encode_with(art.tok, data)[:200]
data = nsc.decode_with(art.tok, tokens.tolist())
T = len(tokens) - 1
inc = IncrementalFeatures(art.o2, cfg.order2_match_order, cfg.match_len_cap)
feats = np.array([inc.push(int(t)) for t in tokens], dtype=np.float64)
print(f"window T={T} tokens, int8 codec")

# (1) GATE: batched encoder tables vs stateful decoder tables (teacher-forced)
penc = C.encode_window(m, cfg, tokens[:T], feats[:T])
dec = C.Int8Decoder(m, cfg, T)
diff = 0; maxdp = 0.0
for t in range(T):
    pd = dec.step(int(tokens[t]), int(feats[t, 0]), float(feats[t, 1]), int(feats[t, 2]), float(feats[t, 3]))
    maxdp = max(maxdp, float(np.abs(pd - penc[t]).max()))
    if not np.array_equal(_quantize(pd, total_bits=QUANT_BITS), _quantize(penc[t], total_bits=QUANT_BITS)):
        diff += 1
print(f"(1) encoder-vs-decoder tables: differing={diff}/{T}  max|dp|={maxdp:.2e}  "
      f"{'BIT-EXACT ✓' if diff == 0 else '✗ NOT LOSSLESS'}")

# (2) full roundtrip: encode with batched probs, decode with stateful decoder
enc = RangeEncoder()
for t in range(T):
    cum = _quantize(penc[t], total_bits=QUANT_BITS)
    s = int(tokens[t + 1])
    enc.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
payload, nbits = enc.finish()

decd = RangeDecoder(payload, nbits)
dec2 = C.Int8Decoder(m, cfg, T)
inc2 = IncrementalFeatures(art.o2, cfg.order2_match_order, cfg.match_len_cap)
toks = [int(tokens[0])]
f = inc2.push(toks[0])
for t in range(T):
    p = dec2.step(toks[t], int(f[0]), float(f[1]), int(f[2]), float(f[3]))
    cum = _quantize(p, total_bits=QUANT_BITS)
    s = decd.decode(cum)
    toks.append(int(s))
    f = inc2.push(int(s))
rec = nsc.decode_with(art.tok, toks)
print(f"(2) roundtrip: payload={len(payload)}B bpc={8*len(payload)/len(data):.3f}  "
      f"tokens_match={toks == [int(x) for x in tokens]}  bytes_match={rec == data}  "
      f"{'✓ BYTE-EXACT' if rec == data else '✗'}")
