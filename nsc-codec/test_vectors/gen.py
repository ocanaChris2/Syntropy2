"""Generate range-coder parity vectors from the REAL Python coder. Writes vec.bin:
  header:  N u32, V u32, n_bits u32
  N steps: symbol u16, then (V+1) u64 cumulative table
  trailer: payload_len u32, payload bytes  (Python's reference encoding)
The Rust harness must reproduce this payload bit-for-bit and decode it back to the symbols.
"""
import os, sys, struct
import numpy as np

ROOT = "/home/christian/Escritorio/compression-algorithms/Syntropy2"
sys.path.insert(0, ROOT)
from neuro_symbolic_compressor import _quantize
from test_compression_integrity import RangeEncoder, RangeDecoder

np.random.seed(42)
V, N, BITS = 256, 2000, 16
cums, syms = [], []
enc = RangeEncoder()
for _ in range(N):
    conc = np.full(V, np.random.uniform(0.1, 1.0))       # varying peakiness
    p = np.random.dirichlet(conc)
    cum = _quantize(p, total_bits=BITS)                  # cumulative freq table, cum[-1]=2^BITS
    s = int(np.random.randint(V))
    enc.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
    cums.append(np.asarray(cum, dtype=np.uint64))
    syms.append(s)
payload, nbits = enc.finish()

# self-check: Python decoder recovers the symbols
dec = RangeDecoder(payload, nbits)
assert [dec.decode(c) for c in cums] == syms, "python roundtrip failed"

out = os.path.join(ROOT, "nsc-codec/test_vectors/vec.bin")
with open(out, "wb") as f:
    f.write(struct.pack("<III", N, V, nbits))
    for cum, s in zip(cums, syms):
        f.write(struct.pack("<H", s))
        f.write(cum.astype("<u8").tobytes())
    f.write(struct.pack("<I", len(payload)))
    f.write(payload)
print(f"wrote {out}: N={N} V={V} nbits={nbits} payload={len(payload)}B")
