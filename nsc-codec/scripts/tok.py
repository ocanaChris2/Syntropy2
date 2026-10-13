#!/usr/bin/env python3
"""Tokenizer shim (bytes <-> token ids) so the Rust codec stays dependency-free. Replicates
neuro_symbolic_compressor.encode_with / decode_with EXACTLY (the latin-1 byte bijection over the
HuggingFace tokenizer.json), so `dec(enc(bytes)) == bytes`.

  python tok.py enc <infile>  <out.ids>  [tokenizer.json]
  python tok.py dec <in.ids>  <outfile>  [tokenizer.json]

The optional 4th argument is the tokenizer to use; when omitted it falls back to $NSC_TOKENIZER
then the repo default. The `nsc` CLI always passes the tokenizer that belongs to the model
version selected for the file, so a file always round-trips through its own tokenizer even after
the default model is upgraded.

.ids layout (shared with the Rust codec): magic 'NIDS' (4) | n u32 | n x token-id u32
"""
import os
import sys
import struct
import numpy as np
from tokenizers import Tokenizer

ROOT = "/home/christian/Escritorio/compression-algorithms/Syntropy2"
DEFAULT_TOKPATH = os.environ.get("NSC_TOKENIZER", ROOT + "/results/export/tokenizer.json")


def _tok(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"tokenizer not found: {path}")
    return Tokenizer.from_file(path)


def _encode(tok, data: bytes) -> np.ndarray:            # == nsc.encode_with (non-ByteTokenizer)
    return np.asarray(tok.encode(data.decode("latin-1")).ids, dtype="<u4")


def _decode(tok, ids) -> bytes:                          # == nsc.decode_with (non-ByteTokenizer)
    return tok.decode([int(i) for i in ids]).encode("latin-1")


def enc(infile: str, outids: str, tokpath: str) -> None:
    tok = _tok(tokpath)
    with open(infile, "rb") as f:
        data = f.read()
    ids = _encode(tok, data)
    if _decode(tok, ids.tolist()) != data:               # lossless precondition (as enforced at compress time)
        raise SystemExit("tokenizer is not a lossless bijection on this input")
    with open(outids, "wb") as f:
        f.write(b"NIDS"); f.write(struct.pack("<I", len(ids))); f.write(ids.tobytes())
    print(f"enc: {len(data):,} bytes -> {len(ids):,} tokens -> {outids}")


def dec(inids: str, outfile: str, tokpath: str) -> None:
    tok = _tok(tokpath)
    with open(inids, "rb") as f:
        b = f.read()
    if b[:4] != b"NIDS":
        raise SystemExit("bad .ids magic")
    n = struct.unpack_from("<I", b, 4)[0]
    ids = np.frombuffer(b, dtype="<u4", count=n, offset=8)
    data = _decode(tok, ids.tolist())
    with open(outfile, "wb") as f:
        f.write(data)
    print(f"dec: {n:,} tokens -> {len(data):,} bytes -> {outfile}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in ("enc", "dec"):
        print("usage: python tok.py [enc <infile> <out.ids> | dec <in.ids> <outfile>] [tokenizer.json]",
              file=sys.stderr)
        sys.exit(2)
    tokpath = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_TOKPATH
    (enc if sys.argv[1] == "enc" else dec)(sys.argv[2], sys.argv[3], tokpath)
