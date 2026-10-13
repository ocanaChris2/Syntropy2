"""Int8 codec: fully-batched encoder (softmax over T with future masked) + a stateful
KV-cache decoder that pads future scores to -inf (T-length softmax) so it reproduces the
encoder's per-position coder tables BIT-EXACTLY. Integer accumulation makes them identical
regardless of batch shape/device. Everything non-matmul stays float (already batch-invariant).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from neuro_symbolic_compressor import build_rope, apply_rope, peaked_distribution

BITS = 8
_QMAX = (1 << (BITS - 1)) - 1


def qrow(t):                                        # per-row int8 quant over last dim
    s = (t.abs().amax(-1, keepdim=True) / _QMAX).clamp_min(1e-30)
    return torch.round(t / s).clamp(-_QMAX - 1, _QMAX), s


_wc = {}
def _qw(W):
    k = (id(W), W.device)                           # device in key: m.to() moves in place (same id)
    if k not in _wc:
        wq, ws = qrow(W); _wc[k] = (wq.float(), ws.squeeze(-1).double())
    return _wc[k]


def MM(x, W, b):                                    # int8 linear via fp32-EXACT matmul
    # int8*int8 summed over k<=1024 <= 1024*127^2 = 16.5M < 2^24 -> fp32 accumulates the
    # integer EXACTLY (verified). Runs on GPU at any batch (no _int_mm m>16 limit) and is
    # bit-identical to int64 accumulate -> same coder tables, lossless.
    xq, xs = qrow(x); wq, ws = _qw(W)
    return (xq @ wq.t()).double().mul_(xs.double() * ws).float() + b


def _film(mod, hid, conf):
    gb = MM(F.gelu(MM(conf, mod[0].weight, mod[0].bias)), mod[2].weight, mod[2].bias)
    g, be = gb.chunk(2, -1); return (1 + g) * hid + be


def _gru_step(m, cfg, x_t, hs):                     # one GRU timestep (list hs mutated-copy)
    inp, new = x_t, []
    gw = {n: p for n, p in m.local.gru.named_parameters()}
    for l in range(cfg.gru_layers):
        gi = MM(inp, gw[f"weight_ih_l{l}"], gw[f"bias_ih_l{l}"])
        gh = MM(hs[l], gw[f"weight_hh_l{l}"], gw[f"bias_hh_l{l}"])
        ir, iz, in_ = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(ir + hr); z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn); hnew = (1 - z) * n + z * hs[l]
        new.append(hnew); inp = hnew
    return new


def _classical(m, cfg, idt, o2c, mc):
    V = cfg.vocab_size
    p1 = m.order1.float()[idt]
    o1_max = p1.max(-1).values
    o1_ent = -(p1 * (p1 + m.eps).log()).sum(-1) / math.log(V)
    conf = torch.stack([o1_max, o1_ent, o2c, mc], -1)
    return p1, o1_max, o1_ent, conf


def _finish(m, cfg, gh, xf_h, resid_gru, resid_xf, o1_max, o2c, mc, p1, feats_slice, idt):
    V = cfg.vocab_size; T = gh.shape[0]; dev = gh.device
    q = MM(torch.cat([gh, xf_h], -1), m.gate.q_proj.weight, m.gate.q_proj.bias)
    ef = torch.stack([torch.full((T,), 0.5, device=dev), o1_max, o2c, mc, gh.abs().mean(-1), xf_h.abs().mean(-1)], -1)
    keys = m.gate.expert_embed[None] + MM(ef.reshape(T * 6, 1), m.gate.feat_proj.weight, m.gate.feat_proj.bias).reshape(T, 6, -1)
    sc = (q.unsqueeze(1) * keys).sum(-1) / math.sqrt(cfg.gate_dim)
    w = sc[:, :4].softmax(-1); g = torch.sigmoid(sc[:, 4:])
    p0 = m.order0.float()[None].expand(T, V)
    p2 = peaked_distribution(torch.tensor(feats_slice[:, 0], dtype=torch.long, device=dev)[None], o2c[None], V)[0]
    pm = peaked_distribution(torch.tensor(feats_slice[:, 2], dtype=torch.long, device=dev)[None], mc[None], V)[0]
    p_cls = w[:, 0:1] * p0 + w[:, 1:2] * p1 + w[:, 2:3] * p2 + w[:, 3:4] * pm
    z = (p_cls + m.eps).log() + g[:, 0:1] * resid_gru + g[:, 1:2] * resid_xf
    return torch.softmax(z, -1).cpu().numpy()


# ============================ BATCHED ENCODER =================================
def encode_window(m, cfg, ids, feats):
    """Fully batched int8 forward over a window -> probs[T,V]. Canonical computation."""
    V, H, D = cfg.vocab_size, cfg.gru_hidden, cfg.d_model
    h, hd = cfg.n_heads, D // cfg.n_heads
    T = len(ids)
    dev = m.embed.weight.device
    idt = torch.tensor(np.asarray(ids), dtype=torch.long, device=dev); E = m.embed(idt)
    o2c = torch.tensor(feats[:, 1], dtype=torch.float32, device=dev); mc = torch.tensor(feats[:, 3], dtype=torch.float32, device=dev)
    p1, o1_max, o1_ent, conf = _classical(m, cfg, idt, o2c, mc)
    # GRU
    hs = [torch.zeros(1, H, device=dev) for _ in range(cfg.gru_layers)]
    gru_out = torch.empty(T, H, device=dev)
    for t in range(T):
        hs = _gru_step(m, cfg, E[t:t+1], hs); gru_out[t] = hs[-1][0]
    gh = _film(m.local.film.to_gb, gru_out, conf)
    resid_gru = MM(gh, m.local.to_resid.weight, m.local.to_resid.bias)
    # transformer (batched masked attention)
    x = E
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
    for blk in m.xformer.blocks:
        qkv = MM(blk.ln1(x), blk.attn.qkv.weight, blk.attn.qkv.bias)
        q, k, v = qkv.split(D, -1)
        q = q.view(T, h, hd).transpose(0, 1); k = k.view(T, h, hd).transpose(0, 1); v = v.view(T, h, hd).transpose(0, 1)
        cos, sin = build_rope(T, hd, x.device, offset=0)
        q = apply_rope(q.unsqueeze(0), cos, sin)[0]; k = apply_rope(k.unsqueeze(0), cos, sin)[0]
        qq, sq = qrow(q); kk, sk = qrow(k); vq, sv = qrow(v)
        sc = (qq @ kk.transpose(-1, -2)).double()
        sc = (sc * (sq.double() * sk.transpose(-1, -2).double())).float().masked_fill(mask, float("-inf")) / math.sqrt(hd)
        att = sc.softmax(-1)
        bq, sb = qrow(att * sv.transpose(-1, -2))
        oc = (bq @ vq).double() * sb.double()
        out = oc.float().transpose(0, 1).reshape(T, D)
        x = x + MM(out, blk.attn.proj.weight, blk.attn.proj.bias)
        x = x + MM(F.gelu(MM(blk.ln2(x), blk.mlp[0].weight, blk.mlp[0].bias)), blk.mlp[2].weight, blk.mlp[2].bias)
    xf_h = _film(m.xformer.film.to_gb, m.xformer.lnf(x), conf)
    resid_xf = MM(xf_h, m.xformer.to_resid.weight, m.xformer.to_resid.bias)
    return _finish(m, cfg, gh, xf_h, resid_gru, resid_xf, o1_max, o2c, mc, p1, feats, idt)


# ============================ STATEFUL DECODER ================================
class Int8Decoder:
    """Per-token; pads future scores to -inf so its length-T softmax matches the
    batched encoder's row t bit-exactly. KV cache holds RoPE'd int8 k and int8 v."""
    def __init__(self, m, cfg, T_window):
        self.m, self.cfg, self.Tw = m, cfg, T_window
        self.h, self.hd, self.D = cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.d_model
        self.reset_window()

    def reset_window(self):
        nb = len(self.m.xformer.blocks)
        self.gru_hs = [torch.zeros(1, self.cfg.gru_hidden) for _ in range(self.cfg.gru_layers)]
        self.kc = [[] for _ in range(nb)]; self.ks = [[] for _ in range(nb)]   # k codes / k scales
        self.vc = [[] for _ in range(nb)]; self.vs = [[] for _ in range(nb)]   # v codes / v scales
        self.pos = 0

    @torch.no_grad()
    def step(self, tok, o2t, o2c_f, mt, mc_f):
        m, cfg = self.m, self.cfg
        D, h, hd = self.D, self.h, self.hd
        idt = torch.tensor([tok], dtype=torch.long); E = m.embed(idt)   # [1,D]
        o2c = torch.tensor([o2c_f], dtype=torch.float32); mc = torch.tensor([mc_f], dtype=torch.float32)
        feats1 = np.array([[o2t, o2c_f, mt, mc_f]], dtype=np.float64)
        p1, o1_max, o1_ent, conf = _classical(m, cfg, idt, o2c, mc)
        self.gru_hs = _gru_step(m, cfg, E, self.gru_hs)
        gh = _film(m.local.film.to_gb, self.gru_hs[-1], conf)
        resid_gru = MM(gh, m.local.to_resid.weight, m.local.to_resid.bias)
        x = E; pos = self.pos
        cos, sin = build_rope(self.Tw, hd, x.device, offset=0)          # full-window RoPE tables
        c1, s1 = cos[:, :, pos:pos+1], sin[:, :, pos:pos+1]
        for bi, blk in enumerate(m.xformer.blocks):
            qkv = MM(blk.ln1(x), blk.attn.qkv.weight, blk.attn.qkv.bias)
            q, k, v = qkv.split(D, -1)
            q = q.view(1, h, hd).transpose(0, 1); k = k.view(1, h, hd).transpose(0, 1); v = v.view(1, h, hd).transpose(0, 1)
            q = apply_rope(q.unsqueeze(0), c1, s1)[0]; k = apply_rope(k.unsqueeze(0), c1, s1)[0]   # [h,1,hd]
            kq, sk = qrow(k); vq, svv = qrow(v)
            self.kc[bi].append(kq); self.ks[bi].append(sk); self.vc[bi].append(vq); self.vs[bi].append(svv)
            Kq = torch.cat(self.kc[bi], 1); SK = torch.cat(self.ks[bi], 1)     # [h,P,hd],[h,P,1]
            Vq = torch.cat(self.vc[bi], 1); SV = torch.cat(self.vs[bi], 1)
            qq, sq = qrow(q)                                                    # [h,1,hd],[h,1,1]
            sc_int = (qq @ Kq.transpose(-1, -2)).double()   # [h,1,P]  fp32-exact int
            sc = (sc_int * (sq.double() * SK.transpose(-1, -2).double())).float().squeeze(1) / math.sqrt(hd)  # [h,P]
            scores = torch.full((h, self.Tw), float("-inf")); scores[:, :pos+1] = sc          # T-pad softmax
            att = scores.softmax(-1)[:, :pos+1]                                                # [h,P]
            bq, sb = qrow(att * SV[:, :, 0])                                                   # [h,P],[h,1]
            oc = (bq.unsqueeze(1) @ Vq).double() * sb.unsqueeze(1).double()  # [h,1,hd]  fp32-exact int
            out = oc.float().squeeze(1).reshape(1, D)
            x = x + MM(out, blk.attn.proj.weight, blk.attn.proj.bias)
            x = x + MM(F.gelu(MM(blk.ln2(x), blk.mlp[0].weight, blk.mlp[0].bias)), blk.mlp[2].weight, blk.mlp[2].bias)
        xf_h = _film(m.xformer.film.to_gb, m.xformer.lnf(x), conf)
        resid_xf = MM(xf_h, m.xformer.to_resid.weight, m.xformer.to_resid.bias)
        probs = _finish(m, cfg, gh, xf_h, resid_gru, resid_xf, o1_max, o2c, mc, p1, feats1, idt)[0]
        self.pos += 1
        return probs
