#!/usr/bin/env python3
"""Parallel roundtrip of files through nsc at LEVEL; prints per-file and per-category deltas vs P1."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
ROOT = '/home/christian/Escritorio/compression-algorithms/Syntropy2/nsc-codec'
S = os.path.dirname(os.path.abspath(__file__)) + '/rtp'
os.makedirs(S, exist_ok=True)
level = os.environ.get('LEVEL', '5')
nsc = os.environ.get('NSC', ROOT + '/target/release/nsc')
P = int(os.environ.get('P', '8'))
def load(name):
    try: d = json.load(open(f'{ROOT}/bench/results/{name}'))
    except Exception: return {}
    out = {}
    for f in d['files']:
        for r in f['results']: out.setdefault(f['path'], {})[r['codec']] = r['bytes']
    return out
p1 = load(f'p1_L{level}.json'); cl = load('p0_classical.json')
def one(f):
    rel = os.path.relpath(os.path.abspath(f), ROOT)
    tag = rel.replace('/', '_')
    t0 = time.time()
    r = subprocess.run([nsc, 'compress', f, f'{S}/{tag}.nsc', '-' + level, '--jobs', '2'], capture_output=True, text=True)
    te = time.time() - t0
    if r.returncode: return (rel, os.path.getsize(f), None, te, 0, 'COMPRESS FAIL ' + r.stderr[-200:])
    t0 = time.time()
    r = subprocess.run([nsc, 'decompress', f'{S}/{tag}.nsc', f'{S}/{tag}.out', '--jobs', '2'], capture_output=True, text=True)
    td = time.time() - t0
    ok = r.returncode == 0 and subprocess.run(['cmp', '-s', f, f'{S}/{tag}.out']).returncode == 0
    new = os.path.getsize(f'{S}/{tag}.nsc')
    try: os.remove(f'{S}/{tag}.out'); os.remove(f'{S}/{tag}.nsc')
    except OSError: pass
    return (rel, os.path.getsize(f), new, te, td, 'OK' if ok else 'MISMATCH ' + r.stderr[-200:])
files = [l.rstrip('\n') for a in sys.argv[1:] for l in (open(a[1:]) if a.startswith('@') else [a + '\n']) if l.strip()]
with ThreadPoolExecutor(P) as ex: res = list(ex.map(one, files))
cats = {}
print(f"{'file':52} {'raw':>10} {'new':>10} {'P1':>10} {'Δ%P1':>7} {'Δ%cl':>7} {'enc':>6} {'dec':>6} rt", flush=True)
for rel, raw, new, te, td, st in res:
    a = p1.get(rel, {}).get(f'nsc-{level}', 0)
    c = min([v for v in cl.get(rel, {}).values() if v], default=0)
    if new is None: print(rel, st); continue
    cat = rel.split('/')[2]
    t = cats.setdefault(cat, [0, 0, 0, 0]); t[0] += raw; t[1] += new; t[2] += a or new; t[3] += c or new
    d1 = f"{100*(new-a)/a:+7.2f}" if a else '      -'
    dc = f"{100*(new-c)/c:+7.2f}" if c else '      -'
    print(f"{rel[13:65]:52} {raw:>10} {new:>10} {a:>10} {d1} {dc} {te:6.1f} {td:6.1f} {st}")
T = [0, 0, 0, 0]
for cat, t in sorted(cats.items()):
    print(f"{'['+cat+']':52} {t[0]:>10} {t[1]:>10} {t[2]:>10} {100*(t[1]-t[2])/t[2]:+7.2f} {100*(t[1]-t[3])/t[3]:+7.2f}  {8*t[1]/t[0]:.3f} bpb")
    for i in range(4): T[i] += t[i]
print(f"{'TOTAL':52} {T[0]:>10} {T[1]:>10} {T[2]:>10} {100*(T[1]-T[2])/T[2]:+7.2f} {100*(T[1]-T[3])/T[3]:+7.2f}  {8*T[1]/T[0]:.3f} bpb")
