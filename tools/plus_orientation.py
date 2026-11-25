#!/usr/bin/env python3
"""Exact relative Neel orientation of the blocks of a severed lattice along an open walk.

For a rung (la, lb) with O = <+| A B |+>, A = e^{-tau H(la)}, B = e^{-tau H(lb)} (B acts first,
the walk origin sits next to |+>), the basis state at three positions of the walk is distributed as
    origin end (next to |+>):   (B A|+>)(z)       middle:  (A|+>)(z) (B|+>)(z)
    far end   (next to <+|):    (A B|+>)(z)
For each block b (rectangles between the cuts in model.json) let m_b(z) be its staggered
magnetization with sign (-1)^(x+y). Reported: the probability that all blocks have m_b != 0 and
the same sign, conditional on all m_b != 0, for all walks and for walks with no hop on a severed
bond (H1 replaced by H1' without the X terms that H2 lacks, as in tools/plus_support.py).

On models/4x4_v2 at the middle rung (0,1), beta = 8:
    all walks       origin 0.1251  middle 0.7919  far end 0.4364
    no severed hop  origin 0.1251  middle 0.8235  far end 0.8235
The support condition changes the orientation statistics only toward the H1 (full) end.

Usage: python3 tools/plus_orientation.py MODEL --beta 8 [--la 0 --lb 1]
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_reference import load_sparse

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('--beta', type=float, required=True)
ap.add_argument('--la', type=float, default=0.0); ap.add_argument('--lb', type=float, default=1.0)
a = ap.parse_args()
m = json.load(open(os.path.join(a.model, 'model.json')))
N, Lx, Ly = m['N'], m['Lx'], m['Ly']; tau = a.beta / 2
xs = [0] + list(m.get('vcut', [])) + [Lx]; ys = [0] + list(m.get('hcut', [])) + [Ly]
blocks = [[x + Lx * y for y in range(ys[j], ys[j + 1]) for x in range(xs[i], xs[i + 1])]
          for j in range(len(ys) - 1) for i in range(len(xs) - 1)]
H1 = load_sparse(os.path.join(a.model, 'H1.txt'), N); H2 = load_sparse(os.path.join(a.model, 'H2.txt'), N)
off1 = (H1 - sp.diags(H1.diagonal())).tocsr(); off2 = (H2 - sp.diags(H2.diagonal())).tocsr()
H1p = (sp.diags(H1.diagonal()) + off1.multiply((abs(off2) > 0).astype(float))).tocsr()
dim = 1 << N; plus = np.ones(dim) / np.sqrt(dim); I = sp.identity(dim, format='csr')
z = np.arange(dim); spin = 1 - 2 * ((z[:, None] >> np.arange(N)) & 1)
stag = np.array([(-1) ** ((i % Lx) + (i // Lx)) for i in range(N)])
mb = np.stack([(spin[:, b] * stag[b]).sum(1) for b in blocks], 1)
ok = (mb != 0).all(1); sg = np.sign(mb); agree = ok & (sg == sg[:, [0]]).all(1)
stat = lambda w: float((np.clip(w, 0, None) * agree).sum() / (np.clip(w, 0, None) * ok).sum())
print(f'{len(blocks)} blocks {blocks}; rung (la, lb) = ({a.la}, {a.lb}), beta = {a.beta}')
for label, H1x in (('all walks', H1), ('no severed hop', H1p)):
    Ha = ((1 - a.la) * H1x + a.la * H2).tocsr(); Hb = ((1 - a.lb) * H1x + a.lb * H2).tocsr()
    sa = float(eigsh(Ha, k=1, which='SA', return_eigenvectors=False)[0])
    sb = float(eigsh(Hb, k=1, which='SA', return_eigenvectors=False)[0])
    A = lambda v: expm_multiply(-tau * (Ha - sa * I), v); B = lambda v: expm_multiply(-tau * (Hb - sb * I), v)
    Ap, Bp = A(plus), B(plus)
    print(f'{label:15s} origin {stat(B(Ap)):.4f}  middle {stat(Ap * Bp):.4f}  far end {stat(A(Bp)):.4f}')
