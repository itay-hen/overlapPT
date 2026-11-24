#!/usr/bin/env python3
"""Exact probability, at every rung, that an open walk uses no severed-bond hop
(i.e. lies in the support of the last rung, H2 on both sides), and the same fraction in kept
samples: a sample drawn at rung k is in that support iff its u at rung K-1 is finite.

Exact: P_k = <+|e^{-tau H'(la)} e^{-tau H'(lb)}|+> / <+|e^{-tau H(la)} e^{-tau H(lb)}|+>, where H'
drops the X-string terms of H1 that H2 does not contain (diagonal terms untouched).

Usage: python3 tools/plus_support.py MODEL --beta 8 --rungs K [RUN_DIR... [--partial]]
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_reference import load_sparse
from plus_bias_check import load

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('--beta', type=float, required=True)
ap.add_argument('--rungs', type=int, required=True); ap.add_argument('runs', nargs='*')
ap.add_argument('--partial', action='store_true')
a = ap.parse_args()
N = json.load(open(os.path.join(a.model, 'model.json')))['N']; K = a.rungs; tau = a.beta / 2
H1 = load_sparse(os.path.join(a.model, 'H1.txt'), N); H2 = load_sparse(os.path.join(a.model, 'H2.txt'), N)
# severed hops: off-diagonal entries of H1 at (row, col) pairs where H2 has none
off1 = (H1 - sp.diags(H1.diagonal())).tocsr(); off2 = (H2 - sp.diags(H2.diagonal())).tocsr()
mask = (abs(off2) > 0).astype(float)
H1p = (sp.diags(H1.diagonal()) + off1.multiply(mask)).tocsr()
dim = 1 << N; plus = np.ones(dim) / np.sqrt(dim); I = sp.identity(dim, format='csr')
pairs = [((0.0, 2 * k / (K - 1)) if k / (K - 1) <= 0.5 else (2 * k / (K - 1) - 1, 1.0)) for k in range(K)]
lams = sorted({x for p in pairs for x in p})
v, vp = {}, {}
for lam in lams:
    H = ((1 - lam) * H1 + lam * H2).tocsr(); Hp = ((1 - lam) * H1p + lam * H2).tocsr()
    s = float(eigsh(H, k=1, which='SA', return_eigenvectors=False)[0])
    v[lam] = expm_multiply(-tau * (H - s * I), plus); vp[lam] = expm_multiply(-tau * (Hp - s * I), plus)
P = np.array([(vp[la] @ vp[lb]) / (v[la] @ v[lb]) for la, lb in pairs])
mc = None
if a.runs:
    data = [load(r, K, a.partial) for r in a.runs]
    mc = np.array([[np.mean(u[:, k, K - 1] < 1e299) for k in range(K)] for u in data])
print('rung  exact      ' + ('MC mean +- SEM over runs   z' if mc is not None else ''))
for k in range(K):
    line = f'{k:3d}  {P[k]:.5f}'
    if mc is not None:
        m, s = mc[:, k].mean(), mc[:, k].std(ddof=1) / np.sqrt(len(mc))
        line += f'    {m:.5f} +- {s:.5f}   {(m - P[k]) / s if s > 0 else 0:+.1f}'
    print(line)
