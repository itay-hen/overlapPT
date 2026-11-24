#!/usr/bin/env python3
"""Exact per-rung sampler observables for the OVERLAP_PLUS ladder.

For each rung k with O_k(eps) = <+| exp(-tau H_eps(lam_a)) exp(-tau H_eps(lam_b)) |+>, where every
off-diagonal (X-string) coefficient is scaled by (1+eps), the open-walk weight of a walk with q
operators scales as (1+eps)^q, so
    <q>_k = d ln O_k / d eps,   <q(q-1)>_k = O_k'' / O_k     (at eps = 0).
These are computed by central finite differences of the sparse exact overlaps.
Also prints the exact marginal of the walk start y0 at one rung on request (--y0-rung).

Usage: python3 tools/plus_observables.py MODEL --beta 8 --rungs 17 [--out FILE.json]
"""
import argparse, json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_reference import load_sparse


def split(H):
    D = sp.diags(H.diagonal())
    return D.tocsr(), (H - D).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('--beta', type=float, required=True)
    ap.add_argument('--rungs', type=int, required=True); ap.add_argument('--h', type=float, default=1e-3)
    ap.add_argument('--y0-rung', type=int, default=-1); ap.add_argument('--out')
    a = ap.parse_args()
    N = json.load(open(os.path.join(a.model, 'model.json')))['N']
    H = [split(load_sparse(os.path.join(a.model, f'H{k}.txt'), N)) for k in (1, 2)]
    dim = 1 << N; plus = np.ones(dim) / np.sqrt(dim); tau = a.beta / 2
    K = a.rungs
    pairs = [((0.0, 2 * k / (K - 1)) if k / (K - 1) <= 0.5 else (2 * k / (K - 1) - 1, 1.0)) for k in range(K)]
    lams = sorted({x for p in pairs for x in p})
    I = sp.identity(dim, format='csr')

    def Hl(lam, eps):
        return ((1 - lam) * (H[0][0] + (1 + eps) * H[0][1]) + lam * (H[1][0] + (1 + eps) * H[1][1])).tocsr()
    shift = {lam: float(eigsh(Hl(lam, 0), k=1, which='SA', return_eigenvectors=False)[0]) for lam in lams}
    res = {}
    for eps in (-a.h, 0.0, a.h):
        v = {lam: expm_multiply(-tau * (Hl(lam, eps) - shift[lam] * I), plus) for lam in lams}
        # lnO up to the eps-independent constant -tau(s_a+s_b)
        res[eps] = [np.log(v[la] @ v[lb]) for la, lb in pairs]
        if eps == 0.0:
            v0 = v
    h = a.h
    out = {'N': N, 'beta': a.beta, 'rungs': K, 'q_mean': [], 'q_var': []}
    for k in range(K):
        lm, l0, lp = res[-h][k], res[0.0][k], res[h][k]
        d1 = (lp - lm) / (2 * h); d2 = (lp - 2 * l0 + lm) / h ** 2  # d2 = (ln O)'' = <q^2>-<q>^2 - <q>
        out['q_mean'].append(d1); out['q_var'].append(d2 + d1)
    if a.y0_rung >= 0:
        la, lb = pairs[a.y0_rung]
        # p(y0) ∝ (<+| e^{-tau H(la)} e^{-tau H(lb)})[y0]; H real symmetric
        w = expm_multiply(-tau * (Hl(lb, 0) - shift[lb] * I), v0[la])
        out['y0_rung'] = a.y0_rung; out['p_y0'] = list(w / w.sum())
    txt = json.dumps(out)
    if a.out: open(a.out, 'w').write(txt + '\n')
    print(json.dumps({k: out[k] for k in ('q_mean', 'q_var')}, indent=0)[:3000])


if __name__ == '__main__':
    main()
