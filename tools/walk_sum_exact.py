#!/usr/bin/env python3
"""Exact sum of the OVERLAP_PLUS walk weights, resolved by walk length q -- no enumeration.

The open-walk ensemble at rung s (pair (lam_a, lam_b) of replica.hpp's OvLambdas) has total
weight  sum_{y0} sum_{walks} W = dim * O(lam_a, lam_b),
    O(lam_a, lam_b) = <+| exp(-tau H(lam_a)) exp(-tau H(lam_b)) |+>,  tau = beta/2,
    H(lam) = (1 - lam) H1 + lam H2.
Every walk of length q carries exactly q off-diagonal amplitudes. Scaling all off-diagonal
terms by g therefore multiplies its weight by g^q, so
    O(lam_a, lam_b; g) = sum_q c_q g^q,   c_q = (1/dim) * sum of W over all walks with |S| = q.
The c_q are read off with an FFT of O on the circle |g| = 1 (M points; aliasing from
c_{q+M} is negligible once M >> the support of P(q)). P(q) = c_q / O is the exact
distribution of q that the sampler at that rung must reproduce, and sum_{q<=Q} c_q is
what tests/enum_test.cpp returns (as ln) by brute force.

Energies are shifted by E_ref (lowest eigenvalue of H1) to keep O(1) in range; this
multiplies every c_q by the same factor and is undone in the reported ln O.

Usage:
  python3 tools/walk_sum_exact.py MODEL_DIR --beta B --rungs K [--which 0,8,16] [--M 512]
                                  [--out FILE.json]
MODEL_DIR holds H1.txt and H2.txt (tools/make_model.py). Feasible to N ~ 18.
"""
import argparse, json, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh


def load_split(path, N):
    """Return (diag, offdiag) of the Hamiltonian in prepare.cpp's text format."""
    d = 2 ** N
    idx = np.arange(d)
    diag = np.zeros(d)
    rows, cols, vals = [], [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        c = float(t[0])
        ops = [(int(t[i]) - 1, t[i + 1].upper()) for i in range(1, len(t), 2)]
        if all(o == 'Z' for _, o in ops):
            s = np.ones(d)
            for q, _ in ops:
                s *= 1 - 2 * ((idx >> q) & 1)
            diag += c * s
        elif all(o == 'X' for _, o in ops):
            m = 0
            for q, _ in ops:
                m |= 1 << q
            rows.append(idx ^ m); cols.append(idx); vals.append(np.full(d, c))
        else:
            raise SystemExit(f'{path}: only pure-Z and pure-X terms are supported')
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(d, d)) if vals else sp.csr_matrix((d, d))
    return diag, X


def lambdas(s):
    return (0.0, 2.0 * s) if s <= 0.5 else (2.0 * s - 1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--beta', type=float, required=True)
    ap.add_argument('--rungs', type=int, required=True)
    ap.add_argument('--which', default=None, help='comma-separated rung indices (default: all)')
    ap.add_argument('--M', type=int, default=512)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    N = json.load(open(f'{a.model}/model.json'))['N']
    D1, X1 = load_split(f'{a.model}/H1.txt', N)
    D2, X2 = load_split(f'{a.model}/H2.txt', N)
    H1 = (sp.diags(D1) + X1).astype(float)
    E_ref = float(np.linalg.eigvalsh(H1.toarray())[0]) if N <= 10 else \
        float(eigsh(H1, k=1, which='SA', return_eigenvectors=False)[0])
    dim = 2 ** N
    plus = np.ones(dim) / np.sqrt(dim)
    tau = a.beta / 2
    K = a.rungs
    which = list(range(K)) if a.which is None else [int(x) for x in a.which.split(',')]
    gs = np.exp(2j * np.pi * np.arange(a.M) / a.M)

    out = {'model': a.model, 'N': N, 'beta': a.beta, 'rungs': K, 'M': a.M, 'E_ref': E_ref, 'rung': []}
    for j in which:
        s = j / (K - 1)
        la, lb = lambdas(s)
        Da = (1 - la) * D1 + la * D2 - E_ref
        Db = (1 - lb) * D1 + lb * D2 - E_ref
        Xa = (1 - la) * X1 + la * X2
        Xb = (1 - lb) * X1 + lb * X2
        Og = np.empty(a.M, dtype=complex)
        for m, g in enumerate(gs):
            Hb = (sp.diags(Db) + g * Xb).tocsr()
            Ha = (sp.diags(Da) + g * Xa).tocsr()
            v = expm_multiply(-tau * Hb, plus.astype(complex))   # segment [0,c) acts first
            v = expm_multiply(-tau * Ha, v)
            Og[m] = plus @ v
        c = np.fft.fft(Og).real / a.M        # c_q = (1/M) sum_m O(g_m) g_m^{-q}
        O1 = Og[0].real
        P = c / O1
        q = np.arange(a.M)
        lnO = np.log(O1) - a.beta * E_ref      # undo the shift: exp(-beta (H - E_ref))
        tail = float(np.abs(P[a.M // 2:]).max())
        rec = {'j': j, 's': s, 'lam_a': la, 'lam_b': lb, 'lnO': float(lnO),
               'mean_q': float((q * P).sum()), 'var_q': float((q * q * P).sum() - (q * P).sum() ** 2),
               'sumP': float(P.sum()), 'alias_tail_max': tail,
               'lnc': [float(np.log(abs(x)) - a.beta * E_ref) if x != 0 else None for x in c],   # ln c_q (sign: c_q >= 0 up to FFT noise)
               'P': [float(x) for x in P]}
        out['rung'].append(rec)
        print(f'rung {j:2d} s={s:.4f} (la,lb)=({la:.4f},{lb:.4f})  lnO={lnO:.8f}  <q>={rec["mean_q"]:.4f}  '
              f'var={rec["var_q"]:.4f}  sumP-1={rec["sumP"] - 1:.1e}  tail={tail:.1e}', flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
