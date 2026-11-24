#!/usr/bin/env python3
"""Exact OVERLAP_PLUS reference by sparse exponential actions (no diagonalization).

For each distinct lambda on the ladder (0,0) -> (0,1) -> (1,1), H(lam) = (1-lam) H1 + lam H2,
computes v_lam = exp(-tau (H(lam) - s_lam)) |+>, tau = beta/2, with scipy's expm_multiply and a
shift s_lam (the lowest eigenvalue, from Lanczos) for numerical range. Then

    ln O(lam_a, lam_b) = ln <+| e^{-tau H(lam_a)} e^{-tau H(lam_b)} |+> = ln(v_a . v_b) - tau (s_a + s_b)

(H is real symmetric). Prints the same 'plus' record as tools/ed_overlap.py: lnF, F and, with
--rungs K, the MBAR-gauge free energies f_k = ln O(last rung) - ln O(rung k).
Memory is a few vectors of length 2^N plus the sparse Hamiltonians; 4x4 (N=16) takes seconds.

Usage: python3 tools/plus_reference.py MODEL_DIR --beta 8 --rungs 17 [--out FILE.json]
"""
import argparse
import json
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh


def load_sparse(path, N):
    """Same term convention as ed_overlap.load: X-strings and Z-strings, Z = +1 on bit 0."""
    dim = 1 << N
    idx = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    off = {}
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        c = float(tok[0])
        xm = zm = 0
        for i in range(1, len(tok), 2):
            q, p = int(tok[i]) - 1, tok[i + 1].upper()
            if p == 'X':
                xm |= 1 << q
            elif p == 'Z':
                zm |= 1 << q
            else:
                raise SystemExit('only X/Z terms supported')
        if xm and zm:
            raise SystemExit('mixed X/Z terms are not supported')
        if xm:
            off[xm] = off.get(xm, 0.0) + c
        else:
            par = np.zeros(dim, dtype=np.int64)
            m = zm
            while m:
                b = m & -m
                par ^= (idx & b) != 0
                m ^= b
            diag += c * (1 - 2 * par)
    rows, cols, vals = [idx], [idx], [diag]
    for xm, c in off.items():
        rows.append(idx ^ xm); cols.append(idx); vals.append(np.full(dim, c))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--beta', type=float, nargs='+', required=True)
    ap.add_argument('--rungs', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args()
    N = json.load(open(os.path.join(a.model, 'model.json')))['N']
    H = [load_sparse(os.path.join(a.model, f'H{k}.txt'), N) for k in (1, 2)]
    plus = np.ones(1 << N) / np.sqrt(1 << N)
    K = a.rungs if a.rungs >= 3 else 3
    lam_pairs = []
    for k in range(K):
        s = k / (K - 1)
        lam_pairs.append((0.0, 2 * s) if s <= 0.5 else (2 * s - 1, 1.0))
    lams = sorted({x for pr in lam_pairs for x in pr})
    shift = {}
    Hl = {}
    for lam in lams:
        Hl[lam] = ((1 - lam) * H[0] + lam * H[1]).tocsr()
        shift[lam] = float(eigsh(Hl[lam], k=1, which='SA', return_eigenvectors=False)[0])
    out = []
    for beta in a.beta:
        tau = beta / 2
        v = {lam: expm_multiply(-tau * (Hl[lam] - shift[lam] * sp.identity(1 << N, format='csr')), plus)
             for lam in lams}
        lnO = lambda la, lb: float(np.log(v[la] @ v[lb]) - tau * (shift[la] + shift[lb]))
        l11, l12, l22 = lnO(0.0, 0.0), lnO(0.0, 1.0), lnO(1.0, 1.0)
        r = dict(beta=beta, lnF=2 * l12 - l11 - l22, F=float(np.exp(2 * l12 - l11 - l22)))
        if a.rungs >= 3:
            lo = [lnO(la, lb) for la, lb in lam_pairs]
            r['f'] = [lo[-1] - x for x in lo]
        out.append(r)
    res = {'N': N, 'method': 'sparse expm_multiply', 'plus': out}
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
