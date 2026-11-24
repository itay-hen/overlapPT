#!/usr/bin/env python3
"""Exact-diagonalization reference for the overlap runs (small N only, N <= ~14).

Reads MODEL_DIR/H1.txt and H2.txt (X/Z Pauli-product terms) and prints:
  * ground energies, spectral gaps, and ground-state degeneracies of H1 and H2,
  * the exact squared ground-state overlap |<g1|g2>|^2,
  * for each requested tau: exact ln Z_ab(tau) = ln Tr[e^{-tau H_a} e^{-tau H_b}]
    for ab = 11, 12, 22, and the finite-tau ratio R(tau) = Z_12 / sqrt(Z_11 Z_22)
    that the Monte Carlo estimates (R -> |<g1|g2>|^2 as tau -> infinity),
  * the closed-form anchor used by the C++ driver vs the exact value at tau_min.

  * for each --beta: the |+> finite-projection fidelity F_beta and ln F_beta
    (the OVERLAP_PLUS estimate), and with --rungs K the exact MBAR-gauge free
    energies f_k = ln O(last rung) - ln O(rung k) along the interpolation ladder
    (0,0) -> (0,1) -> (1,1), H(lam) = (1-lam) H1 + lam H2.

Usage: python3 tools/ed_overlap.py MODEL_DIR --tau 1 2 4 8 [--tau-min-frac 0.01]
                                   [--beta 4 8 --rungs 17]
"""
import argparse
import json
import os
import numpy as np


def load(path, N):
    dim = 1 << N
    idx = np.arange(dim)
    H = np.zeros((dim, dim))
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
        # Z eigenvalue +1 for bit 0, -1 for bit 1 (any consistent choice works for
        # products appearing an even number of times along closed paths; ZZ/XX here)
        parity = np.array([bin(z & zm).count('1') & 1 for z in range(dim)])
        zval = 1 - 2 * parity
        H[idx ^ xm, idx] += c * zval  # (Z_S X_T)|z> = zval(z ^ xm) ... X-only or Z-only terms
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--tau', type=float, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--tau-min-frac', type=float, default=0.01)
    ap.add_argument('--beta', type=float, nargs='*', default=[])
    ap.add_argument('--rungs', type=int, default=0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.model, 'model.json')))
    N = meta['N']
    H = [load(os.path.join(a.model, f'H{k}.txt'), N) for k in (1, 2)]
    for h in H:
        assert np.allclose(h, h.T)
    E, V = zip(*[np.linalg.eigh(h) for h in H])
    out = {'model': meta}
    for k in range(2):
        e = E[k]
        deg = int(np.sum(np.abs(e - e[0]) < 1e-9))
        out[f'H{k+1}'] = dict(E0=e[0], gap=e[deg] - e[0], degeneracy=deg)
    # Ground-state overlap. With a unique ground state on both sides this is |<g1|g2>|^2.
    # If either ground space is degenerate the eigenvector choice is arbitrary, so report
    # instead the target the |+> estimator converges to: the normalized projections of |+>
    # onto the two ground spaces (gsOverlap v19 Sec. III.A). Both are always reported.
    plus = np.ones(2 ** N) / np.sqrt(2 ** N)
    proj = []
    for k in range(2):
        deg = out[f'H{k+1}']['degeneracy']
        G = V[k][:, :deg]
        v = G @ (G.T @ plus)
        nv = np.linalg.norm(v)
        proj.append(v / nv if nv > 1e-12 else None)
    unique = out['H1']['degeneracy'] == 1 and out['H2']['degeneracy'] == 1
    out['overlap_sq'] = float((V[0][:, 0] @ V[1][:, 0]) ** 2) if unique else None
    out['overlap_sq_plus_target'] = (float((proj[0] @ proj[1]) ** 2)
                                     if proj[0] is not None and proj[1] is not None else None)
    if not unique:
        out['note'] = ('degenerate ground space: overlap_sq is undefined; overlap_sq_plus_target is the '
                       'squared overlap of the normalized ground-space projections of |+>')
    ov = out['overlap_sq'] if unique else out['overlap_sq_plus_target']
    M = (V[0].T @ V[1]) ** 2          # |<a_n|b_m>|^2

    def lnZ(tau, a_, b_):
        if a_ == b_:
            x = -2 * tau * E[a_]
            m = x.max()
            return m + np.log(np.exp(x - m).sum())
        x = -tau * (E[a_][:, None] + E[b_][None, :])
        m = x.max()
        return m + np.log((np.exp(x - m) * M).sum())

    rows = []
    for tau in a.tau:
        z11, z12, z22 = lnZ(tau, 0, 0), lnZ(tau, 0, 1), lnZ(tau, 1, 1)
        tmin = tau * a.tau_min_frac
        exact_anchor = {p: lnZ(tmin, *ab) for p, ab in (('11', (0, 0)), ('12', (0, 1)), ('22', (1, 1)))}
        tr = [np.trace(h) / 2 ** N for h in H]
        hh = [[np.trace(H[i] @ H[j]) / 2 ** N for j in range(2)] for i in range(2)]
        series = {}
        for p, (i, j) in (('11', (0, 0)), ('12', (0, 1)), ('22', (1, 1))):
            s = 1 - tmin * (tr[i] + tr[j]) + 0.5 * tmin ** 2 * (hh[i][i] + hh[j][j] + 2 * hh[i][j])
            series[p] = N * np.log(2) + np.log(s)
        rows.append(dict(tau=tau, lnZ11=z11, lnZ12=z12, lnZ22=z22,
                         R=float(np.exp(z12 - 0.5 * (z11 + z22))),
                         anchor_err={p: series[p] - exact_anchor[p] for p in series}))
    out['tau'] = rows

    def O(beta, la, lb):
        Ha = (1 - la) * H[0] + la * H[1]
        Hb = (1 - lb) * H[0] + lb * H[1]
        ea, va = np.linalg.eigh(Ha)
        eb, vb = np.linalg.eigh(Hb)
        sa, sb = ea[0], eb[0]
        x = vb @ (np.exp(-beta / 2 * (eb - sb)) * (vb.T @ plus))
        x = va @ (np.exp(-beta / 2 * (ea - sa)) * (va.T @ x))
        return np.log(plus @ x) - beta / 2 * (sa + sb)

    prow = []
    for beta in a.beta:
        l11, l12, l22 = O(beta, 0, 0), O(beta, 0, 1), O(beta, 1, 1)
        r = dict(beta=beta, lnF=2 * l12 - l11 - l22, F=float(np.exp(2 * l12 - l11 - l22)))
        if a.rungs >= 3:
            K = a.rungs
            lam = []
            for k in range(K):
                sv = k / (K - 1)
                lam.append((0.0, 2 * sv) if sv <= 0.5 else (2 * sv - 1, 1.0))
            lnO = [O(beta, la, lb) for la, lb in lam]
            r['f'] = [lnO[-1] - v for v in lnO]
        prow.append(r)
    out['plus'] = prow
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
