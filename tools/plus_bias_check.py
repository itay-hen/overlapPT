#!/usr/bin/env python3
"""Offline checks on kept OVERLAP_PLUS samples: MBAR re-fits and the exact-score test.

Input: run directories containing the kept reduced-potential matrix written with
-DMBAR_KEEP_SAMPLES=1 (renyi_samples_a2_r0.bin; with --partial, an unfinished run's
renyi_ustream_a2_r0.bin, truncated to whole samples). Layout: sample-major, then the rung
the configuration was drawn at, then u = -ln W at all K rungs (1e300 = zero weight).

Reports
  1. each run's MBAR ln F (should reproduce overlap_plus.dat) and the mean over runs;
  2. one MBAR fit of all runs pooled;
  3. ln F from contiguous pieces of 1/m of each run (m = 1, 2, 4, 8). Each run's piece
     estimates are averaged first, and the error is over runs (pieces of one chain are not
     independent). A finite-sample bias grows roughly in proportion to m.
  4. with --exact-f, the MBAR score at the EXACT free energies:
         g_j = sum over time slices of [ sum over the K configurations of p_j(x) ] - n,
         p_j(x) = exp(f_j - u_j(x)) / sum_k exp(f_k - u_k(x)).
     E[g] = 0 whenever every rung's marginal is correct, even for correlated PT chains.
     The score is converted to the first-order ln F error it implies (one Newton step from
     the exact f) and reported per time block (position in the run) with errors over runs.
     A score that drifts toward zero indicates equilibration; a stable nonzero score
     indicates a sampling or weight-evaluation mismatch; a zero score together with an ln F
     offset indicates finite-sample fitting bias.

ln F = f[0] + f[K-1] - 2 f[mid] with f_k = -ln Z_k (gauge fixed by f[K-1] = 0).

Usage: python3 tools/plus_bias_check.py RUN_DIR... --rungs K [--exact LNF]
           [--exact-f REF.json [--ref-beta 8]] [--blocks 4] [--partial]
"""
import argparse, glob, json, os
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def load(run, K, partial):
    fs = sorted(glob.glob(os.path.join(run, 'renyi_samples_a*_r*.bin')))
    if not fs and partial:
        fs = sorted(glob.glob(os.path.join(run, 'renyi_ustream_a*_r*.bin')))
    if len(fs) != 1:
        raise SystemExit(f'{run}: expected one kept-sample file, found {fs}')
    a = np.fromfile(fs[0], dtype=np.float64)
    n = a.size // (K * K)
    if a.size != n * K * K and not partial:
        raise SystemExit(f'{fs[0]}: size is not a multiple of K*K')
    return a[:n * K * K].reshape(n, K, K)  # [sample, sampling rung, evaluated rung]


def logp(U, f):
    """log posterior rung probabilities for rows U (n x K) at free energies f."""
    a = f[None, :] - U
    return a - logsumexp(a, axis=1)[:, None]


def validate(u, K):
    """Abort unless u is a usable MBAR input: no NaN or -inf; every configuration has
    nonzero weight at the rung it was drawn at; and the rungs are connected, i.e. the
    graph with an edge i-j whenever a sample drawn at i has nonzero weight at j (or the
    reverse) is connected. Without connectivity the relative free energies of the
    components are not identified, and a zero score does not mean a valid fit."""
    if np.isnan(u).any() or np.isneginf(u).any():
        raise SystemExit('u-matrix contains NaN or -inf')
    own = u[:, np.arange(K), np.arange(K)]
    if not np.all(own < 1e299):
        bad = np.unique(np.nonzero(~(own < 1e299))[1])
        raise SystemExit(f'zero or non-finite weight at the sampling rung itself (rungs {bad.tolist()})')
    A = (u < 1e299).any(axis=0)          # A[i, j]: some sample drawn at i has support at j
    A = A | A.T
    seen = {0}; todo = [0]
    while todo:
        i = todo.pop()
        for j in np.nonzero(A[i])[0]:
            if j not in seen:
                seen.add(int(j)); todo.append(int(j))
    if len(seen) != K:
        raise SystemExit(f'rungs not connected by sample support: reached {sorted(seen)} from rung 0')


def mbar_fit(u, K):
    """u: [n, K, K] -> (ln F, f) with f[K-1] = 0. Aborts if the input is invalid
    (see validate) or the fit does not converge."""
    validate(u, K)
    U = np.where(u.reshape(-1, K) > 1e299, np.inf, u.reshape(-1, K))
    n = u.shape[0]

    def obj(x):
        f = np.append(x, 0.0)
        a = f[None, :] - U
        ld = logsumexp(a, axis=1)
        g = np.exp(a - ld[:, None]).sum(0) - n
        return ld.sum() - n * f.sum(), g[:-1]

    r = minimize(obj, np.zeros(K - 1), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=20000, gtol=1e-10 * n, ftol=1e-16))
    f = np.append(r.x, 0.0)
    resid = np.abs(obj(r.x)[1]).max() / n
    if not (np.all(np.isfinite(f)) and np.isfinite(resid)) or resid > 1e-6:
        raise SystemExit(f'MBAR fit did not converge: max |score|/n = {resid:.2e} ({r.message})')
    mid = (K - 1) // 2
    return f[0] + f[-1] - 2 * f[mid], f


def score_terms(u, K, f):
    """Per-slice score contributions g_t (n x K) and the summed Hessian at f."""
    validate(u, K)
    n = u.shape[0]
    U = np.where(u.reshape(-1, K) > 1e299, np.inf, u.reshape(-1, K))
    P = np.exp(logp(U, f))                      # (n*K) x K
    g = P.reshape(n, K, K).sum(1) - 1.0          # per slice: sum over its K configurations
    Hs = np.diag(P.sum(0)) - P.T @ P
    return g, Hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--rungs', type=int, required=True)
    ap.add_argument('--exact', type=float); ap.add_argument('--exact-f')
    ap.add_argument('--ref-beta', type=float, default=8.0)
    ap.add_argument('--blocks', type=int, default=4); ap.add_argument('--partial', action='store_true')
    a = ap.parse_args(); K = a.rungs; mid = (K - 1) // 2; R = len(a.runs)
    data = [load(r, K, a.partial) for r in a.runs]
    nmin = min(u.shape[0] for u in data)
    if a.partial:
        data = [u[:nmin] for u in data]
    ex = a.exact
    dev = (lambda x: f'  ({x - ex:+.4f})') if ex is not None else (lambda x: '')
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))

    print(f'== 1. per-run MBAR ({R} runs, {nmin} samples per rung)')
    full = np.array([mbar_fit(u, K)[0] for u in data])
    for r, l in zip(a.runs, full):
        print(f'   {r}: lnF={l:.5f}{dev(l)}')
    print(f'   mean {full.mean():.5f} +- {sem(full):.5f}{dev(full.mean())}')

    print('== 2. pooled fit of all runs')
    pooled = mbar_fit(np.concatenate(data, axis=0), K)[0]
    print(f'   {pooled:.5f}{dev(pooled)}')

    print('== 3. pieces of 1/m of each run (piece estimates averaged per run; error over runs)')
    for m in (1, 2, 4, 8):
        per_run = np.array([np.mean([mbar_fit(b, K)[0] for b in np.array_split(u, m, axis=0)]) for u in data])
        print(f'   m={m}: {nmin // m} samples/piece  mean {per_run.mean():.5f} +- {sem(per_run):.5f}{dev(per_run.mean())}')

    if a.exact_f:
        ref = json.load(open(a.exact_f))['plus']
        ref = [x for x in ref if abs(x['beta'] - a.ref_beta) < 1e-9][0]
        fx = np.array(ref['f']); fx = fx - fx[-1]
        if len(fx) != K:
            raise SystemExit('reference has a different number of rungs')
        print(f'== 4. MBAR score at the exact free energies (ln F exact {ref["lnF"]:.5f})')
        c = np.zeros(K); c[0] += 1; c[-1] += 1; c[mid] -= 2
        B = a.blocks
        G = np.zeros((R, B, K)); Hsum = np.zeros((K, K)); ntot = 0
        for i, u in enumerate(data):
            g, Hs = score_terms(u, K, fx)
            Hsum += Hs; ntot += u.shape[0]
            for b, gb in enumerate(np.array_split(g, B, axis=0)):
                G[i, b] = gb.mean(0)            # mean score per slice in this block
        Hn = Hsum / ntot                          # Hessian per slice
        Hr = Hn[:-1, :-1]
        # one Newton step from the exact f (gauge f[K-1] = 0): delta = -H^{-1} g, ln F shift = c . delta
        shift = lambda gs: float(c[:-1] @ np.linalg.solve(Hr, -gs[:-1]))
        S_rb = np.array([[shift(G[i, b]) for b in range(B)] for i in range(R)])
        S_r = np.array([shift(G[i].mean(0)) for i in range(R)])
        if not np.all(np.isfinite(G)):
            raise SystemExit('non-finite score at the exact free energies')
        print(f'   implied first-order ln F error, whole runs: {S_r.mean():+.5f} +- {sem(S_r):.5f}'
              f'   (MBAR fits above: {full.mean() - ref["lnF"]:+.5f})')
        print('   by position in run: ' + '  '.join(f'{S_rb[:, b].mean():+.4f}+-{sem(S_rb[:, b]):.4f}' for b in range(B)))
        gm = G.mean(1); gz = gm.mean(0) / (gm.std(0, ddof=1) / np.sqrt(R))
        print('   per-rung mean score / SEM over runs: ' + ' '.join(f'{z:+.1f}' for z in gz))
        # Not a chi^2: the per-rung scores are correlated, and with R runs the run-level
        # covariance has rank <= R-1, so no proper K-1 dimensional test is available.
        print(f'   sum of squared per-rung z (descriptive only; correlated, not chi2): {np.sum(gz[:-1] ** 2):.1f}')


if __name__ == '__main__':
    main()
