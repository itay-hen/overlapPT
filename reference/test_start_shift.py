"""Stationarity check of the reference kernel with the start-shift move (ptmbar.START_SHIFT).

Single-ensemble chains at the rung (lambda_a, lambda_b) = (0, 1/2), beta = 4, against the exact
absolute probabilities P(q) for q <= 4, from enumerating all 16 * M^q configurations
(normalization N * O, with O from dense exponentiation). Runs the kernel with and without the move.
Usage: python3 test_start_shift.py [CHAINS] [SWEEPS]
"""
import itertools, sys
import numpy as np
from scipy.linalg import expm
from multiprocessing import Pool
import ptmbar as P

beta, la, lb, QE = 4.0, 0.0, 0.5, 4
plus = np.ones(P.DIM) / np.sqrt(P.DIM)
O = float(plus @ expm(-beta * P.H_dense(la) / 2) @ expm(-beta * P.H_dense(lb) / 2) @ plus)
exact = np.zeros(QE + 1)
for q in range(QE + 1):
    for G in itertools.product(range(P.M), repeat=q):
        for y0 in range(P.DIM):
            exact[q] += np.exp(P.logW(P.Cfg(y0, G), la, lb, beta))
exact /= P.DIM * O

def chain(args):
    seed, shift, nsw = args
    P.START_SHIFT = shift
    rng = np.random.default_rng(seed); c = P.Cfg(int(rng.integers(P.DIM)), [])
    lw = P.logW(c, la, lb, beta); h = np.zeros(QE + 1); n = 0
    for t in range(nsw):
        lw = P.sweep(c, la, lb, beta, rng, lw)
        if t >= nsw // 10:
            n += 1
            if len(c.G) <= QE: h[len(c.G)] += 1
    return h / n

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 24; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    print('exact P(q), q=0..4:', np.round(exact, 5), ' sum', round(exact.sum(), 5))
    ok = True
    for shift in (False, True):
        with Pool(min(R, 12)) as p:
            H = np.array(p.map(chain, [(1000 * shift + i, shift, NS) for i in range(R)]))
        m, se = H.mean(0), H.std(0, ddof=1) / np.sqrt(R)
        z = (m - exact) / se; zs = (H.sum(1).mean() - exact.sum()) / (H.sum(1).std(ddof=1) / np.sqrt(R))
        good = np.abs(z).max() < 3.5 and abs(zs) < 3
        ok &= good
        print(f'start_shift={shift}: z per q {np.round(z, 2)}  z(P(q<=4)) {zs:+.2f}  ' + ('PASS' if good else 'FAIL'))
    sys.exit(0 if ok else 1)
