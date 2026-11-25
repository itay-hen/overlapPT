#!/usr/bin/env python3
"""Regression tests for tools/plus_bias_check.py's fit validation (external review, 2026-11-25).

An unidentified or invalid MBAR input must abort, not return a fit:
  * two rungs whose samples have disjoint support (relative gauge unidentified);
  * a u-matrix containing NaN;
  * a sample with zero weight at the rung it was drawn at.
A well-posed two-rung problem with known answer must still fit.
Usage: python3 tests/test_plus_bias_check.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import plus_bias_check as pbc

def aborts(u):
    try:
        pbc.mbar_fit(u, u.shape[1])
    except SystemExit:
        return True
    return False

K, n = 2, 50
dis = np.full((n, K, K), 1e300); dis[:, 0, 0] = 0.0; dis[:, 1, 1] = 0.0
nan = np.zeros((n, K, K)); nan[3, 1, 0] = np.nan
own = np.zeros((n, K, K)); own[5, 1, 1] = 1e300
ok = []
ok.append(('disjoint supports abort', aborts(dis)))
ok.append(('NaN aborts', aborts(nan)))
ok.append(('zero own-rung weight aborts', aborts(own)))
# Well posed: u_1(x) = u_0(x) + c for every x, so f_1 - f_0 = c exactly.
rng = np.random.default_rng(1); c = 0.7
x = rng.normal(size=(n, K)); good = np.stack([x, x + c], axis=2)
try:
    _, f = pbc.mbar_fit(good, K)
    ok.append(('constant shift recovered', abs((f[1] - f[0]) - c) < 1e-8))
except SystemExit as e:
    ok.append((f'constant shift recovered ({e})', False))
for name, passed in ok:
    print(('PASS ' if passed else 'FAIL ') + name)
sys.exit(0 if all(p for _, p in ok) else 1)
