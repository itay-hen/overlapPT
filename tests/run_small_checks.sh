#!/bin/bash
# Exact checks of the OVERLAP_PLUS weight and sampler on a 2x2 lattice (= 4-site ring),
# J=1, Gamma=0.8, one vertical cut (2 bonds), beta=2 (tau=1 per segment). A few minutes.
# Uses the SAME compiler flags as the production build (common_flags.sh). Exits non-zero
# if a check fails.
#
#  1. enum_test: the sum of the code's weight over ALL open walks with q <= Q, for every
#     y0, must increase with Q and approach dim * O(lam_a, lam_b) (exact diagonalization)
#     from below.
#  2. hist_test: R independent single-replica chains at the (0,1) rung (s=0.5, the cross
#     term), production move set, must reproduce the exact P(q <= 5) within 3 standard
#     errors of the mean over chains.
#  3. the same chains must reproduce the exact marginal of the walk's start y0
#     (every state within a t-quantile, Bonferroni-corrected). This catches a frozen y0.
#  4. the checkpoint key separates H2 tables that differ only in which bond is kept.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
source "$SRC/common_flags.sh"
R=${R:-8}                 # independent chains for check 2
STEPS=${STEPS:-4000000}   # updates per chain
M="$SRC/models/2x2_check"
python3 "$SRC/tools/make_model.py" --Lx 2 --Ly 2 --J 1 --Gamma 0.8 --vcut 1 --out "$M" >/dev/null
MODEL_FLAGS=(-DOVERLAP_PLUS -DBeta=2 -DMBAR_N_BETA=5 -Dqmax=64)
"$SRC/build_overlap.sh" "$M" "${MODEL_FLAGS[@]}" >/dev/null 2>&1
cd "$M/build"
for t in enum_test hist_test; do
  g++ -O2 -w "${OVERLAP_BASE_FLAGS[@]}" -DSERIAL_MPI "${MODEL_FLAGS[@]}" -I . -I "$SRC/serial" -o $t "$SRC/tests/$t.cpp"
done
# 4. checkpoint key: two H2 tables that differ only in WHICH bond carries the X-term
#    must hash differently (they collided before the generator index was hashed).
mkdir -p hashchk
echo "-0.8 1 X 2 X" > hashchk/Ha.txt; echo "-0.8 1 X 3 X" > hashchk/Hb.txt
HASHES=()
for v in a b; do
  python3 "$SRC/tools/make_tables.py" hamiltonian.hpp "$M/H1.txt" hashchk/H$v.txt > hamiltonian2.hpp   # quoted includes resolve here
  g++ -O0 -w "${OVERLAP_BASE_FLAGS[@]}" -DSERIAL_MPI "${MODEL_FLAGS[@]}" -I . -I "$SRC/serial" -o hashchk/h$v "$SRC/tests/hash_test.cpp"
  HASHES+=("$(hashchk/h$v)")
done
python3 "$SRC/tools/make_tables.py" hamiltonian.hpp "$M/H1.txt" "$M/H2.txt" > hamiltonian2.hpp   # restore
python3 - "$SRC" "$M" <<'PY' > exact.txt
import sys, numpy as np
from scipy.linalg import expm
src, m = sys.argv[1], sys.argv[2]
sys.path.insert(0, src + '/tools')
from ed_overlap import load
H = [load(f'{m}/H{k}.txt', 4) for k in (1, 2)]
plus = np.ones(16) / 4
for la, lb in ((0, 0.5), (0, 1)):
    Ha = (1 - la) * H[0] + la * H[1]; Hb = (1 - lb) * H[0] + lb * H[1]
    print(np.log(plus @ expm(-Ha) @ expm(-Hb) @ plus))
# exact marginal of the walk's start y0 at the (0,1) rung: segment [0,c) (H2) acts on y0 first
py = plus @ expm(-H[0]) @ expm(-H[1]); py /= py.sum()
np.savetxt('exact_y0.txt', py)
PY
LNO_QUARTER=$(sed -n 1p exact.txt); LNO_MID=$(sed -n 2p exact.txt)
for Q in 4 6 8; do ./enum_test $Q 0.25; ./enum_test $Q 0.5; done > enum.txt
ZTOT=$(python3 -c "import math; print(16*math.exp($LNO_MID))")
seq 1 $R | xargs -P 4 -I{} sh -c "./hist_test 0.5 $ZTOT 5 $STEPS > hist_{}.txt"
python3 - "$LNO_QUARTER" "$LNO_MID" "$R" "${HASHES[0]}" "${HASHES[1]}" <<'PY'
import sys, re, glob, math, statistics as st
ex = {0.25: float(sys.argv[1]), 0.5: float(sys.argv[2])}
ok = True
print('== 1. weight enumeration vs exact (must increase with Q and stay below exact)')
rows = [(int(a), float(b), float(c)) for a, b, c in re.findall(r'QMAX=(\d+) s=(\S+) .*ln\(sum/dim\)=(\S+)', open('enum.txt').read())]
for s_ in (0.25, 0.5):
    v = [x for q, s2, x in rows if s2 == s_]
    good = all(b > a for a, b in zip(v, v[1:])) and all(x < ex[s_] for x in v)
    ok &= good
    print(f'   s={s_}: ' + ', '.join(f'{x:.4f}' for x in v) + f'   exact {ex[s_]:.4f}   ' + ('PASS' if good else 'FAIL'))
print('== 2. sampler vs exact P(q<=5) at the (0,1) rung, production move set')
vals = [float(re.search(r'sum<=5 exact=(\S+) sampled=(\S+)', open(f).read()).group(2)) for f in sorted(glob.glob('hist_*.txt'))]
pe = float(re.search(r'sum<=5 exact=(\S+)', open(sorted(glob.glob('hist_*.txt'))[0]).read()).group(1))
m, se = st.mean(vals), st.stdev(vals) / math.sqrt(len(vals))
z = (m - pe) / se
good = abs(z) < 3
ok &= good
print(f'   {len(vals)} chains: mean {m:.5f} +- {se:.5f} (SEM), exact {pe:.5f}, deviation {z:+.2f} SEM   ' + ('PASS' if good else 'FAIL'))
print('== 3. sampler vs exact marginal of the walk start y0 at the (0,1) rung')
import numpy as np
py = np.loadtxt('exact_y0.txt')
S = np.array([[float(x) for x in re.findall(r'y0=\d+ sampled=(\S+)', open(f).read())] for f in sorted(glob.glob('hist_*.txt'))])
mean, sem = S.mean(0), S.std(0, ddof=1) / math.sqrt(len(S))
zs = (mean - py) / np.maximum(sem, 1e-12)
# SEMs come from only R chains, so use the Student-t quantile with a Bonferroni correction over
# the states tested (family-wise false-alarm rate 0.1%). A frozen y0 fails by orders of magnitude.
from scipy.stats import t as student_t
mask = py > 1e-3
zcut = float(student_t.ppf(1 - 0.001 / (2 * mask.sum()), len(S) - 1))
good = bool(np.all(np.abs(zs[mask]) < zcut))
ok &= good
k = int(np.argmax(np.abs(zs)))
print(f'   largest deviation: y0={k} sampled {mean[k]:.5f} +- {sem[k]:.5f}, exact {py[k]:.5f} ({zs[k]:+.2f} SEM; cutoff {zcut:.2f})   ' + ('PASS' if good else 'FAIL'))
print('== 4. checkpoint key distinguishes H2 tables that differ only in the bond index')
good = sys.argv[4] != sys.argv[5]
ok &= good
print(f'   {sys.argv[4]} vs {sys.argv[5]}   ' + ('PASS' if good else 'FAIL'))
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
PY
