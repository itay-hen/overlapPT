#!/bin/bash
# Sampler vs exact walk-length distribution P(q), rung by rung, at production beta.
#
#   tests/run_qdist.sh MODEL_DIR BETA QMAX K "j1 j2 ..." R STEPS [extra compiler flags...]
#
# Exact side: tools/walk_sum_exact.py (sum of the weight over ALL open walks, resolved by q,
# via the g^q expansion -- no enumeration, any beta; agrees with tests/enum_test.cpp's
# brute-force partial sums to 10 digits on 2x2). Rung j of a K-rung ladder is s = j/(K-1).
# Sampler side: R independent single-replica chains (tests/qdist_test.cpp, production move
# kernel, no PT) of STEPS updates each at every rung. Reported per rung: <q> exact vs mean
# over chains +- SEM, and the largest |z| over the q bins with exact P(q) > 1e-3, where z
# uses the between-chain standard error of each bin (replication-based, as in the paper).
# Output goes to MODEL_DIR/qdist/.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
source "$SRC/common_flags.sh"
MODEL="$(cd "$1" && pwd)"; BETA=$2; QMAX=$3; K=$4; JS=$5; R=$6; STEPS=$7; shift 7
EXTRA=("$@")
MODEL_FLAGS=(-DOVERLAP_PLUS -DBeta=$BETA -DMBAR_N_BETA=5 -Dqmax=$QMAX)
"$SRC/build_overlap.sh" "$MODEL" "${MODEL_FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} >/dev/null 2>&1
OUT="$MODEL/qdist"; mkdir -p "$OUT"
g++ -O3 -w "${OVERLAP_BASE_FLAGS[@]}" -DSERIAL_MPI "${MODEL_FLAGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
    -I "$MODEL/build" -I "$SRC/serial" -o "$OUT/qdist_test" "$SRC/tests/qdist_test.cpp"
WHICH=$(echo $JS | tr ' ' ',')
[ -f "$OUT/exact.json" ] || python3 "$SRC/tools/walk_sum_exact.py" "$MODEL" --beta $BETA --rungs $K \
    --which $WHICH --M 512 --out "$OUT/exact.json" > "$OUT/exact.txt"
for j in $JS; do
  s=$(python3 -c "print($j/($K-1))")
  for c in $(seq 1 $R); do echo "$s $j $c"; done
done | xargs -P ${NPAR:-8} -n 3 sh -c "'$OUT/qdist_test' \$0 $STEPS \$((\$1*1000+\$2)) > '$OUT/h_j'\$1'_c'\$2'.txt'"
python3 - "$OUT" <<'PY'
import sys, json, glob, re, numpy as np
out = sys.argv[1]
ex = json.load(open(f'{out}/exact.json'))
for r in ex['rung']:
    j = r['j']; P = np.clip(np.array(r['P']), 0, None)
    fs = sorted(glob.glob(f'{out}/h_j{j}_c*.txt'))
    H = []
    for f in fs:
        h = np.zeros(len(P))
        for ln in open(f):
            if ln.startswith('#'): continue
            q, n = ln.split(); h[int(q)] += float(n)
        H.append(h / h.sum())
    H = np.array(H); R = len(H)
    q = np.arange(len(P))
    m = H @ q
    mean, sem = m.mean(), m.std(ddof=1) / np.sqrt(R)
    sel = P > 1e-3
    pm, ps = H.mean(0), H.std(0, ddof=1) / np.sqrt(R)
    z = (pm[sel] - P[sel]) / np.where(ps[sel] > 0, ps[sel], np.inf)
    tv = 0.5 * np.abs(pm - P).sum()
    print(f"rung {j:2d} s={r['s']:.4f}  <q> exact {r['mean_q']:.3f}  sampled {mean:.3f} +- {sem:.3f} "
          f"(z={(mean - r['mean_q']) / sem:+.2f})  max|z|_bins={np.abs(z).max():.2f} over {sel.sum()} bins  TV={tv:.4f}  chains={R}")
PY
