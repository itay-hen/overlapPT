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
# Output goes to MODEL_DIR/qdist/. The exact reference is cached as exact_<key>.json, where
# <key> hashes H1.txt, H2.txt, beta, K, the requested rungs and the series order, and the
# file's own "key" field is checked before use. Histograms go to a fresh per-invocation
# directory, and only that directory is aggregated, so stale files cannot enter.
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
M=512
KEY=$(python3 - "$MODEL" "$BETA" "$K" "$WHICH" "$M" <<'PY'
import sys, hashlib
model, beta, K, which, M = sys.argv[1:]
h = hashlib.sha256()
for f in ('H1.txt', 'H2.txt'):
    h.update(open(f'{model}/{f}', 'rb').read())
h.update(f'beta={float(beta)!r} K={int(K)} which={which} M={int(M)}'.encode())
print(h.hexdigest()[:16])
PY
)
REF="$OUT/exact_$KEY.json"
if [ ! -f "$REF" ]; then
  python3 "$SRC/tools/walk_sum_exact.py" "$MODEL" --beta $BETA --rungs $K \
      --which $WHICH --M $M --out "$REF.tmp" > "$OUT/exact_$KEY.txt"
  python3 - "$REF.tmp" "$KEY" "$BETA" "$K" "$WHICH" <<'PY'
import sys, json
p, key, beta, K, which = sys.argv[1:]
d = json.load(open(p)); d['key'] = key
d['key_inputs'] = dict(beta=float(beta), K=int(K), which=which)
json.dump(d, open(p, 'w'))
PY
  mv "$REF.tmp" "$REF"
fi
RUNDIR=$(mktemp -d "$OUT/hist_XXXXXX")
for j in $JS; do
  s=$(python3 -c "print($j/($K-1))")
  for c in $(seq 1 $R); do echo "$s $j $c"; done
done | xargs -P ${NPAR:-8} -n 3 sh -c "'$OUT/qdist_test' \$0 $STEPS \$((\$1*1000+\$2)) > '$RUNDIR/h_j'\$1'_c'\$2'.txt'"
python3 - "$REF" "$KEY" "$RUNDIR" "$JS" "$R" <<'PY'
import sys, json, numpy as np
ref, key, hdir, js, R0 = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split()], int(sys.argv[5])
ex = json.load(open(ref))
if ex.get('key') != key:
    raise SystemExit(f'{ref}: key {ex.get("key")} does not match this invocation ({key})')
if sorted(r['j'] for r in ex['rung']) != sorted(js):
    raise SystemExit(f'{ref}: rungs {[r["j"] for r in ex["rung"]]} differ from the requested {js}')
print(f'reference {ref} (key {key}); histograms {hdir}')
for r in ex['rung']:
    j = r['j']; P = np.clip(np.array(r['P']), 0, None)
    fs = [f'{hdir}/h_j{j}_c{c}.txt' for c in range(1, R0 + 1)]
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
