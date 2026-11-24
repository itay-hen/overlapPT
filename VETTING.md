# overlapPT — notes for external vetting

Ground-state overlaps |<g1|g2>|^2 with PMR-QMC + parallel tempering + MBAR, adapted
from the November 2026 Rényi-entropy code (`mbarptNov`). The method is described in
the manuscript "Calculating ground-state overlaps with parallel tempering" (gsOverlap
v19). Section and equation numbers below refer to v19.

**Status: partially validated.** Correct on every exact check at 4 qubits. The
|+> open-walk mode, after the start-shift fix, lands 1.6 SEM low at 12 qubits (result 6)
and 2.1 SEM low at 16 qubits (result 7). Each is acceptable alone, but together they hint
at a small negative bias, roughly 5-10% in F, that is not yet explained (see "Open items").
Please treat it as code under review.

## What to look at

Relative to the original code, the physics changes are confined to `replica.hpp`,
`mainQMC.hpp`, `parameters.hpp` and `mbarpt_mpi.cpp`; `mbar_common.hpp` adds the extended
checkpoint key; `divdiff.hpp` only gains a bounds guard; `prepare.cpp` is unchanged. Everything new is under
`#ifdef OVERLAP_MODE` / `#ifdef OVERLAP_PLUS`, so the Rényi build is unaffected. A
line-by-line diff against the original tree is available on request.

| where | what | paper |
|---|---|---|
| `replica.hpp`, `OvWeight` | open-walk weight, cut summed analytically, segment [0,c) at H(lam_b), [c,q) at H(lam_a), tau = Beta/2 each, H(lam) = (1-lam)H1 + lam H2 | Eq. (W), Thm. 1, Sec. III.F |
| `replica.hpp`, `OvPrecompute`, `OvEnergyH`, `OvDH` | energies and hopping amplitudes of H1 and H2 along the walk (d evaluated at the post-hop state, as in the original code) | Eqs. (dj), (dprod) |
| `replica.hpp`, `update()` | new single-operator insertion/removal sub-moves and the start-shift pair that moves y0 (Rfactor = Nop and 1/Nop); cyclic rotation disabled for open walks | Eq. (props), Sec. IV.B |
| `replica.hpp`, `CalcCompositeEnergy`, `calc_d_for_config` | per-copy Hamiltonian (trace variant only) | Sec. III.B |
| `mbarpt_mpi.cpp`, `pipeline_stat` | reported statistic ln F = f[0] + f[K-1] - 2 f[mid], used for the point estimate and every bootstrap replicate | Sec. III.F |
| `mbarpt_mpi.cpp`, `main` | interpolation ladder s_k = k/(K-1) -> (lam_a, lam_b); trace-variant branch with closed-form anchor | Eqs. (anchor), (trace) |
| `tools/make_tables.py` | H1 and H2 tables on H1's permutation list; H2 gets d = 0 on removed bonds; rejects positive hopping coefficients | Sec. II (ii), Prop. 2 |
| `mbarpt_mpi.cpp`, after the main MBAR solve | refuses to report a result if the solve did not converge, any f is non-finite, a sample has zero weight at its own rung, or adjacent rungs are disconnected; prints adjacent-rung MBAR overlaps and a bootstrap block-length scan | — |
| `mbar_common.hpp`, `CheckpointKey` | key includes overlap mode, ordered Hamiltonian pair, both Hamiltonians and the actual ladder | — |

Packaging relative to the original tree: the Rényi-only Slurm script `recipe_sweep_job.sh`
and the generated example header `examples/renyi_H32/hamiltonian.hpp` are omitted (the original
tree has them); the single-process MPI stand-in is `serial/serial_mpi.h`, used only when compiled
with `-DSERIAL_MPI`; a few identifiers and comments were renamed (e.g. `hamiltonian_checksum`)
with no change in behaviour.

Points that deserve a skeptical read:
1. **Detailed balance of the composite update with the new single-operator moves.**
   They are added as sub-moves of the existing composite chain (`cont`/`Rfactor`
   logic inherited from the original code). The replicated histogram test below passes
   at 0.1% precision for both the production kernel and a reduced kernel, but only on
   one small system.
2. **Remaining moves on open walks.** Pair insertion/removal, adjacent swaps,
   fundamental-cycle substitution and loop/segment exchange were written for closed
   walks; they are used unchanged. Only the cyclic rotation was found invalid and
   disabled.
3. **Ladder mixing.** See the 4x3 results below.
4. **Error bars.** The per-run bootstrap error is ~10x smaller than the run-to-run
   scatter (also reported in the paper for the MBAR asymptotic error). One mechanism:
   the block length is chosen from the autocorrelation of the own-rung reduced
   potential, which decorrelates faster than the ln F contrast; the bootstrap error then
   grows with block length (the run now prints this scan and warns). Use replication.

## Reproducing

Requirements: a C++17 compiler, Python 3 with numpy/scipy. MPI is optional:
`build_overlap.sh` uses `mpicxx` if present, otherwise the single-process stub in
`serial/serial_mpi.h`, enabled with `-DSERIAL_MPI` (one rank; every rung in one process).

    tests/run_small_checks.sh          # ~3 min: exact checks 1 and 2 below; exits non-zero on failure

    python3 tools/make_model.py --Lx 4 --Ly 3 --J 1 --Gamma 0.8 --vcut 2 --out models/4x3_v2
    ./build_overlap.sh models/4x3_v2 -DOVERLAP_PLUS -DBeta=8 -DMBAR_N_BETA=17 -Dqmax=192 \
        -DMBAR_T_EQUIL=10000 -DMBAR_N_SAMPLES=10000 -DMBAR_KEEP_SAMPLES=0
    mkdir run1 && (cd run1 && ../models/4x3_v2/build/overlap.bin)   # ~1 h per run, single core
    python3 tools/ed_overlap.py models/4x3_v2 --tau 1 --beta 8 --rungs 17   # exact lnF and f_k (~5 min)

Run several independent copies in separate directories and take the mean and standard
error of `lnF` across runs. (The checkpoint key now refuses to resume a different mode,
pair, Hamiltonian or ladder, but one directory per run keeps the files apart.)

## Results so far

Model: J sum ZZ - Gamma sum XX, J=1, Gamma=0.8, open boundaries; H2 removes the bonds
crossing one vertical cut. Exact values from `tools/ed_overlap.py`. Per-run numbers
are in `results/per_run_results.txt`.

1. **Weight vs exhaustive enumeration (2x2, beta=2).** Sum of the code's weight over
   all y0 and all open walks with q <= Q approaches dim * O(lam_a, lam_b) from below:
   ln O(0,1/2): 7.536, 7.884, 8.009 for Q=4,6,8, exact 8.0533;
   ln O(0,1): 6.913, 7.232, 7.332, exact 7.3593. (`tests/enum_test.cpp`)
2. **Sampler vs exact P(q) (2x2, beta=2, the (0,1) rung).** Single-replica chains of
   4x10^6 updates, no PT, 8 independent chains each, exact P(q <= 5) = 0.78235:
   production kernel 0.78312 +- 0.00065 (+1.2 SEM); reduced kernel without cycle,
   loop and segment moves 0.78115 +- 0.00085 (-1.4 SEM). (`tests/hist_test.cpp`.)
   *Correction:* an earlier version of this file quoted single chains run without the
   production flag `SEGMENT_EXCHANGE_MAX_ATTEMPTS=1`, i.e. with the inherited 64-retry
   segment exchange, which violates detailed balance. The test now shares the
   production flags (`common_flags.sh`).
3. **Full PT+MBAR, 2x2 (= the paper's 4-site ring), beta=4, 9 rungs, 6 runs.**
   ln F = -0.316 +- 0.043 (SEM), exact -0.3234 (+0.18 SEM). Run-to-run s.d. 0.105
   (the paper reports 0.107 for the same system); per-run bootstrap ~0.011.
4. **Full PT+MBAR, 4x3, beta=8, 17 rungs, 12 runs of 10^4 samples.**
   ln F = -0.894 +- 0.208, exact -0.5416 (-1.70 SEM); F = 0.409, 95% t-interval
   [0.259, 0.646], exact 0.5818. Run-to-run s.d. 0.72, per-run bootstrap 0.067.
   **Not conclusive.** The runs split by ladder mixing: the 7 runs with 207-256
   round trips give ln F in [-0.65, +0.03]; the 5 with 99-172 give [-2.30, -0.95].
   The mean residual is concentrated on the first leg (0,0) -> (0,1).
   **Long runs (12 x 5x10^4 samples) did not converge either:** ln F = -1.06 +- 0.26,
   range [-3.14, -0.06]; correlation with round trips 0.97. Per-rung comparison with
   ED: the second leg (0,1) -> (1,1) is correct in every run (residual s.d. 0.04); the
   whole error sits on the first leg (residual -0.53 +- 0.86, as large as -2.55).
   **Cause: the walk's start y0 was effectively frozen.** In open mode y0 could change
   only by a single spin flip, which flips that spin along the whole walk and was
   accepted ~10^-4 of the time; the cyclic rotation that moves y0 in closed walks is
   disabled. Across 5x10^4 samples each rung saw only 18-19 distinct y0 values (of
   2048 reachable), i.e. the replicas' initial states shuffled by PT. Each replica
   then samples the ensemble conditioned on its y0, which is a different
   distribution. On 2x2 at beta <= 4 spin flips are accepted often enough, which is
   why the small checks passed.
   **Fix:** a start-shift move pair: prepend P_p and set y0' = P_p y0, or the reverse,
   which keeps every old intermediate state (proposal ratio Nop). A third exact
   check now compares the sampled marginal of y0 with ED.
6. **4x3 with the start-shift move, beta=8, 17 rungs, 12 runs of 10^4 samples.**
   ln F = -0.598 +- 0.034, exact -0.5416 (-1.64 SEM); F = 0.550 vs 0.5818. Run-to-run
   s.d. 0.119 (was 0.72), per-run bootstrap 0.05-0.09; no correlation with round trips
   (r = 0.04). Every rung's mean free energy agrees with ED within 1.2 SEM; both legs
   scatter symmetrically. Each rung now visits ~900 distinct y0 values in 10^4 samples
   (was 18-19 in 5x10^4). These runs were built before the second-review changes
   below; the speed-ups reproduce the same trajectory and the bootstrap change only
   affects the reported per-run error. Logs: `models/4x3_v2/runs_plus_b8_shift/`
   (and the frozen-y0 long runs in `runs_plus_b8_long/`).
7. **4x4 (both midline cuts, 8 bonds severed), start-shift move, beta=8, 25 rungs,
   12 runs of 2x10^4 samples, qmax=160.** ln F = -1.654 +- 0.055 (run-to-run s.d. 0.191),
   exact F_beta: -1.5389 (-2.08 SEM); F = 0.191 +0.011/-0.010 vs 0.2146 (ground-state limit
   0.214617). 36-49 PT round trips per run, no significant correlation with ln F
   (r = -0.30, n = 12). Minimum adjacent MBAR overlap ~0.11 (rungs 15-16); q_99.9 = 99.
   Every run has stderr_status 1: the bootstrap error grows ~4x from block 28 to block 1750,
   so the per-run errors (0.16-0.40 at the largest block) are not usable, and only the
   run-to-run spread is quoted. Per rung, relative to the middle rung: leg 1 (f0 - f12)
   is -1.3 SEM off and leg 2 (f12 - f24) +0.7 SEM off; both push ln F down, and the leg-1
   residual drifts smoothly from 0 at the middle rung to -1.3 SEM at rung 0. The distinct
   lattice states per rung fall from ~9000 on leg 1 to ~2800 on leg 2. Each run took
   ~3 h on one core (12 in parallel on one laptop). Logs: `models/4x4_v2/runs_plus_b8/`;
   reference: `models/4x4_v2/plus_ref_b8_k25.json`.
5. **Trace variant (closed walks summed over y0), 4x3, beta=4, 5 runs per ensemble.**
   (1/2) ln R = -0.5340 +- 0.0493, exact -0.5348 (+0.02 SEM); one of the three
   anchored free energies, ln Z11, is +2.5 SEM off. Paper Table tab:cpp.

## Fixes after the code review of 2026-11-22

Each item was confirmed in the source before fixing; none changes a previously
reported number (all 33 reported runs converged at MBAR iterations 7-63, none was
resumed from a checkpoint, and the trace runs had short walks).

| issue | fix | verification |
|---|---|---|
| Trace mode: hopping products in `std::complex<double>` underflow to an exact 0 that the `prefix_unsafe` guard does not catch | extended-exponent prefix/suffix products in overlap mode | H1=H2=-1e-5 XX, tau=4e6, q=80: previous code returns 0, fixed code log10 W = 33.392471 = exact log10(80^80/80!) |
| Checkpoint key omitted mode, pair, H2 and ladder | added to the key; format version 4 | — |
| MBAR could report non-converged, non-finite or disconnected results | solver returns convergence status; post-solve checks abort before any result is written | a two-rung ladder (beta 1e-4 and 8) is refused: overlap 0.0000, exit code 3, no result file |
| Tests did not use the production flags; no tolerance | shared `common_flags.sh`; 8 replicated chains, 3-SEM tolerance; enumeration checked for monotone approach from below | passes (see result 2) |
| Progress diagnostic printed f[0] in open mode | prints ln F | — |
| `ExExFloat` table lookup out of bounds for exponent gaps >= 100000 | bounds-checked lookup | — |
| No sign check on hopping coefficients | `make_tables.py` rejects positive X-string coefficients | rejects `0.5 X X` |
| ED helper picked arbitrary eigenvectors for degenerate ground spaces | reports the |+>-projected target in that case | 3x3 example: 0.51256 |

## Fixes and speed-ups after the second code review (archive d)

| issue | change | verification |
|---|---|---|
| Checkpoint key: two H2 tables differing only in which bond is kept hashed identically (the off-diagonal Z-masks are all zero, and the generator index was not hashed) | hash includes the Hamiltonian index, generator index and per-generator term counts; checkpoint format 5 | archive d: `-0.8 X1X2` and `-0.8 X1X3` give the same hash; now different (check 4 of `run_small_checks.sh`) |
| Bootstrap error included failed or disconnected replicates; the exported error ignored the block-length scan | a replicate counts only if its solve converged, its statistic is finite and adjacent rungs stay connected (overlap >= 1e-6); if any fails the exported `stderr` is `nan`; if the scan grows by > 30% the exported `stderr` is the largest scanned value. The result files gain `stderr_status` (0 ok, 1 scan-grown, 2 unavailable) and `stderr_default_block` | — |
| `OvWeight` recomputed both half-weights at every rung | prefix/suffix half-weights cached per walk for lam = 0 and 1 (leg 1 shares the H1 suffix, leg 2 the H2 prefix); passes stop at the first exactly-zero hop | 3000 random 4x3 walks x 17 rungs: identical zero pattern, max \|d ln W\| = 5.7e-14 against archive d; 2.2x less time in `OvWeight` |
| (both speed-ups together) | — | short 4x3 run (beta=8, 17 rungs, 2000 samples, same seed): 668 s -> 445 s (1.50x), identical ln F, i.e. the same trajectory |
| `ExExFloat::normalize` called `frexp` | exponent read from the IEEE-754 bits for finite normal values, `frexp` otherwise | bitwise identical to `frexp` on 2x10^7 random bit patterns plus zeros, subnormals, inf, nan |
| Exact open-walk references needed dense diagonalization | `tools/plus_reference.py`: sparse `expm_multiply` on \|+> per distinct lambda | 4x3: all 17 f_k within 1.4e-13 of `ed_overlap.py`, 0.9 s. 4x4 with both midline cuts, beta=8: F = 0.2146230061, 10 s (`models/4x4_v2/plus_ref_b8.json`) |

## Open items

* **Possible small negative bias in the open-walk mode** (focused write-up: `BIAS_REVIEW.md`). 4x3 is -1.64 SEM and 4x4 is
  -2.08 SEM; the pooled deviation is about -2.6 sigma. Candidates, none tested yet:
  (a) a slow mode not removed by 10^4 equilibration sweeps; the block-length growth of
  the bootstrap error shows such a mode exists. Test: longer equilibration, or keep the
  samples and compare the halves. (b) a detailed-balance error in a move that only shows
  at larger q; check 3 tests the y0 marginal only on 2x2. Test: the same marginal and
  q-distribution checks on 4x3 at one rung against `plus_reference.py`-style exact
  marginals. (c) MBAR finite-sample bias on the long first leg. Test: more rungs on leg 1.
* The 2x2 PT result (3) predates the start-shift move; spin flips mix y0 there, but it
  has not been re-run.
* Multi-rank MPI builds are untested in the overlap modes (all runs used the stub);
  the rung partition uses equal costs in OVERLAP_PLUS.
* `make_tables.py` supports only pure X-string and pure Z-string terms; lattice
  states are packed into 64-bit words for checkpoints and MPI swaps, so N <= 64.
* Odd N with this model: every level is two-fold degenerate, so only the
  trial-state-dependent target is defined; `models/3x3_v2` is kept as an example.
* The trace anchor's truncation error is O(beta_min^3) in general; the O(beta_min^4)
  quoted in the paper requires the trace cancellations stated there.

## Chain benchmarks with exact free-fermion references (2026-11-22, later)

* `tools/chain_freefermion.py`: exact sector-resolved overlaps for the open chain
  J ZZ - Gamma XX cut at its midline, by a Y-string Jordan-Wigner map to two decoupled
  Majorana chains; every sign comes from explicit Pauli-string algebra. Gives F, in-sector
  gaps, p_a = |<+|g_a>|^2, and (Gaussian imaginary-time Riccati flow) the exact finite-beta
  ln O(lam_a, lam_b) on the PT ladder, i.e. the MBAR f_k. Checked against ED for
  N = 4..20 (energies, gaps, p_a, F to ~1e-13; F_beta to ~1e-14). Restricted to N divisible
  by 4 (odd half-chains: prod X of a block spans two Majorana sub-chains; not handled).
  Valid to at least N = 512; N = 1024 fails (zero-mode splitting below double precision).
* Exact mean expansion order <q> = Gamma d ln O / d Gamma agrees with the sampler at N = 16
  (55.2 exact, 55.1-56.2 sampled).
* **Code change: N <= 32 -> N <= 64.** `RenyiReplica::pic_key` packed the B-state into 32 bits
  (static_assert N <= 32; the earlier note "N <= 64" was wrong). It is now a 128-bit key
  (`pic_key_t`, `PicKeyHash`): B-state << 64 | D-zero count; `BClassUpdatePosition` reads
  key >> 64. Verified: with -DEXACTLY_REPRODUCIBLE, patched and unpatched builds give
  byte-identical logs on chain16 (beta=8, 17 rungs, 500+300 steps).
* Models: `models/chain{16,64}` (Gamma=0.8), `models/chain{32,64}_G05` (Gamma=0.5); exact
  references `exact_*.json` in each.

### 2026-11-23: chain benchmarks fail at N >= 32 -- sampler does not equilibrate

Runs (all Gamma=0.5, J=1, open chain, cut at the middle bond, OVERLAP_PLUS):

| N | beta | rungs | T_equil / samples | MC lnF | exact lnF_beta | PT round trips |
|---|------|-------|-------------------|--------|----------------|----------------|
| 16 | 16 | 33 | 4000 / 2000 | -1.41 +- 0.12 | -0.620 | 195 |
| 32 | 8  | 17 | 4000 / 2000 | -2.68 +- 0.15 | -0.700 | 103 |
| 32 | 16 | 33 | 2000 / 400  | -5.73 +- 0.14 | -0.737 | 3 |
| 32 | 16 | 33 | 20000 / 1500| -4.54 +- 0.11 | -0.737 | 5 |
| 64 | 16 | 33 | 2000 / 400  | -9.24 +- 0.10 | -0.745 | 2 |

(The N=16, Gamma=0.8, beta=8 smoke run is fine: -0.40 +- 0.12 vs -0.321.)

What was checked:
* Exact references re-derived by brute-force sparse expm on the C++ input files H1.txt/H2.txt
  (scratch ed_ladder.py, N=16, beta=16): leg1 -6.8447, leg2 -6.2248, lnF -0.61989, identical
  to tools/chain_freefermion.py. The references are right.
* hamiltonian2.hpp tables: H2's zero hop sits on the correct operator (the cut bond).
* OvWeight / OvPrefix / OvSuffix, the slot caches, InvalidateCache on spin flips and on the
  single/start-shift moves, and the single/start-shift proposal ratios read correct.
* N=8, beta=16 with chaining off (COMPOSITE_UPDATE_BREAK_PROBABILITY=1.0, now #ifndef-guarded)
  and with loop/segment exchange off: no significant change; all consistent with exact.
* N=32 patched (128-bit pic_key) vs unpatched replica.hpp at beta=4: same within noise.
  The N<=64 patch is not the cause.

Diagnosis: the per-rung <q> is wrong ALREADY AT RUNG 0 (plain full chain, no zero hops):
exact <q> = Gamma d lnO/dGamma is 42.8 (N=32, beta=8) and 77.0 (N=32, beta=16); MC gives 48-49
and 81.6, and the q-occupancy trace is still relaxing toward the exact value at the end of every
N=32 run (52 -> 47.7; 89.5 -> 87.5). tau_q is 20-64 samples even at beta=4. The equilibration
time of the open-walk sampler grows steeply with N (plausibly diffusive domain-wall annealing of
y0 in the antiferromagnetic chain, flip acceptance ~1%), and the per-run bootstrap error bars
(block length 2-5) grossly underestimate the real error. MBAR then propagates the
non-equilibrium rung distributions into a large, stable-looking bias, worst on leg 2.

Consequence: the code as is cannot produce a trustworthy F at N >= 32 on a laptop budget; the
2^64 benchmark needs a better move set (e.g. cluster/domain flips of y0 with the walk, or
loop updates) and a convergence check against the exact per-rung <q> before any production.

### 2026-11-23 (cont.): fix -- OV_CUT_FLIP block move + OV_INIT_CLASSICAL start

Two opt-in options (parameters.hpp; OV_CUT_MASK generated by tools/make_tables.py as the
component of H2's interaction graph containing the last qubit):
* OV_CUT_FLIP: flip every spin on one side of the cut in y0, hence along the whole walk.
  Self-inverse, symmetric proposal, plain Metropolis; replaces a single spin flip with
  probability OV_CUT_FLIP_PROB (0.5). Acceptance (N=32, beta=8): 1.0 at s=1 (exactly free on
  severed segments), ~0.02 at s=1/2, ~0 at s=0.
* OV_INIT_CLASSICAL: y0 annealed to a minimum of H1's diagonal energy (removes the ~N/2
  walk-spanning domain walls a random y0 seeds).

N=32, Gamma=0.5, beta=8, 17 rungs, T_equil 4000, 2000 samples (one run each unless noted):

| variant | lnF | leg1 (exact -3.649) | leg2 (exact -2.949) | rung-0 <q> (exact 42.85) |
|---|---|---|---|---|
| old | -2.68 | -3.39 | -0.71 | 49.4 |
| init only | +0.01 | -3.53 | -3.53 | 43.3 |
| cut flip only | -0.87 | -3.87 | -3.00 | 46.0 |
| both, 5 replicas | -0.739 +- 0.067 (sd 0.150) | -3.67 | -2.93 | 42.6 |

exact lnF_beta = -0.6997: the fixed sampler agrees (z = -0.58). Both pieces are needed: the
cut flip supplies the misaligned-halves sector that the severed rungs require, the init
removes the seeded walls. N=8, beta=16 with both: lnF -0.363 +- 0.12 vs -0.358.
Kernel-level check of the new move against the exact P(q): tests/run_qdist.sh (below).

### 2026-11-23 (cont.): exact walk sum and kernel-level P(q) test (paper Sec. "Exact summation over all walks")

tools/walk_sum_exact.py: H(g) = D0 + g * offdiag => O(g) = sum_q c_q g^q, c_q = (1/dim) sum of W over
all walks of length q; c_q by FFT on |g|=1 (M=512). Checks:
* 2x2 (four-site ring), beta=2: partial sums ln sum_{q<=6,7,8} c_q at s=0.25,0.5 equal
  tests/enum_test.cpp to 10 digits; full lnO equals ED.
* beta=4: P(q<=6) at (0,0) = 0.3098921 (= paper Table tab:pq exact); ratios at (0,1/4) for
  q_max=6,7,8 = 0.3514515, 0.4614129, 0.5797091 (= paper's 0.351/0.461/0.580).
* aliasing tail and |sumP-1| < 2e-16 everywhere.

tests/run_qdist.sh: 8 independent single-ensemble chains per rung vs exact P(q).
N=8 open chain, Gamma=0.5, beta=16, rungs (0,0),(0,1),(1,1), 2e6 updates/chain:

| kernel | rung | <q> exact | sampled | max|z| bins | TV |
|---|---|---|---|---|---|
| default | (0,0) | 22.433 | 22.458 +- 0.052 | 1.54 | 0.0027 |
| default | (0,1) | 25.073 | 25.073 +- 0.046 | 2.95 | 0.0031 |
| default | (1,1) | 27.290 | 27.318 +- 0.028 | 2.29 | 0.0028 |
| CUT_FLIP+INIT | (0,0) | 22.433 | 22.446 +- 0.037 | 2.67 | 0.0021 |
| CUT_FLIP+INIT | (0,1) | 25.073 | 25.103 +- 0.028 | 2.95 | 0.0037 |
| CUT_FLIP+INIT | (1,1) | 27.290 | 27.301 +- 0.044 | 4.34 | 0.0031 |

Both kernels have the right stationary distribution (max|z| over 37 bins with 7-dof t is
consistent). 4x3 (Gamma=0.8, beta=8, qmax=160, 1e6 updates/chain): exact <q> = 40.593, 44.839,
48.442 at (0,0),(0,1),(1,1); sampler comparison: see below when finished.

4x3, Gamma=0.8, beta=8, 8 chains x 1e6 updates:

| kernel | rung | <q> exact | sampled | max|z| bins | TV |
|---|---|---|---|---|---|
| default | (0,0) | 40.593 | 40.944 +- 0.347 | 2.41 | 0.0135 |
| default | (0,1) | 44.839 | 44.990 +- 0.283 | 1.86 | 0.0098 |
| default | (1,1) | 48.442 | 48.563 +- 0.184 | 2.22 | 0.0081 |
| CUT_FLIP+INIT | (0,0) | 40.593 | 40.452 +- 0.208 | 5.73 (q=42: 0.0321 vs 0.0341) | 0.0130 |
| CUT_FLIP+INIT | (0,1) | 44.839 | 45.096 +- 0.347 | 2.13 | 0.0101 |
| CUT_FLIP+INIT | (1,1) | 48.442 | 48.166 +- 0.148 | 3.48 | 0.0146 |

Single-ensemble chains mix slowly on 4x3 (chain means scatter sd ~0.5-1.0), so 8-chain bin SEs are
noisy. The q=42 outlier is unresolved: rerunning CUT_FLIP+INIT with 24 chains x 2e6 at (0,0),(1,1)
(models/qd_4x3cf2).

### 2026-11-23 (cont.): beta=16 production with the fixed sampler

N=32, Gamma=0.5, beta=16, 33 rungs, T_equil 4000, 1500 samples, OV_CUT_FLIP+OV_INIT_CLASSICAL
(models/c32b16_fix), exact lnF_beta = -0.73672:
* s1: lnF = -0.929 +- 0.060 (351 PT round trips), 5.2 h wall
* s2: lnF = -0.782 +- 0.069 (343 round trips)
* mean -0.855, run-to-run sd 0.104 (vs old sampler: -5.73 / -4.54). s1 alone is 3.2 of its own
  bootstrap sigma off; two runs cannot separate bias from optimistic per-run errors -- more
  replicas needed before quoting.
N=64 (c64b16_fix): both runs were killed by session teardown (no state checkpoint written);
restarted from scratch under nohup.

4x3 CUT_FLIP+INIT rerun, 24 chains x 2e6 (models/qd_4x3cf2): (0,0) <q> 40.572 +- 0.111 (z=-0.18),
max|z| 2.39 over 59 bins, TV 0.0045; (1,1) 48.545 +- 0.098 (z=+1.05), max|z| 2.34, TV 0.0050.
The 8-chain q=42 outlier (|z|=5.7) was noise. Both kernels pass on 4x3. Paper Table tab:walksum
updated.
2026-11-23 20:35: N=64 restart (15:12) was killed again at ~20:03 by session teardown (nohup is not
enough; checkpoints are written only on a caught SIGTERM). Relaunched N=64 s1,s2 and N=32 s3,s4,s5
fully detached (setsid double-fork + caffeinate -i; launcher in session scratchpad detach_run.py).

N=32, beta=16, five replicas (s1-s5; s3-s5 run 2.5 h each when 5 jobs share the machine):
lnF = -0.929, -0.782, -0.976, -0.402, -0.855 (per-run bootstrap 0.060-0.119).
Mean -0.789 +- 0.102 (SEM), sd 0.229, 95% t-interval [-1.07, -0.50]; exact -0.73672, z = -0.51.
Unbiased within error, but the run-to-run sd is 2.8x the mean per-run bootstrap error (0.081):
per-run errors are optimistic here too (per-run z from -3.2 to +3.7). Quote replication-based errors only.

N=64, beta=16, fixed sampler, 2 runs (6.2-6.4 h each, 1 core): lnF = -0.860 +- 0.061 (451 round trips),
-0.872 +- 0.083 (348); exact -0.74488. Mean -0.866, deviation -0.12 nats. Two runs give no usable
error bar (their 0.008 scatter is luck: N=32 shows run-to-run sd 0.23), so this is consistent with
exact at the N=32 scatter but not yet quotable. Note 6 of the 7 beta=16 runs (N=32,64) fall below
exact; a small negative bias ~0.05-0.1 nats is not excluded. Needs >= 5 replicas.
