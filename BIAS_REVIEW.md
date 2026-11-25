# Request: diagnose a small negative bias in the OVERLAP_PLUS estimator

This is a focused request, separate from the general notes in `VETTING.md` (read its
"What to look at" and results 6-7 for context). Section numbers refer to the gsOverlap v19
manuscript. Everything needed to reproduce is in this archive. The exact references are
computed by `tools/plus_reference.py`; the 4x4 computation takes 10 s.

## Outcome (2026-11-24)

**4x4: removed; explanation supported, not proven.** The offset is consistent with slow mixing
of the relative Neel orientation of the four quadrants of the severed lattice. Without a dedicated
move a quadrant is reoriented only through many unfavourable single-spin flips. The link to support
is localized along the walk (`tools/plus_orientation.py`, middle rung): the probability that all
four quadrants agree is 0.125 at the origin (H2 side) with or without the no-severed-hop condition,
and 0.436 -> 0.824 at the H1 end. Walks with no severed hop favour aligned quadrants at the H1 end
but do not require them.

* **Decisive test (`tools/plus_support.py`).** The fraction of walks at each rung with no
  severed-bond hop is computed exactly by propagating with the severed bonds' X terms removed.
  Eight long-warm-up runs without the move gave 0.01586 at the middle rung against the exact
  0.01811 (-4.9 sigma), with -2.3 and -2.5 sigma two rungs to either side. This was the
  reviewer's "support" observation, seen in the middle of the ladder rather than at the end.
* **Fix.** `-DOV_CUT_FLIP_ALL=1` flips one H2 component chosen uniformly (`OV_CUT_MASKS` from
  `tools/make_tables.py`). It is a fixed involution with a state-independent choice, accepted
  with the plain Metropolis ratio.
* **Result.** Twelve runs (move plus classical initialization) give ln F = -1.48895 +- 0.05396
  vs exact -1.53887 (+0.93 SEM), s.d. 0.187. Support agrees at all 25 rungs (max |z| 1.7). The
  exact-score test implies +0.048 +- 0.054, with no drift. The comparison also changes the
  initialization, warm-up length and build provenance. See `VETTING.md` result 8.

**Excluded along the way:**
* **Walk-start parity.** Odd-parity walkers clear within 1000 warm-up steps. The exact odd mass
  is about 1e-6.
* **MBAR finite-sample bias as the main effect.** Splitting runs moves the mean by up to 0.0205,
  pooling by 0.0022, against an SEM of 0.054 (this is not an upper bound).
* **A detailed-balance error in start-shift.** The reviewer found none.

**4x3:** a newer group of 12 runs is -0.24 SEM, and its score test is consistent with zero.
The original -1.64 SEM is therefore most likely chance.

Everything below is the original request, kept as written.

## The observation

The |+> open-walk mode was run as 12 independent runs per lattice, with the start-shift move, beta = 8, and
`-DOVERLAP_PLUS -Dqmax=160 -DMBAR_T_EQUIL=10000`. The error below is the standard error over runs.

| lattice | rungs | samples/rung | ln F (MC) | exact ln F_beta | deviation | run-to-run s.d. |
|---|---|---|---|---|---|---|
| 4x3 (12 qubits) | 17 | 10^4 | -0.598 +- 0.034 | -0.5416 | -1.64 SEM | 0.119 |
| 4x4 (16 qubits, both midline cuts) | 25 | 2x10^4 | -1.654 +- 0.055 | -1.5389 | -2.08 SEM | 0.191 |

Each deviation is tolerable on its own. Both have the same sign, and the pooled significance is
about 2.6 sigma. The offset is about 4.0x (4x3) and 3.1x (4x4) the run-to-run variance.

**Where the deviation sits on the ladder.** The free energies below are relative to the middle
rung, (0,1), as mean residual (SEM) over 12 runs. Because ln F = (f_0 - f_mid) + (f_{K-1} - f_mid),
the two end columns add up to the ln F deviation.

```
4x3  rung: 0      1      2      3      4      5      6      7      8 |  9     10     11     12     13     14     15     16
     res : -.004  -.004  -.005  -.005  -.006  -.006  -.007  -.006  0  | -.000 -.000  -.002  -.004  -.010  -.019  -.032  -.053
     sem :  .036   .032   .029   .026   .023   .020   .015   .009  0  |  .004  .008   .013   .017   .022   .028   .035   .044
4x4  rung: 0     2     4     6     8     10    12 | 14    16    18    20    22    23    24
     res : -.069 -.059 -.050 -.037 -.018 -.000  0 | +.004 +.002 -.002 -.004 -.007 -.022 -.046
     sem :  .053  .051  .049  .047  .038  .020  0 |  .007  .014  .022  .036  .055  .062  .066
```

The residual grows toward the ends of the ladder, (0,0) and (1,1), on both lattices. At
the (1,1) end, walks that contain an X operator on a severed bond have zero weight, so that
ensemble's support is strictly smaller than its neighbour's. This pattern may be a clue, or
it may be noise: every per-rung entry is within 1.3 SEM.

## Facts that may help

* **Diagnostics in every run's log:**
  * MBAR converges in 16-63 iterations.
  * Adjacent-rung overlaps are at least 0.11 on 4x4 and higher on 4x3.
  * PT swap acceptance is 0.63-0.84; there are 36-49 round trips per run on 4x4.
  * No zero-weight rows. q_99.9 = 99 against qmax = 160.
* **The bootstrap error is not stable.** In every 4x4 run it grows about 4x from block length
  28 to 1750 (`stderr_status 1`). Some slow mode decorrelates over thousands of samples.
* **Mean walk length matches exact values at every rung.**
  * Exact values: <q>_k = d ln O_k/d eps, where every off-diagonal coefficient is scaled by
    (1+eps). Computed by `tools/plus_observables.py`, stored as `models/*/plus_obs_b8_k*.json`.
  * MC means over the 12 runs: chi^2 = 7.6 over 17 rungs (4x3) and 15.7 over 25 rungs (4x4).
  * On 4x4, rungs 7-14 sit together ~0.15 low (about 1 SEM each, correlated through PT).
* **Exact checks on 2x2 pass** (`tests/run_small_checks.sh`):
  * weight enumeration;
  * P(q <= 5) from single-replica chains;
  * the marginal of the walk start y0;
  * the checkpoint hash.
* **The earlier, much larger failure is fixed.** The walk start y0 was effectively frozen, which
  gave a run-to-run s.d. of 0.72 and a correlation of 0.97 with round trips. The start-shift
  move fixed it. See `VETTING.md` result 4 and the `runs_plus_b8_long` logs. The
  residual effect reported here is roughly 10x smaller.

## Hypotheses (none confirmed or excluded yet)

1. **MBAR finite-sample bias.** The statistic is a nonlinear function of the f_k. With an
   effective sample of ~700 per rung (tau_int up to ~27 samples), an O(1/N_eff) bias of this
   size is plausible, and it would scale with the run-to-run variance, as observed.
   Prediction: pooling runs or using longer runs shrinks it; splitting runs into m pieces
   multiplies it by ~m.
2. **Incomplete equilibration of a slow mode.** The block-length growth of the bootstrap
   error shows such a mode exists. Prediction: early parts of a run deviate more than late parts.
3. **A detailed-balance error that only shows at large q or N.** Candidates: the start-shift pair,
   the single-operator insertion/removal pair, and their interaction with the composite update
   and `qmax`. The y0 marginal has been checked against exact values only on 2x2. Prediction: a
   single-rung chain on 4x3 misses the exact y0 marginal or var(q).
4. **A ladder or support effect at the ends** (see above). Possibilities: MBAR handling of the zero-weight
   region at (1,1), or under-resolution of the end segments. Prediction: more rungs near the
   ends change the result.
5. **Something in the reweighting itself.** Every sample's weight is re-evaluated at all K rungs
   (`OvWeight` with cached prefix and suffix). The cache was checked against the
   uncached code on random walks (max |d ln W| = 5.7e-14), and the uncached code against
   enumeration on 2x2. It has not been checked against enumeration on a larger model.

## Running now (results to follow)

* 12 runs on 4x3 with the raw u-matrix kept (`-DMBAR_KEEP_SAMPLES=1`, 2x10^4 samples). They will
  be analysed by `tools/plus_bias_check.py`: per-run, pooled, split into m = 1, 2, 4, 8
  pieces, and by position in the run. This separates hypotheses 1 and 2.
* Four single-rung chains on 4x3 (`tests/marg_test.cpp`, 10^7 updates each) at rungs
  s = 0.25 and 0.5, compared with the exact y0 marginal and var(q) (hypothesis 3).

## Where to look in the code

| what | where |
|---|---|
| open-walk weight, cut sum, prefix/suffix cache | `replica.hpp`: `OvPrefix`, `OvSuffix`, `OvWeight` (around line 1090-1150) |
| move set, including start-shift (`t_fins`/`t_frem`) and single insert/remove | `replica.hpp`: `update()` (around line 3369-3660); acceptance uses `newWeight*Rfactor` |
| u-matrix fill (u = -ln W at every rung, 1e300 = zero weight) | `mbarpt_mpi.cpp` around line 940-960 |
| MBAR solve, reported statistic, bootstrap | `mbarpt_mpi.cpp`: `solve_mbar_mpi` (line 207), `pipeline_stat`, bootstrap section |
| ladder mapping s -> (lam_a, lam_b) | `OvLambdas` in `replica.hpp`; `tools/plus_reference.py` (same mapping) |

## Reproducing

    ./build_overlap.sh models/4x4_v2 -DOVERLAP_PLUS -DBeta=8 -DMBAR_N_BETA=25 -Dqmax=160 \
        -DMBAR_T_EQUIL=10000 -DMBAR_N_SAMPLES=20000 -DMBAR_KEEP_SAMPLES=1
    mkdir r1 && (cd r1 && ../models/4x4_v2/build/overlap.bin > log.txt)      # ~3 h on one core
    python3 tools/plus_reference.py   models/4x4_v2 --beta 8 --rungs 25       # exact ln F and f_k
    python3 tools/plus_observables.py models/4x4_v2 --beta 8 --rungs 25       # exact <q>_k, var(q)_k
    python3 tools/plus_bias_check.py r1 r2 ... --rungs 25 --exact -1.5388722499

On 4x3 (`models/4x3_v2`, 17 rungs, 10^4 samples) a run takes about 45 min. The 2x2 checks
are `tests/run_small_checks.sh`. Logs of all reported runs are under `models/*/runs_*`, and a
one-line summary of each run is in `results/per_run_results.txt`.

## What would help most

* A reason, from the code, why the estimator could be biased by about 5-10% in F at this
  sample size, or an argument that hypothesis 1 fully accounts for it.
* Any detailed-balance problem in the move set: proposal ratios, the composite-update
  acceptance, or the `qmax` boundary.
* Any problem with how MBAR treats states that have zero weight at some rungs.
* A cheaper decisive test than the ones above.
