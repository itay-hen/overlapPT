# overlapPT: ground-state overlaps with PMR-QMC + parallel tempering + MBAR

A copy of `mbarptNov-main` (the Rényi-entropy code) adapted to compute the squared
overlap between the ground states of two Hamiltonians H1 and H2. The original
README is kept as `README.md`; the Rényi example Hamiltonian moved to
`examples/renyi_H32/` so it can never be picked up by a build by accident.

## Two modes

* **`-DOVERLAP_PLUS` (recommended; gsOverlap v19 Sec. III.F).** |+> trial state,
  open walks, fixed projection `Beta` (the paper's beta; tau = Beta/2 per segment),
  ONE parallel-tempering run over the interpolation ladder
  (0,0) -> (0,1) -> (1,1) with H(lam) = (1-lam) H1 + lam H2. Reports
  ln F_beta = f[0] + f[K-1] - 2 f[mid], which is gauge invariant: no anchor, no
  separate runs. `MBAR_N_BETA` must be odd. Converges with the in-sector gap.
* **`-DOVERLAP_MODE` alone (trace variant, below).** Converges only with the full
  spectral gap; kept for reference and for Table tab:cpp of the paper.

Compiler flags shared by the build and the tests are in `common_flags.sh`. Every run
checks its MBAR solve before reporting (convergence, finite free energies, nonzero
own-rung weights, connected ladder), prints the adjacent-rung MBAR overlaps and, in the
overlap modes, how the bootstrap error changes with block length. Run
`tests/run_small_checks.sh` after any change; it exits non-zero on failure.

Overlap-plus usage:

    ./build_overlap.sh models/4x3_v2 -DOVERLAP_PLUS -DBeta=8 -DMBAR_N_BETA=17 -Dqmax=160 \
        -DMBAR_T_EQUIL=10000 -DMBAR_N_SAMPLES=10000
    mkdir run1 && (cd run1 && ../models/4x3_v2/build/overlap.bin)     # writes overlap_plus.dat
    python3 tools/ed_overlap.py models/4x3_v2 --tau 1 --beta 8 --rungs 17   # exact lnF and f_k
    python3 tools/plus_reference.py models/4x3_v2 --beta 8 --rungs 17     # same, sparse; fine at N=16

`overlap_plus.dat` columns: beta, lnF, stderr, N, rungs, samples, stderr_status (0 usable;
1 the error grew with block length and `stderr` is the largest scanned value; 2 a bootstrap
replicate failed and `stderr` is nan), and the default-block-length error.

Changes for this mode: `OvWeight` in `replica.hpp` (cut-summed open-walk weight on
the interpolated Hamiltonian, with the pure-H1 suffix and pure-H2 prefix cached per
walk), single-operator insertion/removal moves (needed because pair moves never
change the walk's endpoint), a start-shift move pair that changes the walk's start
y0 (without it y0 is frozen at large beta), the cyclic-rotation move switched off
(invalid for open walks), and `pipeline_stat` in `mbarpt_mpi.cpp`.

**Error bars: use replication.** On the 2x2 (= 4-site ring) test at beta=4 the
per-run bootstrap error is ~0.011 while the run-to-run scatter is 0.105, the
same factor ~10 the paper reports for MBAR asymptotic errors. Quote the
standard error over independent runs, never the per-run value.

**qmax.** Each divided-difference update costs O(qmax), so set qmax to about
1.5x the `[qmax sizing]` q_99.9 the run reports (the run stops if it is hit).

## What is computed (trace variant)

For one pair ab in {11, 12, 22}, the program estimates

    Z_ab(tau) = Tr[ exp(-tau H_a) exp(-tau H_b) ]

on a geometric ladder of tau values, and anchors it at tau_min with a closed form.
Three runs combine into

    R(tau) = Z_12(tau) / sqrt( Z_11(tau) Z_22(tau) )  ->  |<g1|g2>|^2   as tau -> infinity,

assuming both ground states are unique. No trial state is needed. Z_aa(tau) is
the ordinary partition function at 2*tau.

## How it maps onto the Rényi code

The Rényi alpha=2 sampler already uses one operator sequence cut into two
segments, each with its own divided-difference kernel and its own d-factors, and
sums over the cut position. Overlap mode (`-DOVERLAP_MODE`) uses it with:

* subsystem A = every qubit, B = empty, so both copies carry the same state, the
  cut is continuous, and closure means the full sequence returns to z;
* a different Hamiltonian per copy: copy 0 (segment [0,c), acts first) uses H_b
  and copy 1 (segment [c,q)) uses H_a (`ham_of_copy` in `replica.hpp`, tables
  `OV_*` in the generated `hamiltonian2.hpp`);
* the closed-form anchor ln Z_ab(tau_min) = N ln2 + ln[1 - tau (t_a+t_b)
  + tau^2/2 (<H_a^2> + <H_b^2> + 2<H_a H_b>)], with t = Tr(H)/2^N and
  <XY> = Tr(XY)/2^N computed from the Pauli coefficients. Its error is
  O(tau_min^4), so keep tau_min small (0.01 is a good default).

The Monte Carlo moves, parallel tempering, MBAR solver, bootstrap errors and
checkpointing are unchanged.

## Usage

    python3 tools/make_model.py --Lx 4 --Ly 3 --J 1 --Gamma 0.8 --vcut 2 --out models/4x3_v2
    ./build_overlap.sh models/4x3_v2 -DBeta=2 -DMBAR_BETA_MIN=0.01 -DMBAR_N_BETA=16 \
        -DMBAR_T_EQUIL=20000 -DMBAR_N_SAMPLES=20000
    # one directory per pair (checkpoint files would otherwise collide):
    mkdir p11 p12 p22
    (cd p11 && ../models/4x3_v2/build/overlap.bin 11)    # likewise 12, 22
    python3 tools/combine_overlap.py p11 p12 p22 --ed models/4x3_v2/ed.json
    python3 tools/ed_overlap.py models/4x3_v2 --tau 2 4 8   # exact reference, N <= ~14

`Beta` is tau, the projection time of each segment. Without `mpicxx` the build
script uses the single-process stand-in `serial/serial_mpi.h` (laptop testing only).

## Files added or changed

| file | change |
|---|---|
| `replica.hpp` | per-copy Hamiltonian in `CalcCompositeEnergy` / `calc_d_for_config` |
| `mainQMC.hpp` | includes `hamiltonian2.hpp` and declares `g_ov_ham` in overlap mode |
| `parameters.hpp` | overlap-mode masks; `Beta`, `qmax`, `MBAR_*` overridable with `-D` |
| `mbarpt_mpi.cpp` | overlap branch in `main`: pair argument, anchor, result file |
| `build_overlap.sh` | prepare + table generation + compile, per model directory |
| `tools/make_model.py` | full and severed lattice Hamiltonians |
| `tools/make_tables.py` | H1 and H2 tables on H1's permutation list, plus traces |
| `tools/ed_overlap.py` | exact Z_ab(tau), R(tau), overlap, gaps, anchor error |
| `tools/plus_reference.py` | sparse exact \|+> reference (lnF and every rung's f_k) |
| `tools/plus_observables.py` | exact per-rung \<q\> and var(q) (and optionally the y0 marginal) for the |+> ladder |
| `tools/plus_bias_check.py` | offline MBAR on kept samples: per run, pooled, split into pieces, by position; rejects NaN, zero own-rung weight and disconnected supports |
| `tests/test_plus_bias_check.py` | regression tests for that validation |
| `tools/plus_support.py` | exact vs sampled fraction of walks with no severed-bond hop, per rung |
| `tools/plus_orientation.py` | exact relative Neel orientation of the severed blocks at the origin, middle and far end of the walk |
| `tools/detach_run.py` | detached run launcher; writes `run_manifest.json` (binary hash + build manifest), `--seed N` |
| `tools/record_build.py` | build manifest: exact compiler command, compiler version, git revision, source and binary hashes |
| `tests/marg_test.cpp` | single-rung chain on any model: block means of q, q^2 and the y0 histogram |
| `tools/combine_overlap.py` | R(tau) with error from three runs, optional ED comparison |
| `serial/serial_mpi.h` | single-process MPI stand-in for local builds (`-DSERIAL_MPI`) |

## Caveats

* Odd N: every level of the J ZZ - Gamma XX model is exactly two-fold degenerate
  (the two Z2 symmetries anticommute), so the overlap is not defined; use even N.
* For J > Gamma the model is Néel-ordered and the lowest gap shrinks quickly with
  L (0.12 already at 4x3), so the tau needed for R(tau) to converge grows fast.
* Only X-product and Z-product terms are supported by `make_tables.py`.
* N <= 64 (lattices are packed into 64-bit words for checkpoints and MPI swaps).
