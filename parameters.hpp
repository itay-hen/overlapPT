//
// This program is introduced in the paper:
// Lev Barash, Arman Babakhani, Itay Hen, A quantum Monte Carlo algorithm for arbitrary spin-1/2 Hamiltonians, Physical Review Research 6, 013281 (2024).
//
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//

//
// Below are the parameter values:
//

#define Tsteps 1000000               // number of Monte-Carlo initial equilibration updates
#define steps  10000000              // number of Monte-Carlo updates
#define stepsPerMeasurement 10       // number of Monte-Carlo updates per measurement

// ═══════════════════════════════════════════════════════════════════════════════
// [VALIDATION RUN — TEMPORARY] Beta and the MBAR_* block below (MBAR_ALPHA,
// MBAR_N_BETA, MBAR_T_EQUIL, MBAR_N_SAMPLES) are currently set to a fast
// smoke-test config for validating the P0 audit fixes (F1/F2/F3/F5/F6/F7/F13/F17)
// against the exact-diagonalization table in MBARPT_audit.txt / tools/ed_reference.py,
// NOT the production config. Beta=3 is chosen because it's on the ED plateau
// (S2=0.855, S3=0.691 for beta>=3), so a single beta value validates the whole
// high-beta regime without needing per-beta recompiles.
//
// Production defaults to restore after validation passes:
//   Beta = 5, MBAR_ALPHA = 2, MBAR_N_BETA = 64 (still 32 if that's now the
//   regular working config), MBAR_T_EQUIL = 1000000, MBAR_N_SAMPLES = 500000.
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef Beta
#define Beta   3                  // inverse temperature (in OVERLAP_MODE: tau, the projection time of EACH segment)
#endif

//
// Below is the list of standard observables:
//

// #define MEASURE_H                    // <H>             is measured when this line is not commented
// #define MEASURE_H2                   // <H^2>           is measured when this line is not commented
// #define MEASURE_HDIAG                // <H_{diag}>      is measured when this line is not commented
// #define MEASURE_HDIAG2               // <H_{diag}^2>    is measured when this line is not commented
// #define MEASURE_HOFFDIAG             // <H_{offdiag}>   is measured when this line is not commented
//#define MEASURE_HOFFDIAG2            // <H_{offdiag}^2> is measured when this line is not commented
// #define MEASURE_Z_MAGNETIZATION      // Z-magnetization is measured when this line is not commented
// #define MEASURE_Z_MAGNETIZATION2     // Z-magnetization^2 is measured when this line is not commented
#define MEASURE_RATIO                // <ratio>         is measured when this line is not commented

//
// Below are the implementation parameters:
//

// ─────────────────────────────────────────────────────────────────────────────
// SUBSYSTEM MASKS  (Paper Section II.D, Eq. 17)
//
// The whole quantum system is split into two parts: subsystem A and subsystem B.
// This bipartition is the heart of the Rényi entropy calculation — we measure
// how "entangled" A and B are.
//
// Each mask is a string of 0s and 1s, one character per qubit.
// A '1' means that qubit belongs to that subsystem.
//
// Example for a 9-qubit system (indices 0–8):
//   SUBSYSTEM_A_MASK "000011011"  → qubits 0,1,3,4  belong to A
//   SUBSYSTEM_B_MASK "111100100"  → qubits 2,5,6,7,8 belong to B
//
// The two masks must not overlap (no qubit can be in both A and B).
// Paper: "Each system was partitioned into two contiguous subregions A and B"
//        (Section III.A)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef OVERLAP_PLUS
#ifndef OVERLAP_MODE
#define OVERLAP_MODE
#endif
#define OV_SINGLE_MOVES 1   // [OVERLAP_PLUS] open walks: single-operator insert/remove on, rotation off
#else
#define OV_SINGLE_MOVES 0
#endif
// [OVERLAP_PLUS] OV_CUT_FLIP: block move flipping every spin on one side of the cut
// (OV_CUT_MASK from tools/make_tables.py) in y0, hence along the whole walk. An involution
// with a symmetric proposal, accepted by plain Metropolis; free on fully severed segments.
// Attempted in place of a single spin flip with probability OV_CUT_FLIP_PROB.
// OV_INIT_CLASSICAL: start each replica's y0 at an annealed minimum of H1's diagonal
// energy instead of a random string (a random y0 seeds ~N/2 domain walls that span the
// whole walk and anneal out only slowly; see VETTING.md 2026-11-23).
#ifndef OV_CUT_FLIP
#define OV_CUT_FLIP 0
#endif
// OV_CUT_FLIP_ALL: with OV_CUT_FLIP, flip a uniformly chosen connected block of H2
// (OV_CUT_MASKS) instead of only the block containing the last qubit. Needed when H2 has more
// than two blocks (4x4 with both midline cuts: four quadrants).
#ifndef OV_CUT_FLIP_ALL
#define OV_CUT_FLIP_ALL 0
#endif
#ifndef OV_CUT_FLIP_PROB
#define OV_CUT_FLIP_PROB 0.5
#endif
#ifndef OV_INIT_CLASSICAL
#define OV_INIT_CLASSICAL 0
#endif
#ifdef OVERLAP_MODE
// [OVERLAP] Ground-state overlap mode: no bipartition. Every qubit is in A
// (shared by both copies) and B is empty, so the two copies always carry the
// same state, the cut between the H_b segment and the H_a segment is
// continuous, and closure means the whole sequence returns to z. The alpha=2
// weight is then the cut-sum expansion of Tr[exp(-tau H_a) exp(-tau H_b)].
#define SUBSYSTEM_A_MASK  std::string(N, '1')
#define SUBSYSTEM_B_MASK  std::string(N, '0')
#else
#define SUBSYSTEM_A_MASK  "000011011"
#define SUBSYSTEM_B_MASK "111100100"
#endif
#define Nsubsystems 5

// ─────────────────────────────────────────────────────────────────────────────
// qmax — maximum operator sequence length  (Paper Section II.D, Eq. 16)
//
// In the PMR (Permutation Matrix Representation) framework, the partition
// function is written as an infinite series expansion. Each term corresponds
// to a "sequence" of q quantum operators applied one after another.
// Think of it like a path through quantum space: the path visits q steps.
//
// [qmax fix] qmax=100 is now FATAL, not silent, if the sampler ever needs to
// go beyond it (mbar_common.hpp's scan_qmax/print_qmax_fatal_message) --
// job 428227 (testerjob_mpi_validate.sh, beta_target=3, alpha=3, N_BETA=32)
// hit it on 5/32 replicas during thermalization on a PRE-qmax-fix binary,
// and its final S_3 came out 1.684 vs the exact-diagonalization value of
// 0.691 (~6 sigma off despite a converged-looking MBAR solve) -- consistent
// with qmax truncation silently biasing the highest-beta rungs low on Z_3.
// Bumped 100 -> 300 as a reasoned-but-NOT-precisely-derived interim value:
// the old binary never printed how far past 100 those 5 replicas wanted to
// go, only that they hit it -- MBARPT_audit.txt records a comparable ~25%
// hit rate (16/64) at qmax=100, beta=5 on a similarly-sized system, so 3x
// headroom is a reasonable first guess, not a measured one. The new
// print_qmax_progress() output (mean/max q every progress interval, both
// drivers) is how to actually verify this empirically on the next run: if
// max q stays comfortably under 300 throughout, this is enough; if the
// fatal check fires again, raise it further and rerun. Confirmed safe from
// F10's "factorial[] is inf for k>=171" latent issue: factorial[] is only
// ever indexed by small values independent of qmax -- factorial[p]/
// factorial[r] (cycle-related, bounded by cycle_max_len, a property of the
// compiled Hamiltonian's fixed cycle structure) and factorial[len] with
// len=P.count()<=N (an operator-support popcount, not a cycle length, but
// likewise capped by N and independent of qmax) inside the legacy
// Replica::measure_observable's #ifdef MEASURE_CUSTOM_OBSERVABLES path,
// which the MBAR+PT drivers never call -- never by q itself, so raising
// qmax alone doesn't reach it.
//
// Paper Eq. 16: Z_1(β) = Σ_z Σ_{q=0}^∞ Σ_{i_q} ⟨z|S_{i_q}|z⟩ D(z,S_{i_q}) e^{-β[...]}
//
// [beta=5 sweep, 2026-08-02] Bumped 300 -> 500 before adding beta=5 to the
// pilot sweep (sweep_beta_alpha.sh). qmax must scale ~alpha*beta*N
// (MBARPT_audit.txt line 65); job 493509 (beta=3, alpha=3, this Hamiltonian)
// measured q_99.9=154 -> recommended qmax>=231 (current 300 barely cleared
// this). Linear beta scaling projects q_99.9(beta=5,alpha=3) ~ 154*(5/3) ~
// 257 -> recommended qmax>=385. 500 gives ~30% headroom over that estimate
// on top of the extrapolation being unverified (no beta=5 data on THIS
// Hamiltonian/N exists yet -- MBARPT_audit.txt's own beta=5 data point,
// qmax=100 "marginal" at 16/64 replicas, is from a different-sized system).
// As always: this is a reasoned interim value, not a measured one -- check
// each beta=5 job's own "[qmax sizing]" line and the qmax_achieved fatal
// gate before trusting the result; raise further and rerun if it fires.
// ─────────────────────────────────────────────────────────────────────────────
// ── STUDY NOTE [fixes landed 2026-07-12 · reviewed 2026-09-10] ──────────────
// Physics: q counts how many terms of the infinite PMR series expansion
// (Eq. 16 above) actually get included for a given sample. qmax truncates
// that infinite sum to a finite one -- too small a qmax means silently
// dropping the tail of a physically real series, not just adding noise.
// Plain-English: think of qmax as "how many terms we bother computing before
// giving up." Hitting the cap doesn't round gracefully -- it just silently
// throws away the rest of the physics for that sample, biasing results LOW
// (measured: S2~35 vs the exact-diagonalization value of 0.855 at beta=10,
// back when this was silent). It's now a hard, loud stop instead of a quiet
// truncation (mbar_common.hpp's scan_qmax/print_qmax_fatal_message), and a
// completed run tells you, from its own data, what qmax the NEXT run needs.
// ─────────────────────────────────────────────────────────────────────────
#ifndef qmax
#define qmax     500
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Nbins — bins for statistical error estimation  [OPTIMIZATION — not in paper]
//
// This is purely a statistical analysis tool. After the simulation, we want
// to know not just the answer, but how confident we are in it (the error bar).
// We divide all measurements into 250 "bins" and look at how much the average
// varies between bins. This is called "binning analysis" or "block averaging."
// It accounts for the fact that consecutive MC samples are not fully independent.
// ─────────────────────────────────────────────────────────────────────────────
#define Nbins    250

// ─────────────────────────────────────────────────────────────────────────────
// EXHAUSTIVE_CYCLE_SEARCH  [OPTIMIZATION — not directly in paper]
//
// During the "cycle completion" Monte Carlo move (see update() in replica.hpp),
// the code needs to find cycles in the operator algebra. With this flag enabled,
// it searches ALL available cycles. Commented out, it restricts the search to a
// narrower range of cycle lengths. The exhaustive search is more thorough and
// leads to better mixing of the Markov chain (the sampler explores more broadly).
// ─────────────────────────────────────────────────────────────────────────────
#define EXHAUSTIVE_CYCLE_SEARCH

// ─────────────────────────────────────────────────────────────────────────────
// GAPS_GEOMETRIC_PARAMETER — controls cycle completion move  [OPTIMIZATION]
//
// During the cycle completion update, a random "gap length" u is drawn from a
// geometric distribution with this parameter. Geometric(0.8) strongly favors
// small gaps (u=0 with prob 0.2, u=1 with prob 0.16, etc.), keeping the moves
// local and efficient. Tuning this affects how aggressively the sampler
// rearranges the operator sequence.
// ─────────────────────────────────────────────────────────────────────────────
#define GAPS_GEOMETRIC_PARAMETER 0.8

// ─────────────────────────────────────────────────────────────────────────────
// COMPOSITE_UPDATE_BREAK_PROBABILITY  [OPTIMIZATION — not in paper]
//
// The composite update chains together multiple small MC moves before deciding
// whether to accept or reject the whole chain. At each step in the chain, there
// is a 90% chance of stopping and a 10% chance of doing one more sub-move.
// This creates variable-length "macro-moves" that can traverse configuration
// space more efficiently than single small moves, while still satisfying
// detailed balance when taken as a whole.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef COMPOSITE_UPDATE_BREAK_PROBABILITY
#define COMPOSITE_UPDATE_BREAK_PROBABILITY  0.9
#endif

// #define WORM_UPDATE                  // uncomment this line to employ worm update rather than composite update
//
// [F3 fix] ENABLE_TRUE_LOOP_UPDATES used to gate three loop-type composite-update
// sub-moves together. Only two of them (ApplyLoopExchange, ApplySegmentExchange)
// are genuinely symmetric proposals: they permute whole blocks of operators, so
// the multiset of intermediate composite states -- and hence the proposal
// probability -- is identical before and after the move. ApplyInnerLoopReversal
// REVERSES the operator order within a segment; the endpoints still close (proof:
// reversing a sequence of involutions whose forward composition is the identity
// still composes to the identity), but the INTERIOR composite states generally
// change, so the occurrence-map group sizes the move was drawn from differ
// between the forward and reverse proposals. Plain Metropolis (no Hastings
// correction) is therefore a detailed-balance violation of unquantified size.
// Split into two flags so the two symmetric moves can stay on while the
// asymmetric one is disabled until it gets a proposal-ratio correction.
#define ENABLE_LOOP_EXCHANGE_UPDATES 1   // ApplyLoopExchange / ApplySegmentExchange -- symmetric, safe
#define ENABLE_INNER_LOOP_REVERSAL   0   // ApplyInnerLoopReversal -- DISABLED, no Hastings correction (F3)
// [DIAG segex] ApplySegmentExchange redraws its random quadruple up to 64 times
// until the two pairs interleave.  A proposal conditioned on eventual success has
// probability w/p_succ(S), and p_succ depends on the configuration (it grows with
// the number of repeated composite states, i.e. with q), so plain Metropolis on it
// is NOT detailed-balanced: the stationary distribution is pulled toward
// W(S)*p_succ(S)^theta -- an excess of long sequences (fable_audit/BIAS_AUDIT_REPORT
// addendum, Test J).  Both knobs are overridable with -D at compile time:
//   -DENABLE_SEGMENT_EXCHANGE=0          -> move disabled (loop exchange stays on)
//   -DSEGMENT_EXCHANGE_MAX_ATTEMPTS=1    -> single draw, no retry (detailed-balanced)
// Defaults reproduce the production behaviour exactly.
#ifndef ENABLE_SEGMENT_EXCHANGE
#define ENABLE_SEGMENT_EXCHANGE      ENABLE_LOOP_EXCHANGE_UPDATES
#endif
#ifndef SEGMENT_EXCHANGE_MAX_ATTEMPTS
#define SEGMENT_EXCHANGE_MAX_ATTEMPTS 64
#endif
// [DIAG moves] Per-move disable flags for single_rung_check.cpp experiments
// (-DDISABLE_MOVE_SWAP=1 etc.).  A disabled sub-move becomes a no-op at its own
// slot (the other sub-moves keep their selection probabilities, so each remaining
// move is still a valid Metropolis move on its own).  DISABLE_MOVE_CYCLE also
// removes ergodicity over plaquette loops -- interpret that one with care.
// Defaults (0) reproduce production.
#ifndef DISABLE_MOVE_SWAP
#define DISABLE_MOVE_SWAP   0
#endif
#ifndef DISABLE_MOVE_CYCLE
#define DISABLE_MOVE_CYCLE  0
#endif
#ifndef DISABLE_MOVE_LOOPEX
#define DISABLE_MOVE_LOOPEX 0
#endif
#ifndef DISABLE_MOVE_BLOCK
#define DISABLE_MOVE_BLOCK  0
#endif
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: same detailed-balance requirement as F2 below -- a Metropolis move
// needs the forward and reverse proposal probabilities to be correctly
// accounted for (or provably symmetric), or the sampler converges to a
// subtly wrong distribution instead of the true thermal one.
// Plain-English: this move takes a chunk of the operator sequence and
// reverses its order. The two endpoints still line up correctly afterward,
// but reversing the middle changes how many equivalent ways there were to
// propose that exact move vs. its reverse -- and the code was treating them
// as equally likely, which isn't proven true. Left disabled until someone
// derives the correction; its two siblings (loop exchange, segment exchange,
// both whole-block permutations) ARE provably symmetric and stay enabled.
// ─────────────────────────────────────────────────────────────────────────
// #define ABS_WEIGHTS                  // uncomment this line to employ absolute values of weights rather than real parts of weights
// #define EXACTLY_REPRODUCIBLE         // uncomment this to always employ the same RNG seeds and reproduce exactly the same results

//
// Uncomment or comment the macros below to enable or disable the ability to checkpoint and restart
//

// #define SAVE_COMPLETED_CALCULATION   // save detailed data to "qmc_data_*.dat" when calculaiton is completed
// #define SAVE_UNFINISHED_CALCULATION  // save calculation state to the files "qmc_data_*.dat" prior to exiting when SIGTERM signal is detected
// #define RESUME_CALCULATION           // attempt to read data from "qmc_data_*.dat" to resume the previous calculation
// #define HURRY_ON_SIGTERM             // break composite update on SIGTERM signal to speed up saving data; this is usually not necessary

// ═══════════════════════════════════════════════════════════════════════════════
// MBAR + PARALLEL TEMPERING PARAMETERS
// (Paper: "A Universal QMC Framework for Rényi Entanglement Entropies",
//  Kalev, Barash, Hen (2025))
//
// HOW TO RUN:
//   1. Set Beta (inverse temperature, above) to the temperature you want.
//   2. Set MBAR_ALPHA to the Rényi order you want (1, 2, or 3).
//   3. Recompile and run.
//   4. If MBAR_ALPHA > 1, also run with MBAR_ALPHA=1 first to get LN_Z1_TARGET.
// ═══════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_ALPHA — Rényi index α  (Paper Eq. 1, Eq. 2)
//
// This is the order of the Rényi entropy we compute. Think of it as a knob
// that controls which "version" of entanglement entropy we measure:
//   α=1  → approaches the standard von Neumann entropy (and gives Z_1 as a check)
//   α=2  → second Rényi entropy, the most commonly computed in QMC
//   α=3  → third Rényi entropy
//
// Paper Eq. 1: S_α(ρ_A) = 1/(1-α) · log TrA[ρ_A^α]
// Paper Eq. 2: S̃_α = TrA[ρ_A^α] = Z_α / Z_1^α
//
// Internally, α also equals the number of independent copies ("replicas") of
// subsystem B that are simulated simultaneously. For α=2 there are 2 B-copies,
// for α=3 there are 3 B-copies (Paper Eqs. 24-29).
// ─────────────────────────────────────────────────────────────────────────────
#ifdef OVERLAP_MODE
#define MBAR_ALPHA           2   // [OVERLAP] always two segments (H_b then H_a)
#endif
#ifndef MBAR_ALPHA
#define MBAR_ALPHA           3   // [VALIDATION RUN] alpha=3 exercises F1 (the headline
                                  // negative-S3 bug) directly; also recompile with
                                  // MBAR_ALPHA=2 afterward as a regression check.
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_N_BETA — number of points on the inverse-temperature ladder
// (Paper Section II.A, Eq. 4 "telescoping product methodology")
//
// Rather than computing Z_α(β_target) directly (very hard), we break the
// problem into many small steps. We build a "ladder" of β values from a
// very small β (easy classical limit) up to our target β. Each rung of the
// ladder is close enough to the next that the ratio Z_α(β_{r-1})/Z_α(β_r)
// can be estimated accurately by Monte Carlo.
//
// More rungs → smoother ladder → more accurate, but also more computation.
// 32 rungs is the value used in the paper's benchmark simulations
// (Section III.A: "Nβ = 32 inverse-temperature values").
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MBAR_N_BETA
#define MBAR_N_BETA          32  // [regular working value for MPI runs -- not reduced for the smoke test]
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_BETA_TARGET — the inverse temperature β we actually want  (Paper Eq. 1)
//
// This inherits the value of Beta defined at the top of this file, so you only
// need to change Beta in one place. β = 1/T where T is temperature.
// High β = low temperature (more quantum effects, stronger entanglement).
// Low β = high temperature (approaches classical limit, weaker entanglement).
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_BETA_TARGET     Beta     // inherits Beta defined above — one place to change target

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_BETA_MIN — the lowest β on the ladder (the "classical anchor")
// (Paper Section II.A, Eq. 5)
//
// At β=0, the quantum system becomes purely classical and Z_α(0) can be
// computed exactly with a simple formula (Paper Eq. 5):
//   Z_α(0) = (dim H_B)^α · dim H_A
//
// We don't use β=0 exactly; instead we use β_min = 1% of β_target.
// At this tiny β, quantum corrections are of order β_min² ≈ 0.01% —
// negligible. This gives us an exact "anchor" to set the absolute scale of
// all our free energy estimates.
//
// [OPTIMIZATION]: The 1/100 ratio is a practical implementation choice
// (not specified in the paper) that ensures negligible quantum corrections
// while keeping the 100× ladder range constant as Beta changes.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MBAR_BETA_MIN
#define MBAR_BETA_MIN        (Beta / 100.0)
#endif
                                             // auto-adjusts with Beta: always 1% of target
                                             // keeps quantum corrections O((Beta/100)^2) negligible
                                             // and maintains a consistent 100x ladder range

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_T_EQUIL — thermalization (burn-in) steps per replica  (Paper Section II.F)
//
// Before we start collecting samples, each replica needs time to "forget" its
// random starting configuration and reach the correct thermal equilibrium.
// These 200,000 steps are discarded — they are just warmup.
//
// The paper states: "For each α, a full parallel-tempering simulation was
// performed" (Section III.A). Thermalization is an implicit prerequisite.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MBAR_T_EQUIL
#define MBAR_T_EQUIL         100000   // [VALIDATION RUN] 10x smaller than production (1000000)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_N_SAMPLES — number of configurations saved per ladder rung  (Paper Eq. 10)
//
// After warmup, each replica collects this many independent snapshots of its
// configuration. These snapshots are fed into the MBAR solver.
//
// Paper Section III.A: "generating a combined dataset of ~10^6 statistically
// independent configurations." With 32 rungs × 100,000 samples = 3.2 million
// total. The paper's stated ~10^6 reflects the effective independent sample
// count after autocorrelation.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MBAR_N_SAMPLES
#define MBAR_N_SAMPLES       200000   // [2026-09-13] 10x the 20000 validation value, per Prof. Hen's
#endif
                                      // "10x as many steps for .1 - 1" note -- more RECORDED samples is
                                      // what raises N_eff (MBAR_STEPS_PER_SAMP only changes spacing, which
                                      // OPT-5's adaptive stride already tunes). Production was 500000.

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_STEPS_PER_SAMP — MC steps between recorded samples  [OPTIMIZATION]
//
// Consecutive MC configurations are highly correlated (each differs from the
// previous by only one small move). To get approximately independent samples,
// we wait 10 steps between recordings. This "thinning" reduces statistical
// error without wasting storage.
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_STEPS_PER_SAMP  10

// MBAR_KEEP_SAMPLES — keep every recorded u-matrix row after a run completes
// (renamed to renyi_samples_a<alpha>_r<rank>.bin) instead of deleting it, so the
// point estimate and error bar can be re-checked offline with
// tools/ustream_recheck.py. Disk: N_SAMPLES * N_BETA * 8 bytes per rank per alpha
// chain (~51 MB/rank, ~1.6 GB per chain at 200000 x 32). An alpha>=2 job that
// solves Z1 fresh keeps both its alpha=1 and alpha chains. Set 0 to delete as before.
#ifndef MBAR_KEEP_SAMPLES
#define MBAR_KEEP_SAMPLES    1
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_SWAP_INTERVAL — attempt a parallel tempering swap every N steps
// (Paper Section II.C, "Periodic exchange moves between neighboring replicas")
//
// Parallel Tempering works by occasionally swapping the configurations of two
// neighboring replicas on the β-ladder. This lets configurations "visit" many
// temperatures, greatly improving mixing. Every 5 steps we attempt all
// neighboring swaps (alternating even/odd pairs).
//
// The 30–50% swap acceptance rate reported in the paper (Section III.B)
// indicates this interval is well-tuned.
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_SWAP_INTERVAL   5

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_SOLVER_TOL — convergence tolerance for the MBAR self-consistency loop
// (Paper Section II.C, Eq. 12)
//
// MBAR works by iteratively solving a set of self-consistency equations (Eq. 12)
// until the free energy estimates stop changing. We stop when the largest change
// in any f_j between iterations is less than 10^-6.
//
// Paper Section III.A: "iteratively solving the MBAR self-consistency equations
// until convergence to 10^-6 relative precision was achieved."
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_SOLVER_TOL      1e-6

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_SOLVER_MAXITER — safety cap on MBAR iterations  [OPTIMIZATION]
//
// If MBAR hasn't converged after this many iterations, we stop anyway to
// prevent an infinite loop. In practice, convergence typically happens well
// before 50,000 iterations (usually within a few thousand).
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_SOLVER_MAXITER  50000

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_N_BOOT — number of bootstrap replicates for uncertainty estimation
//               (Code Review §1.2 — closes the publication gap)
//
// After the main MBAR solve, we resample the collected configurations per rung
// with replacement, re-solve MBAR (warm-started from the main solution), and
// record f[0] = ln Z_α(β_target) − ln Z_α(β_min) for each replicate.
// The standard deviation of these B values is the bootstrap standard error.
//
// 200 replicates gives a reliable estimate of the standard error while keeping
// the overhead manageable (each replicate is a warm-started MBAR solve on the
// already-collected data — no new MC sampling is needed).
// ─────────────────────────────────────────────────────────────────────────────
#ifndef MBAR_N_BOOT
#define MBAR_N_BOOT          200
#endif

// ─────────────────────────────────────────────────────────────────────────────
// MBAR_LN_Z1_TARGET — SUPERSEDED; no longer read by mbarpt.cpp
//
// ln Z_1(β_target) is now loaded automatically at runtime from z1_anchor.dat.
// When MBAR_ALPHA=1, the result is written there; when MBAR_ALPHA>1, it is
// read back (running the α=1 pipeline first if no matching entry exists).
//
// This constant is kept here only as a human-readable reference for the
// 9-qubit Heisenberg Hamiltonian (H3.txt) at β=0.3: ln Z_1 ≈ 14.5563.
// It is NOT used by any code path — changing it has no effect.
// ─────────────────────────────────────────────────────────────────────────────
#define MBAR_LN_Z1_TARGET    14.5563  // reference only — see z1_anchor.dat
