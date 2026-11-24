// ═══════════════════════════════════════════════════════════════════════════════
// mbar_common.hpp — shared MBAR+PT infrastructure for mbarpt.cpp / mbarpt_mpi.cpp
// ═══════════════════════════════════════════════════════════════════════════════
//
// The OpenMP driver (mbarpt.cpp) and the MPI driver (mbarpt_mpi.cpp) duplicated
// ~400 lines of identical logic: the beta ladder, the classical/high-T anchor,
// autocorrelation analysis, Anderson mixing, the PT swap acceptance test, and the
// z1_anchor.dat cache I/O. Extracted here so every fix to that logic applies once,
// not twice. Functions that are genuinely different between the two parallelism
// models (solve_mbar vs solve_mbar_mpi, mpi_boundary_swap, the pipeline drivers,
// main()) stay in their respective .cpp files.
//
// Must be included AFTER mainQMC.hpp (needs RenyiReplica, Replica, N, Nop, rng,
// val(), ExExFloat, mpi_rank, mpi_size, rng_seed) and after parameters.hpp's
// MBAR_* macros are visible.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef MBAR_COMMON_HPP
#define MBAR_COMMON_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <algorithm>
#include <random>
#include <unistd.h>   // getpid() -- POSIX only; this codebase compiles on Linux clusters exclusively

// ═══════════════════════════════════════════════════════════════════════════════
// PARAMETERS — local aliases for the #defines in parameters.hpp (Item: identical
// aliases previously declared separately in mbarpt.cpp and mbarpt_mpi.cpp).
// ═══════════════════════════════════════════════════════════════════════════════
static const int    ALPHA          = MBAR_ALPHA;
static const int    N_BETA         = MBAR_N_BETA;
static const double BETA_MIN       = MBAR_BETA_MIN;
static const double BETA_TARGET    = MBAR_BETA_TARGET;
static const int    T_EQUIL        = MBAR_T_EQUIL;
static const int    N_SAMPLES      = MBAR_N_SAMPLES;
static const int    STEPS_PER_SAMP = MBAR_STEPS_PER_SAMP;
static const int    SWAP_INTERVAL  = MBAR_SWAP_INTERVAL;
static const double MBAR_TOL       = MBAR_SOLVER_TOL;
static const int    MBAR_MAX_ITER  = MBAR_SOLVER_MAXITER;
static const int    N_BOOT         = MBAR_N_BOOT;

// ═══════════════════════════════════════════════════════════════════════════════
// build_beta_ladder()  —  Paper Section II.A, Eq. 4
// ═══════════════════════════════════════════════════════════════════════════════
inline std::vector<double> build_beta_ladder() {
    std::vector<double> ladder(N_BETA);
    double log_min = std::log(BETA_MIN);
    double log_max = std::log(BETA_TARGET);
    for (int k = 0; k < N_BETA; k++) {
        double t = (N_BETA == 1) ? 1.0 : (double)k / (N_BETA - 1);
        ladder[k] = std::exp(log_min + t * (log_max - log_min));
    }
    ladder[0]        = BETA_MIN;
    ladder[N_BETA-1] = BETA_TARGET;
    return ladder;
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_classical_renyi_Z()  —  Paper Eq. 5, exact classical (q=0) sector at any beta
// ═══════════════════════════════════════════════════════════════════════════════
inline double compute_classical_renyi_Z(double beta, int alpha_val) {
    std::bitset<N> mask_A(SUBSYSTEM_A_MASK);
    std::bitset<N> mask_B(SUBSYSTEM_B_MASK);
    std::vector<int> A_bits, B_bits;
    for (int i = 0; i < N; i++) {
        if (mask_A.test(i)) A_bits.push_back(i);
        if (mask_B.test(i)) B_bits.push_back(i);
    }
    int N_A = A_bits.size(), N_B = B_bits.size();
    int dim_A = 1 << N_A, dim_B = 1 << N_B;
    Replica tmp;
    double Z_alpha_cl = 0.0;
    for (int a = 0; a < dim_A; a++) {
        std::bitset<N> cfg_A;
        for (int i = 0; i < N_A; i++)
            if ((a >> i) & 1) cfg_A.set(A_bits[i]);
        double Z_B = 0.0;
        for (int b = 0; b < dim_B; b++) {
            std::bitset<N> cfg_B;
            for (int i = 0; i < N_B; i++)
                if ((b >> i) & 1) cfg_B.set(B_bits[i]);
            tmp.lattice = cfg_A | cfg_B;
            Z_B += std::exp(-beta * tmp.CalcEnergy());
        }
        Z_alpha_cl += std::pow(Z_B, alpha_val);
    }
    return Z_alpha_cl;
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_anchor_lnZ()  —  Paper Eq. 5, exact at beta=0 (kept as a printed reference)
// ═══════════════════════════════════════════════════════════════════════════════
inline double compute_anchor_lnZ(int N_A, int N_B, int alpha) {
    return (N_A + alpha * N_B) * std::log(2.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_offdiag_anchor_terms()  —  [F7 fix] off-diagonal-only O(beta^2) anchor
// correction terms, computed once and reused for every alpha.
//
// The OLD anchor was (N_A+alpha*N_B)*ln2 + (beta^2/2)*alpha*<E^2>_inf: a beta=0
// reference plus a DIAGONAL-ONLY Taylor term. That's wrong at its claimed order
// two ways: (1) the diagonal/classical sector doesn't need a truncated Taylor
// term at all -- compute_classical_renyi_Z sums it EXACTLY, to all orders in
// beta; (2) it entirely omits the off-diagonal (quantum) contribution, which
// ED measurement shows is 2x the size of the included diagonal term for this
// model.
//
// This function computes the two off-diagonal quantities needed ON TOP OF the
// exact classical anchor:
//
//   od2_inf               -- linear-in-alpha piece.
//     (1/dim) * sum_z sum_k |d_k(z)|^2 = <z|H^2|z> - E(z)^2 averaged over all
//     z. This is exactly what the diagonal-only formula missed: the classical
//     term only ever captured E(z)^2, never the off-diagonal contribution to
//     <H^2>.
//
//   intra_A_sq_over_dimA  -- alpha(alpha-1) piece, from operators that never
//     touch a B qubit ("intra-A hopping": P_matrix[k] & subsystem_B_mask == 0).
//     Tr_A[(V_A^intra)^2] / dim_A, where V_A^intra is the dim_A x dim_A matrix
//     of those operators' matrix elements alone.
//
// DERIVATION SKETCH: write M(beta) = Tr_B[exp(-beta H)] (a dim_A x dim_A
// matrix; Z_alpha(beta) = Tr[M(beta)^alpha]). M(0) = dim_B * I commutes with
// everything, so expanding Tr[M(beta)^alpha] to O(beta^2):
//   Tr[M^alpha] ~= dim_A*dim_B^alpha
//              + (beta^2/2)*alpha*dim_B^{alpha-1}*Tr[M2]
//              + (beta^2/2)*alpha*(alpha-1)*dim_B^{alpha-2}*Tr[M1^2]
// where M1 = -Tr_B[H], M2 = Tr_B[H^2]. Tr[M2] = Tr_full[H^2] = dim*<H^2>_inf
// (the beta^1 term Tr[M1] vanishes: <E>_inf=0 by particle-hole symmetry).
// M1 splits into a piece diagonal-in-zA (the classical per-zA energy -- ALREADY
// captured exactly, to all orders, by compute_classical_renyi_Z's own Taylor
// content) plus an off-diagonal-in-zA piece equal to dim_B * V_A^intra (terms
// touching a B qubit can never connect two states with equal zB, so they
// contribute nothing to Tr_B[H]'s off-diagonal-in-zA part at all). The
// diagonal/off-diagonal cross term in Tr[M1^2] vanishes exactly (V_A^intra has
// zero diagonal). Converting Tr[M2] and Tr[M1^2] into ln(Tr[M^alpha]) via
// ln(1+x)~=x gives the two terms above.
//
// Validated against exact diagonalization of the compiled 9-qubit test
// Hamiltonian: reduces the worst-case anchor bias (beta_min=0.1) from -0.80 to
// -0.077, a ~10x tightening, and to -0.014 (14x) at the beta_min=0.05 operating
// point most runs actually use. See tools/validate_f7_anchor.py.
//
// Cost: O(2^N * Nop) -- one extra factor of Nop over compute_classical_renyi_Z,
// paid once at startup. Fine at the N~9-20 scale this codebase targets; for
// much larger N this would need the same estimate-or-shrink-beta_min treatment
// the audit already flags for compute_classical_renyi_Z itself.
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: this anchor is Z_alpha at the near-classical BOTTOM of the whole
// 32-rung temperature ladder. Every other rung is only ever computed
// RELATIVE to this one absolute number (MBAR gives free-energy differences,
// not absolutes) -- so any bias here shifts every single result on the
// ladder by the same amount, like a ruler that's miscalibrated at zero.
// Plain-English: the old anchor measured the size of quantum jitter using
// only its "diagonal" (classical-looking) half and forgot the "off-diagonal"
// (genuinely quantum) half entirely -- which direct measurement shows is
// actually TWICE as big as the half that was kept. Fixed two ways: (1) sum
// the purely classical part EXACTLY (compute_classical_renyi_Z, a brute-force
// sum over all 512 classical states -- cheap at N=9) instead of Taylor-
// approximating it, and (2) add the missing off-diagonal quantum term below.
// ─────────────────────────────────────────────────────────────────────────
inline void compute_offdiag_anchor_terms(double& od2_inf, double& intra_A_sq_over_dimA) {
    std::bitset<N> mask_A(SUBSYSTEM_A_MASK);
    std::bitset<N> mask_B(SUBSYSTEM_B_MASK);
    std::vector<int> A_bits, B_bits;
    for (int i = 0; i < N; i++) {
        if (mask_A.test(i)) A_bits.push_back(i);
        if (mask_B.test(i)) B_bits.push_back(i);
    }
    int N_A = (int)A_bits.size();
    int N_B = (int)B_bits.size();
    int dim_A = 1 << N_A, dim_B = 1 << N_B;
    int dim = 1 << N;

    // TrB_V[a][a'] accumulates sum_zB <a,zB| V |a',zB> for intra-A operators V
    // only (V = the off-diagonal part of H). Equals dim_B * V_A^intra[a,a'].
    std::vector<std::vector<std::complex<double>>> TrB_V(
        dim_A, std::vector<std::complex<double>>(dim_A, 0.0));
    double sum_od2 = 0.0;
    Replica tmp;

    for (int a = 0; a < dim_A; a++) {
        std::bitset<N> cfg_A;
        for (int i = 0; i < N_A; i++) if ((a >> i) & 1) cfg_A.set(A_bits[i]);
        for (int b = 0; b < dim_B; b++) {
            std::bitset<N> cfg_B;
            for (int i = 0; i < N_B; i++) if ((b >> i) & 1) cfg_B.set(B_bits[i]);
            std::bitset<N> z = cfg_A | cfg_B;

            for (int k = 0; k < Nop; k++) {
                std::bitset<N> zprime = z ^ P_matrix[k];
                // Evaluate the D-coefficient at the POST-application state,
                // matching RenyiReplica::calc_d_for_config's convention.
                tmp.lattice = zprime;
                std::complex<double> d = tmp.calc_d(k);
                sum_od2 += std::norm(d);

                if (!(P_matrix[k] & mask_B).any()) {  // intra-A: never touches a B qubit
                    int aprime = 0;
                    for (int i = 0; i < N_A; i++) if (zprime.test(A_bits[i])) aprime |= (1 << i);
                    TrB_V[a][aprime] += d;
                }
            }
        }
    }

    od2_inf = sum_od2 / (double)dim;

    double intra_sq = 0.0;
    for (int a = 0; a < dim_A; a++)
        for (int ap = 0; ap < dim_A; ap++)
            if (a != ap)
                intra_sq += std::norm(TrB_V[a][ap] / (double)dim_B);
    intra_A_sq_over_dimA = intra_sq / (double)dim_A;
}

// Combines the two off-diagonal terms above into the O(beta^2) correction to
// ADD ON TOP OF ln(compute_classical_renyi_Z(beta, alpha_val)). Not valid on
// its own -- the classical (diagonal) sector must come from the exact sum,
// not a Taylor term, or the alpha=1 diagonal content gets double-counted.
inline double compute_quantum_anchor_correction(double beta, int alpha_val,
                                                 double od2_inf, double intra_A_sq_over_dimA) {
    return (beta * beta * 0.5) * (alpha_val * od2_inf
                                  + alpha_val * (alpha_val - 1) * intra_A_sq_over_dimA);
}

// ═══════════════════════════════════════════════════════════════════════════════
// compute_tau_int()  —  integrated autocorrelation time (Madras-Sokal windowing)
// ═══════════════════════════════════════════════════════════════════════════════
inline double compute_tau_int(const std::vector<double>& x) {
    int n = (int)x.size();
    if (n < 20) return 1.0;
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double C0 = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; C0 += d * d; }
    C0 /= n;
    if (C0 < 1e-15) return 1.0;
    double tau = 0.5;
    int max_lag = n / 4;
    for (int t = 1; t <= max_lag; t++) {
        double Ct = 0.0;
        for (int i = 0; i < n - t; i++) Ct += (x[i] - mean) * (x[i + t] - mean);
        Ct /= n;
        tau += Ct / C0;
        if ((double)t >= 5.0 * tau) break;
    }
    return std::max(0.5, tau);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adaptive recording stride  [OPT-5, audit F16 item 10]
//
// Problem: u is recorded every STEPS_PER_SAMP MC steps, but tau_int is
// typically hundreds to thousands of RECORDED samples at production beta --
// i.e. ~2*tau-fold oversampling, and each record costs a full K-way
// reweighting (K weight evaluations per replica). The audit measured ~80% of
// wall time going to this discarded-on-arrival reweighting.
//
// Fix: keep N_SAMPLES records and every existing downstream layout (u,
// ustream, tau_series, block bootstrap) completely unchanged -- only the
// number of MC steps BETWEEN records adapts, via a stride_factor multiplying
// STEPS_PER_SAMP. Every RECAL recorded samples, measure tau on the CURRENT
// recorded grid (tau_series already holds exactly this) and, if it's still
// well above the target spacing, multiply stride_factor up so the NEXT
// batch of recorded samples is drawn from a longer chain and therefore less
// correlated. This is a simple converging proportional controller with a
// deadband (only reacts when tau_rec > 4, avoiding oscillation once close to
// the target of ~2) -- not a compound formula, deliberately: correctness of
// MBAR/bootstrap doesn't depend on the stride being optimal, only on
// tau_series/u/ustream meaning what they've always meant (samples on
// whatever grid was actually recorded), so a simple, self-correcting rule is
// preferable to a precisely-tuned one.
// ═══════════════════════════════════════════════════════════════════════════════
static const int OPT5_RECAL_INTERVAL = 2000;   // recheck stride every this many recorded samples
static const double OPT5_TARGET_TAU_REC = 2.0; // aim for tau ~ this many samples on the recorded grid
static const int OPT5_MAX_STRIDE = 64;

// Given the current stride_factor and a freshly-measured tau_rec (max over
// rungs, already Allreduce(MAX)'d across ranks by the caller in the MPI
// driver), returns the new stride_factor. Pure function -- callers are
// responsible for calling it identically on every rank (see mbarpt_mpi.cpp)
// and for logging the change.
inline int adaptive_stride_update(int stride_factor, double tau_rec) {
    const double deadband = 2.0 * OPT5_TARGET_TAU_REC;  // == 4.0 at the default target of 2.0
    if (tau_rec <= deadband) return stride_factor;       // already close enough -- don't churn
    int factor = (int)std::ceil(tau_rec / deadband);
    long long new_stride = (long long)stride_factor * factor;
    if (new_stride < 1) new_stride = 1;
    if (new_stride > OPT5_MAX_STRIDE) new_stride = OPT5_MAX_STRIDE;
    return (int)new_stride;
}

// ═══════════════════════════════════════════════════════════════════════════════
// anderson_solve_linsys()  —  KKT solve for Anderson mixing coefficients
// ═══════════════════════════════════════════════════════════════════════════════
static const int ANDERSON_M = 5;

inline bool anderson_solve_linsys(
    const std::vector<std::vector<double>>& r_hist,
    int K,
    std::vector<double>& alpha)
{
    int h = (int)r_hist.size();
    std::vector<std::vector<double>> B(h, std::vector<double>(h, 0.0));
    for (int i = 0; i < h; i++)
        for (int j = i; j < h; j++) {
            double dot = 0.0;
            for (int k = 0; k < K; k++) dot += r_hist[i][k] * r_hist[j][k];
            B[i][j] = B[j][i] = dot;
        }
    int n = h + 1;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < h; j++)
            A[i][j] = B[i][j];
    for (int i = 0; i < h; i++) { A[i][h] = 1.0; A[h][i] = 1.0; }
    b[h] = 1.0;
    for (int col = 0; col < n; col++) {
        int pivot_row = col;
        double pivot_val = std::abs(A[col][col]);
        for (int row = col + 1; row < n; row++)
            if (std::abs(A[row][col]) > pivot_val) { pivot_val = std::abs(A[row][col]); pivot_row = row; }
        if (pivot_val < 1e-14) return false;
        std::swap(A[col], A[pivot_row]);
        std::swap(b[col], b[pivot_row]);
        double inv_pivot = 1.0 / A[col][col];
        for (int row = col + 1; row < n; row++) {
            double factor = A[row][col] * inv_pivot;
            for (int j = col; j < n; j++) A[row][j] -= factor * A[col][j];
            b[row] -= factor * b[col];
        }
    }
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    alpha.assign(x.begin(), x.begin() + h);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// log_exexfloat()  —  underflow-safe log(|w|) from an ExExFloat
// ═══════════════════════════════════════════════════════════════════════════════
inline double log_exexfloat(ExExFloat w) {
    w.normalize();
    double m = w.get_mantissa();
    if (m == 0.0) return -1e300;
    return std::log(std::abs(m)) + (double)w.get_exponent() * 0.6931471805599453;
}

// ═══════════════════════════════════════════════════════════════════════════════
// attempt_pt_swap()  —  Paper Section II.C, Metropolis test for a PT exchange move.
//
// [F17 fix] On ACCEPT, this now swaps the two replicas' CONFIGURATIONS (via
// SwapConfigWith) instead of their temperatures. Each slot's beta is pinned
// permanently, so slot k == rung k always — the acceptance MATH is unchanged
// (the ratio is symmetric either way: swap-configs and swap-temperatures are
// the same Metropolis test, just differing in which object ends up holding
// which physical state afterward), only the post-accept bookkeeping differs.
// See RenyiReplica::SwapConfigWith for the full rationale.
//
// Used directly by mbarpt.cpp (all pairs) and by mbarpt_mpi.cpp (intra-rank
// pairs); cross-rank boundary pairs use the MPI-specific mpi_boundary_swap()
// instead, since that needs to exchange the configuration over the network.
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: parallel tempering only works if "ladder slot k" reliably means
// "temperature beta_k" forever. That's what lets a cold, hard-to-mix replica
// occasionally borrow a well-mixed configuration from a hot, easy one --
// the whole point of running 32 temperatures side by side instead of just one.
// Plain-English: the ACCEPT/REJECT decision (the math right below) was always
// correct. But on an accepted swap, the OLD code exchanged which TEMPERATURE
// each replica object held, not which CONFIGURATION it held. After the very
// first accepted swap anywhere on the ladder, "slot k and slot k+1" no
// longer meant "neighboring temperatures" -- so later pairing rounds were
// often swapping physically distant temperatures, wasting the attempt and
// letting configurations get stuck. Fixed by swapping the physical
// configuration instead (SwapConfigWith, replica.hpp) and pinning each
// slot's temperature permanently -- "slot k == rung k" becomes a standing
// invariant, no bookkeeping needed anywhere.
// ─────────────────────────────────────────────────────────────────────────
inline bool attempt_pt_swap(RenyiReplica* rA, RenyiReplica* rB) {
    double beta_A = rA->beta, beta_B = rB->beta;  // fixed per-slot, restored below either way

    ExExFloat W_A_old = rA->currWeight;
    ExExFloat W_B_old = rB->currWeight;
    double log_W_A = log_exexfloat(W_A_old);
    double log_W_B = log_exexfloat(W_B_old);

    // Cross-probe: what would each configuration's weight be at the other's beta?
    rA->set_beta(beta_B);  // rA->currWeight = W(config_A; beta_B)
    rB->set_beta(beta_A);  // rB->currWeight = W(config_B; beta_A)

    double log_W_A2 = log_exexfloat(rA->currWeight);
    double log_W_B2 = log_exexfloat(rB->currWeight);

    double log_acc = (log_W_A2 + log_W_B2) - (log_W_A + log_W_B);
    if (std::log(val(rng)) < log_acc) {
        // ACCEPT: exchange configurations; slots keep their fixed beta.
        // Slot A should end up holding config_B evaluated at beta_A -- that's
        // exactly rB->currWeight right now (from the cross-probe above).
        // Slot B should end up holding config_A evaluated at beta_B -- rA->currWeight.
        ExExFloat weight_for_slot_A = rB->currWeight;
        ExExFloat weight_for_slot_B = rA->currWeight;

        rA->SwapConfigWith(*rB);  // physically exchange lattice/z/Sq/q/currEnergy

        rA->beta = beta_A;  rA->currWeight = weight_for_slot_A;  rA->InvalidateBeta();
        rB->beta = beta_B;  rB->currWeight = weight_for_slot_B;  rB->InvalidateBeta();
        return true;
    }

    // REJECT: configs never moved; just restore beta/weight (same as before).
    rA->beta = beta_A;
    rA->currWeight = W_A_old;
    rA->InvalidateBeta();

    rB->beta = beta_B;
    rB->currWeight = W_B_old;
    rB->InvalidateBeta();
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// qmax diagnostics  —  [qmax fix, audit P1 item 7]
//
// qmax truncation (a replica's move sequence needing length >= qmax and being
// silently capped, replica.hpp's qmax_achieved flag) was previously only ever
// checked once, post-thermalization, in mbarpt_mpi.cpp, and never treated as
// fatal anywhere. ED-quantified impact: qmax=100 gives S2~35 vs exact 0.855 at
// beta=10 (MBARPT_audit.txt). A replica silently capped at qmax is sampling
// from the WRONG (truncated) weight function for the rest of the run, not just
// producing noisier output -- this is a correctness bug, not a statistics one,
// so (unlike F4) there is no "honest wider error bar" version of it: it must
// stop the run rather than continue.
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fixes landed 2026-07-09 · reviewed 2026-09-10] ──────────────
// Physics: q is literally how many terms of the PMR series (Eq. 16) got
// included for one sample -- qmax_achieved means the sampler NEEDED more
// terms than the compiled qmax cap allows and got truncated instead.
// Plain-English: a truncated sample isn't "a bit noisier" -- it's sampled
// from a DIFFERENT (wrong, shorter-series) weight function for the rest of
// the run. Before this fix, hitting the cap was checked once, only in the
// MPI driver, only after thermalization, and was never fatal -- a run could
// finish looking perfectly converged while quietly reporting garbage (ED-
// measured: S2~35 vs the true 0.855 at beta=10). Now it's checked every
// single sample in both drivers and immediately stops the run.
// ─────────────────────────────────────────────────────────────────────────
struct QmaxStats {
    bool any_achieved = false;
    long long sum_q = 0;
    int max_q = 0;
    int n = 0;
};

inline QmaxStats scan_qmax(const std::vector<RenyiReplica*>& replicas) {
    QmaxStats s;
    s.n = (int)replicas.size();
    for (auto* r : replicas) {
        if (r->qmax_achieved) s.any_achieved = true;
        s.sum_q += r->q;
        if (r->q > s.max_q) s.max_q = r->q;
    }
    return s;
}

inline void print_qmax_progress(const QmaxStats& s, std::ostream& out) {
    out << "    q occupancy: mean=" << (s.n > 0 ? (double)s.sum_q / s.n : 0.0)
        << "  max=" << s.max_q << "  (qmax=" << qmax << ")\n";
}

// [DIAG swap-fallback] Sums RenyiReplica::swap_total_count/swap_fallback_count
// across the local rank's replicas. Callers diff two snapshots (taken at
// successive progress checkpoints) to get this block's fallback RATE, since
// the counters are cumulative for the whole run, not per-block.
// ── STUDY NOTE [instrumentation added 2026-09-09/10, NOT a fix yet · this
//    note written 2026-09-10] ────────────────────────────────────────────
// Physics: connects directly to the D_zero_prefix drift bug further down
// this file's history (see PartialUpdateAfterSwap in replica.hpp) -- this is
// how often that fix's "cheap patch" bails out to the "expensive full
// resync." q (sequence length) grows with beta*alpha, so the resync gets
// pricier exactly at the high-beta rungs that are already the slowest part
// of the ladder to sample.
// Plain-English: this is a stopwatch, not a fix. It just counts, per block
// of samples, what fraction of swap moves needed the expensive resync
// instead of the cheap patch -- so we can tell whether that cost is a real
// driver of the "high-beta/high-alpha runs are very slow" problem, or a
// red herring, before spending time optimizing it.
// ─────────────────────────────────────────────────────────────────────────
struct SwapFallbackStats {
    unsigned long long total = 0;
    unsigned long long fallback = 0;
};

inline SwapFallbackStats scan_swap_fallback(const std::vector<RenyiReplica*>& replicas) {
    SwapFallbackStats s;
    for (auto* r : replicas) {
        s.total    += r->swap_total_count;
        s.fallback += r->swap_fallback_count;
    }
    return s;
}

inline void print_swap_fallback_progress(const SwapFallbackStats& prev,
                                          const SwapFallbackStats& now,
                                          std::ostream& out) {
    unsigned long long d_total    = now.total    - prev.total;
    unsigned long long d_fallback = now.fallback - prev.fallback;
    double pct = d_total > 0 ? 100.0 * (double)d_fallback / (double)d_total : 0.0;
    out << "    swap fallback: " << d_fallback << "/" << d_total
        << " this block (" << std::fixed << std::setprecision(1) << pct << "%)\n";
}

inline void print_qmax_fatal_message(std::ostream& out) {
    out << "\n"
           "═══════════════════════════════════════════════════\n"
           "  FATAL [qmax]: qmax=" << qmax << " truncation hit during sampling.\n"
           "  At least one replica's move sequence needed length >= qmax and was\n"
           "  silently capped -- continuing would silently corrupt Z_alpha (audit:\n"
           "  qmax=100 gave S2~35 vs exact 0.855 at beta=10). See the qmax sizing\n"
           "  line printed just above (from this run's own q-histogram) for what\n"
           "  to raise qmax to; increase it in parameters.hpp and recompile.\n"
           "═══════════════════════════════════════════════════\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// qmax sizing recommendation
//
// MBARPT_audit.txt Sec. B's own prescribed methodology: "qmax > max over
// rungs of q_99.9 x 1.5 ... must be re-derived whenever N, beta_target, or
// alpha changes." A single hand-picked constant can never satisfy that for
// every future config -- it has to come from data. This tracks a per-rung
// histogram of q as actually observed (thermalization AND sampling, since
// job 428227 showed qmax_achieved firing already during thermalization),
// and reports the ACTUAL number the audit's rule calls for, instead of a
// generic "raise it and guess again" reminder. Printed automatically both
// when qmax-fatal fires (using whatever data was collected up to that
// point -- still informative even on an aborted run) and at the end of a
// successful run (so a DIFFERENT future beta_target/alpha/N has real data
// to size qmax from, per the audit's "re-derived whenever ... changes").
//
// Memory: num_replicas*(qmax+1) longs -- e.g. 32*301*8 =~ 77KB at this
// codebase's current scale. Negligible next to the u-matrix.
// ═══════════════════════════════════════════════════════════════════════════════
struct QmaxHistogram {
    std::vector<std::vector<long long>> counts;  // counts[k][q], k=0..num_replicas-1
};

inline QmaxHistogram make_qmax_histogram(int num_replicas) {
    QmaxHistogram h;
    h.counts.assign(num_replicas, std::vector<long long>(qmax + 1, 0));
    return h;
}

inline void record_qmax_histogram(QmaxHistogram& h, const std::vector<RenyiReplica*>& replicas) {
    for (size_t k = 0; k < replicas.size(); k++) {
        int q = replicas[k]->q;
        if (q < 0) q = 0;
        if (q > qmax) q = qmax;  // defensive only -- the fatal check should already have fired by now
        h.counts[k][q]++;
    }
}

// q value at which the cumulative count first reaches the given percentile
// (e.g. 0.999 for q_99.9) of that rung's total observations. Returns 0 if
// the rung has no observations yet (e.g. a fatal trigger on the very first
// checked point).
inline int qmax_percentile(const std::vector<long long>& counts, double percentile) {
    long long total = 0;
    for (long long c : counts) total += c;
    if (total == 0) return 0;
    long long target = (long long)std::ceil(percentile * (double)total);
    long long cum = 0;
    for (int q = 0; q < (int)counts.size(); q++) {
        cum += counts[q];
        if (cum >= target) return q;
    }
    return (int)counts.size() - 1;
}

// Local (this process's own rungs) max q_99.9 -- for the MPI driver, combine
// across ranks with MPI_Allreduce(MPI_MAX) on the returned value (max over
// ALL rungs = max over each rank's max over its own local rungs; no need to
// exchange full histograms, just this one int).
inline int local_max_q99_9(const QmaxHistogram& h) {
    int m = 0;
    for (const auto& row : h.counts) {
        int p = qmax_percentile(row, 0.999);
        if (p > m) m = p;
    }
    return m;
}

// censored=true on the qmax-fatal path: when qmax binds, every observed q is
// capped at the OLD qmax by construction (an insert that would exceed it is
// rejected before it can be recorded), so the histogram's q_99.9 -- and
// therefore the recommendation below -- is itself a lower bound on what the
// audit's rule actually calls for, not the final answer. The end-of-run
// (uncensored) print is the one to trust once a run completes without ever
// hitting the cap. [qmax fix audit] max_q99_9==0 (e.g. a resumed run whose
// own histogram never got any data before reaching this check) would
// otherwise print a nonsense "recommended qmax >= 0" -- report that plainly
// instead.
inline void print_qmax_recommendation(int max_q99_9, std::ostream& out, bool censored = false) {
    if (max_q99_9 == 0) {
        out << "  [qmax sizing] no q data collected this invocation -- no recommendation "
               "available (e.g. this run resumed straight past thermalization before any "
               "q was ever recorded).\n";
        return;
    }
    int recommended = (int)std::ceil(1.5 * (double)max_q99_9);
    out << "  [qmax sizing] observed max q_99.9 across all rungs = " << max_q99_9
        << "  ->  recommended qmax >= " << recommended
        << " (audit rule: qmax > 1.5 * q_99.9; current qmax=" << qmax << ")";
    if (censored)
        out << "  -- AT LEAST: this run's q values were capped at qmax=" << qmax
               << ", so the true q_99.9 may be higher than what got recorded; treat this "
                  "as a lower bound, not the final answer (the end-of-run report from a "
                  "run that completes without hitting the cap gives the uncensored number)";
    out << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// check_numerics_bounds()  —  [F10 fix] cheap one-time startup guards
//
// F10 (audit: SUSPECTED/latent, "low now, high later"): divdiff.hpp's
// invPowersOf2[] is indexed by ExExFloat exponent differences with NO bound
// check (maxexp=100000) -- an out-of-bounds read is possible once beta*N*(max
// coefficient) grows large enough (N>~10^3 sites, or very large beta*energy
// scale). mainQMC.hpp's factorial[] is a plain double, inf for k>=171.
//
// Deliberately NOT fixed by adding a branch inside ExExFloat's +/- operators:
// those run in the hottest of hot loops (every weight evaluation, every MC
// step), and this codebase's current regime (N~9-20, qmax=100) never actually
// triggers either issue (per the audit's own "SUSPECTED, not CONFIRMED" and
// "currently harmless" framing) -- paying a per-operation cost for a
// not-currently-observed failure mode is the wrong tradeoff. Instead: estimate
// the worst-case exponent scale ONCE at startup from the compiled Hamiltonian
// and BETA_TARGET, and warn loudly before the run if it's approaching the
// unchecked bound, so a future larger-N build fails loudly at startup instead
// of silently reading garbage mid-run.
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: not physics -- floating point safety. The weight machinery uses a
// custom extended-range float (ExExFloat, divdiff.hpp) specifically because
// Rényi weights can span hundreds of orders of magnitude at large beta*N,
// far beyond what an ordinary double can represent without overflow.
// Plain-English: two internal lookup tables (one for the extended-float
// math, one for factorials) have hard size limits that are never checked at
// the point of use. At this codebase's current scale (N=9) those limits
// aren't remotely close to being hit -- this is a one-time startup estimate
// that warns loudly if a much bigger future system starts approaching the
// danger zone, instead of silently reading past the end of an array mid-run.
// ─────────────────────────────────────────────────────────────────────────
inline void check_numerics_bounds() {
    double max_coeff = 0.0;
    for (int i = 0; i < D0_size; i++) max_coeff = std::max(max_coeff, std::abs(D0_coeff[i]));
    for (int k = 0; k < Nop; k++)
        for (int i = 0; i < D_size[k]; i++)
            max_coeff = std::max(max_coeff, std::abs(D_coeff[k][i]));

    // Conservative: beta*N*max_coeff bounds the largest single-term exponent
    // (base e) that CalcEnergy/weight evaluation can produce; convert to base 2.
    double est_max_exponent = BETA_TARGET * N * max_coeff * 1.4426950408889634;
    if (est_max_exponent > 0.5 * maxexp) {
        std::cerr << "\n  *** WARNING [F10]: estimated max ExExFloat exponent (~"
                  << est_max_exponent << ") is approaching divdiff.hpp's unchecked "
                  << "maxexp=" << maxexp << " bound. An out-of-bounds invPowersOf2[] "
                     "read is possible. Consider a smaller beta_target/N for this build, "
                     "or raise maxexp in divdiff.hpp.\n";
    }
    if (qmax >= 171) {
        // [qmax fix audit] Downgraded from "verify this" to a one-line note:
        // already verified, not just asserted. factorial[] is read only at
        // replica.hpp's factorial[p]/factorial[r] (cycle-related, bounded by
        // cycle_max_len -- a property of the compiled Hamiltonian's fixed
        // cycle structure) and factorial[len] with len=P.count()<=N (an
        // operator-support popcount, not a cycle length, but likewise
        // independent of qmax) inside #ifdef MEASURE_CUSTOM_OBSERVABLES in
        // the legacy Replica::measure_observable -- which the MBAR+PT
        // drivers never call. RenyiReplica never touches factorial[] at all.
        // So raising qmax past 171 cannot reach the inf-valued high entries;
        // re-verify this note if factorial[] ever gets a new caller.
        std::cerr << "\n  [F10 note] qmax=" << qmax << " >= 171 -- factorial[k] (mainQMC.hpp, "
                     "plain double) is inf for k>=171, but verified benign: it's never indexed "
                     "by q, only by small cycle-length/operator-support values independent of "
                     "qmax. See mbar_common.hpp's check_numerics_bounds for the verification.\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Block bootstrap  —  [F4 fix, audit P1 item 6]
//
// OLD bug: when thinning by stride=ceil(max_tau) would leave < MIN_NK=max(K,100)
// samples/rung, the code silently reverted to the FULL unthinned series and fed
// it to a plain row-bootstrap (independent draws per (rung,row)) as if the rows
// were i.i.d. -- precisely when mixing is worst (the audit found this fired in
// six separate production jobs). The row-bootstrap also drew independently
// PER RUNG, discarding whatever cross-rung correlation exists from replicas
// evolving in lockstep with periodic PT config exchange at the same sweep index.
//
// Fix: a moving-BLOCK bootstrap, block length = ceil(max_tau) (same quantity
// the old code called "stride"), applied to the RAW (unthinned) series, with
// the same block-start draw reused across every rung for a given resampled
// slot -- resampling whole sweeps, not independent rows. This is used for the
// bootstrap regardless of what the point-estimate solve did with its own
// (separately, still-thinned-for-compute-cost) copy of the data, so the
// uncertainty estimate is always honest even when the point estimate falls
// back to raw data for numerical-conditioning reasons.
//
// Per-driver usage: call plan_block_bootstrap() once (values are identical
// on every MPI rank, since block_len derives from a globally-Allgathered
// max_tau and N_SAMPLES is a compile-time constant -- no communication
// needed). Then for each of N_BOOT replicates, call draw_bootstrap_block_starts()
// with an RNG seeded IDENTICALLY on every rank (no mpi_rank in the seed) so
// every rank's local copy of a rung is resampled at the same time-slices as
// every other rank's rungs -- this is what "preserves cross-rung correlation"
// in a distributed setting without any extra communication.
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: not physics -- pure statistics. But it decides whether the
// error bars you quote against exact diagonalization can be trusted, or are
// falsely confident.
// Plain-English: consecutive Monte Carlo samples are correlated with their
// close-in-time neighbors, over a timescale tau_int. A bootstrap that
// resamples individual rows quietly pretends every row is an independent
// coin flip -- which UNDERSTATES uncertainty, worst exactly when correlation
// is worst (high beta). The audit found this understated some S2 error bars
// by 10-20x at beta=5. Fix: always resample whole correlated CHUNKS (length
// ~ tau_int) instead of single rows, for the error bar specifically --
// regardless of how the separate (unbiased either way) point estimate is
// computed.
// ─────────────────────────────────────────────────────────────────────────
struct BlockBootstrapPlan {
    int block_len;       // = old "stride"; block length in ORIGINAL (unthinned) samples
    int n_blocks_full;   // number of non-overlapping blocks available in the raw series
    int n_blocks_boot;   // <= n_blocks_full; capped so a bootstrap replicate stays bounded in memory
    int Nk_boot;          // = n_blocks_boot * block_len; samples/rung in one bootstrap replicate
};

inline BlockBootstrapPlan plan_block_bootstrap(double max_tau, int n_samples,
                                                int nk_boot_max_target = 5000) {
    BlockBootstrapPlan p;
    p.block_len = std::max(1, (int)std::ceil(max_tau));
    p.n_blocks_full = std::max(1, n_samples / p.block_len);
    p.n_blocks_boot = p.n_blocks_full;
    if ((long long)p.n_blocks_full * p.block_len > (long long)nk_boot_max_target) {
        p.n_blocks_boot = std::max(1, nk_boot_max_target / p.block_len);
    }
    p.Nk_boot = p.n_blocks_boot * p.block_len;
    return p;
}

// Draws n_blocks_boot block-start indices (into the ORIGINAL raw series, units
// of samples) with replacement from the n_blocks_full available non-overlapping
// blocks. Deterministic given rng's state -- call with an identically-seeded
// rng on every MPI rank to get identical draws without communication.
// [OPT-11] Out-param overload: fills a caller-owned vector instead of
// returning by value, so a driver looping over N_BOOT replicates can hoist
// the vector above the loop and reuse its capacity every replicate instead
// of allocating a fresh one each time (resizes only if size doesn't already
// match -- true after the first replicate).
inline void draw_bootstrap_block_starts(const BlockBootstrapPlan& p,
                                         std::mt19937& rng,
                                         std::vector<int>& out) {
    std::uniform_int_distribution<int> block_idx(0, p.n_blocks_full - 1);
    if ((int)out.size() != p.n_blocks_boot) out.resize(p.n_blocks_boot);
    for (int i = 0; i < p.n_blocks_boot; i++) out[i] = block_idx(rng) * p.block_len;
}

inline std::vector<int> draw_bootstrap_block_starts(const BlockBootstrapPlan& p,
                                                      std::mt19937& rng) {
    std::vector<int> starts;
    draw_bootstrap_block_starts(p, rng, starts);
    return starts;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sign diagnostics  —  [F9 fix]
//
// Negative-weight configurations are silently mapped to u=1e300 ("impossible")
// during u-matrix construction with no sign statistics reported anywhere. For
// the compiled bipartite AFM Hamiltonian this codebase currently targets, all
// closed Rényi configurations provably have even q -> weights are positive and
// this is benign (MBARPT_audit.txt F9). For any frustrated/non-bipartite
// Hamiltonian it would silently bias results instead of failing. Per the
// audit's own recommended scope ("at minimum, count and report negative-weight
// events" -- a full fix needs sign reweighting per the paper's footnote 1,
// which is out of scope here): this only adds counting/reporting, it does not
// change how negative-weight samples are used (still mapped to u=1e300).
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: the "sign problem" -- for some quantum Hamiltonians, individual
// Monte Carlo weights come out negative, which breaks the "weight = a
// probability" interpretation the whole sampling method relies on. THIS
// Hamiltonian is bipartite/antiferromagnetic, which provably keeps every
// closed configuration's weight non-negative -- a property of this specific
// physics, not something the code enforces or checks for on its own.
// Plain-English: negative weights were already silently treated as
// "impossible" with zero record kept. Now the code counts and reports how
// often that happens. For this Hamiltonian it should always read exactly
// zero -- this counter is a tripwire that would fire immediately if someone
// ever swapped in a different (frustrated) Hamiltonian where that's no
// longer true, instead of silently producing biased results.
// ─────────────────────────────────────────────────────────────────────────
struct SignStats {
    long long pos = 0;
    long long neg = 0;
};

inline void report_sign_stats(const SignStats& s, std::ostream& out) {
    if (s.neg == 0) return;
    double avg_sign = (double)(s.pos - s.neg) / (double)(s.pos + s.neg);
    out << "\n  *** WARNING [F9]: " << s.neg << " / " << (s.pos + s.neg)
        << " weight evaluations were NEGATIVE (sign problem detected).\n"
        << "  <sign> = " << avg_sign << "  These are mapped to u=1e300 (treated as "
           "zero-weight) in the MBAR input, same as before this warning existed.\n"
        << "  This codebase assumes positive-definite weights (true for the bipartite "
           "AFM Hamiltonian this was audited against, per MBARPT_audit.txt F9). If you "
           "changed the Hamiltonian, this may now be silently biasing results -- a real "
           "fix needs sign reweighting (paper footnote 1), not just this warning.\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Output sanity gate  —  [audit P1 item 8]
//
// A Rényi entropy S_alpha must satisfy 0 <= S_alpha <= N_A*ln2 (max entanglement
// entropy of an N_A-qubit subsystem) for ANY valid quantum state. A result
// outside this range is not "noisy", it's definitely wrong -- print a loud,
// unmissable failure message instead of letting a broken number quietly flow
// downstream into a results table (this is exactly the class of bug the
// original audit had to reverse-engineer from ED cross-checks after the fact).
// ═══════════════════════════════════════════════════════════════════════════════
// ── STUDY NOTE [fix landed 2026-07-09 · reviewed 2026-09-10] ────────────────
// Physics: a Rényi entropy S_alpha of an N_A-qubit subsystem MUST satisfy
// 0 <= S_alpha <= N_A*ln2 for literally any physically valid quantum state --
// that is a hard mathematical bound, not a rule of thumb or a sanity guess.
// Plain-English: this is the simplest possible bug catcher -- if the final
// number breaks a bound that CANNOT be broken by any real quantum state,
// something upstream is definitely wrong, full stop. Before this existed,
// impossible numbers (-3.2, 35.3) were printed without comment and would
// have silently gone into a results table -- this is exactly how several of
// the bugs in this file were originally caught by hand during the audit.
// ─────────────────────────────────────────────────────────────────────────
inline bool renyi_entropy_sane(double S_alpha, int N_A) {
    const double eps = 1e-6;  // tolerance for floating-point/statistical noise at the boundary
    return S_alpha > -eps && S_alpha < N_A * std::log(2.0) + eps;
}

inline void print_sanity_violation(double S_alpha, int alpha_val, int N_A, std::ostream& out) {
    // Job logs commonly merge stdout+stderr into one file. The results block
    // the drivers print to std::cout just before calling this may still sit
    // in cout's buffer while this message (std::cerr) is unbuffered, splicing
    // the FATAL text mid-line into later stdout output (seen in job 474004's
    // log, where the message appeared truncated at "[0, "). Flush cout first
    // so the merged log stays ordered, and flush `out` at the end.
    std::cout.flush();
    out << "\n"
           "═══════════════════════════════════════════════════\n"
           "  FATAL [output sanity gate]: S_" << alpha_val << " = " << S_alpha
        << " is outside the physically required range [0, N_A*ln2] = [0, "
        << (N_A * std::log(2.0)) << "].\n"
           "  This result is definitely wrong, not just noisy -- do not use it. Likely\n"
           "  causes: qmax truncation, insufficient thermalization/sampling, a stale or\n"
           "  mismatched z1_anchor.dat entry, or a bug. See MBARPT_audit.txt.\n"
           "═══════════════════════════════════════════════════\n"
        << std::flush;
}

// ═══════════════════════════════════════════════════════════════════════════════
// hamiltonian_checksum()  —  [F6 fix] structural hash of the compiled Hamiltonian
//
// Recompiling against a different H.txt (or regenerating hamiltonian.hpp with
// different prepare.cpp conventions) changes P_matrix/D0_coeff/D_coeff etc.
// silently -- there is no other provenance trail tying a z1_anchor.dat entry to
// the physics that produced it. Hashes every compiled table (FNV-1a over each
// bitset's string form and each coefficient's raw bits) so a Hamiltonian change
// changes the key and a stale entry is never matched.
// ═══════════════════════════════════════════════════════════════════════════════
inline uint64_t fnv1a_str(uint64_t h, const std::string& s) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
inline uint64_t fnv1a_double(uint64_t h, double v) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "double must be 8 bytes");
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; i++) { h ^= (bits >> (i * 8)) & 0xFFu; h *= 1099511628211ULL; }
    return h;
}
inline uint64_t hamiltonian_checksum() {
    uint64_t h = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
    h = fnv1a_str(h, std::to_string(N) + "," + std::to_string(Nop) + "," + std::to_string(D0_size));
    for (int i = 0; i < Nop; i++) h = fnv1a_str(h, P_matrix[i].to_string());
    for (int i = 0; i < D0_size; i++) {
        h = fnv1a_double(h, D0_coeff[i]);
        h = fnv1a_str(h, D0_product[i].to_string());
    }
    for (int k = 0; k < Nop; k++) {
        h = fnv1a_str(h, std::to_string(D_size[k]));
        for (int i = 0; i < D_size[k]; i++) {
            h = fnv1a_double(h, D_coeff[k][i]);
            h = fnv1a_str(h, D_product[k][i].to_string());
        }
    }
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════════
// z1_anchor.dat cache — I/O  [F5/F6 fix]
//
// Key now includes a Hamiltonian checksum, qmax, T_EQUIL, and STEPS_PER_SAMP
// on top of the original (beta_target, beta_min, N_BETA, N_SAMPLES, masks) --
// any of these changing invalidates a cached entry instead of silently reusing
// a stale one (F6). The bootstrap stderr is now stored alongside ln Z1 and
// returned on a cache hit, so S_alpha's error bar always includes the ln Z1
// contribution instead of silently treating a cache hit as zero-uncertainty
// (F5). Writes go through a per-process temp file + atomic rename instead of
// read-modify-write in place, so concurrent jobs sharing a directory can never
// observe (or produce) a half-written file (F6) -- the remaining race, two
// writers finishing at nearly the same time, resolves as last-writer-wins,
// which is a correct cache semantics (just not a merge), not corruption.
// ═══════════════════════════════════════════════════════════════════════════════
// [job 480705, 2026-07-23/24 audit finding] z1_anchor.dat's key previously
// had NO way to detect that replica.hpp's WEIGHT-COMPUTATION LOGIC changed --
// only physics parameters (beta_target, N_SAMPLES, Hamiltonian hash, etc.)
// invalidated an entry. The 2026-07-22 D_zero_prefix drift fix changed how
// weights are computed but NOT any of those physics parameters, so a
// z1_anchor entry cached by a PRE-fix binary kept silently matching and being
// reused by POST-fix runs. This was empirically confirmed: job 480705's
// alpha=3 pipeline used a fresh, correct ln Z_3 estimate (171.9233, close to
// ED-exact 171.6726) combined with a STALE cached ln Z_1 anchor
// (58.0826 -- independently checked against ED-exact 57.6851, a 4.66-sigma
// deviation given its own claimed stderr), producing S_3=1.162 vs the
// ED-exact 0.691 -- a ~3.6-sigma miss that looked like a correctness
// regression but was actually a stale-cache bug. ALARMING because
// ComputeSingleSegmentWeight (replica.hpp, the alpha=1 weight path) reads
// D_zero_prefix directly (`D_zero_prefix[replica][end] != ...[start]`) --
// the exact quantity the drift bug corrupted -- so alpha=1 runs were never
// immune to begin with, contrary to the intuition that the bug "only
// mattered for alpha>=2 cut enumeration."
//
// Fix: WEIGHT_LOGIC_VERSION folds into both this key and CheckpointKey below.
// Bump it whenever a change to replica.hpp's weight/enumeration logic could
// change results for an already-cached (beta_target, N_SAMPLES, Hamiltonian,
// ...) combination -- the 2026-07-22 fix is retroactively "version 1"; the
// NEXT such fix must bump this to 2, and so on. This makes stale entries
// mechanically inert instead of relying on someone remembering to delete
// z1_anchor.dat/the checkpoint files by hand after every correctness fix.
// ── STUDY NOTE [fix landed 2026-07-29 · reviewed 2026-09-10] ────────────────
// Physics: ties directly to F5/F6 below -- a stale cached ln Z_1 poisons
// every S_alpha computed from it (S_alpha = (alpha*lnZ1 - lnZalpha)/(alpha-1)),
// since ln Z_1 isn't recomputed on a cache hit, it's just trusted.
// Plain-English: the D_zero_prefix drift fix in replica.hpp changed HOW
// weights get computed internally, but changed none of the run's physics
// *settings* (same beta, same N_SAMPLES, ...) -- so the cache's old matching
// rule below couldn't tell "before the fix" apart from "after the fix," and
// kept serving a pre-fix (subtly wrong) cached answer to post-fix runs. This
// integer is a version tag that must be bumped BY HAND every time the
// weight-computation logic itself changes, so a future fix like this one
// can't repeat the exact same silent-staleness mistake.
// ─────────────────────────────────────────────────────────────────────────
static const int WEIGHT_LOGIC_VERSION = 1;

static const char* Z1_ANCHOR_FILE = "z1_anchor.dat";

// ── STUDY NOTE [F6+F5 fix landed 2026-07-09 · reviewed 2026-09-10] ──────────
// Physics: not physics directly -- a caching/provenance bug. But Z_1 feeds
// EVERY S_alpha calculation, so a stale or under-reported Z_1 poisons
// whatever alpha you compute afterward.
// F6, plain-English: this cache file remembers "the answer to the expensive
// Z_1 calculation" so future alpha>=2 runs don't have to redo it. But the
// old key only recorded a few of the settings that go into that answer
// (beta, sample count, masks) -- not which Hamiltonian was compiled in, nor
// qmax/thermalization settings. Recompiling with different physics could
// silently reuse an answer computed under DIFFERENT physics, no error at
// all. Fixed by including a checksum of the whole compiled Hamiltonian in the key.
// F5, plain-English: separately, even a correctly-matched cache hit set the
// cached ln Z_1's uncertainty to exactly zero -- as if it were known
// perfectly. The S_alpha error formula has an alpha^2 multiplier on that
// term, so silently dropping it hid up to a +-1.4 contribution in one case.
// Now the cache stores and returns the uncertainty alongside the value.
// ─────────────────────────────────────────────────────────────────────────
struct Z1AnchorKey {
    double beta_target, beta_min;
    int n_beta, n_samples, qmax_val, t_equil, steps_per_samp, weight_logic_version;
    uint64_t ham_hash;
};

inline Z1AnchorKey current_z1_anchor_key() {
    Z1AnchorKey k;
    k.beta_target = BETA_TARGET;
    k.beta_min = BETA_MIN;
    k.n_beta = N_BETA;
    k.n_samples = N_SAMPLES;
    k.qmax_val = qmax;
    k.t_equil = T_EQUIL;
    k.steps_per_samp = STEPS_PER_SAMP;
    k.weight_logic_version = WEIGHT_LOGIC_VERSION;
    k.ham_hash = hamiltonian_checksum();
    return k;
}

inline bool z1_key_matches(const Z1AnchorKey& a, double bt, double bm, int nb, int ns,
                            int qm, int te, int sps, int wlv, uint64_t hash,
                            const std::string& ma, const std::string& mb) {
    return (std::abs(bt - a.beta_target) <= 1e-10 * std::abs(a.beta_target) + 1e-15)
        && (std::abs(bm - a.beta_min)    <= 1e-10 * std::abs(a.beta_min)    + 1e-15)
        && nb == a.n_beta && ns == a.n_samples && qm == a.qmax_val
        && te == a.t_equil && sps == a.steps_per_samp
        && wlv == a.weight_logic_version && hash == a.ham_hash
        && ma == SUBSYSTEM_A_MASK && mb == SUBSYSTEM_B_MASK;
}

// Returns true and fills ln_z1/stderr_out if a matching cache entry exists.
// Lines in an older format (pre-F6 short format, or pre-weight_logic_version)
// never match the new key -- the trailing `>> wlv` extraction itself fails on
// a line that doesn't have that field, so they become inert, forcing a fresh
// (correctly-keyed) run rather than risking a silent stale match.
inline bool read_z1_anchor(double& ln_z1, double& stderr_out) {
    std::ifstream f(Z1_ANCHOR_FILE);
    if (!f.is_open()) return false;

    Z1AnchorKey want = current_z1_anchor_key();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',') c = ' ';
        std::istringstream ss(line);
        double bt, bm, lnz, stderr_val; int nb, ns, qm, te, sps, wlv; unsigned long long hash;
        std::string ma, mb;
        if (!(ss >> bt >> bm >> nb >> ns >> ma >> mb >> qm >> te >> sps >> wlv >> hash >> lnz >> stderr_val))
            continue;
        if (z1_key_matches(want, bt, bm, nb, ns, qm, te, sps, wlv, (uint64_t)hash, ma, mb)) {
            ln_z1 = lnz;
            stderr_out = stderr_val;
            return true;
        }
    }
    return false;
}

// Writes (or overwrites) the ln Z1 + stderr entry for the current run parameters.
inline void write_z1_anchor(double ln_z1, double stderr_val) {
    Z1AnchorKey want = current_z1_anchor_key();

    std::vector<std::string> lines;
    {
        std::ifstream in_file(Z1_ANCHOR_FILE);
        if (in_file.is_open()) {
            std::string l;
            while (std::getline(in_file, l)) lines.push_back(l);
        }
    }

    std::ostringstream oss;
    oss << std::setprecision(15)
        << want.beta_target << "," << want.beta_min << ","
        << want.n_beta << "," << want.n_samples << ","
        << SUBSYSTEM_A_MASK << "," << SUBSYSTEM_B_MASK << ","
        << want.qmax_val << "," << want.t_equil << "," << want.steps_per_samp << ","
        << want.weight_logic_version << "," << want.ham_hash << ","
        << ln_z1 << "," << stderr_val;
    std::string new_entry = oss.str();

    bool replaced = false;
    for (auto& l : lines) {
        if (l.empty() || l[0] == '#') continue;
        std::string tmp_l = l;
        for (char& c : tmp_l) if (c == ',') c = ' ';
        std::istringstream ss(tmp_l);
        double bt, bm, lnz_dummy, stderr_dummy; int nb, ns, qm, te, sps, wlv; unsigned long long hash;
        std::string ma, mb;
        if (!(ss >> bt >> bm >> nb >> ns >> ma >> mb >> qm >> te >> sps >> wlv >> hash >> lnz_dummy >> stderr_dummy))
            continue;  // old-format or malformed line -- leave as inert history, don't touch
        if (z1_key_matches(want, bt, bm, nb, ns, qm, te, sps, wlv, (uint64_t)hash, ma, mb)) {
            l = new_entry;
            replaced = true;
        }
    }
    if (!replaced) {
        if (lines.empty()) {
            lines.push_back("# z1_anchor.dat — cached ln Z1(beta_target) values for MBAR+PT");
            lines.push_back("# Format: beta_target,beta_min,n_beta,n_samples,mask_a,mask_b,"
                             "qmax,t_equil,steps_per_samp,weight_logic_version,hamiltonian_hash,ln_z1,stderr");
        }
        lines.push_back(new_entry);
    }

    // [F6 fix] Atomic write: temp file (unique per process) + rename(), so a
    // concurrent reader never observes a partially-written file, and two
    // concurrent writers resolve as last-writer-wins instead of interleaved
    // corruption. rename() is atomic on the same filesystem on POSIX (the
    // only platform this codebase compiles on).
    std::ostringstream tmp_name_ss;
    tmp_name_ss << Z1_ANCHOR_FILE << ".tmp." << (long)getpid();
    std::string tmp_name = tmp_name_ss.str();
    {
        std::ofstream out_file(tmp_name);
        for (const auto& l : lines) out_file << l << "\n";
    }
    if (std::rename(tmp_name.c_str(), Z1_ANCHOR_FILE) != 0) {
        std::cerr << "WARNING: failed to atomically rename " << tmp_name
                   << " to " << Z1_ANCHOR_FILE << " -- z1_anchor cache not updated.\n";
        return;
    }
    std::cout << "  Saved ln Z1 = " << std::scientific << std::setprecision(8) << ln_z1
              << "  (stderr=" << stderr_val << ")  ->  " << Z1_ANCHOR_FILE << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpointing  —  [F20 fix]
//
// Nothing intermediate was ever saved before this: the legacy Replica class
// has a working save/load scaffold (replica.hpp's save_QMC_data/load_QMC_data)
// but it was never wired to a signal handler or ported to RenyiReplica/the
// MBAR+PT drivers -- SAVE_UNFINISHED_CALCULATION/RESUME_CALCULATION are both
// commented out in parameters.hpp, so it's fully inert even for the legacy
// path. The audit logged >=5 multi-day jobs lost this way, two of which died
// inside thermalization.
//
// Per the user's scope decision: SIGTERM-triggered only (Slurm's
// `--signal=B:TERM@<secs>` gives a warning before the hard kill at the time
// limit) -- NOT a periodic autosave. This does not help against an ungraceful
// kill with no warning (OOM, node failure); it specifically targets the
// "job hit its time limit" failure mode, which was the majority of the
// audit's incident log.
//
// The u-matrix is streamed to disk incrementally (append_ustream_sample,
// called once per sample from the sampling loop) rather than only being
// written at checkpoint time -- so the SIGTERM handler's checkpoint write is
// always small (replica states only, no multi-GB flush) regardless of how
// much sampling has happened, which matters because SIGTERM grace periods
// are typically short. The in-memory `u`/`u_local` array is NOT removed --
// it's still built up during sampling exactly as before (Phase 3/3.5's data
// flow is unchanged); the stream file is a redundant durability copy, read
// back ONLY when resuming a checkpoint that was already partway through
// sampling, to reconstruct the in-memory prefix before the sampling loop
// continues filling in the rest.
// ═══════════════════════════════════════════════════════════════════════════════
#include <csignal>

// volatile sig_atomic_t is the only type the standard guarantees is safe to
// read/write from a signal handler without a data race. Checked (not
// cleared) by both drivers at existing synchronization points -- see
// mbarpt.cpp/mbarpt_mpi.cpp's thermalization/sampling loops.
// ── STUDY NOTE [F20/F21 fixes landed 2026-07-10 · reviewed 2026-09-10] ──────
// Physics: none -- pure infrastructure. Doesn't change a single number the
// simulation produces.
// Plain-English: a multi-day cluster job that hits its wall-time limit gets
// hard-killed by Slurm -- and until this fix, that meant losing 100% of the
// progress with nothing to resume from (at least 5 multi-day jobs lost this
// way, two of them mid-thermalization). Now the code catches Slurm's
// warning signal (SIGTERM, sent before the hard kill) at the same cadence
// it already checks for PT swaps, saves just enough state (replica
// configs + the samples already collected) to pick back up exactly where it
// left off, and exits with a specific code meaning "resubmit the same job."
// ─────────────────────────────────────────────────────────────────────────
volatile sig_atomic_t g_checkpoint_requested = 0;

// [F20 diagnostic, added after job 480826] Set once at startup (before the
// handler is armed), read-only afterward -- not a race despite being a plain
// int, since the write always happens-before install_checkpoint_signal_handler().
// -1 for the single-process OpenMP driver (mbarpt.cpp has no rank concept).
static int g_signal_handler_rank = -1;

// job 480826 hit its Slurm --time limit and was hard-killed mid-sampling with
// NO "[checkpoint] SIGTERM received" line anywhere in the log -- meaning
// either (a) SIGTERM never reached this handler at all (plausible: the job
// launches via plain `mpirun ... &` from inside the batch script rather than
// `srun`, so Slurm's --signal delivery to remote-node ranks depends on
// mpirun's own signal propagation, not Slurm's job-step tracking), or (b) it
// reached the handler but write_checkpoint()+MPI_Finalize() didn't complete
// before the grace period ran out. Case (a) and (b) are indistinguishable
// from the job log alone. This handler now does an async-signal-safe raw
// write(2) the INSTANT the signal is caught -- before anything else touches
// g_checkpoint_requested or takes the Allreduce path -- so the next run's log
// unambiguously answers "did the handler even fire" independent of whether
// the full checkpoint sequence had time to finish. Only async-signal-safe
// calls are used here (write(); no iostream/printf, which are NOT safe to
// call from a signal handler).
inline void handle_checkpoint_signal(int) {
    static const char prefix[] = "[checkpoint] SIGTERM caught by handler, rank=";
    char buf[64];
    size_t n = sizeof(prefix) - 1;
    memcpy(buf, prefix, n);
    int r = g_signal_handler_rank;
    char digits[12];
    int nd = 0;
    if (r < 0) { digits[nd++] = '-'; r = -r; }
    char tmp[12]; int nt = 0;
    if (r == 0) tmp[nt++] = '0';
    while (r > 0) { tmp[nt++] = '0' + (r % 10); r /= 10; }
    while (nt > 0) digits[nd++] = tmp[--nt];
    memcpy(buf + n, digits, nd); n += nd;
    buf[n++] = '\n';
    ssize_t written = write(STDERR_FILENO, buf, n);
    (void)written;  // nothing sensible to do with a failed diagnostic write
    g_checkpoint_requested = 1;
}

inline void install_checkpoint_signal_handler(int rank = -1) {
    g_signal_handler_rank = rank;
    std::signal(SIGTERM, handle_checkpoint_signal);
    // [2026-11-03] SIGUSR1 as well.  Open MPI 5 (PRRTE) aborts the whole job within a fraction of a second when mpirun receives SIGTERM, but it
    // forwards SIGUSR1 to every rank untouched, so the Slurm job script now asks Slurm for --signal=B:USR1@120.  Same handler: it only sets the
    // flag that the sampling loop's Allreduce already checks, so nothing else in the program changes (the log line still says SIGTERM).
    std::signal(SIGUSR1, handle_checkpoint_signal);
}

// [OPT-5] Bumped whenever the checkpoint file's binary LAYOUT changes (not
// just the physics parameters CheckpointKey already tracks). Kept as the
// FIRST field so a file written by an OLDER layout (without this field, or
// with fields in a different order/size) is read into a struct of the wrong
// total size -- the version check below then fails on essentially-random
// leftover bytes, giving a clean "stale checkpoint, start fresh" rather than
// silently misinterpreting a phase/step/stride_factor that was never there.
// v3: added weight_logic_version -- see WEIGHT_LOGIC_VERSION's comment above
// (job 480705 finding): a checkpoint's saved replica state (SaveState/
// LoadState) was captured under whatever weight-computation logic the
// binary that wrote it had; resuming that state under a binary with DIFFERENT
// weight logic risks the same class of silent staleness the z1_anchor bug
// exhibited. Adding the field also changes the struct SIZE, which alone
// already invalidates every pre-existing checkpoint via the size-mismatch
// path this comment block describes -- the explicit field is for clarity/
// defense-in-depth, not the only thing making old files fail.
#define CHECKPOINT_FORMAT_VERSION 5  // v5: overlap table hash includes generator indices; v4: overlap fields; v2: added stride_factor (OPT-5); v3: added weight_logic_version
                                     // v4: [OVERLAP] overlap mode, Hamiltonian pair, both Hamiltonians, actual ladder

// [OVERLAP] Checksum of the ladder actually used by the current pipeline (set by the
// driver before any checkpoint is read or written). Part of the checkpoint key, so a
// restart can never append samples taken on a different ladder.
inline uint64_t g_ckpt_ladder_hash = 0;
inline uint64_t ladder_checksum(const std::vector<double>& ladder) {
    uint64_t h = 1469598103934665603ULL;
    for (double v : ladder) h = fnv1a_double(h, v);
    return h;
}
// [OVERLAP] Checksum of both overlap Hamiltonians (OV_* tables); constant outside overlap mode.
inline uint64_t overlap_tables_checksum() {
    uint64_t h = 1469598103934665603ULL;
#ifdef OVERLAP_MODE
    // Hamiltonian index, generator index and term counts are hashed too: the
    // off-diagonal Z-masks are often all zero, so without the generator index two
    // H2 tables with the same coefficients on different bonds collided.
    for (int a = 0; a < OV_NHAM; a++) {
        h = fnv1a_str(h, "H" + std::to_string(a) + ":" + std::to_string(OV_D0_size[a]));
        for (int k = 0; k < OV_D0_size[a]; k++) {
            h = fnv1a_double(h, OV_D0_coeff[a][k]);
            h = fnv1a_str(h, OV_D0_product[a][k].to_string());
        }
        for (int j = 0; j < Nop; j++) {
            h = fnv1a_str(h, "g" + std::to_string(j) + ":" + std::to_string(OV_D_size[a][j]));
            for (int k = 0; k < OV_D_size[a][j]; k++) {
                h = fnv1a_double(h, OV_D_coeff[a][j][k]);
                h = fnv1a_str(h, OV_D_product[a][j][k].to_string());
            }
        }
    }
#endif
    return h;
}

struct CheckpointKey {
    int format_version;
    int alpha_val, n_beta, n_samples, t_equil, steps_per_samp, qmax_val, num_replicas;
    int weight_logic_version;
    double beta_target, beta_min;
    uint64_t ham_hash;
    int overlap_mode;          // [OVERLAP] 0 = Renyi, 1 = trace variant, 2 = |+> open walks
    int ov_pair;               // [OVERLAP] 10*(Hamiltonian of copy 0) + (Hamiltonian of copy 1); -1 otherwise
    uint64_t ov_tables_hash;   // [OVERLAP] checksum of both Hamiltonians
    uint64_t ladder_hash;      // checksum of the actual ladder values
};

inline CheckpointKey current_checkpoint_key(int alpha_val, int num_replicas) {
    CheckpointKey k;
    k.format_version = CHECKPOINT_FORMAT_VERSION;
    k.alpha_val = alpha_val;
    k.n_beta = N_BETA;
    k.n_samples = N_SAMPLES;
    k.t_equil = T_EQUIL;
    k.steps_per_samp = STEPS_PER_SAMP;
    k.qmax_val = qmax;
    k.num_replicas = num_replicas;
    k.weight_logic_version = WEIGHT_LOGIC_VERSION;
    k.beta_target = BETA_TARGET;
    k.beta_min = BETA_MIN;
    k.ham_hash = hamiltonian_checksum();
#if defined(OVERLAP_PLUS)
    k.overlap_mode = 2;
#elif defined(OVERLAP_MODE)
    k.overlap_mode = 1;
#else
    k.overlap_mode = 0;
#endif
#ifdef OVERLAP_MODE
    k.ov_pair = 10 * g_ov_ham[0] + g_ov_ham[1];
#else
    k.ov_pair = -1;
#endif
    k.ov_tables_hash = overlap_tables_checksum();
    k.ladder_hash = g_ckpt_ladder_hash;
    return k;
}

// beta_target/beta_min compared exactly (not with a tolerance, unlike
// z1_anchor's key match) -- a checkpoint is only ever read back by the same
// kind of run that wrote it (no cross-run reuse the way z1_anchor.dat is
// deliberately shared across alpha values), so there's no legitimate reason
// for these to differ by rounding at all; an exact mismatch is exactly the
// "don't resume into the wrong beta-ladder" case this key exists to catch.
inline bool checkpoint_key_matches(const CheckpointKey& a, const CheckpointKey& b) {
    return a.format_version == b.format_version
        && a.alpha_val == b.alpha_val && a.n_beta == b.n_beta && a.n_samples == b.n_samples
        && a.t_equil == b.t_equil && a.steps_per_samp == b.steps_per_samp
        && a.qmax_val == b.qmax_val && a.num_replicas == b.num_replicas
        && a.weight_logic_version == b.weight_logic_version
        && a.beta_target == b.beta_target && a.beta_min == b.beta_min
        && a.ham_hash == b.ham_hash
        && a.overlap_mode == b.overlap_mode && a.ov_pair == b.ov_pair
        && a.ov_tables_hash == b.ov_tables_hash && a.ladder_hash == b.ladder_hash;
}

enum class CheckpointPhase : int { THERMALIZING = 0, SAMPLING = 1 };

// rank < 0 means "no rank suffix" (the OpenMP driver, a single process).
inline std::string checkpoint_state_path(int alpha_val, int rank) {
    std::ostringstream oss;
    oss << "renyi_checkpoint_a" << alpha_val;
    if (rank >= 0) oss << "_r" << rank;
    oss << ".dat";
    return oss.str();
}
inline std::string checkpoint_ustream_path(int alpha_val, int rank) {
    std::ostringstream oss;
    oss << "renyi_ustream_a" << alpha_val;
    if (rank >= 0) oss << "_r" << rank;
    oss << ".bin";
    return oss.str();
}

// Writes header + one RenyiReplica::SaveState() block per replica. Atomic
// temp-file+rename, same reasoning as z1_anchor.dat (F6): a resubmitted job
// starting up concurrently with... itself finishing (shouldn't happen, but
// cheap to make impossible) never observes a half-written file.
inline void write_checkpoint(int alpha_val, int rank,
                              CheckpointPhase phase, int step_or_samp,
                              int stride_factor,  // [OPT-5] 1 during thermalization
                              const std::vector<RenyiReplica*>& replicas) {
    std::string path = checkpoint_state_path(alpha_val, rank);
    std::ostringstream tmp_name_ss;
    tmp_name_ss << path << ".tmp." << (long)getpid();
    std::string tmp_name = tmp_name_ss.str();
    {
        std::ofstream out(tmp_name, std::ios::binary);
        CheckpointKey key = current_checkpoint_key(alpha_val, (int)replicas.size());
        out.write((const char*)&key, sizeof(key));
        int phase_i = (int)phase;
        out.write((const char*)&phase_i, sizeof(phase_i));
        out.write((const char*)&step_or_samp, sizeof(step_or_samp));
        out.write((const char*)&stride_factor, sizeof(stride_factor));  // [OPT-5]
        for (auto* r : replicas) r->SaveState(out);
    }
    if (std::rename(tmp_name.c_str(), path.c_str()) != 0) {
        std::cerr << "WARNING: failed to atomically rename " << tmp_name << " to " << path
                   << " -- checkpoint not saved.\n";
        return;
    }
    std::cerr << "[checkpoint] saved " << path << "  (phase="
              << (phase == CheckpointPhase::THERMALIZING ? "thermalizing" : "sampling")
              << ", progress=" << step_or_samp << ")\n";
}

// Returns true and fills phase/step_or_samp/replicas on success. Returns
// false (leaving replicas untouched -- caller starts fresh) if no checkpoint
// file exists, or its key doesn't match the current run parameters -- same
// "stale file is silently inert, not corrupting" philosophy as z1_anchor
// (F6): recompiling with different parameters just starts over instead of
// crashing or silently resuming into the wrong run.
inline bool read_checkpoint(int alpha_val, int rank,
                             CheckpointPhase& phase, int& step_or_samp,
                             int& stride_factor,  // [OPT-5]
                             std::vector<RenyiReplica*>& replicas) {
    std::string path = checkpoint_state_path(alpha_val, rank);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;

    CheckpointKey file_key;
    in.read((char*)&file_key, sizeof(file_key));
    if (!in) return false;
    CheckpointKey want = current_checkpoint_key(alpha_val, (int)replicas.size());
    if (!checkpoint_key_matches(file_key, want)) return false;

    int phase_i;
    in.read((char*)&phase_i, sizeof(phase_i));
    in.read((char*)&step_or_samp, sizeof(step_or_samp));
    in.read((char*)&stride_factor, sizeof(stride_factor));  // [OPT-5]
    if (!in) return false;
    phase = (CheckpointPhase)phase_i;

    for (auto* r : replicas) {
        r->LoadState(in);
        r->set_beta(r->beta);  // rebuild beta_pow_factorial/currWeight/caches from loaded state
    }
    return true;
}

// Opens the u-stream file for appending. If already_written_samples > 0
// (resuming), first truncates the file to EXACTLY that many samples' worth
// of bytes -- discards any possible torn trailing write from an ungraceful
// kill that raced past the last graceful (SIGTERM-triggered) checkpoint --
// then appends. Otherwise (fresh start) truncates to empty first, so a stale
// file from an abandoned earlier attempt (different parameters -- the
// checkpoint key already rejected it) is never appended onto.
inline std::ofstream open_ustream_for_append(const std::string& path,
                                              int already_written_samples,
                                              int block_doubles) {
    if (already_written_samples > 0) {
        truncate(path.c_str(),
                 (long long)already_written_samples * block_doubles * (long long)sizeof(double));
    } else {
        std::ofstream(path, std::ios::binary | std::ios::trunc);  // create-or-empty, then closes
    }
    return std::ofstream(path, std::ios::binary | std::ios::app);
}

// Appends one sample's worth of data to the stream file, reading directly
// from the in-memory u array's already-filled slice for this samp (k-major:
// rung 0's K_total values, then rung 1's, ...) -- one source of truth for
// what "this sample's data" means, no separate buffer to keep in sync.
inline void append_ustream_sample(std::ofstream& out, const double* u, int samp,
                                   int num_replicas, int K_total, int N_SAMPLES_TOTAL) {
    for (int k = 0; k < num_replicas; k++)
        out.write((const char*)(u + ((size_t)k * N_SAMPLES_TOTAL + samp) * K_total),
                   (std::streamsize)K_total * sizeof(double));
}

// Reads back n_samples_to_read samples from the stream file (on-disk layout
// matches append_ustream_sample: samp-major, then k, then K_total columns)
// and repacks into the in-memory k-major layout u[(k*N_SAMPLES_TOTAL+samp)*
// K_total+j]. Used only once, right after a mid-sampling checkpoint is
// loaded, to reconstruct the prefix an earlier process invocation streamed
// before this invocation's own sampling loop continues filling in the rest.
//
// Returns true only if all n_samples_to_read samples were read in full.
// Returns false (u's prefix left partially/not filled -- do not use it) if
// the file is missing or shorter than the checkpoint claims, e.g. the
// checkpoint (.dat) and stream (.bin) files were separated (one deleted or
// restored from a backup without the other). The caller MUST treat false as
// fatal: silently proceeding would feed zero-filled (physically bogus) rows
// into MBAR, exactly the class of silent corruption the qmax-fatal/output-
// sanity-gate checks elsewhere exist to prevent. Deliberately returns a
// status instead of aborting here, since the right abort mechanism differs
// by driver (MPI needs MPI_Abort so other ranks waiting on the next
// collective aren't left hanging; OpenMP can just exit()).
inline bool read_ustream_prefix(const std::string& path, int n_samples_to_read,
                                 int num_replicas, int K_total, int N_SAMPLES_TOTAL,
                                 double* u) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    std::vector<double> row_block((size_t)num_replicas * K_total);
    std::streamsize row_bytes = (std::streamsize)(row_block.size() * sizeof(double));
    for (int samp = 0; samp < n_samples_to_read; samp++) {
        in.read((char*)row_block.data(), row_bytes);
        if (in.gcount() != row_bytes) return false;  // missing/truncated -- caller must abort
        for (int k = 0; k < num_replicas; k++)
            std::copy_n(row_block.data() + (size_t)k * K_total, K_total,
                        u + ((size_t)k * N_SAMPLES_TOTAL + samp) * K_total);
    }
    return true;
}

// Deletes a run's checkpoint + stream files once it completes normally --
// otherwise a stale-but-key-matching checkpoint from a finished run would
// make the NEXT identical-parameters run think it can "resume" past
// N_SAMPLES samples already done, skipping straight to Phase 3 on bogus
// (actually-empty) data.
//
// [2026-09-14] With MBAR_KEEP_SAMPLES (parameters.hpp) the u-stream is RENAMED
// to renyi_samples_a<alpha>[_r<rank>].bin instead of deleted, so completed runs
// can be re-analysed offline (tools/ustream_recheck.py). Deleting the state
// file alone already prevents the bogus-resume problem above: read_checkpoint()
// returns false without it, and a fresh start truncates any ustream file.
inline std::string kept_samples_path(int alpha_val, int rank) {
    std::ostringstream oss;
    oss << "renyi_samples_a" << alpha_val;
    if (rank >= 0) oss << "_r" << rank;
    oss << ".bin";
    return oss.str();
}

inline void clear_checkpoint(int alpha_val, int rank) {
    std::remove(checkpoint_state_path(alpha_val, rank).c_str());
#if defined(MBAR_KEEP_SAMPLES) && MBAR_KEEP_SAMPLES
    std::rename(checkpoint_ustream_path(alpha_val, rank).c_str(),
                kept_samples_path(alpha_val, rank).c_str());
#else
    std::remove(checkpoint_ustream_path(alpha_val, rank).c_str());
#endif
}

#endif // MBAR_COMMON_HPP
