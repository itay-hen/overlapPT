// ═══════════════════════════════════════════════════════════════════════════════
// mbarpt_mpi.cpp — MPI version of MBAR + Parallel Tempering for Rényi Entropy
// ═══════════════════════════════════════════════════════════════════════════════
//
// Replaces OpenMP shared-memory parallelism with MPI distributed-memory:
//
//   OpenMP model: all K replicas on one node, threads share memory.
//   MPI model:    K replicas distributed across P ranks in variable-size
//                 contiguous blocks sized to balance per-block COST, not
//                 count -- see partition_rungs_by_cost() [OPT-8] -- since q
//                 (and hence per-step cost) grows with beta, an equal K/P
//                 split makes the coldest rank the straggler for every
//                 collective (audit F12).
//                 Each rank evolves its local replicas sequentially.
//                 PT swaps within a rank are sequential (no change).
//                 PT swaps at rank boundaries use MPI_Sendrecv to exchange
//                 CONFIGURATIONS (q, Sq[], lattice[]) on acceptance -- each
//                 rank's slot keeps its beta permanently fixed (slot == rung
//                 always), matching the local-swap redesign in
//                 RenyiReplica::SwapConfigWith / attempt_pt_swap (see F17 in
//                 MBARPT_audit.txt). Message sizes are bounded by qmax/alpha.
//                 MBAR solver is fully distributed: each rank holds K_local*N_SAMPLES
//                 rows of the u-matrix and contributes to the global MPI_Allreduce.
//                 Bootstrap is also distributed (no gather to rank 0 needed).
//
// All optimizations from mbarpt.cpp are preserved:
//   Anderson mixing in MBAR, lazy cache invalidation in RenyiReplica,
//   EvaluateWeightAt for O(1) multi-beta reweighting, autocorrelation thinning,
//   bootstrap uncertainty, exact-classical + off-diagonal anchor correction, etc.
//
// Compile:
//   mpicxx -O3 -std=c++17 -o mbarpt_mpi.bin mbarpt_mpi.cpp
// Run:
//   mpirun -np <P> ./mbarpt_mpi.bin
//   P must evenly divide MBAR_N_BETA (e.g., P=1,2,4,8,16,32 for N_BETA=32).
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef SERIAL_MPI           // single-process build without an MPI installation (serial/serial_mpi.h)
#include "serial_mpi.h"
#else
#include <mpi.h>       // must come first so MPI_VERSION is defined before mainQMC.hpp/replica.hpp
#endif
#include "mainQMC.hpp"

#include <vector>
#include <set>       // [DIAG lattice] distinct-lattice count
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <limits>

#include "mbar_common.hpp"

// [DIAG fixed-stride] OPT-IN, compile-time, off by default.  Compile with -DMBAR_FIXED_STRIDE_FACTOR=F (an integer >= 1) to
// PIN the recording stride at F * MBAR_STEPS_PER_SAMP MC steps per recorded sample from the very first sample and to SKIP the
// automatic stride calibration (adaptive_stride_update and the tau_rec Allreduce) entirely.  Without the macro nothing below
// changes: the stride starts at 1 and is recalibrated every OPT5_RECAL_INTERVAL samples exactly as before.  Purpose: the
// fixed-stride A/B test (alpha1_beta1p5_stride_ab.sh) -- record every 10 steps (F=1) versus every 640 steps (F=64) with the
// same total MC work, to see whether recording itself perturbs the chain.  Nothing else depends on the stride except the
// number of MC steps between records (see the code read in the 2026-10-07 notes).
#if defined(MBAR_FIXED_STRIDE_FACTOR)
static_assert(MBAR_FIXED_STRIDE_FACTOR >= 1, "MBAR_FIXED_STRIDE_FACTOR must be an integer >= 1");
#endif
// build_beta_ladder, compute_classical_renyi_Z, compute_anchor_lnZ,
// compute_offdiag_anchor_terms, compute_quantum_anchor_correction,
// compute_tau_int, anderson_solve_linsys (+ANDERSON_M),
// log_exexfloat, attempt_pt_swap, and the z1_anchor.dat read/write functions are
// all defined in mbar_common.hpp (shared with mbarpt.cpp). solve_mbar_mpi below
// is genuinely MPI-specific (MPI_Allreduce-based) and has no OpenMP counterpart,
// so it stays local to this file.

// ─────────────────────────────────────────────────────────────────────────────
// mpi_boundary_swap  — PT swap between last replica of rank r and first of r+1
//
// [F17 fix] On accept, exchanges the CONFIGURATION over MPI instead of leaving
// beta swapped -- the MPI-driver equivalent of RenyiReplica::SwapConfigWith /
// attempt_pt_swap's local-swap redesign. Each rank's slot keeps its fixed beta
// permanently, so slot k_l on a rank (global rung rung_block_start[rank]+k_l --
// see partition_rungs_by_cost() [OPT-8]) never needs rung
// bookkeeping, on either the OpenMP or MPI driver. The acceptance test itself
// (steps 1-4) is unchanged; step 5 now moves q/Sq[]/lattice[] instead of a
// rung label -- a small, bounded-size message (q <= qmax ints, alpha bitsets).
//
// After adopting the neighbor's configuration, the weight at my_beta is
// RECOMPUTED locally (UpdateWeight()) rather than transmitted, avoiding any
// need to serialize ExExFloat's internal representation over MPI -- one extra
// weight evaluation per accepted boundary swap, negligible next to a full
// STEPS_PER_SAMP sweep.
//
// [N<=64 assumption] lattice bits are packed into one unsigned long long per
// replica-copy via to_ullong() -- matches this codebase's current scale
// (N=9); would need multi-word packing for N>64, same latent limit as
// elsewhere in this codebase's int-indexed dim_A/dim_B computations.
//
// Protocol (4 MPI_Sendrecv/Send/Recv calls per attempted swap on reject,
// 7 on accept -- tags 200-206 below, one Sendrecv/Send/Recv per tag):
//   1. Exchange betas and current log-weights
//   2. Each side computes weight at the neighbor's beta
//   3. Exchange cross-weights
//   4. Lower rank makes the Metropolis decision, sends it to upper rank
//   5. On accept: exchange q, then Sq[0..q-1], then lattice[] bits
// ─────────────────────────────────────────────────────────────────────────────
static bool mpi_boundary_swap(RenyiReplica* rep, int nbr_rank,
                               bool is_lower,
                               int* accept_ctr, int* total_ctr) {
    double my_beta    = rep->beta;  // fixed per-slot; always restored below
    double my_log_W   = log_exexfloat(rep->currWeight);
    ExExFloat saved_w = rep->currWeight;

    // 1. Exchange betas and current weights
    double nbr_beta, nbr_log_W;
    MPI_Sendrecv(&my_beta,   1, MPI_DOUBLE, nbr_rank, 200,
                 &nbr_beta,  1, MPI_DOUBLE, nbr_rank, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&my_log_W,  1, MPI_DOUBLE, nbr_rank, 201,
                 &nbr_log_W, 1, MPI_DOUBLE, nbr_rank, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 2. Compute weight at neighbor's beta
    rep->set_beta(nbr_beta);
    double my_log_W_cross = log_exexfloat(rep->currWeight);

    // 3. Exchange cross-weights
    double nbr_log_W_cross;
    MPI_Sendrecv(&my_log_W_cross,  1, MPI_DOUBLE, nbr_rank, 202,
                 &nbr_log_W_cross, 1, MPI_DOUBLE, nbr_rank, 202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 4. Lower rank decides, broadcasts result
    double log_acc = (my_log_W_cross + nbr_log_W_cross) - (my_log_W + nbr_log_W);
    int accepted;
    if (is_lower) {
        if (total_ctr) (*total_ctr)++;
        accepted = (std::log(val(rng)) < log_acc) ? 1 : 0;
        MPI_Send(&accepted, 1, MPI_INT, nbr_rank, 203, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&accepted, 1, MPI_INT, nbr_rank, 203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (!accepted) {
        // Restore beta and weight; configuration never moved.
        rep->beta = my_beta;
        rep->currWeight = saved_w;
        rep->InvalidateBeta();
        return false;
    }

    // 5. ACCEPT: exchange configurations.
    int alpha_val = rep->alpha;
    // [DIAG lattice] q and the PT round-trip label travel together (2 ints, same tag).
    int my_qlab[2] = { rep->q, rep->pt_label }, nbr_qlab[2];
    MPI_Sendrecv(my_qlab, 2, MPI_INT, nbr_rank, 204, nbr_qlab, 2, MPI_INT, nbr_rank, 204,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int my_q = my_qlab[0], nbr_q = nbr_qlab[0];
    rep->pt_label = nbr_qlab[1];

    std::vector<int> nbr_Sq(std::max(nbr_q, 1));
    MPI_Sendrecv(rep->Sq, my_q, MPI_INT, nbr_rank, 205,
                 nbr_Sq.data(), nbr_q, MPI_INT, nbr_rank, 205,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<unsigned long long> my_lat(alpha_val), nbr_lat(alpha_val);
    for (int r = 0; r < alpha_val; r++) my_lat[r] = rep->lattice[r].to_ullong();
    MPI_Sendrecv(my_lat.data(), alpha_val, MPI_UNSIGNED_LONG_LONG, nbr_rank, 206,
                 nbr_lat.data(), alpha_val, MPI_UNSIGNED_LONG_LONG, nbr_rank, 206,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Adopt the neighbor's configuration. z (reference config) is reset equal
    // to the newly-adopted lattice: UpdateReferenceConfigs() keeps z==lattice
    // at every completed MC move, and PT swaps only ever happen between
    // update() calls, so that invariant already holds for the config we just
    // received.
    rep->q = nbr_q;
    for (int i = 0; i < nbr_q; i++) rep->Sq[i] = nbr_Sq[i];
    for (int r = 0; r < alpha_val; r++) {
        rep->lattice[r] = std::bitset<N>(nbr_lat[r]);
        rep->z[r] = rep->lattice[r];
    }
    rep->beta = my_beta;  // restore to fixed slot beta
    rep->InvalidateCache();
    rep->InvalidateBeta();          // beta_pow_factorial is stale (built for nbr_beta above)
    rep->currWeight = rep->UpdateWeight();  // recompute for the newly-adopted config at my_beta

    if (is_lower && accept_ctr) (*accept_ctr)++;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_mbar_mpi  — distributed MBAR using MPI_Allreduce
//
// Replaces the two OpenMP parallel-for loops in solve_mbar() with two rounds
// of MPI_Allreduce, keeping all other logic (Anderson mixing, normalization,
// convergence check) identical and computed locally on every rank.
//
// Communication per MBAR iteration:
//   Pass 1: zero MPI (log_denom is local — each rank has its own samples)
//   Pass 2: 2 × MPI_Allreduce of K doubles (max + sum for log-sum-exp)
//   Anderson mixing: zero MPI (f[] is synchronized after each iteration)
//
// INPUT:
//   u_local[n * K + j]  — reduced potential, n = 0..S_local-1 (local samples)
//   K                   — total rungs (global)
//   S_local             — local sample count = K_local * Nk
//   Nk                  — samples per rung (same for all rungs globally)
//   f                   — K-dim free energy vector, initialized on all ranks,
//                         same value on all ranks throughout (synchronized by
//                         the Allreduce-based update)
// ─────────────────────────────────────────────────────────────────────────────
static bool solve_mbar_mpi(const std::vector<double>& u_local,
                            int K, int S_local, int Nk,
                            std::vector<double>& f,
                            bool verbose = true) {
    const double log_Nk = std::log((double)Nk);
    std::vector<double> f_new(K, 0.0);
    std::vector<double> log_denom_local(S_local);

    std::vector<std::vector<double>> g_hist, r_hist;
    g_hist.reserve(ANDERSON_M);
    r_hist.reserve(ANDERSON_M);

    if (verbose && mpi_rank == 0)
        std::cout << "  Running MBAR (distributed MPI+Anderson, m=" << ANDERSON_M << ")...\n";

    bool converged = false;   // [OVERLAP fix] report whether the solve actually converged
    for (int iter = 0; iter < MBAR_MAX_ITER; iter++) {

        // [BUGFIX u==1e300-for-every-column] See mbarpt.cpp's solve_mbar Pass 1
        // for the full derivation. Empirically (production diagnostics), 99%+
        // of samples have u_local==1e300 at EVERY j -- not from nt==0 (measured
        // 0%) but from a beta-independent structural zero (e.g.
        // REALW(D0*D1*D2)==0), and 1e300's ULP (~2.2e284) swamps the O(10-200)
        // log_Nk+f[l] term, making max_t and hence log_denom_local[n] round to
        // EXACTLY -1e300. Pass 2 then computes -u[n,j]-log_denom[n] =
        // -1e300-(-1e300) = 0.0 exactly for every j -- a phantom weight of 1.0
        // contributed identically to every rung by a "zero-weight" row,
        // instead of the ~0 the sentinel was meant to represent. At this
        // prevalence it single-handedly flattens f[j] across the whole beta
        // ladder. Fix: mark an all-1e300 row with a real -infinity so Pass 2
        // can skip it outright.
        //
        // Pass 1: log_denom for each local sample (no MPI needed)
        for (int n = 0; n < S_local; n++) {
            double max_t = -1e300;
            bool all_invalid = true;
            for (int l = 0; l < K; l++) {
                if (u_local[n * K + l] < 1e299) all_invalid = false;
                double t = log_Nk + f[l] - u_local[n * K + l];
                if (t > max_t) max_t = t;
            }
            if (all_invalid) {
                log_denom_local[n] = -std::numeric_limits<double>::infinity();
                continue;
            }
            double sum_e = 0.0;
            for (int l = 0; l < K; l++)
                sum_e += std::exp(log_Nk + f[l] - u_local[n * K + l] - max_t);
            log_denom_local[n] = max_t + std::log(sum_e);
        }

        // Pass 2: f_new[j] via distributed log-sum-exp reduction
        // Step 2a: local max of (-u[n][j] - log_denom[n]) per j
        std::vector<double> local_max(K, -1e300);
        for (int n = 0; n < S_local; n++) {
            if (!std::isfinite(log_denom_local[n])) continue;  // [BUGFIX] all-1e300 row, carries no info
            for (int j = 0; j < K; j++) {
                double lc = -u_local[n * K + j] - log_denom_local[n];
                if (lc > local_max[j]) local_max[j] = lc;
            }
        }

        // Global max across all ranks
        std::vector<double> global_max(K);
        MPI_Allreduce(local_max.data(), global_max.data(), K, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Step 2b: partial sum using global max for numerical stability
        std::vector<double> partial_sum(K, 0.0);
        for (int n = 0; n < S_local; n++) {
            if (!std::isfinite(log_denom_local[n])) continue;  // [BUGFIX] all-1e300 row, carries no info
            for (int j = 0; j < K; j++)
                partial_sum[j] += std::exp(-u_local[n * K + j] - log_denom_local[n] - global_max[j]);
        }

        // Global sum across all ranks
        std::vector<double> total_sum(K);
        MPI_Allreduce(partial_sum.data(), total_sum.data(), K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Compute f_new (same result on all ranks)
        for (int j = 0; j < K; j++)
            f_new[j] = -(global_max[j] + std::log(total_sum[j]));

        // Normalize: pin f[K-1] = 0
        double shift = f_new[K - 1];
        for (int j = 0; j < K; j++) f_new[j] -= shift;

        // Residual
        std::vector<double> r(K);
        double max_resid = 0.0;
        for (int j = 0; j < K; j++) {
            r[j] = f_new[j] - f[j];
            double ar = std::abs(r[j]);
            if (ar > max_resid) max_resid = ar;
        }

        if (verbose && mpi_rank == 0 && (iter % 10 == 0 || max_resid < MBAR_TOL))
            std::cout << "    iter " << std::setw(4) << iter
                      << "  max_residual = " << max_resid << "\n";

        if (!std::isfinite(max_resid)) {   // [OVERLAP fix] NaN/inf: never report convergence
            if (verbose && mpi_rank == 0)
                std::cout << "    MBAR: non-finite residual at iter " << iter << " -- not converged\n";
            return false;
        }
        if (max_resid < MBAR_TOL) {
            f = f_new;
            if (verbose && mpi_rank == 0)
                std::cout << "    Converged at iter " << iter
                          << "  (max_residual = " << max_resid << ")\n";
            converged = true;
            break;
        }

        // Anderson mixing (runs identically on all ranks — f[] is synchronized)
        if ((int)g_hist.size() >= ANDERSON_M) {
            g_hist.erase(g_hist.begin());
            r_hist.erase(r_hist.begin());
        }
        g_hist.push_back(f_new);
        r_hist.push_back(r);

        int h = (int)g_hist.size();
        if (h >= 2) {
            std::vector<double> alpha_coeffs;
            if (anderson_solve_linsys(r_hist, K, alpha_coeffs)) {
                f.assign(K, 0.0);
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < K; j++)
                        f[j] += alpha_coeffs[i] * g_hist[i][j];
                double shift2 = f[K - 1];
                for (int j = 0; j < K; j++) f[j] -= shift2;
            } else {
                g_hist.clear(); r_hist.clear();
                f = f_new;
            }
        } else {
            f = f_new;
        }
    }
    for (double v : f) if (!std::isfinite(v)) converged = false;   // [OVERLAP fix]
    if (!converged && verbose && mpi_rank == 0)
        std::cout << "    MBAR: NOT converged within " << MBAR_MAX_ITER << " iterations\n";
    return converged;
}

// z1_anchor.dat read/write functions are defined in mbar_common.hpp (only called
// on rank 0 below, same as mbarpt.cpp calls them unconditionally).

// ─────────────────────────────────────────────────────────────────────────────
// partition_rungs_by_cost()  [OPT-8]
//
// Problem (audit F12): equal-size contiguous rung blocks (K_local = K/P)
// make the COLDEST rank the straggler for every collective, because q --
// and hence per-step cost -- grows with beta (roughly ~alpha*beta*N), so
// the highest-beta block is far more expensive than the lowest-beta one
// even though both get the same number of rungs. Measured 74.8% efficiency
// in the audit's reference job.
//
// Fix: partition the K rungs into exactly `mpi_size` CONTIGUOUS (preserving
// the F17 slot==rung design -- no strided/scattered assignment) blocks that
// minimize the MAXIMUM total cost of any one block, using beta_ladder[k] as
// a monotone proxy for rung k's cost (cheap, deterministic, needs no pilot
// run; a measured-q proxy could replace it later without touching the
// algorithm below). Every rank computes this from the same beta_ladder with
// no communication, so the result is identical everywhere by construction.
//
// Standard "minimize the maximum contiguous partition sum" DP:
//   dp[p][i]     = min possible (max block cost) partitioning rungs [0,i)
//                  into p non-empty contiguous blocks
//   choice[p][i] = the split point achieving that minimum, for reconstruction
// O(K^2 * P); at realistic K,P (tens to low hundreds) this is microseconds,
// negligible next to anything else in a pipeline that runs for hours.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// pipeline_stat  — the scalar a pipeline run reports (point estimate and every
// bootstrap replicate). In this code's MBAR gauge f[j] = ln Z(rung K-1) - ln Z(rung j).
//   default:        f[0]  (= ln Z(beta_target) - ln Z(beta_min); the anchor is added)
//   OVERLAP_PLUS:   ln F_beta = 2 ln O(0,1) - ln O(0,0) - ln O(1,1)
//                             = f[0] + f[K-1] - 2 f[(K-1)/2],
//                   which is gauge invariant (needs odd K, mid rung = pair (0,1)).
// ─────────────────────────────────────────────────────────────────────────────
static double pipeline_stat(const std::vector<double>& f) {
#ifdef OVERLAP_PLUS
    int K = (int)f.size();
    return f[0] + f[K - 1] - 2.0 * f[(K - 1) / 2];
#else
    return f[0];
#endif
}

static void partition_rungs_by_cost(const std::vector<double>& beta_ladder, int P,
                                     std::vector<int>& rung_block_start,
                                     std::vector<int>& rung_block_len) {
    int K = (int)beta_ladder.size();
    std::vector<double> prefix(K + 1, 0.0);
#ifdef OVERLAP_PLUS
    // [OVERLAP_PLUS] rungs are interpolation points at fixed beta: equal cost
    for (int i = 0; i < K; i++) prefix[i + 1] = prefix[i] + 1.0;
#else
    for (int i = 0; i < K; i++) prefix[i + 1] = prefix[i] + beta_ladder[i];
#endif
    auto block_cost = [&](int a, int b) { return prefix[b] - prefix[a]; };

    const double INF = 1e300;
    std::vector<std::vector<double>> dp(P + 1, std::vector<double>(K + 1, INF));
    std::vector<std::vector<int>> choice(P + 1, std::vector<int>(K + 1, -1));
    dp[0][0] = 0.0;
    for (int p = 1; p <= P; p++) {
        for (int i = p; i <= K; i++) {          // need >= p rungs for p non-empty blocks
            for (int j = p - 1; j < i; j++) {   // previous boundary
                if (dp[p - 1][j] >= INF) continue;
                double cost = std::max(dp[p - 1][j], block_cost(j, i));
                if (cost < dp[p][i]) { dp[p][i] = cost; choice[p][i] = j; }
            }
        }
    }

    rung_block_start.assign(P, 0);
    rung_block_len.assign(P, 0);
    int i = K;
    for (int p = P; p >= 1; p--) {
        int j = choice[p][i];
        rung_block_start[p - 1] = j;
        rung_block_len[p - 1]   = i - j;
        i = j;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_mbar_pt_pipeline_mpi  — full MPI-distributed PT+MBAR simulation
//
// Replica distribution:
//   Rank r owns local replicas k_l = 0..K_local-1 (K_local = rung_block_len[r],
//   generally different per rank -- see partition_rungs_by_cost() [OPT-8]),
//   corresponding to global beta_ladder[rung_block_start[r] + k_l].
//
// PT swap protocol:
//   Within-rank: sequential (same as mbarpt.cpp, no MPI).
//   Boundary between rank r and r+1: mpi_boundary_swap (exchanges configurations
//   on accept, per the F17 fix -- each rank's slot keeps a permanently fixed beta).
//   Active boundaries for each parity round are processed sequentially via a
//   for-loop over boundary indices; non-participating ranks skip. This is
//   deadlock-free because MPI_Sendrecv is used and each rank is involved in
//   at most two boundary swaps per round (left and right neighbors).
//
// u-matrix stays distributed: each rank holds K_local * Nk rows throughout.
// MBAR and bootstrap both run distributed (no gather to rank 0).
// ─────────────────────────────────────────────────────────────────────────────
// [OVERLAP fix] MBAR overlap between adjacent rungs, O_{i,i+1} = Nk sum_n W_ni W_n,i+1
// (Shirts & Chodera), for a distributed u-matrix of K_local*Nk rows. Returns the
// minimum over i (identical on every rank); ov_out, if given, receives all K-1 values.
static double mbar_min_adjacent_overlap(const std::vector<double>& u_loc, int K, size_t n_rows,
                                        int Nk, const std::vector<double>& f,
                                        int* min_i_out = nullptr, std::vector<double>* ov_out = nullptr) {
    const double logN = std::log((double)Nk);
    std::vector<double> ov_local(K, 0.0), ov(K, 0.0), w(K);
    for (size_t n = 0; n < n_rows; n++) {
        const double* u = &u_loc[n * K];
        double mx = -1e300;
        for (int l = 0; l < K; l++) if (u[l] < 1e299) mx = std::max(mx, logN + f[l] - u[l]);
        if (mx <= -1e299) continue;
        double sum = 0.0;
        for (int l = 0; l < K; l++) if (u[l] < 1e299) sum += std::exp(logN + f[l] - u[l] - mx);
        double logden = mx + std::log(sum);
        for (int l = 0; l < K; l++) w[l] = (u[l] < 1e299) ? std::exp(f[l] - u[l] - logden) : 0.0;
        for (int i = 0; i + 1 < K; i++) ov_local[i] += w[i] * w[i + 1];
    }
    MPI_Allreduce(ov_local.data(), ov.data(), K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double min_ov = 1e300; int min_i = 0;
    for (int i = 0; i + 1 < K; i++) { ov[i] *= Nk; if (!(ov[i] >= min_ov)) { min_ov = ov[i]; min_i = i; } }
    if (min_i_out) *min_i_out = min_i;
    if (ov_out) { ov.resize(K - 1); *ov_out = ov; }
    return min_ov;
}

// [OVERLAP fix] Status of the reported error bar, set by run_mbar_pt_pipeline_mpi:
// 0 = usable; 1 = the block-length scan found it growing (the exported stderr is then the
// largest scanned value); 2 = unavailable (a bootstrap replicate failed: non-converged,
// non-finite, or disconnected) -- the exported stderr is then NaN.
static int g_stderr_status = 0;
static double g_stderr_short = 0.0;   // the default-block-length bootstrap error

static double run_mbar_pt_pipeline_mpi(int alpha_val,
                                        const std::vector<double>& beta_ladder,
                                        double log_Z_anchor,
                                        double wall_start,
                                        double* boot_stderr_out = nullptr) {
    int K = (int)beta_ladder.size();

    // [OPT-8] Variable-size contiguous blocks, not K/mpi_size equal split --
    // see partition_rungs_by_cost() above. rung_block_start/rung_block_len are
    // computed identically on every rank (pure function of beta_ladder +
    // mpi_size, no communication); this rank's own slice is
    // rung_block_start[mpi_rank] / rung_block_len[mpi_rank]. (Named distinctly
    // from the unrelated scalar `block_len` used later for bootstrap/thinning
    // block sizing -- same word, different concept, different scope.)
    std::vector<int> rung_block_start, rung_block_len;
    partition_rungs_by_cost(beta_ladder, mpi_size, rung_block_start, rung_block_len);
    g_ckpt_ladder_hash = ladder_checksum(beta_ladder);  // [OVERLAP] checkpoint key includes the actual ladder
    int K_local    = rung_block_len[mpi_rank];   // replicas on THIS rank (may differ across ranks)
    int rank_start = rung_block_start[mpi_rank]; // first global replica index on this rank

    if (mpi_rank == 0)
        std::cout << "\n── MBAR+PT pipeline (MPI)  alpha=" << alpha_val
                  << "  beta_target=" << beta_ladder[K-1]
                  << "  ranks=" << mpi_size
                  << "  K_local=" << K_local << " ──────────────\n";

    // Allocate K_local replicas on this rank
    std::vector<RenyiReplica*> local_replicas(K_local);
    for (int k = 0; k < K_local; k++) {
        local_replicas[k] = new RenyiReplica(alpha_val);
        local_replicas[k]->init_renyi(beta_ladder[rank_start + k], rank_start + k);  // [F13] global slot index
    }

    // [OPT-2] Per-rung beta-factorial tables, built ONCE for the whole pipeline
    // (identical on every rank since beta_ladder is identical on every rank).
    // The u-fill loop below evaluates every local replica's weight at ALL K
    // global betas, so this table has K rows regardless of K_local. Removes
    // the O(q) stack rebuild EvaluateWeightAt used to do on every one of those
    // K_local*K calls per sample.
    std::vector<std::vector<ExExFloat>> bpf(K, std::vector<ExExFloat>(qmax));
    for (int j = 0; j < K; j++) {
        bpf[j][0] = ExExFloat(1.0);
        for (int t = 1; t < qmax; t++) bpf[j][t] = bpf[j][t-1] * ((-beta_ladder[j]) / t);
    }

    // [qmax fix] Tracks q as actually observed (thermalization + sampling) so
    // a qmax-fatal trigger -- or a successful run's end-of-run report -- can
    // tell you the ACTUAL qmax the audit's own sizing rule calls for, instead
    // of a generic "raise it and guess again" reminder. Only this rank's own
    // rungs; combined across ranks via MPI_Allreduce(MPI_MAX) where used.
    QmaxHistogram qmax_hist = make_qmax_histogram(K_local);

    // [F20 fix] Resume from a checkpoint if one exists and matches this run's
    // parameters. Each rank has its own checkpoint file (checkpoint_state_path
    // includes mpi_rank), written together at the same SIGTERM event -- so in
    // normal operation every rank's read_checkpoint() call agrees. Verify that
    // explicitly (Bcast rank 0's outcome, compare) rather than assuming it:
    // proceeding with SOME ranks resumed and others not would silently produce
    // a corrupted run.
    install_checkpoint_signal_handler(mpi_rank);
    CheckpointPhase resume_phase = CheckpointPhase::THERMALIZING;
    int resume_progress = 0;
    int stride_factor = 1;  // [OPT-5] restored below if resuming mid-sampling; must end up rank-uniform
    bool resumed = read_checkpoint(alpha_val, mpi_rank, resume_phase, resume_progress, stride_factor, local_replicas);
    {
        // [OPT-5] stride_factor added to the cross-rank agreement check --
        // it's now part of what "every rank resumed consistently" means.
        int local_agree[4] = { resumed ? 1 : 0, (int)resume_phase, resume_progress, stride_factor };
        int root_agree[4] = { local_agree[0], local_agree[1], local_agree[2], local_agree[3] };
        MPI_Bcast(root_agree, 4, MPI_INT, 0, MPI_COMM_WORLD);
        if (local_agree[0] != root_agree[0] ||
            (root_agree[0] && (local_agree[1] != root_agree[1] || local_agree[2] != root_agree[2]
                                || local_agree[3] != root_agree[3]))) {
            std::cerr << "[rank " << mpi_rank << "] FATAL: checkpoint state disagrees with rank 0 "
                         "(resumed=" << local_agree[0] << " vs " << root_agree[0]
                      << ") -- checkpoint files are inconsistent across ranks; delete all "
                         "renyi_checkpoint_a" << alpha_val << "_r*.dat / renyi_ustream_a"
                      << alpha_val << "_r*.bin and restart.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        resumed = root_agree[0] != 0;
        resume_phase = (CheckpointPhase)root_agree[1];
        resume_progress = root_agree[2];
        stride_factor = root_agree[3];
    }
#if defined(MBAR_FIXED_STRIDE_FACTOR)
    // [DIAG fixed-stride] Pin the stride.  A checkpoint written mid-sampling by a run with a DIFFERENT stride cannot be resumed
    // into a pinned run (the recorded grid would change halfway); stop with a clear message instead.  stride_factor is already
    // identical on every rank here (it came from the Bcast above), so this branch is taken by all ranks or none.
    if (resumed && resume_phase == CheckpointPhase::SAMPLING && stride_factor != MBAR_FIXED_STRIDE_FACTOR) {
        if (mpi_rank == 0)
            std::cerr << "FATAL: checkpoint was written with stride factor " << stride_factor
                      << " but this build pins it at " << MBAR_FIXED_STRIDE_FACTOR
                      << "; delete the renyi_checkpoint_* / renyi_ustream_* files and restart.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    stride_factor = MBAR_FIXED_STRIDE_FACTOR;
    if (mpi_rank == 0)
        std::cout << "  [FIXED STRIDE] recording stride pinned at factor " << stride_factor << " = "
                  << (STEPS_PER_SAMP * stride_factor) << " MC steps per recorded sample; automatic stride calibration is OFF.\n";
#endif
    if (resumed && mpi_rank == 0)
        std::cout << "  [checkpoint] Resumed: phase="
                  << (resume_phase == CheckpointPhase::THERMALIZING ? "thermalizing" : "sampling")
                  << "  progress=" << resume_progress << "\n";

    // [F17 fix] PT swaps (both do_local_swap and mpi_boundary_swap) now exchange
    // CONFIGURATIONS, so local slot k_l's beta never changes -- it permanently
    // represents global rung rank_start+k_l. No rung-tracking bookkeeping needed.

    // [DIAG lattice] Per-PAIR swap statistics (pair p = rungs p,p+1; counted by
    // the rank that owns/decides the lower member), and round-trip counters:
    // pt_round_trips = label-1 configurations arriving at rung 0 (a complete
    // bottom->top->bottom excursion), pt_top_arrivals = label-0 configurations
    // arriving at rung K-1.  Reduced across ranks after sampling.
    std::vector<long long> pair_acc(K - 1, 0), pair_tot(K - 1, 0);
    long long pt_round_trips = 0, pt_top_arrivals = 0;
    bool diag_count_pairs = false;   // switched on for the sampling phase only

    // ── Local PT swap helper ─────────────────────────────────────────────────
    auto do_local_swap = [&](int k_l, int* ac, int* tc) -> bool {
        if (tc) (*tc)++;
        if (diag_count_pairs) pair_tot[rank_start + k_l]++;
        if (attempt_pt_swap(local_replicas[k_l], local_replicas[k_l + 1])) {
            if (ac) (*ac)++;
            if (diag_count_pairs) pair_acc[rank_start + k_l]++;
            return true;
        }
        return false;
    };

    // [DIAG lattice] Update the round-trip labels of the configurations that
    // currently sit at the two ends of the ladder; call after every swap round.
    auto update_pt_labels = [&]() {
        if (rank_start == 0) {                       // this rank owns rung 0
            RenyiReplica* r0 = local_replicas[0];
            if (r0->pt_label == 1 && diag_count_pairs) pt_round_trips++;
            r0->pt_label = 0;
        }
        if (rank_start + K_local == K) {             // this rank owns rung K-1
            RenyiReplica* rt = local_replicas[K_local - 1];
            if (rt->pt_label == 0 && diag_count_pairs) pt_top_arrivals++;
            rt->pt_label = 1;
        }
    };

    // ── Boundary PT swaps for one parity round ───────────────────────────────
    // Iterates all num_ranks-1 possible boundaries; each rank participates only
    // in the boundaries where it is the lower (b) or upper (b+1) partner.
    // Non-participating ranks skip immediately. No barrier needed: MPI_Sendrecv
    // synchronizes the participating pair, and the loop continues for the next b.
    //
    // [OPT-7] Sub-phased by boundary-index (b) parity, not just round_parity.
    // Problem: when K_local is even, EVERY boundary shares the same
    // round_parity, so a single `for b=0..mpi_size-2` loop puts all
    // mpi_size-1 boundary swaps in one round -- and since rank r+1 is the
    // upper partner of boundary r AND the lower partner of boundary r+1, it
    // must finish boundary r's Sendrecv before program order reaches boundary
    // r+1's, serializing the whole round into a chain P-1 deep. Splitting by
    // b-parity (pass=0: b=0,2,4,...; pass=1: b=1,3,5,...) makes every pair of
    // boundaries active within ONE pass use four DISTINCT ranks (boundary b
    // uses ranks {b,b+1}; boundaries of the same parity never share a rank),
    // so all swaps within a pass are independent and can run concurrently --
    // cutting round latency from O(P) to O(2) sequential Sendrecv hops.
    // The SET of swaps executed per round_parity call is unchanged; only the
    // program-order/wall-clock scheduling across ranks changes.
    auto do_boundary_swaps = [&](int round_parity, int* ac, int* tc) {
        for (int pass = 0; pass < 2; pass++) {
            for (int b = pass; b < mpi_size - 1; b += 2) {
                // [OPT-8] Global index of the boundary pair's lower element.
                // Was (b+1)*K_local-1, which assumed every rank owns the same
                // number of rungs; with variable-size blocks the boundary
                // between rank b and b+1 is simply the last global index in
                // rank b's block, read from the shared rung_block_start table
                // (identical on every rank, so this needs no communication).
                int boundary_global = rung_block_start[b + 1] - 1;
                if (boundary_global % 2 != round_parity) continue;

                if (mpi_rank == b) {
                    bool acc = mpi_boundary_swap(local_replicas[K_local - 1], b + 1, true, ac, tc);
                    if (diag_count_pairs) { pair_tot[boundary_global]++; if (acc) pair_acc[boundary_global]++; }
                } else if (mpi_rank == b + 1) {
                    mpi_boundary_swap(local_replicas[0], b, false, ac, tc);
                }
            }
        }
    };

    // ── Combined swap step: even pairs then odd pairs ────────────────────────
    auto do_all_swaps = [&](int* ac, int* tc) {
        // Even global pairs within this rank
        for (int k_l = 0; k_l < K_local - 1; k_l++)
            if ((rank_start + k_l) % 2 == 0) do_local_swap(k_l, ac, tc);
        // Even global boundary pairs
        do_boundary_swaps(0, ac, tc);

        // Odd global pairs within this rank
        for (int k_l = 0; k_l < K_local - 1; k_l++)
            if ((rank_start + k_l) % 2 == 1) do_local_swap(k_l, ac, tc);
        // Odd global boundary pairs
        do_boundary_swaps(1, ac, tc);
        update_pt_labels();   // [DIAG lattice]
    };

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 1: THERMALIZATION
    // ════════════════════════════════════════════════════════════════════════
    if (resumed && resume_phase == CheckpointPhase::SAMPLING) {
        if (mpi_rank == 0) std::cout << "\n[1/3] Thermalization already complete (resumed past it).\n";
    } else {
        int therm_start = resumed ? resume_progress : 0;
        if (mpi_rank == 0)
            std::cout << "\n[1/3] Thermalizing " << K << " replicas (" << T_EQUIL << " steps each"
                      << (therm_start > 0 ? ", resuming from step " + std::to_string(therm_start) : "")
                      << ")...\n";

#if defined(DUMP_WARMUP_Q) && DUMP_WARMUP_Q
        // [DUMP warm-up q] (only when compiled with -DDUMP_WARMUP_Q=1; no effect otherwise)
        // Records every local rung's q every WARMUP_Q_EVERY warm-up steps, so ONE run shows the whole
        // relaxation curve from the q=0 start through the end of warm-up.  Written to
        // renyi_warmq_a<alpha>_r<global rung>.bin (int32) right after the warm-up loop.
        const int WARMUP_Q_EVERY = 500;
        std::vector<std::vector<int>> warm_q(K_local);
#endif
        for (int step = therm_start; step < T_EQUIL; step++) {
            for (int k = 0; k < K_local; k++) local_replicas[k]->update();
#if defined(DUMP_WARMUP_Q) && DUMP_WARMUP_Q
            if (step % WARMUP_Q_EVERY == 0)
                for (int k = 0; k < K_local; k++) warm_q[k].push_back(local_replicas[k]->q);
#endif

            if (step % SWAP_INTERVAL == 0) {
                do_all_swaps(nullptr, nullptr);

                // [qmax fix] Sampled at swap cadence -- plenty of points for a
                // percentile over T_EQUIL steps, negligible added cost.
                record_qmax_histogram(qmax_hist, local_replicas);

                // [F20 fix] Allreduce agreement (not a local-only check like
                // qmax): SIGTERM is a graceful-shutdown signal, so every rank
                // should checkpoint at the SAME loop iteration and exit
                // together, not have one rank race ahead of others still
                // waiting on a collective. Piggybacks on the swap cadence --
                // already a synchronization point, no new collective added.
                int local_ckpt = g_checkpoint_requested ? 1 : 0, global_ckpt = 0;
                MPI_Allreduce(&local_ckpt, &global_ckpt, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                if (global_ckpt) {
                    // [OPT-5] stride_factor is always 1 here (only meaningful
                    // during sampling); passed through so every message has the same layout.
                    write_checkpoint(alpha_val, mpi_rank, CheckpointPhase::THERMALIZING,
                                      step + 1, stride_factor, local_replicas);
                    if (mpi_rank == 0)
                        std::cout << "  [checkpoint] SIGTERM received; saved and exiting "
                                     "(resubmit to continue).\n";
                    MPI_Finalize();
                    std::exit(75);  // EX_TEMPFAIL (sysexits.h): temporary, please retry
                }
            }

#if defined(OVERLAP_PLUS)
            // [DIAG parity] XX hops, start-shifts and PT swaps preserve the global Z-parity of
            // the walk start y0; only single-spin flips change it. Report how many walkers are
            // still in the odd sector (equilibrium mass there is tiny at large beta).
            if (step % 1000 == 0 || step == T_EQUIL - 1) {
                int odd_l = 0, odd_g = 0;
                for (int k = 0; k < K_local; k++) odd_l += (int)(local_replicas[k]->lattice[0].count() & 1);
                MPI_Allreduce(&odd_l, &odd_g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if (mpi_rank == 0)
                    std::cout << "  [DIAG parity] warm-up step " << step << ": odd-parity walkers " << odd_g << "/" << K << "\n";
            }
#endif
            if (step > 0 && step % 20000 == 0 && mpi_rank == 0) {
                double t = (double)clock() / CLOCKS_PER_SEC - wall_start;
                std::cout << "  step " << step << "/" << T_EQUIL
                          << "  elapsed=" << std::fixed << std::setprecision(1) << t << "s\n";
                std::cout.flush();
            }
        }
#if defined(DUMP_WARMUP_Q) && DUMP_WARMUP_Q
        for (int k = 0; k < K_local; k++) {
            std::ostringstream wname;
            wname << "renyi_warmq_a" << alpha_val << "_r" << (rank_start + k) << ".bin";
            std::ofstream wf(wname.str().c_str(), std::ios::binary);
            wf.write(reinterpret_cast<const char*>(warm_q[k].data()),
                     (std::streamsize)(warm_q[k].size() * sizeof(int)));
        }
#endif
        if (mpi_rank == 0) std::cout << "  Thermalization complete.\n";
    }
    {
        // [qmax fix] Was informational-only before; now fatal (audit P1 item
        // 7) -- a replica already at qmax after thermalization means the
        // WHOLE run would sample from a truncated weight function from the
        // start. Local-only check + MPI_Abort (see the per-sample check below
        // for why that's deliberate, not an oversight).
        QmaxStats qs_local = scan_qmax(local_replicas);
        int local_hit = qs_local.any_achieved ? 1 : 0;
        int global_qmax_hits = 0;
        MPI_Reduce(&local_hit, &global_qmax_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0)
            std::cout << "  qmax_achieved on " << global_qmax_hits << "/" << mpi_size << " ranks\n";

        // [qmax fix] Combine across ranks (max over ALL rungs of q_99.9 = max
        // over each rank's max over its own local rungs) so a fatal message's
        // sizing number reflects the whole ladder, not just one rank's rungs.
        // Called unconditionally, by every rank, regardless of whether THIS
        // rank hit qmax -- MPI_Allreduce is a collective, so it would deadlock
        // if only the ranks with any_achieved==true called it.
        int local_q99_9 = local_max_q99_9(qmax_hist), global_q99_9 = 0;
        MPI_Allreduce(&local_q99_9, &global_q99_9, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (qs_local.any_achieved) {
            // [qmax fix audit] Printed by the ABORTING rank itself, not
            // gated on mpi_rank==0 -- qmax truncation fires on the highest-
            // beta rungs, owned by the LAST rank (blocks are contiguous and
            // beta_ladder is increasing, so the last block -- whatever size
            // [OPT-8]'s cost-balanced partition gave it -- always covers the
            // coldest/most-expensive rungs), so rank 0 essentially never
            // enters this branch in the realistic failure mode. Gating the
            // print on rank 0 meant
            // it almost never actually printed, defeating the whole point
            // of the unconditional Allreduce above. Every rank already has
            // the same global_q99_9 value, so any rank printing it is
            // correct (harmless if more than one rank hits qmax and prints
            // it redundantly -- each copy is rank-prefixed).
            std::cerr << "[rank " << mpi_rank << "] ";
            print_qmax_recommendation(global_q99_9, std::cerr, /*censored=*/true);
            std::cerr << "[rank " << mpi_rank << "] ";
            print_qmax_fatal_message(std::cerr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2: SAMPLING
    // ════════════════════════════════════════════════════════════════════════
    int S_local = K_local * N_SAMPLES;
    int samp_start = (resumed && resume_phase == CheckpointPhase::SAMPLING) ? resume_progress : 0;
    if (mpi_rank == 0)
        std::cout << "\n[2/3] Collecting " << (K * N_SAMPLES) << " total samples ("
                  << N_SAMPLES << " per rung)"
                  << (samp_start > 0 ? ", resuming from sample " + std::to_string(samp_start) : "")
                  << "...\n";

    // Flat layout: u_local[s_idx * K + j], s_idx = k_l * N_SAMPLES + samp
    std::vector<double> u_local(S_local * K, 0.0);
    int swap_accept = 0, swap_total = 0;
    long long sign_pos = 0, sign_neg = 0;  // [F9] negative-weight diagnostics

    // [DIAG zero-weight] Per-rung count of samples where nt==0 (EnumerateTuples
    // found A doesn't close / no valid cut at all -- identically zero weight at
    // EVERY beta, so u_local[s_idx*K+0..K-1] is 1e300 across the WHOLE row).
    // Tracked to check a suspected solve_mbar_mpi bug: a fully-1e300 row makes
    // log_denom_local[n] itself evaluate to exactly -1e300 (the O(10) log_Nk+f[l]
    // term is 290 orders of magnitude below the ULP at 1e300 and is silently
    // dropped, not just "small"), so Pass 2's "-u_local[n,j] - log_denom[n]"
    // becomes exactly 1e300 - 1e300 = 0.0 (bit-exact IEEE754 cancellation, not
    // approximate) for every j -- i.e. this sample contributes a "phantom"
    // weight of exactly 1.0 to every rung identically, instead of the ~0 its
    // sentinel was meant to represent. If this fraction is large, it would
    // explain f[j] coming out nearly flat across the whole beta ladder.
    std::vector<long long> zero_weight_local(K_local, 0);

    // [DIAG m==0] The nt==0 theory measured 0/640000 on real hardware and is
    // dead -- but nt==0 and m<0.0 (sign_neg) are NOT the only ways a cell
    // becomes the 1e300 sentinel: m==0.0 exactly (a genuine, non-sign-problem
    // exact-zero weight, e.g. if REALW(D0*D1*D2) happens to be exactly zero
    // for a structural/symmetry reason independent of target_beta) falls into
    // the same `else` branch but was never counted separately. If that fires
    // for EVERY j in a row simultaneously (a beta-INDEPENDENT structural
    // zero, not per-beta noise), it reproduces the exact same whole-row
    // 1e300-1e300 cancellation mechanism in solve_mbar_mpi via a different
    // trigger. Tracked per local slot k_l, split into "every column zero"
    // (the dangerous case) vs "some but not all" (comparatively harmless,
    // handled correctly by the log-sum-exp as long as >=1 column is finite).
    std::vector<long long> zero_mantissa_full_local(K_local, 0);
    std::vector<long long> zero_mantissa_partial_local(K_local, 0);

    // [DIAG u-spread] Most direct test of "does the per-sample weight even
    // vary with beta": running sum/sumsq/count of u_local, PER TARGET-BETA
    // COLUMN j, restricted to genuinely finite (m>0.0) cells only. If real
    // physics is at work, mean[j] should vary substantially and monotonically
    // across the ladder (colder rungs = larger u typically); if it's nearly
    // identical for every j, the per-sample weight evaluation itself isn't
    // beta-differentiated, independent of which specific mechanism is why.
    std::vector<double> u_sum_local(K, 0.0), u_sumsq_local(K, 0.0);
    std::vector<long long> u_count_local(K, 0);

    // [F17 fix] tau_series[k_l] = self-energy time series for local slot k_l,
    // which IS global rung rank_start+k_l permanently (PT swaps move
    // configurations between slots, not temperatures).
    std::vector<std::vector<double>> tau_series(K_local);
    for (auto& s : tau_series) s.reserve(N_SAMPLES);

    // [F20 fix] Per-rank stream file (durability copy; u_local above is still
    // the primary data source, unchanged). On resume, reconstruct this rank's
    // in-memory prefix from what it streamed before an earlier invocation was
    // killed, including tau_series for those samples.
    std::string ustream_path = checkpoint_ustream_path(alpha_val, mpi_rank);
    if (samp_start > 0) {
        if (!read_ustream_prefix(ustream_path, samp_start, K_local, K, N_SAMPLES, u_local.data())) {
            std::cerr << "[rank " << mpi_rank << "] FATAL: checkpoint claims " << samp_start
                      << " samples were already streamed, but " << ustream_path << " is "
                         "missing or shorter than that. Delete all renyi_checkpoint_a"
                      << alpha_val << "_r*.dat / renyi_ustream_a" << alpha_val
                      << "_r*.bin and restart.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int samp = 0; samp < samp_start; samp++)
            for (int k_l = 0; k_l < K_local; k_l++) {
                int r = rank_start + k_l;
                tau_series[k_l].push_back(u_local[(k_l * N_SAMPLES + samp) * K + r]);
            }
    }
    std::ofstream ustream_out = open_ustream_for_append(ustream_path, samp_start, K_local * K);

    // [DIAG swap-fallback] Snapshot at the start of sampling (post-thermalization
    // counts aren't reset, so this is nonzero on a fresh run too -- only the
    // per-block DELTA printed below matters).
    SwapFallbackStats prev_swap_stats = scan_swap_fallback(local_replicas);

    // [DIAG lattice] Per-slot series of the lattice sector at every recorded
    // sample: E(z_0) of replica 0 (Energies_replicas[0][0], valid after
    // EnsureCachesValid), the packed lattice[0] bits, and q.  These are what
    // the u-series autocorrelation cannot see (a slowly-varying lattice shifts
    // u by a constant), and E(z_0) has an exact ED reference
    // (tools/ed_observables.py).  Memory: 3*8*N_SAMPLES bytes per slot.
    std::vector<std::vector<double>> E0_series(K_local);
    std::vector<std::vector<unsigned long long>> lat_series(K_local);
    std::vector<std::vector<int>> q_series(K_local);
    for (int k = 0; k < K_local; k++) {
        E0_series[k].reserve(N_SAMPLES); lat_series[k].reserve(N_SAMPLES); q_series[k].reserve(N_SAMPLES);
    }
    diag_count_pairs = true;     // per-pair PT counters and round trips: sampling phase only
    for (auto* r : local_replicas) r->pt_label = -1;

    long long total_mc_steps_phase2 = 0;  // [OPT-5] varies with stride_factor -- printed at the end
    for (int samp = samp_start; samp < N_SAMPLES; samp++) {
        // [OPT-5] steps_this_samp = STEPS_PER_SAMP * stride_factor: the ONLY
        // thing that adapts is how many MC steps run between recorded
        // samples -- see mbarpt.cpp's OpenMP counterpart for the full
        // rationale. stride_factor is kept rank-uniform by the recalibration
        // block below (Allreduce(MAX) on tau_rec before any rank updates it).
        int steps_this_samp = STEPS_PER_SAMP * stride_factor;
        total_mc_steps_phase2 += steps_this_samp;
        for (int step = 0; step < steps_this_samp; step++) {
            for (int k = 0; k < K_local; k++) local_replicas[k]->update();
            if (step % SWAP_INTERVAL == 0) do_all_swaps(&swap_accept, &swap_total);
        }

        // Fill u_local: evaluate weight at all K beta values for each local replica
        // [OPT-1] EnumerateTuples() once per sample (beta-independent gate +
        // OPT-3/4 D-product tables), EvaluateEnumeratedAt() per beta for just
        // the beta-dependent divdiff part -- see mbarpt.cpp's OpenMP
        // counterpart for the full rationale. nt==0 (identically-zero
        // weight) skips the whole K-loop, matching what per-beta evaluation
        // would have produced anyway.
        for (int k_l = 0; k_l < K_local; k_l++) {
            int s_idx = k_l * N_SAMPLES + samp;
            local_replicas[k_l]->EnsureCachesValid();
            // [DIAG lattice] record the lattice sector at this sample
            E0_series[k_l].push_back(local_replicas[k_l]->Energies_replicas[0][0]);
            lat_series[k_l].push_back(local_replicas[k_l]->lattice[0].to_ullong());
            q_series[k_l].push_back(local_replicas[k_l]->q);
            int nt = local_replicas[k_l]->EnumerateTuples();
            if (nt == 0) zero_weight_local[k_l]++;  // [DIAG zero-weight] whole row is 1e300
            int zero_mantissa_this_row = 0;  // [DIAG m==0] columns hit via m==0.0 this row
            for (int j = 0; j < K; j++) {
                if (nt == 0) {
                    u_local[s_idx * K + j] = 1e300;
                    continue;
                }
                // [OPT-2] Pass the precomputed row instead of rebuilding it per call.
                ExExFloat w = local_replicas[k_l]->EvaluateEnumeratedAt(beta_ladder[j], bpf[j].data());
                w.normalize();
                double m = w.get_mantissa();
                if (m > 0.0) {
                    u_local[s_idx * K + j] = -std::log(m) - (double)w.get_exponent() * 0.6931471805599453;
                    sign_pos++;
                    // [DIAG u-spread] accumulate only genuinely finite cells.
                    u_sum_local[j] += u_local[s_idx * K + j];
                    u_sumsq_local[j] += u_local[s_idx * K + j] * u_local[s_idx * K + j];
                    u_count_local[j]++;
                } else {
                    u_local[s_idx * K + j] = 1e300;
                    if (m < 0.0) sign_neg++;  // [F9] true sign-problem event, not a genuine zero
                    else zero_mantissa_this_row++;  // [DIAG m==0] m == 0.0 exactly, nt!=0
                }
            }
            // [DIAG m==0] nt==0 already forced the whole row to 1e300 above and
            // is counted separately -- only classify m==0.0 clustering when
            // nt!=0 (this loop actually evaluated all K columns for real).
            if (nt != 0) {
                if (zero_mantissa_this_row == K) zero_mantissa_full_local[k_l]++;
                else if (zero_mantissa_this_row > 0) zero_mantissa_partial_local[k_l]++;
            }
        }

        // [F17 fix] Record self-energy at each local slot's own (permanent) rung.
        for (int k_l = 0; k_l < K_local; k_l++) {
            int r = rank_start + k_l;  // global rung of local slot k_l -- never changes
            tau_series[k_l].push_back(u_local[(k_l * N_SAMPLES + samp) * K + r]);
        }

        // [OPT-5] Recalibrate the recording stride every OPT5_RECAL_INTERVAL
        // recorded samples -- see mbarpt.cpp's OpenMP counterpart for the
        // full rationale. MUST be rank-uniform: the recal condition depends
        // only on `samp`, which is identical on every rank by construction
        // (same loop trip count, same samp_start after the Bcast-verified
        // resume above), so every rank enters this block on the same
        // iteration -- collective-safe. Allreduce(MAX) the LOCAL tau_rec
        // (each rank only has its own K_local rungs' tau_series) before
        // computing new_stride, so every rank's adaptive_stride_update() call
        // sees the identical tau_rec and therefore produces the identical
        // new_stride with no further communication.
#if !defined(MBAR_FIXED_STRIDE_FACTOR)   // [DIAG fixed-stride] pinned builds skip the whole recalibration block
        if ((samp + 1) % OPT5_RECAL_INTERVAL == 0) {
            double local_tau_rec = 0.0;
            for (int k_l = 0; k_l < K_local; k_l++) {
                double t = compute_tau_int(tau_series[k_l]);
                if (t > local_tau_rec) local_tau_rec = t;
            }
            double tau_rec = 0.0;
            MPI_Allreduce(&local_tau_rec, &tau_rec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            int new_stride = adaptive_stride_update(stride_factor, tau_rec);
            if (new_stride != stride_factor) {
                if (mpi_rank == 0)
                    std::cout << "  recording stride -> " << new_stride
                              << "  (tau_rec=" << std::fixed << std::setprecision(1) << tau_rec << ")\n";
                stride_factor = new_stride;
            }
        }
#endif  // !MBAR_FIXED_STRIDE_FACTOR

        // [F20 fix] Stream this sample's row to disk (durability copy).
        append_ustream_sample(ustream_out, u_local.data(), samp, K_local, K, N_SAMPLES);

        // [qmax fix] Local-only check + immediate MPI_Abort: cheap (K_local
        // flag reads, no collective) every sample, and MPI_Abort(MPI_COMM_WORLD,
        // ...) terminates the whole job regardless of which rank calls it --
        // an Allreduce-based agreement first isn't needed and would add a
        // collective call every single sample for no benefit.
        QmaxStats qs = scan_qmax(local_replicas);
        record_qmax_histogram(qmax_hist, local_replicas);
        if (qs.any_achieved) {
            // Deliberately this rank's own rungs only, NOT cross-rank-combined
            // like the post-thermalization check -- getting a global max here
            // would need an Allreduce that only THIS rank (the one that hit
            // qmax) would call, which is exactly the deadlock this check's
            // local-only design exists to avoid. Still useful: it's a lower
            // bound on the true across-ladder q_99.9.
            std::cerr << "[rank " << mpi_rank << "] ";
            print_qmax_recommendation(local_max_q99_9(qmax_hist), std::cerr, /*censored=*/true);
            std::cerr << "[rank " << mpi_rank << "] (this rank's own rungs only -- "
                         "the true max across the whole ladder may be higher)\n";
            print_qmax_fatal_message(std::cerr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // [F20 fix] Allreduce agreement (see the thermalization loop's
        // comment for why, unlike qmax above). samp+1 because this sample
        // (and its ustream row, just written above) is fully done.
        {
            int local_ckpt = g_checkpoint_requested ? 1 : 0, global_ckpt = 0;
            MPI_Allreduce(&local_ckpt, &global_ckpt, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (global_ckpt) {
                ustream_out.close();  // std::exit() skips destructors -- flush explicitly first
                write_checkpoint(alpha_val, mpi_rank, CheckpointPhase::SAMPLING, samp + 1,
                                  stride_factor, local_replicas);
                if (mpi_rank == 0)
                    std::cout << "  [checkpoint] SIGTERM received; saved and exiting "
                                 "(resubmit to continue).\n";
                MPI_Finalize();
                std::exit(75);  // EX_TEMPFAIL (sysexits.h): temporary, please retry
            }
        }

        if (samp % (N_SAMPLES / 10) == 0 && samp > 0) {
            double local_sum_q = (double)qs.sum_q, local_max_q = (double)qs.max_q;
            double sum_q_g = 0.0, max_q_g = 0.0;
            MPI_Reduce(&local_sum_q, &sum_q_g, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max_q, &max_q_g, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            // [DIAG swap-fallback] Local delta since the last print, then sum
            // across ranks -- cheap (2 long longs), collective already needed
            // here anyway for q occupancy, so no extra synchronization cost.
            SwapFallbackStats now_swap_stats = scan_swap_fallback(local_replicas);
            unsigned long long local_d_total    = now_swap_stats.total    - prev_swap_stats.total;
            unsigned long long local_d_fallback = now_swap_stats.fallback - prev_swap_stats.fallback;
            unsigned long long d_total_g = 0, d_fallback_g = 0;
            MPI_Reduce(&local_d_total, &d_total_g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_d_fallback, &d_fallback_g, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            prev_swap_stats = now_swap_stats;

            if (mpi_rank == 0) {
                double elapsed = (double)clock() / CLOCKS_PER_SEC - wall_start;
                double fallback_pct = d_total_g > 0
                    ? 100.0 * (double)d_fallback_g / (double)d_total_g : 0.0;
                std::cout << "  " << samp << "/" << N_SAMPLES
                          << " samples  elapsed=" << elapsed << "s\n"
                          << "    q occupancy: mean=" << (sum_q_g / K) << "  max=" << max_q_g
                          << "  (qmax=" << qmax << ")\n"
                          << "    swap fallback: " << d_fallback_g << "/" << d_total_g
                          << " this block (" << std::fixed << std::setprecision(1)
                          << fallback_pct << "%)\n";
            }
        }
    }

    // [F20 fix] swap_accept/swap_total/sign_pos/sign_neg are NOT restored from
    // the checkpoint on resume (unlike tau_series/u_local, which are
    // reconstructed from the streamed prefix) -- there's nowhere they were
    // persisted, since a rejected/accepted swap count and a sign event aren't
    // recoverable from the u-matrix after the fact. This doesn't affect
    // S_alpha or its error bar (those come from the correctly-reconstructed
    // u_local), only these two diagnostic prints -- label them honestly
    // instead of silently understating a resumed run's true totals.

    // Reduce swap statistics to rank 0 for display
    {
        // Per-rank counts stay int (<= ~2 attempts per swap round), but their SUM
        // over 32 ranks can pass INT_MAX once N_SAMPLES/STEPS_PER_SAMP grow ~10-100x,
        // which would print a garbage acceptance % -- reduce as long long instead.
        long long local_accept = swap_accept, local_total = swap_total;
        long long global_accept = 0, global_total = 0;
        MPI_Reduce(&local_accept, &global_accept, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_total,  &global_total,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            std::cout << "  Sampling complete.  PT swap acceptance: "
                      << (global_total > 0 ? 100.0 * global_accept / global_total : 0.0) << "%";
            if (samp_start > 0)
                std::cout << "  (since resuming at sample " << samp_start
                           << " -- excludes samples before that)";
            std::cout << "\n";
            // [OPT-5] stride_factor is rank-uniform (Allreduce'd at every
            // recalibration), so total_mc_steps_phase2 is already identical
            // on every rank -- no reduction needed, just print rank 0's copy.
            std::cout << "  Effective MC steps this phase: " << total_mc_steps_phase2
                      << "  (final recording stride=" << stride_factor << ")\n";
        }
    }

    // [F9] Reduce sign statistics to rank 0 for the diagnostic report.
    {
        long long global_pos = 0, global_neg = 0;
        MPI_Reduce(&sign_pos, &global_pos, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sign_neg, &global_neg, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            report_sign_stats(SignStats{global_pos, global_neg}, std::cout);
            if (samp_start > 0 && global_neg > 0)
                std::cout << "  (sign stats above are since resuming at sample " << samp_start
                           << " only -- not restorable from the checkpoint, see mbar_common.hpp)\n";
        }
    }

    // [qmax fix] Printed unconditionally, not just on a fatal trigger -- per
    // MBARPT_audit.txt Sec. B, qmax "must be re-derived whenever N, beta_target,
    // or alpha changes," so even a run that never hit the cap gives real data
    // for sizing the NEXT (different) config instead of another guess. All
    // ranks call the Allreduce (unlike the per-sample fatal check, this is a
    // single one-time call after the loop, safe to combine across ranks).
    {
        int local_q99_9 = local_max_q99_9(qmax_hist), global_q99_9 = 0;
        MPI_Allreduce(&local_q99_9, &global_q99_9, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            print_qmax_recommendation(global_q99_9, std::cout);
            if (resumed)
                std::cout << "  (qmax sizing above is since resuming this invocation only -- "
                             "the histogram isn't checkpointed, so it doesn't include steps/"
                             "samples from before the resume)\n";
        }
    }

    // ── DIAGNOSTIC [erratic-deviation investigation, added 2026-09-10] ───────
    // Same test as mbarpt.cpp's OpenMP driver -- see that file's comment
    // right above the identical block for the full physics/plain-English
    // writeup (also renyicomp/ERRATIC_DEVIATION_FINDINGS.md). MPI-specific:
    // solve_mbar_mpi is collective (Allreduce-based), so every rank must call
    // it on its own K_local-row prefix; only rank 0 prints, since the result
    // is already identical on every rank by construction.
    {
        if (mpi_rank == 0)
            std::cout << "\n[DIAG bias-vs-N] Progressive ln_Z_" << alpha_val
                      << "(beta_target) using increasing sample-count PREFIXES (as if the "
                         "run had\n  stopped early) -- checking whether the estimate trends "
                         "toward the truth as N grows,\n  the signature of a finite-sample "
                         "MBAR/BAR bias:\n";
        for (double frac : {0.25, 0.5, 0.75, 1.0}) {
            int Nk_frac = std::min(N_SAMPLES, std::max(K, (int)(N_SAMPLES * frac)));
            std::vector<double> u_local_frac((size_t)K_local * Nk_frac * K);
            for (int k = 0; k < K_local; k++)
                std::copy_n(u_local.data() + (size_t)k * N_SAMPLES * K, (size_t)Nk_frac * K,
                            u_local_frac.data() + (size_t)k * Nk_frac * K);
            std::vector<double> f_frac(K, 0.0);
            solve_mbar_mpi(u_local_frac, K, K_local * Nk_frac, Nk_frac, f_frac, false);
            if (mpi_rank == 0) {
                double lnZ_frac = log_Z_anchor + pipeline_stat(f_frac);   // [OVERLAP fix] ln F in open mode
                std::cout << "  " << std::setw(3) << (int)(frac * 100) << "% (" << Nk_frac
#ifdef OVERLAP_PLUS
                          << "/rung, raw/unthinned):  ln F = "
#else
                          << "/rung, raw/unthinned):  ln_Z_" << alpha_val << " = "
#endif
                          << std::scientific << std::setprecision(8) << lnZ_frac << std::fixed << "\n";
            }
        }
        if (mpi_rank == 0)
#if defined(MBAR_THIN_POINT_ESTIMATE)
            std::cout << "  (the 100% row above is a raw/unthinned point estimate -- it should "
                         "be CLOSE to, but not necessarily bit-identical to, the official "
                         "RESULTS line below, which may use thinned data for its own point "
                         "estimate.)\n";
#else
            std::cout << "  (the 100% row above is the all-sample point estimate: the same solver on the same data as the "
                         "official RESULTS line below, which since 2026-10-09 also uses all recorded samples -- the two "
                         "ln Z values should be identical.)\n";
#endif
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2.5: AUTOCORRELATION ANALYSIS
    // ════════════════════════════════════════════════════════════════════════
    if (mpi_rank == 0) std::cout << "\n[2.5/3] Autocorrelation analysis...\n";

    std::vector<double> tau_local(K_local);
    for (int k = 0; k < K_local; k++)
        tau_local[k] = compute_tau_int(tau_series[k]);
    { std::vector<std::vector<double>> tmp; tau_series.swap(tmp); }  // free

    // Gather all tau values to all ranks for a consistent block_len computation.
    // [OPT-8] Allgatherv, not Allgather: block sizes vary across ranks now, so
    // a single uniform recvcount would be wrong. rung_block_len/rung_block_start
    // (built identically on every rank by partition_rungs_by_cost()) are
    // exactly the per-rank counts/displacements needed -- blocks are
    // contiguous and in rank order, so displacement[p] == rung_block_start[p]
    // lines up each rank's contribution with its global rung range in all_tau.
    std::vector<double> all_tau(K, 0.0);
    MPI_Allgatherv(tau_local.data(), K_local, MPI_DOUBLE,
                   all_tau.data(), rung_block_len.data(), rung_block_start.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double max_tau = *std::max_element(all_tau.begin(), all_tau.end());

    // [DIAG zero-weight] Same Allgatherv pattern as tau above -- gather each
    // rung's nt==0 ("A never closes, identically zero weight at every beta")
    // sample count so we can see, per rung, what fraction of that rung's
    // N_SAMPLES rows are the fully-1e300 sentinel row implicated in the
    // suspected solve_mbar_mpi log-sum-exp cancellation bug.
    std::vector<long long> zero_weight_all(K, 0);
    MPI_Allgatherv(zero_weight_local.data(), K_local, MPI_LONG_LONG,
                   zero_weight_all.data(), rung_block_len.data(), rung_block_start.data(), MPI_LONG_LONG,
                   MPI_COMM_WORLD);
    long long zero_weight_total = 0;
    for (int r = 0; r < K; r++) zero_weight_total += zero_weight_all[r];

    // [DIAG m==0] Same Allgatherv pattern -- these are owned per local slot
    // k_l (the replica being evaluated), same as zero_weight_local.
    std::vector<long long> zero_mantissa_full_all(K, 0), zero_mantissa_partial_all(K, 0);
    MPI_Allgatherv(zero_mantissa_full_local.data(), K_local, MPI_LONG_LONG,
                   zero_mantissa_full_all.data(), rung_block_len.data(), rung_block_start.data(), MPI_LONG_LONG,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(zero_mantissa_partial_local.data(), K_local, MPI_LONG_LONG,
                   zero_mantissa_partial_all.data(), rung_block_len.data(), rung_block_start.data(), MPI_LONG_LONG,
                   MPI_COMM_WORLD);
    long long zero_mantissa_full_total = 0, zero_mantissa_partial_total = 0;
    for (int r = 0; r < K; r++) {
        zero_mantissa_full_total += zero_mantissa_full_all[r];
        zero_mantissa_partial_total += zero_mantissa_partial_all[r];
    }

    // [DIAG u-spread] Unlike tau/zero_weight (owned per rung), every rank
    // contributes to every COLUMN j (each rank's K_local replicas are each
    // evaluated at all K target betas) -- so this needs a full elementwise
    // SUM across ranks, not a disjoint-ownership Allgatherv.
    std::vector<double> u_sum_all(K, 0.0), u_sumsq_all(K, 0.0);
    std::vector<long long> u_count_all(K, 0);
    MPI_Allreduce(u_sum_local.data(), u_sum_all.data(), K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(u_sumsq_local.data(), u_sumsq_all.data(), K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(u_count_local.data(), u_count_all.data(), K, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        for (int r = 0; r < K; r++) {
            double mean = u_count_all[r] > 0 ? u_sum_all[r] / u_count_all[r] : 0.0;
            double var  = u_count_all[r] > 0 ? std::max(0.0, u_sumsq_all[r] / u_count_all[r] - mean * mean) : 0.0;
            std::cout << "    rung " << std::setw(2) << r
                      << "  beta=" << std::fixed    << std::setprecision(4) << beta_ladder[r]
                      << "  tau_int=" << std::fixed << std::setprecision(2) << all_tau[r]
                      << " samples"
                      << "  zero_weight=" << zero_weight_all[r] << "/" << N_SAMPLES
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * zero_weight_all[r] / N_SAMPLES) << "%)"
                      << "  m0_full=" << zero_mantissa_full_all[r]
                      << "  m0_partial=" << zero_mantissa_partial_all[r]
                      << "  [DIAG u-spread] u_mean=" << std::scientific << std::setprecision(4) << mean
                      << "  u_stddev=" << std::sqrt(var)
                      << "  n_finite=" << u_count_all[r] << std::fixed << "\n";
        }
        std::cout << "  [DIAG zero-weight] total = " << zero_weight_total << "/" << (K * N_SAMPLES)
                  << "  (" << std::fixed << std::setprecision(1)
                  << (100.0 * zero_weight_total / ((double)K * N_SAMPLES)) << "%) -- "
                     "fully-1e300 rows suspected of contaminating solve_mbar_mpi's "
                     "log-sum-exp via exact 1e300-1e300 cancellation (see comment at "
                     "zero_weight_local's declaration).\n";
        std::cout << "  [DIAG m==0] full-row (all K columns m==0.0, nt!=0) = " << zero_mantissa_full_total
                  << "/" << (K * N_SAMPLES) << "  (" << std::fixed << std::setprecision(1)
                  << (100.0 * zero_mantissa_full_total / ((double)K * N_SAMPLES)) << "%)"
                  << "  -- these are the dangerous ones: a beta-INDEPENDENT structural zero "
                     "(e.g. REALW(D0*D1*D2)==0 for every j) reproduces the same whole-row "
                     "1e300-1e300 cancellation as the already-falsified nt==0 theory, just "
                     "via a different trigger nt==0 doesn't see.\n"
                  << "  [DIAG m==0] partial-row (some but not all K columns) = " << zero_mantissa_partial_total
                  << "/" << (K * N_SAMPLES) << "  (" << std::fixed << std::setprecision(1)
                  << (100.0 * zero_mantissa_partial_total / ((double)K * N_SAMPLES)) << "%)"
                  << "  -- comparatively harmless: >=1 finite column keeps log_denom[n] finite, "
                     "so these are correctly suppressed rather than phantom-weighted.\n"
                  << "  [DIAG u-spread] compare u_mean/u_stddev across rungs above: if they're "
                     "essentially flat despite beta spanning " << beta_ladder[0] << " to "
                  << beta_ladder[K-1] << ", the per-sample weight itself isn't beta-differentiated "
                     "-- independent of which specific mechanism explains why.\n";
        if (samp_start > 0)
            std::cout << "  (zero_weight/m==0/u-spread counts above are since resuming at sample "
                       << samp_start << " only, but the /" << N_SAMPLES << " and /" << (K * N_SAMPLES)
                       << " denominators are NOT reduced to match -- not restorable from the "
                          "checkpoint, same as sign stats -- so the printed percentages "
                          "understate the true fraction for a resumed run.)\n";
    }

#if defined(DUMP_QSERIES) && DUMP_QSERIES
    // [DUMP q] (only when compiled with -DDUMP_QSERIES=1; no effect otherwise)
    // Writes each local rung's recorded q series (int32, one value per recorded sample, in
    // sampling order) to renyi_qseries_a<alpha>_r<global rung>.bin, so that <q> per rung can be
    // re-analysed offline with any burn-in cut (the kept u-matrix files do not contain q).
    // After a checkpoint resume the series covers the post-resume samples only.
    for (int k = 0; k < K_local; k++) {
        std::ostringstream qname;
        qname << "renyi_qseries_a" << alpha_val << "_r" << (rank_start + k) << ".bin";
        std::ofstream qf(qname.str().c_str(), std::ios::binary);
        qf.write(reinterpret_cast<const char*>(q_series[k].data()),
                 (std::streamsize)(q_series[k].size() * sizeof(int)));
    }
#endif

    // ── [DIAG lattice] Lattice-sector report, per-pair PT acceptance, round trips ──
    // Per rung (from this rank's own series, then Allgatherv'd): mean and
    // 20-block standard error of E(z_0), tau_int of the E(z_0) series and of
    // the q series, mean q, number of distinct lattice[0] values visited and
    // number of lattice changes between consecutive recorded samples, and the
    // acceptance fractions of the spin-flip / block-swap / composite moves over
    // the WHOLE run (counters are never reset).  Compare the E(z_0) and q
    // columns with `python3 tools/ed_observables.py --beta <target> --alpha <a>`
    // (exact ED at every rung of this ladder).  A cold rung whose lattice
    // changes only a few dozen times, whose <E(z_0)> misses ED by more than its
    // blocking error, and whose flip acceptance is ~0 is the frozen-spin
    // signature discussed in fable_audit/BIAS_AUDIT_REPORT.md (F-A).
    {
        const int NCOL = 11;   // [10] = fraction of samples with odd-parity y0 (OVERLAP_PLUS)
        std::vector<double> lat_local((size_t)K_local * NCOL, 0.0);
        for (int k = 0; k < K_local; k++) {
            const std::vector<double>& E = E0_series[k];
            int n = (int)E.size();
            double meanE = 0.0, meanq = 0.0;
            for (int i = 0; i < n; i++) { meanE += E[i]; meanq += q_series[k][i]; }
            if (n > 0) { meanE /= n; meanq /= n; }
            // 20-block standard error of the mean of E(z_0)
            double seE = 0.0;
            { int nb = 20, L = n / nb;
              if (L > 0) {
                  double s = 0.0, s2 = 0.0;
                  for (int b = 0; b < nb; b++) {
                      double m = 0.0;
                      for (int i = b * L; i < (b + 1) * L; i++) m += E[i];
                      m /= L; s += m; s2 += m * m;
                  }
                  double var = (s2 / nb - (s / nb) * (s / nb)) * nb / (nb - 1.0);
                  seE = std::sqrt(std::max(0.0, var) / nb);
              } }
            std::vector<double> qd(q_series[k].begin(), q_series[k].end());
            double tauE = compute_tau_int(E);
            double tauq = compute_tau_int(qd);
            std::set<unsigned long long> distinct(lat_series[k].begin(), lat_series[k].end());
            long long changes = 0;
            for (int i = 1; i < n; i++) if (lat_series[k][i] != lat_series[k][i - 1]) changes++;
            RenyiReplica* r = local_replicas[k];
            double* row = &lat_local[(size_t)k * NCOL];
            row[0] = meanE; row[1] = seE; row[2] = tauE; row[3] = meanq; row[4] = tauq;
            row[5] = (double)distinct.size(); row[6] = (double)changes;
#if defined(OVERLAP_PLUS)
            { long long odd = 0; for (unsigned long long y : lat_series[k]) odd += __builtin_popcountll(y) & 1;
              row[10] = n > 0 ? (double)odd / n : 0.0; }
#endif
            row[7] = r->flip_attempts      ? (double)r->flip_accepts      / r->flip_attempts      : -1.0;
            row[8] = r->block_attempts     ? (double)r->block_accepts     / r->block_attempts     : -1.0;
#if defined(OVERLAP_PLUS) && OV_CUT_FLIP
            row[8] = r->cutflip_attempts   ? (double)r->cutflip_accepts   / r->cutflip_attempts   : -1.0;   // block_acc column = cut-flip acceptance
#endif
            row[9] = r->composite_attempts ? (double)r->composite_accepts / r->composite_attempts : -1.0;
        }
        std::vector<int> cnt(mpi_size), disp(mpi_size);
        for (int p = 0; p < mpi_size; p++) { cnt[p] = rung_block_len[p] * NCOL; disp[p] = rung_block_start[p] * NCOL; }
        std::vector<double> lat_all((size_t)K * NCOL, 0.0);
        MPI_Allgatherv(lat_local.data(), K_local * NCOL, MPI_DOUBLE,
                       lat_all.data(), cnt.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        std::vector<long long> pair_acc_g(K - 1, 0), pair_tot_g(K - 1, 0);
        MPI_Allreduce(pair_acc.data(), pair_acc_g.data(), K - 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(pair_tot.data(), pair_tot_g.data(), K - 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        long long rt_g = 0, ta_g = 0;
        MPI_Allreduce(&pt_round_trips, &rt_g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&pt_top_arrivals, &ta_g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            std::cout << "\n  [DIAG lattice] per-rung lattice sector (sampling phase; compare E0/q with tools/ed_observables.py):\n"
                      << "    rung     beta    <E0>     se(E0)  tau_E     <q>   tau_q  distinct  changes  flip_acc  block_acc  comp_acc  PTacc(pair r,r+1)"
#if defined(OVERLAP_PLUS)
                      << "  [odd-parity fraction of y0 at end of line]"
#endif
                      << "\n";
            for (int r = 0; r < K; r++) {
                const double* row = &lat_all[(size_t)r * NCOL];
                std::cout << "    " << std::setw(4) << r
                          << std::fixed << std::setprecision(4) << std::setw(9) << beta_ladder[r]
                          << std::setprecision(4) << std::setw(9) << row[0]
                          << std::setprecision(4) << std::setw(9) << row[1]
                          << std::setprecision(1) << std::setw(8) << row[2]
                          << std::setprecision(2) << std::setw(8) << row[3]
                          << std::setprecision(1) << std::setw(8) << row[4]
                          << std::setw(10) << (long long)row[5]
                          << std::setw(9)  << (long long)row[6]
                          << std::setprecision(4) << std::setw(10) << row[7]
                          << std::setw(11) << row[8]
                          << std::setw(10) << row[9];
                if (r < K - 1)
                    std::cout << std::setprecision(3) << std::setw(10)
                              << (pair_tot_g[r] > 0 ? (double)pair_acc_g[r] / pair_tot_g[r] : -1.0)
                              << " (" << pair_acc_g[r] << "/" << pair_tot_g[r] << ")";
#if defined(OVERLAP_PLUS)
                std::cout << "  odd=" << std::setprecision(4) << row[10];
#endif
                std::cout << "\n";
            }
            std::cout << "  [DIAG lattice] PT round trips (bottom->top->bottom) during sampling: " << rt_g
                      << "   bottom->top arrivals: " << ta_g
                      << "   (a healthy ladder makes >= 10 round trips per run; 0-2 means the top rungs are effectively isolated)\n";
        }
    }

    // [F4 fix] block_len (was "stride") is identical on every rank -- derived
    // from the Allgathered max_tau and the compile-time N_SAMPLES, no extra
    // communication needed. This is what lets the Phase 3.5 block bootstrap
    // below stay synchronized across ranks without any communication either
    // (see BlockBootstrapPlan's doc comment in mbar_common.hpp).
    int block_len = std::max(1, (int)std::ceil(max_tau));

#if defined(MBAR_THIN_POINT_ESTIMATE)
    int Nk_thin   = N_SAMPLES / block_len;
    if (mpi_rank == 0)
        std::cout << "  max tau_int = " << std::fixed << std::setprecision(2) << max_tau
                  << " samples  →  block_len=" << block_len
                  << "  thinned Nk (point estimate)=" << Nk_thin
                  << "  (" << (100 - 100 * Nk_thin / N_SAMPLES) << "% reduction)\n";
#else
    // [all-sample point estimate, 2026-10-09] block_len is now used ONLY as the block length of the bootstrap error bar.
    if (mpi_rank == 0)
        std::cout << "  max tau_int = " << std::fixed << std::setprecision(2) << max_tau
                  << " samples  →  block_len=" << block_len
                  << "  (bootstrap block length; the point estimate uses all " << N_SAMPLES << " samples per rung, not thinned)\n";
#endif

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3: MBAR SOLVER (point estimate, distributed across all ranks)
    // ════════════════════════════════════════════════════════════════════════
    // [F4 fix] Thinning here is a compute-cost optimization only -- MBAR's
    // point estimate is unbiased regardless of sample correlation. Falling
    // back to the raw series when thinning would leave too few rows (a
    // numerical-conditioning concern) no longer affects the error bar: that
    // always comes from the block bootstrap in Phase 3.5, on the raw series,
    // regardless of which branch this takes. Warn loudly, don't abort -- a
    // short run still gets an honest, if wide, error bar.
    if (mpi_rank == 0) std::cout << "\n[3/3] Running MBAR (distributed)...\n";

    std::vector<double> u_pe_local;   // thinned copy; stays EMPTY in the default (all-sample) build
    int Nk_eff;

#if !defined(MBAR_THIN_POINT_ESTIMATE)
    // [all-sample point estimate, 2026-10-09] DEFAULT.  The reported point estimate (the RESULTS line, the z1 anchor cache) is
    // the MBAR solution on ALL recorded samples.  Thinning to every block_len-th sample only threw away half the data and added
    // noise to the shipped number (all-sample minus thinned: +0.015 at alpha=1 beta=5, +0.015 at alpha=2 beta=3, -0.009 at
    // alpha=1 beta=1.5, +0.046 at alpha=3 beta=1.5); it was never needed for the error bar, which comes from the block bootstrap
    // below (resampling whole blocks of the raw series, block length = ceil(max tau_int)).  Compile with
    // -DMBAR_THIN_POINT_ESTIMATE to get the old thinned point estimate back (to compare with runs made before this change).
    Nk_eff = N_SAMPLES;
    if (mpi_rank == 0)
        std::cout << "  Point estimate: ALL " << N_SAMPLES << " recorded samples per rung (not thinned).\n";
#else
    const int MIN_NK_WARN = std::max(K, 100);
    if (mpi_rank == 0)
        std::cout << "  Point estimate: THINNED to every block_len-th sample (compiled with -DMBAR_THIN_POINT_ESTIMATE).\n";
    if (block_len > 1 && Nk_thin >= MIN_NK_WARN) {
        if (mpi_rank == 0)
            std::cout << "  Building thinned u-matrix ("
                      << K_local << "x" << Nk_thin << " per rank) for the point estimate...\n";
        u_pe_local.resize((size_t)K_local * Nk_thin * K);
        for (int k = 0; k < K_local; k++)
            for (int i = 0; i < Nk_thin; i++)
                std::copy_n(u_local.data() + (k * N_SAMPLES + i * block_len) * K, K,
                            u_pe_local.data() + (k * Nk_thin + i) * K);
        Nk_eff = Nk_thin;
    } else {
        if (block_len > 1 && mpi_rank == 0)
            std::cout << "  WARNING: thinning by block_len=" << block_len << " would leave only "
                      << Nk_thin << " samples/rung (<" << MIN_NK_WARN
                      << ") — using all " << N_SAMPLES << " raw samples for the point estimate "
                         "instead (the block bootstrap below reports an honest, correlation-aware "
                         "error bar regardless of this choice).\n";
        Nk_eff = N_SAMPLES;
    }
#endif  // MBAR_THIN_POINT_ESTIMATE

    int S_eff_local = K_local * Nk_eff;
    std::vector<double> f(K, 0.0);
    bool main_converged = solve_mbar_mpi(Nk_eff == N_SAMPLES ? u_local : u_pe_local, K, S_eff_local, Nk_eff, f);
    { std::vector<double> tmp; u_pe_local.swap(tmp); }  // free thinned copy (if any); raw u_local kept for Phase 3.5

    // ════════════════════════════════════════════════════════════════════════
    // [OVERLAP fix] Validity checks before any result can be reported. A failed
    // check aborts the run, so no result file is written.
    //  (a) the MBAR solve converged and every f is finite;
    //  (b) every recorded sample has a finite reduced potential at its OWN rung
    //      (a sampled configuration always has positive weight there; anything else
    //      is a bug);
    //  (c) the MBAR overlap between adjacent rungs, O_{i,i+1} = N sum_n W_ni W_n,i+1
    //      (Shirts & Chodera), is not negligible. O ~ 0 means the empirical ensembles
    //      are effectively disconnected and the free-energy differences undefined.
    // ════════════════════════════════════════════════════════════════════════
    {
        auto fatal = [&](const std::string& msg) {
            if (mpi_rank == 0) std::cerr << "\nFATAL [MBAR validity]: " << msg
                                         << "\n  No result is reported.\n";
            MPI_Abort(MPI_COMM_WORLD, 3);
        };
        if (!main_converged) fatal("the MBAR solve did not converge (or produced non-finite free energies).");
        long long bad_own_local = 0, bad_own = 0;
        for (int k_l = 0; k_l < K_local; k_l++) {
            int r = rank_start + k_l;
            for (int smp = 0; smp < N_SAMPLES; smp++)
                if (!(u_local[((size_t)k_l * N_SAMPLES + smp) * K + r] < 1e299)) bad_own_local++;
        }
        MPI_Allreduce(&bad_own_local, &bad_own, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (bad_own > 0) fatal(std::to_string(bad_own) + " recorded samples have zero weight at their own rung.");

        std::vector<double> ov; int min_i = 0;
        double min_ov = mbar_min_adjacent_overlap(u_local, K, (size_t)K_local * N_SAMPLES, N_SAMPLES, f, &min_i, &ov);
        if (mpi_rank == 0) {
            std::cout << "\n  [MBAR overlap] adjacent-rung overlaps O(i,i+1):";
            for (int i = 0; i + 1 < K; i++) std::cout << (i % 8 == 0 ? "\n    " : " ")
                                                      << std::fixed << std::setprecision(3) << ov[i];
            std::cout << "\n  [MBAR overlap] minimum " << std::setprecision(4) << min_ov
                      << " between rungs " << min_i << " and " << min_i + 1 << "\n";
            if (min_ov < 0.03)
                std::cout << "  WARNING: weak overlap between adjacent rungs (< 0.03); add rungs near "
                          << min_i << "-" << min_i + 1 << " -- free-energy differences there are poorly determined.\n";
        }
        if (!(min_ov >= 1e-6)) fatal("adjacent rungs " + std::to_string(min_i) + " and " + std::to_string(min_i + 1)
                                      + " have MBAR overlap " + std::to_string(min_ov) + " (effectively disconnected).");
    }

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 3.5: BLOCK BOOTSTRAP UNCERTAINTY (distributed)  [F4 fix]
    //
    // Every rank seeds boot_rng from the SAME broadcast value (unlike the old
    // per-rank-independent design) and calls plan_block_bootstrap()/
    // draw_bootstrap_block_starts() with the same (block_len, N_SAMPLES)
    // inputs -- both already rank-uniform -- so every rank draws the EXACT
    // SAME sequence of block-start indices for a given replicate b, with no
    // further communication. Each rank applies that shared draw to its OWN local
    // rungs' raw data. This is what preserves cross-rung correlation across
    // the whole distributed ladder: rung k on rank p and rung k' on rank p'
    // get resampled at the same time-slices for every replicate, exactly like
    // the single-process version, without moving any sample data between
    // ranks. solve_mbar_mpi aggregates via Allreduce as before, so f_b[0] is
    // identical on all ranks.
    // ════════════════════════════════════════════════════════════════════════
    if (mpi_rank == 0)
        std::cout << "\n[3.5/3] Block bootstrap uncertainty (" << N_BOOT << " replicates)...\n";

    BlockBootstrapPlan bp = plan_block_bootstrap(max_tau, N_SAMPLES);
    if (mpi_rank == 0) {
        if (bp.n_blocks_full < 20)
            std::cout << "  WARNING: only " << bp.n_blocks_full << " independent-ish blocks "
                         "(length " << bp.block_len << ", from tau_int) are available for the "
                         "bootstrap — the error bar below is honest but may itself be noisy. "
                         "Consider a longer run.\n";
        if (bp.n_blocks_boot < bp.n_blocks_full)
            std::cout << "  (subsampling to " << bp.n_blocks_boot << "/" << bp.n_blocks_full
                      << " blocks/replicate for memory; scaling stderr by sqrt("
                      << bp.n_blocks_boot << "/" << bp.n_blocks_full << "))\n";
    }

    int Nk_boot = bp.Nk_boot;
    int S_boot_local = K_local * Nk_boot;

    // rng_seed itself is NOT rank-uniform (mainQMC.hpp's init_rng sets it to
    // 1000+mpi_rank under EXACTLY_REPRODUCIBLE, or an independent
    // std::random_device draw per rank otherwise) -- it deliberately gives
    // each rank's own physics RNG an independent stream. The bootstrap needs
    // the OPPOSITE property (see comment above), so broadcast rank 0's value
    // once and seed every rank's boot_rng from that shared value instead.
    unsigned int shared_boot_seed = rng_seed;
    MPI_Bcast(&shared_boot_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Fixed salt so the bootstrap stream differs from the physics stream seeded by the same value.
    const unsigned int kBootstrapSeedSalt = 3203386110u;
    std::mt19937 boot_rng(shared_boot_seed ^ kBootstrapSeedSalt);  // IDENTICAL on every rank

    std::vector<double> boot_f0(N_BOOT);
    int n_boot_fail = 0;   // [OVERLAP fix] bootstrap replicates that failed (see boot_ok)
    // A replicate counts only if its solve converged, its statistic is finite, and its
    // adjacent rungs stay connected (a disconnected replicate can "converge" with
    // undetermined relative free energies).
    auto boot_ok = [&](bool conv, const std::vector<double>& u_b, size_t rows, int nk, const std::vector<double>& fb) {
        if (!conv || !std::isfinite(pipeline_stat(fb))) return false;
        return mbar_min_adjacent_overlap(u_b, K, rows, nk, fb) >= 1e-6;
    };
    // [OPT-11] u_boot_local/f_b/starts hoisted above the loop and refilled
    // each replicate instead of allocated fresh -- see mbarpt.cpp's OpenMP
    // counterpart for the full rationale (u_boot_local's copy loop below
    // always overwrites every element, so reuse is safe).
    std::vector<int> starts;
    std::vector<double> u_boot_local(S_boot_local * K);
    std::vector<double> f_b = f;
    for (int b = 0; b < N_BOOT; b++) {
        draw_bootstrap_block_starts(bp, boot_rng, starts);  // same draw on every rank
        for (int blk = 0; blk < bp.n_blocks_boot; blk++) {
            int src_start = starts[blk];
            for (int k = 0; k < K_local; k++)
                std::copy_n(u_local.data() + (k * N_SAMPLES + src_start) * K, (size_t)bp.block_len * K,
                            u_boot_local.data() + (k * Nk_boot + blk * bp.block_len) * K);
        }

        f_b = f;  // warm start from converged main solution (assignment, not construction)
        { bool conv = solve_mbar_mpi(u_boot_local, K, S_boot_local, Nk_boot, f_b, false);
          if (!boot_ok(conv, u_boot_local, (size_t)S_boot_local, Nk_boot, f_b)) n_boot_fail++; }  // [OVERLAP fix]
        // f_b[0] is the same on all ranks (solve_mbar_mpi uses Allreduce)
        boot_f0[b] = pipeline_stat(f_b);

        if ((b + 1) % std::max(1, N_BOOT / 4) == 0 && mpi_rank == 0)
            std::cout << "  bootstrap " << (b + 1) << "/" << N_BOOT << "\n";
    }
#ifdef OVERLAP_MODE
    // [OVERLAP fix] Block-length stability of the bootstrap error on the reported statistic.
    // The default block length comes from the autocorrelation of the own-rung reduced
    // potential, which can be much shorter than that of the reported contrast. Repeat the
    // bootstrap with blocks 4x, 16x and 64x longer (when at least 10 blocks remain). An error
    // that keeps growing with block length means the per-run error bar is not trustworthy;
    // quote replication errors instead.
    double scan_se_max = 0.0;   // largest error over the longer block lengths
    {
        std::vector<std::pair<int,double>> scan;
        for (int factor : {4, 16, 64}) {
            BlockBootstrapPlan bq = plan_block_bootstrap(max_tau * factor, N_SAMPLES);
            if (bq.n_blocks_full < 10) break;
            int nrep = std::max(20, N_BOOT / 2);
            std::vector<double> u_b((size_t)K_local * bq.Nk_boot * K), f_q = f, vals(nrep);
            for (int b = 0; b < nrep; b++) {
                draw_bootstrap_block_starts(bq, boot_rng, starts);
                for (int blk = 0; blk < bq.n_blocks_boot; blk++)
                    for (int k = 0; k < K_local; k++)
                        std::copy_n(u_local.data() + ((size_t)k * N_SAMPLES + starts[blk]) * K, (size_t)bq.block_len * K,
                                    u_b.data() + ((size_t)k * bq.Nk_boot + (size_t)blk * bq.block_len) * K);
                f_q = f;
                { bool conv = solve_mbar_mpi(u_b, K, K_local * bq.Nk_boot, bq.Nk_boot, f_q, false);
                  if (!boot_ok(conv, u_b, (size_t)K_local * bq.Nk_boot, bq.Nk_boot, f_q)) n_boot_fail++; }
                vals[b] = pipeline_stat(f_q);
            }
            double m = 0, v = 0;
            for (double x : vals) m += x;
            m /= nrep;
            for (double x : vals) v += (x - m) * (x - m);
            scan.push_back({bq.block_len, std::sqrt(v / (nrep - 1)) * std::sqrt((double)bq.n_blocks_boot / bq.n_blocks_full)});
            scan_se_max = std::max(scan_se_max, scan.back().second);
        }
        if (mpi_rank == 0 && !scan.empty()) {
            double var0 = 0, m0 = 0;
            for (double x : boot_f0) m0 += x;
            m0 /= N_BOOT;
            for (double x : boot_f0) var0 += (x - m0) * (x - m0);
            double se0 = std::sqrt(var0 / (N_BOOT - 1)) * std::sqrt((double)bp.n_blocks_boot / bp.n_blocks_full);
            std::cout << "  [bootstrap block scan] block length -> stderr of reported statistic:\n"
                      << "    " << bp.block_len << " -> " << std::scientific << std::setprecision(3) << se0 << "\n";
            double se_max = se0;
            for (auto& pr : scan) { std::cout << "    " << pr.first << " -> " << pr.second << "\n"; se_max = std::max(se_max, pr.second); }
            if (se_max > 1.3 * se0)
                std::cout << "  WARNING: the bootstrap error grows with block length (x" << std::fixed << std::setprecision(2)
                          << se_max / se0 << "); the per-run error bar is not stable -- use the spread over independent runs.\n";
        }
    }
#endif
    { std::vector<double> tmp; u_local.swap(tmp); }  // free raw u_local -- nothing below needs it
    if (n_boot_fail > 0 && mpi_rank == 0)
        std::cout << "  WARNING: " << n_boot_fail << " bootstrap replicates failed (non-converged, non-finite or "
                     "disconnected); the error bar is reported as unavailable (NaN).\n";

    double mean_f0 = 0.0;
    for (double v : boot_f0) mean_f0 += v;
    mean_f0 /= N_BOOT;
    double var_f0 = 0.0;
    for (double v : boot_f0) { double d = v - mean_f0; var_f0 += d * d; }
    double boot_stderr = std::sqrt(var_f0 / (N_BOOT - 1))
                       * std::sqrt((double)bp.n_blocks_boot / bp.n_blocks_full);

    if (mpi_rank == 0)
        std::cout << "  Bootstrap stderr(ln Z_" << alpha_val << ") = "
                  << std::scientific << std::setprecision(4) << boot_stderr
                  << "  (boot mean f[0]=" << mean_f0 << "  main f[0]=" << f[0] << ")\n";
    // [OVERLAP fix] What is exported: NaN if any replicate failed; otherwise the largest
    // error over the block-length scan when it grew by more than 30%.
    g_stderr_short = boot_stderr;
    g_stderr_status = 0;
    double reported_stderr = boot_stderr;
    if (n_boot_fail > 0) { g_stderr_status = 2; reported_stderr = std::numeric_limits<double>::quiet_NaN(); }
    else if (scan_se_max > 1.3 * boot_stderr) { g_stderr_status = 1; reported_stderr = scan_se_max; }
    if (boot_stderr_out) *boot_stderr_out = reported_stderr;

    // ── Compute absolute ln Z_alpha(beta_target) ─────────────────────────────
    double log_Z_alpha_target = log_Z_anchor + pipeline_stat(f);

    if (mpi_rank == 0) {
        std::cout << "\n  ln Z_" << alpha_val << "(beta_target) = "
                  << std::scientific << std::setprecision(8) << log_Z_alpha_target
                  << "  (anchor=" << log_Z_anchor << "  f[0]=" << f[0] << ")\n";
        std::cout << "\n  Full beta-ladder (f[j] = ln Z_" << alpha_val
                  << "(beta_target) - ln Z_" << alpha_val << "(beta_j)):\n";
        for (int j = 0; j < K; j++)
            std::cout << "    beta=" << std::fixed    << std::setprecision(5) << beta_ladder[j]
                      << "  f="     << std::scientific << std::setprecision(4) << f[j]
                      << "  ln_Z="  << (log_Z_alpha_target - f[j]) << "\n";
    }

    // [F20 fix] This pipeline completed normally -- clear this rank's
    // checkpoint/stream files so an identical-parameters rerun doesn't
    // mistake them for an in-progress run to resume.
    clear_checkpoint(alpha_val, mpi_rank);

    for (int k = 0; k < K_local; k++) delete local_replicas[k];
    return log_Z_alpha_target;
}

// ─────────────────────────────────────────────────────────────────────────────
// main  — MPI orchestration (mirrors mbarpt.cpp main, with MPI init/finalize)
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Set global mpi_rank/mpi_size (declared in mainQMC.hpp, used by init_rng)
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // [OPT-8] Rungs are now partitioned into variable-size contiguous blocks
    // (partition_rungs_by_cost()), so exact divisibility is no longer
    // required -- only that every rank gets at least one non-empty block,
    // i.e. mpi_size <= N_BETA.
    if (mpi_size > N_BETA) {
        if (mpi_rank == 0)
            std::cerr << "ERROR: num_ranks (" << mpi_size << ") exceeds MBAR_N_BETA ("
                      << N_BETA << ") -- every rank needs at least one rung.\n";
        MPI_Finalize();
        return 1;
    }

    double wall_start = (double)clock() / CLOCKS_PER_SEC;

    // ── One-time initialization of shared QMC infrastructure ─────────────────
    divdiff_init();
    init_rng();  // uses mpi_rank for EXACTLY_REPRODUCIBLE seed variation
    {
        double curr = 1.0; factorial[0] = curr;
        for (int k = 1; k < qmax; k++) { curr *= k; factorial[k] = curr; }
    }
    for (int i = 0; i < Ncycles; i++) cycle_len[i] = cycles[i].count();
    cycle_min_len = 64;
    for (int i = 0; i < Ncycles; i++) cycle_min_len = std::min(cycle_min_len, cycle_len[i]);
    cycle_max_len = 0;
    for (int i = 0; i < Ncycles; i++) cycle_max_len = std::max(cycle_max_len, cycle_len[i]);
    for (int i = 0; i < Nop; i++)
        for (int j = 0; j < Ncycles; j++)
            if (cycles[j].test(i)) P_in_cycles[i].set(j);

    // [F10 fix] one-time startup guard; deterministic/identical on every rank,
    // so only rank 0 needs to print it.
    if (mpi_rank == 0) check_numerics_bounds();

    // ── Subsystem geometry ────────────────────────────────────────────────────
    std::bitset<N> mask_A(SUBSYSTEM_A_MASK), mask_B(SUBSYSTEM_B_MASK);
    int N_A = (int)mask_A.count(), N_B = (int)mask_B.count();

    // ── Build beta-ladder ─────────────────────────────────────────────────────
    std::vector<double> beta_ladder = build_beta_ladder();
    int K = N_BETA;

#ifdef OVERLAP_MODE
    // ═════════════════════════════════════════════════════════════════════════
    // [OVERLAP] Ground-state overlap mode.
    //
    // Estimates ln Z_ab(tau) with Z_ab(tau) = Tr[exp(-tau H_a) exp(-tau H_b)]
    // for one pair ab in {11, 12, 22} (first program argument, default 12).
    // The ladder variable ("beta" everywhere else in this file) is tau, the
    // projection time of EACH of the two segments. Combine three runs with
    // tools/combine_overlap.py:
    //     |<g1|g2>|^2 = lim_{tau->inf} Z_12(tau) / sqrt(Z_11(tau) Z_22(tau)),
    // where Z_aa(tau) = Tr exp(-2 tau H_a).
    //
    // Anchor at tau_min (closed form, exact through O(tau^2); for traceless
    // Hamiltonians without three-body Pauli closures the next term is O(tau^4)):
    //   Z_ab/2^N = 1 - tau (t_a + t_b) + tau^2/2 (<H_a^2> + <H_b^2> + 2<H_a H_b>)
    // with t = Tr(H)/2^N and <XY> = Tr(XY)/2^N from tools/make_tables.py.
    // ═════════════════════════════════════════════════════════════════════════
#ifdef OVERLAP_PLUS
    // ═════════════════════════════════════════════════════════════════════════
    // [OVERLAP_PLUS] |+> trial state, fixed beta = Beta, one PT/MBAR run over the
    // interpolation ladder s_k = k/(K-1): s in [0,1/2] is the pair (0, 2s),
    // s in [1/2,1] the pair (2s-1, 1), so rung 0 = O_11, mid rung = O_12,
    // last rung = O_22. Output: ln F_beta (gauge invariant, no anchor).
    // ═════════════════════════════════════════════════════════════════════════
    {
        if (K % 2 == 0 || K < 3) {
            if (mpi_rank == 0) std::cerr << "FATAL: OVERLAP_PLUS needs an odd MBAR_N_BETA >= 3\n";
            MPI_Finalize();
            return 1;
        }
        if (N_B != 0 || N_A != N) {
            if (mpi_rank == 0) std::cerr << "FATAL: OVERLAP_PLUS needs A = all qubits, B = empty\n";
            MPI_Finalize();
            return 1;
        }
        for (int k = 0; k < K; k++) beta_ladder[k] = (double)k / (K - 1);
        if (mpi_rank == 0)
            std::cout << "═══════════════════════════════════════════════════\n"
                      << "  OVERLAP_PLUS  |+> trial state, beta=" << (double)(Beta)
                      << " (tau=" << 0.5 * (double)(Beta) << " per segment)\n"
                      << "  N=" << N << "  interpolation ladder: " << K
                      << " rungs (0,0) -> (0,1) -> (1,1)\n"
                      << "  T_equil=" << T_EQUIL << "  N_samples=" << N_SAMPLES
                      << "  steps_per_sample=" << STEPS_PER_SAMP << "  ranks=" << mpi_size << "\n"
                      << "═══════════════════════════════════════════════════\n";
        double err = 0.0;
        double lnF = run_mbar_pt_pipeline_mpi(ALPHA, beta_ladder, 0.0, wall_start, &err);
        if (mpi_rank == 0) {
            std::cout << "\n  RESULT  beta=" << (double)(Beta)
                      << std::scientific << std::setprecision(10)
                      << "  lnF=" << lnF << "  stderr=" << err
                      << "  F=" << std::exp(lnF) << "\n";
            std::ofstream out("overlap_plus.dat");
            out << std::setprecision(17)
                << "# beta lnF stderr N n_rungs n_samples stderr_status stderr_default_block\n"
                << "# stderr_status: 0 usable; 1 grew with block length (stderr = largest scanned value);"
                   " 2 unavailable (stderr = nan). Quote the spread over independent runs.\n"
                << (double)(Beta) << " " << lnF << " " << err << " "
                << N << " " << K << " " << N_SAMPLES << " " << g_stderr_status << " " << g_stderr_short << "\n";
        }
        divdiff_clear_up();
        MPI_Finalize();
        return 0;
    }
#endif
    {
        int pair = (argc > 1) ? std::atoi(argv[1]) : 12;
        int ha = pair / 10 - 1, hb = pair % 10 - 1;
        if (!(pair == 11 || pair == 12 || pair == 21 || pair == 22)) {
            if (mpi_rank == 0) std::cerr << "usage: " << argv[0] << " {11|12|22}\n";
            MPI_Finalize();
            return 1;
        }
        if (N_B != 0 || N_A != N) {
            if (mpi_rank == 0) std::cerr << "FATAL: OVERLAP_MODE needs A = all qubits, B = empty\n";
            MPI_Finalize();
            return 1;
        }
        g_ov_ham[0] = hb;   // copy 0: segment [0,c), acts first
        g_ov_ham[1] = ha;   // copy 1: segment [c,q)

        const double t = BETA_MIN;
        double series = 1.0 - t * (OV_TR_H[ha] + OV_TR_H[hb])
                      + 0.5 * t * t * (OV_TR_HH[ha][ha] + OV_TR_HH[hb][hb] + 2.0 * OV_TR_HH[ha][hb]);
        double anchor = N * std::log(2.0) + std::log(series);

        if (mpi_rank == 0)
            std::cout << "═══════════════════════════════════════════════════\n"
                      << "  OVERLAP MODE  Z_" << pair << "(tau) = Tr[exp(-tau H" << ha + 1
                      << ") exp(-tau H" << hb + 1 << ")]\n"
                      << std::scientific << std::setprecision(6)
                      << "  N=" << N << "  tau ladder: " << K << " points ["
                      << beta_ladder[0] << " to " << beta_ladder[K-1] << "]\n"
                      << "  anchor ln Z(tau_min) = " << anchor << "\n"
                      << "  T_equil=" << T_EQUIL << "  N_samples=" << N_SAMPLES
                      << "  steps_per_sample=" << STEPS_PER_SAMP << "  ranks=" << mpi_size << "\n"
                      << "═══════════════════════════════════════════════════\n";

        double err = 0.0;
        double lnZ = run_mbar_pt_pipeline_mpi(ALPHA, beta_ladder, anchor, wall_start, &err);

        if (mpi_rank == 0) {
            std::cout << "\n  RESULT  pair=" << pair << "  tau=" << BETA_TARGET
                      << std::scientific << std::setprecision(10)
                      << "  lnZ=" << lnZ << "  stderr=" << err << "\n";
            std::ofstream out("overlap_lnZ_" + std::to_string(pair) + ".dat");
            out << std::setprecision(17)
                << "# pair tau lnZ stderr N n_beta n_samples stderr_status stderr_default_block\n"
                << pair << " " << BETA_TARGET << " " << lnZ << " " << err << " "
                << N << " " << K << " " << N_SAMPLES << " " << g_stderr_status << " " << g_stderr_short << "\n";
        }
        divdiff_clear_up();
        MPI_Finalize();
        return 0;
    }
#endif

    // ── Compute anchors: exact classical Z_α(β_min) + off-diagonal O(β_min²)
    //    quantum correction  (F7 fix; see compute_offdiag_anchor_terms).
    //    All ranks compute identically -- O(2^N * Nop) is trivially fast. ────
    double od2_inf, intra_A_sq_over_dimA;
    compute_offdiag_anchor_terms(od2_inf, intra_A_sq_over_dimA);

    double anchor_alpha = std::log(compute_classical_renyi_Z(BETA_MIN, ALPHA))
                        + compute_quantum_anchor_correction(BETA_MIN, ALPHA, od2_inf, intra_A_sq_over_dimA);
    double anchor_1     = std::log(compute_classical_renyi_Z(BETA_MIN, 1))
                        + compute_quantum_anchor_correction(BETA_MIN, 1, od2_inf, intra_A_sq_over_dimA);

    if (mpi_rank == 0) {
        std::cout << "  N_A=" << N_A << "  N_B=" << N_B << "  alpha=" << ALPHA << "\n"
                  << std::scientific << std::setprecision(6)
                  << "  beta=0 reference (Eq. 5, sanity check only): "
                  << compute_anchor_lnZ(N_A, N_B, ALPHA) << "\n"
                  << "  od2_inf=" << od2_inf << "  intra_A_sq/dim_A=" << intra_A_sq_over_dimA << "\n"
                  << "  Anchor ln Z_" << ALPHA << "(β_min): classical="
                  << std::log(compute_classical_renyi_Z(BETA_MIN, ALPHA))
                  << "  +quantum_corr=" << compute_quantum_anchor_correction(BETA_MIN, ALPHA, od2_inf, intra_A_sq_over_dimA)
                  << "  total=" << anchor_alpha << "\n"
                  << "  Anchor ln Z_1(β_min):  classical="
                  << std::log(compute_classical_renyi_Z(BETA_MIN, 1))
                  << "  +quantum_corr=" << compute_quantum_anchor_correction(BETA_MIN, 1, od2_inf, intra_A_sq_over_dimA)
                  << "  total=" << anchor_1 << "\n";

        std::cout << "═══════════════════════════════════════════════════\n"
                  << "  MBAR+PT MPI  RenyiReplica(alpha=" << ALPHA << ")"
                  << "  beta_target=" << BETA_TARGET << "\n"
                  << "  beta-ladder: " << K << " points  ["
                  << beta_ladder[0] << " to " << beta_ladder[K-1] << "]\n"
                  << "  T_equil=" << T_EQUIL
                  << "  N_samples=" << N_SAMPLES
                  << "  steps_per_sample=" << STEPS_PER_SAMP
                  << "  ranks=" << mpi_size << "\n"
                  << "═══════════════════════════════════════════════════\n";
    }

    // ── Obtain ln Z_1(beta_target) — rank 0 reads/writes file, broadcasts ─────
    double ln_z1 = std::numeric_limits<double>::quiet_NaN();
    double boot_stderr_1 = 0.0;  // [F5 fix] loaded from the cache on a hit, not left at 0
    if (ALPHA > 1) {
        int found = 0;
        if (mpi_rank == 0) found = read_z1_anchor(ln_z1, boot_stderr_1) ? 1 : 0;

        // Broadcast found-flag, ln_z1, and boot_stderr_1 together to all ranks.
        MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ln_z1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&boot_stderr_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (!found) {
            if (mpi_rank == 0)
                std::cout << "\n  [z1_anchor] No cached Z1 found — running α=1 pipeline first...\n";
            ln_z1 = run_mbar_pt_pipeline_mpi(1, beta_ladder, anchor_1,
                                              wall_start, &boot_stderr_1);
            if (mpi_rank == 0) {
                write_z1_anchor(ln_z1, boot_stderr_1);
                std::cout << "  [z1_anchor] α=1 complete.  ln Z1 = "
                          << std::scientific << std::setprecision(8) << ln_z1 << "\n\n";
            }
        } else {
            if (mpi_rank == 0)
                std::cout << "  [z1_anchor] Loaded cached ln Z1 = "
                          << std::scientific << std::setprecision(8) << ln_z1
                          << "  (stderr=" << boot_stderr_1 << ")  from " << Z1_ANCHOR_FILE << "\n";
        }
    }

    // ── Run primary ALPHA pipeline ────────────────────────────────────────────
    double boot_stderr_alpha = 0.0;
    double log_Z_alpha_target = run_mbar_pt_pipeline_mpi(ALPHA, beta_ladder,
                                                          anchor_alpha, wall_start,
                                                          &boot_stderr_alpha);

    // ── If ALPHA==1, cache result for future ALPHA>1 runs ─────────────────────
    if (ALPHA == 1) {
        ln_z1 = log_Z_alpha_target;
        if (mpi_rank == 0) write_z1_anchor(ln_z1, boot_stderr_alpha);
    }

    // ── Rényi entropy (Paper Eq. 14) ──────────────────────────────────────────
    // STUDY NOTE: same formula/F5 fix as mbarpt.cpp's main() -- see that
    // file's comment right above this identical block for the full physics
    // and plain-English writeup (fix landed 2026-07-09, reviewed 2026-09-10).
    double S_alpha = 0.0, stderr_S = 0.0;
    if (ALPHA > 1) {
        S_alpha = ((double)ALPHA * ln_z1 - log_Z_alpha_target) / (ALPHA - 1.0);
        double inv_am1 = 1.0 / (ALPHA - 1.0);
        stderr_S = inv_am1 * std::sqrt(
            boot_stderr_alpha * boot_stderr_alpha
            + (double)ALPHA * (double)ALPHA * boot_stderr_1 * boot_stderr_1);
    }

    // ── Output final results (rank 0 only) ────────────────────────────────────
    if (mpi_rank == 0) {
        double wall_end = (double)clock() / CLOCKS_PER_SEC;
        std::cout << "\n═══════════════════════════════════════════════════\n"
                  << "  RESULTS  alpha=" << ALPHA << "  beta_target=" << BETA_TARGET << "\n"
                  << "───────────────────────────────────────────────────\n"
                  << std::scientific << std::setprecision(6)
                  << "  ln Z_" << ALPHA << "(beta_target) = " << log_Z_alpha_target
                  << " ± " << boot_stderr_alpha << "\n"
                  << "  Z_"    << ALPHA << "(beta_target) = " << std::exp(log_Z_alpha_target)
                  << " ± " << std::exp(log_Z_alpha_target) * boot_stderr_alpha << "\n";
        if (ALPHA > 1) {
            std::cout << "  ln Z_1(beta_target)  = " << ln_z1;
            if (boot_stderr_1 > 0.0)
                std::cout << " ± " << boot_stderr_1;
            else
                std::cout << "  (loaded from cache — no bootstrap)";
            std::cout << "\n";
            // STUDY NOTE: same reasoning as mbarpt.cpp's OpenMP driver -- see
            // that file's comment right above the identical block (added
            // 2026-09-10, per Prof. Hen's guidance): Z_1 printed directly
            // because an absolute error on ln Z is a RELATIVE error on Z
            // itself, which "delta/ln_Z" style reporting badly understates.
            std::cout << "  Z_1(beta_target)    = " << std::exp(ln_z1);
            if (boot_stderr_1 > 0.0)
                std::cout << " ± " << std::exp(ln_z1) * boot_stderr_1;
            else
                std::cout << "  (loaded from cache — no bootstrap)";
            std::cout << "\n"
                      << "  S_" << ALPHA << "(beta_target) = " << S_alpha
                      << " ± " << stderr_S << "\n";
            if (boot_stderr_1 == 0.0)
                std::cout << "  (S_α error bar includes only ln Z_α uncertainty; "
                             "rerun with MBAR_ALPHA=1 to include ln Z_1 term)\n";
        }
        std::cout << "  Wall time: " << (wall_end - wall_start) << " s\n"
                  << "  MPI ranks: " << mpi_size << "\n"
                  << "═══════════════════════════════════════════════════\n";
    }

    // [output sanity gate] 0 <= S_alpha <= N_A*ln2 for any valid quantum state.
    // S_alpha is identical on every rank (log_Z_alpha_target/ln_z1 come from
    // Allreduce-synchronized MBAR solves and an explicit Bcast on a cache hit),
    // so every rank reaches the same verdict with no extra communication
    // needed; only rank 0 prints the message, but every rank exits with the
    // same non-zero code on a violation so the job fails visibly either way.
    bool sane = (ALPHA == 1) || renyi_entropy_sane(S_alpha, N_A);
    if (!sane && mpi_rank == 0) print_sanity_violation(S_alpha, ALPHA, N_A, std::cerr);

    divdiff_clear_up();
    MPI_Finalize();
    return sane ? 0 : 1;
}
