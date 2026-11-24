> **This copy (overlapPT):** see `README_OVERLAP.md` and `VETTING.md`. The Rényi-specific Slurm
> script `recipe_sweep_job.sh` and the generated `hamiltonian.hpp` of the Rényi example are not
> included here (the original tree has them); the rest of this file describes the original code.

# mbarptNov: the working MBAR + parallel-tempering Rényi code, November 2026

This is the source tree that produced every "fixed code" result in the November update: the
recipe runs at α = 1, 2, 3 for β = 0.3 to 1.5, α = 2 at β = 3 and 5, and the checkpoint stop-and-resume
test. Nothing here is experimental; the diagnostic scripts, audit logs and the unfinished hardening
draft are left out on purpose.

## Files

| file | role |
|---|---|
| `mbarpt_mpi.cpp` | the program: 32 MPI ranks, one temperature rung each, MBAR over the rungs |
| `mbar_common.hpp` | MBAR solver, parallel tempering, checkpoint save/restore, sample stream |
| `replica.hpp` | the PMR-QMC replica: configuration, weights, all Monte Carlo moves (`RenyiReplica`) |
| `mainQMC.hpp`, `divdiff.hpp` | PMR-QMC core (divided differences, operator bookkeeping) |
| `parameters.hpp` | compile-time parameters; the job script patches β, α, warm-up and sample count per run |
| `hamiltonian.hpp` | generated from `H32.txt` by `prepare.cpp` (the 9-qubit model used throughout) |
| `prepare.cpp`, `H32.txt` | regenerate `hamiltonian.hpp` with `g++ -O3 -o prepare prepare.cpp && ./prepare H32.txt` |
| `recipe_sweep_job.sh` | the Slurm job script: patches parameters, compiles, runs the α = 1 anchor then α, reads the result against ED, checkpoints on the timer signal and resumes |
| `RECIPE.md` | the production recipe and why each item is there |
| `tools/` | Python readers: `ed_reference.py` (exact values), `recipe_check.py`, `pair_estimates.py`, `early_block_checks.py`, `sweep_analysis.py`, and the rest |
| `SEGMENT_EXCHANGE_FIX.md` | the retry-loop defect: mechanism, code before and after, tests |
| `README_PMRQMC_original.md` | the original PMR-QMC readme (Barash, Babakhani, Hen, PRR 6, 013281) |

## How the validated runs were built

The source is compiled by the job script with exactly these flags:

```
mpicxx -O3 -std=c++17 -DSEGMENT_EXCHANGE_MAX_ATTEMPTS=1 -DMBAR_FIXED_STRIDE_FACTOR=64 \
       -DDUMP_QSERIES=1 -DDUMP_WARMUP_Q=1 -o mbarpt_mpi.bin mbarpt_mpi.cpp
```

- `-DSEGMENT_EXCHANGE_MAX_ATTEMPTS=1`: the segment-exchange move draws once (the fix for defect 1).
  Without the flag the source reproduces the old 64-retry behaviour, so the same tree can build both.
- `-DMBAR_FIXED_STRIDE_FACTOR=64`: the recording interval is pinned from the first sample
  (the fix for defect 2); without it the old auto-calibration runs.
- The two `DUMP` flags only write diagnostic q-series files; they do not change the sampling.
- The estimate uses all recorded samples by default (`-DMBAR_THIN_POINT_ESTIMATE` opts back into thinning).

Per run the script sets, through `sed` on `parameters.hpp`: `Beta`, `MBAR_ALPHA`, `MBAR_T_EQUIL`
(warm-up steps, chosen as at least 10 tempering round trips) and `MBAR_N_SAMPLES`. The values in
the file as shipped are placeholders. Example submission:

```
sbatch --job-name=a2b5_s1 --time=2-12:00:00 --qos=long \
       --export=ALL,ALPHA_RUN=2,BETA_RUN=5,T_EQUIL_RUN=4370000,SEED_LABEL=s1 recipe_sweep_job.sh
```

The script is written for Slurm with Open MPI 5 (the checkpoint stop uses `--signal=B:USR1@120`,
which `mpirun` forwards to the ranks; a SIGTERM would abort the job before anything is saved).
On a Grid Engine cluster the signal, resubmit and accounting parts need porting.

## Checksums (md5)

```
782744833402174d541dfe1eaf44c896  divdiff.hpp
982c2a0f5e4fcc3d607b4d8ffa579e58  hamiltonian.hpp
718d6951356285a694233bdcd2b4e6a0  mainQMC.hpp
fbf124c424130074387eefcb32c5f38c  mbar_common.hpp
d8e439557aa3b1ced18dc2ae9389f3e3  mbarpt_mpi.cpp
ce3b328fe9e9046f2cb6e1b38af3b67d  parameters.hpp
99deb4457f03aabdb9b2b9b211e1d3fb  replica.hpp
a63b8d3b636e5d8724d61f32a5b80c38  recipe_sweep_job.sh
89a172e4bdda0120b893025dfaf46132  H32.txt
```

`replica.hpp`, `mbarpt_mpi.cpp`, `mainQMC.hpp`, `divdiff.hpp` and `hamiltonian.hpp` are byte-identical
to the `build/` copies saved with every validated run. `mbar_common.hpp` differs from the runs before
November 3 by one line, the registration of SIGUSR1 as a second checkpoint signal, which does not
touch the sampling. `parameters.hpp` differs from each run's copy only in the four values the job
script patches.

## Not included

- `hardened_draft/`: checked file writes for the checkpoint, drafted and tested in Docker only,
  not yet run on the cluster.
- The single-point experiment scripts (`alpha*_ladder.sh`, `*_recipe_check.sh`, `segex_test_*.sh`,
  `single_rung_check.cpp`) and the audit logs. They are in the diagnostic tree if needed.
