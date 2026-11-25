# Python reference implementation and figure scripts

Copied from the manuscript folder (`gsOverlap/tex/`) so they are versioned with the C++ code.

- `ptmbar.py`, `runmc.py`: Python reference implementation used for Tables V–VII and Fig. 3.
- `fig_mc.py`, `figs.py`, `fig2.py`: figure scripts. `figs.py` writes `severed.pdf` and `severed3.pdf`;
  `fig2.py` writes `ed_scaling.pdf` (the exact-diagonalization scaling figure in the manuscript).
  The older ED panel still in `figs.py` is written to `ed_scaling_legacy.pdf` and is not used.

Recorded runs (regenerated 2026-11-25 after the external review; the original Fig. 3 data were never stored):
- `ring_b4/`, `ring_b2/`, `ring_b6/`: 24 runs each of `python3 runmc.py 60000 SEED [BETA]`, seeds 1-24,
  four-site ring, nine-ensemble ladder. Each `res_<seed>.npz` holds the run's raw reduced potentials
  `u[k, n]`, the rung index of every stored configuration, the slot-0 walk-length series, swap
  acceptances and round trips. `run_manifest.json` has the SHA-256 of `ptmbar.py` and `runmc.py` as run
  and the Python/NumPy/SciPy versions. (The beta=4 runs used `runmc.py` before its optional BETA
  argument was added; the default is beta=4, so the code path is identical.)
- `python3 fig_mc.py ring_b4` recomputes every number of Table tab:mc and writes `mc_summary.json` and
  `mc_validation.pdf` (Fig. 3); `python3 fig_mc.py ring_b2` and `ring_b6` give the other rows of
  Table tab:beta.
- beta=6 is NOT consistent with exact (-2.62 SEM); lnF correlates with round trips (r = 0.89) and
  the residual builds up on the first leg: incomplete mixing of this kernel (no start-shift move).

Known gaps:
- Table tab:pq (30 single-ensemble chains at (0,0)) has no recorded data or script.
