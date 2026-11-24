# Python reference implementation and figure scripts

Copied from the manuscript folder (`gsOverlap/tex/`) so they are versioned with the C++ code.

- `ptmbar.py`, `runmc.py`: Python reference implementation used for Tables V–VII and Fig. 3.
- `fig_mc.py`, `figs.py`, `fig2.py`: figure scripts. `figs.py` writes `severed.pdf` and `severed3.pdf`;
  `fig2.py` writes `ed_scaling.pdf` (the exact-diagonalization scaling figure in the manuscript).
  The older ED panel still in `figs.py` is written to `ed_scaling_legacy.pdf` and is not used.

Known gaps:
- `fig_mc.py` loads `mc_summary.npz`, which is not present anywhere; it must be regenerated from the reference runs.
