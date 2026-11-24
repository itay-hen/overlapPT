# Python reference implementation and figure scripts

Copied from the manuscript folder (`gsOverlap/tex/`) so they are versioned with the C++ code.

- `ptmbar.py`, `runmc.py`: Python reference implementation used for Tables V–VII and Fig. 3.
- `fig_mc.py`, `figs.py`, `fig2.py`: figure scripts.

Known gaps:
- `fig_mc.py` loads `mc_summary.npz`, which is not present anywhere; it must be regenerated from the reference runs.
- `figs.py` and `fig2.py` both write `ed_scaling.pdf`.
