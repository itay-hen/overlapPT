"""Fig. 3 (mc_validation.pdf) and every number of Table tab:mc from the reference runs.

Usage: python3 fig_mc.py RUN_DIR [OUT_DIR]
RUN_DIR holds res_<seed>.npz written by runmc.py (raw reduced potentials included).
Writes OUT_DIR/mc_validation.pdf and OUT_DIR/mc_summary.json (default OUT_DIR = RUN_DIR) and
prints the table.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptmbar import tau_int

run_dir = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else run_dir
files = sorted(glob.glob(os.path.join(run_dir, 'res_*.npz')), key=lambda p: int(p.split('_')[-1][:-4]))
R = [np.load(p) for p in files]; n = len(R)
lnF = np.array([float(r['lnF']) for r in R]); sdm = np.array([float(r['sd_lnF']) for r in R])
f = np.array([r['f'] for r in R]); fex = R[0]['f_ex']; Fex = float(R[0]['F_ex']); lnFex = np.log(Fex)
K = f.shape[1]; i00, i01, i11 = 0, 4, K - 1
m, s = lnF.mean(), lnF.std(ddof=1); sem = s / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
res = (f - fex).mean(0)
tau_u = []
for r in R:
    sel = r['idx'] == 0
    tau_u.append(tau_int(r['u'][i01, sel] - r['u'][i00, sel]))
summ = {
    'runs': n, 'seeds': [int(r['seed']) for r in R], 'sweeps': int(R[0]['ns']),
    'lnF_exact': lnFex, 'F_exact': Fex,
    'lnF_mean': m, 'lnF_sem': sem, 'deviation': m - lnFex, 'deviation_sem': (m - lnFex) / sem,
    'lnF_ci95': [m - t * sem, m + t * sem],
    'F_geo': float(np.exp(m)), 'F_ci95': [float(np.exp(m - t * sem)), float(np.exp(m + t * sem))],
    'F_arith': float(np.exp(lnF).mean()), 'F_lognormal': float(np.exp(m + s * s / 2)),
    'run_sd': s, 'mbar_sd_mean': float(sdm.mean()), 'sd_ratio': float(s / sdm.mean()),
    'swap_acc_range': [float(min(r['acc'].min() for r in R)), float(max(r['acc'].max() for r in R))],
    'roundtrips_range': [int(min(r['rt'] for r in R)), int(max(r['rt'] for r in R))],
    'tau_q_range': [float(min(r['tau'] for r in R)), float(max(r['tau'] for r in R))],
    'tau_udiff_mean': float(np.mean(tau_u)), 'tau_udiff_range': [float(min(tau_u)), float(max(tau_u))],
    'mean_q_range': [float(min(r['q'].mean() for r in R)), float(max(r['q'].mean() for r in R))],
    'ladder_residual_max': float(np.abs(res).max()), 'ladder_residuals': res.tolist(),
    'combo_residual': float(res[i00] - 2 * res[i01] + res[i11]),
}
json.dump(summ, open(os.path.join(out, 'mc_summary.json'), 'w'), indent=1)
for k, v in summ.items():
    if k != 'ladder_residuals': print(f'{k:22s} {v}')

plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
lab = ['(0,0)', '(0,¼)', '(0,½)', '(0,¾)', '(0,1)', '(¼,1)', '(½,1)', '(¾,1)', '(1,1)']
x = np.arange(K)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
axs[0].plot(x, fex, 'k-', lw=1, label='exact')
axs[0].errorbar(x, f.mean(0), yerr=f.std(0, ddof=1) / np.sqrt(n), fmt='o', ms=3.5, capsize=2,
                color='#b03a2e', label=f'PT--MBAR ({n} runs)')
axs[0].set_xticks(x); axs[0].set_xticklabels(lab, rotation=45, fontsize=6)
axs[0].set_ylabel(r'$f_k$ (gauge $f_0=0$)'); axs[0].set_xlabel('ladder ensemble')
axs[0].legend(frameon=False, fontsize=7); axs[0].set_title('(a) ladder free energies', fontsize=8)
ax = axs[1]; xr = np.arange(1, n + 1)
ax.axhspan(m - sem, m + sem, color='#1e8449', alpha=0.3, label='mean $\\pm$ SEM')
ax.axhline(lnFex, color='k', lw=1, label='exact')
ax.errorbar(xr, lnF, yerr=sdm, fmt='o', ms=3, capsize=1.5, color='0.45', label='per run (MBAR sd)')
ax.set_xlabel('run'); ax.set_ylabel(r'$\ln F_\beta$'); ax.set_xlim(0, n + 1)
ax.legend(frameon=False, fontsize=6.5, loc='lower right')
ax.set_title(f'(b) $\\ln F_\\beta$, {n} runs', fontsize=8)
fig.tight_layout(pad=.3); fig.savefig(os.path.join(out, 'mc_validation.pdf'))
print('wrote', os.path.join(out, 'mc_validation.pdf'))
