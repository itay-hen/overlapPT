"""
PMR mixed-walk parallel-tempering + MBAR estimator for ground-state overlaps.
Demonstration system: 4-site ring, H = J sum ZZ - Gamma sum XX.
  H(0) = full ring (4 bonds)      H(1) = severed (bonds 1,3 removed -> two dimers)
Trial state |+>  (open walks).   Target: F_beta = O(0,1)^2 / (O(0,0) O(1,1)).
Ensemble (a,b): weight of  <+| e^{-b H(la)/2} e^{-b H(lb)/2} |+>,
                pre-cut segment governed by H(lb), post-cut by H(la).
"""
import numpy as np
from scipy.linalg import expm

N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
SEVERED = {1, 3}
J, GAM = 1.0, 0.8
M = len(EDGES)
MASKS = np.array([(1 << i) | (1 << j) for i, j in EDGES])
DIM = 1 << N
Z = np.array([[1 - 2 * ((s >> i) & 1) for i in range(N)] for s in range(DIM)])

# diagonal energies and hopping amplitudes of the two endpoint Hamiltonians
E_full = J * sum(Z[:, i] * Z[:, j] for i, j in EDGES).astype(float)
E_sev = J * sum(Z[:, i] * Z[:, j] for k, (i, j) in enumerate(EDGES)
                if k not in SEVERED).astype(float)
d_full = np.array([-GAM] * M)
d_sev = np.array([0.0 if k in SEVERED else -GAM for k in range(M)])

def E_of(lam):  return (1 - lam) * E_full + lam * E_sev
def d_of(lam):  return (1 - lam) * d_full + lam * d_sev

def H_dense(lam):
    H = np.diag(E_of(lam))
    d = d_of(lam)
    for k, m in enumerate(MASKS):
        for s in range(DIM):
            H[s ^ m, s] += d[k]
    return H

# ---------------------------------------------------------------- divided differences
_DDC = {}
def _dd_multiset(vals, counts, b, tag):
    """exp(-b[.]) over the multiset with the given multiplicities (order irrelevant).
    `tag` identifies the energy alphabet: alphabets of equal length must not collide."""
    key = (b, tag, counts)
    v = _DDC.get(key)
    if v is None:
        nodes = np.repeat(vals, counts)
        n = nodes.size - 1
        T = np.diag(nodes) + (np.diag(np.ones(n), 1) if n else 0)
        v = float(expm(-b * T)[0, n])
        _DDC[key] = v
    return v

def dd_row_col(codes, vals, b, tag):
    """pref[c]=exp(-b[E_0..E_c]) and suff[c]=exp(-b[E_c..E_q]) for the coded energy
    sequence.  These are row 0 and column q of exp(-bT) for one bidiagonal T; here
    they are accumulated through the multiset of visited energies."""
    nv = len(vals); q = len(codes) - 1
    pref = np.empty(q + 1); suff = np.empty(q + 1)
    cnt = [0] * nv
    for c in range(q + 1):
        cnt[codes[c]] += 1
        pref[c] = _dd_multiset(vals, tuple(cnt), b, tag)
    cnt = [0] * nv
    for c in range(q, -1, -1):
        cnt[codes[c]] += 1
        suff[c] = _dd_multiset(vals, tuple(cnt), b, tag)
    return pref, suff

_CODE = {}
def coder(lam):
    v = _CODE.get(lam)
    if v is None:
        E = E_of(lam); vals = np.unique(np.round(E, 12))
        v = (vals, np.searchsorted(vals, np.round(E, 12)))
        _CODE[lam] = v
    return v

# ---------------------------------------------------------------- configuration weight
class Cfg:
    __slots__ = ("y0", "G")
    def __init__(self, y0, G): self.y0, self.G = y0, list(G)

def states(c):
    y = np.empty(len(c.G) + 1, dtype=np.int64); y[0] = c.y0
    for r, k in enumerate(c.G): y[r + 1] = y[r] ^ MASKS[k]
    return y

_DPAR = {}
def dpar(lam):
    v = _DPAR.get(lam)
    if v is None:
        v = tuple(float(x) for x in d_of(lam)); _DPAR[lam] = v
    return v

def logW(c, la, lb, beta, y=None):
    """log of the cut-marginalised mixed weight; -inf if the weight vanishes.

    W = sum_c  pref_c * Dpre_c * suff_c * Dsuf_c , with pref/suff the prefix and
    suffix divided differences (row 0 and column q of one bidiagonal exp) and
    Dpre/Dsuf the partitioned hopping products."""
    if y is None: y = states(c)
    G = c.G; q = len(G)
    vb, cb = coder(lb); va, ca = coder(la)
    b = beta / 2
    codes_b = [cb[t] for t in y]; codes_a = [ca[t] for t in y]
    nb = len(vb); na = len(va)
    db = dpar(lb); da = dpar(la)
    # prefix divided differences under H(lb) and prefix hopping products
    cnt = [0] * nb; pref = []; dp = 1.0; Dpre = []
    for cpos in range(q + 1):
        cnt[codes_b[cpos]] += 1
        pref.append(_dd_multiset(vb, tuple(cnt), b, lb))
        Dpre.append(dp)
        if cpos < q: dp *= db[G[cpos]]
    # suffix divided differences under H(la) and suffix hopping products
    cnt = [0] * na; suff = [0.0] * (q + 1); Dsuf = [0.0] * (q + 1); ds = 1.0
    for cpos in range(q, -1, -1):
        cnt[codes_a[cpos]] += 1
        suff[cpos] = _dd_multiset(va, tuple(cnt), b, la)
        Dsuf[cpos] = ds
        if cpos > 0: ds *= da[G[cpos - 1]]
    W = 0.0
    for cpos in range(q + 1):
        W += pref[cpos] * Dpre[cpos] * suff[cpos] * Dsuf[cpos]
    return np.log(W) if W > 0 else -np.inf

# ---------------------------------------------------------------- Monte Carlo
# START_SHIFT = True adds the start-shift move (paper Sec. "Local updates", move 6): with
# probability 0.2 per attempt, either prepend a generator P_k (k uniform) and move the origin to
# y0' = P_k y0, so every configuration of the old walk is kept after the new first hop, or delete
# the first hop and move the origin to y1 (each branch with probability 1/2). Proposal ratio M,
# as for insertion and removal. Move probabilities then become 0.32 / 0.32 / 0.16 / 0.2.
START_SHIFT = False

def sweep(c, la, lb, beta, rng, lw):
    for _ in range(4):
        q = len(c.G)
        if START_SHIFT:
            if rng.random() < 0.2:
                if rng.random() < 0.5:                # prepend P_k, origin y0 -> y0 ^ P_k
                    k = int(rng.integers(M)); oy = c.y0
                    c.G.insert(0, k); c.y0 = int(oy ^ MASKS[k]); lw2 = logW(c, la, lb, beta)
                    if np.log(rng.random() + 1e-300) < lw2 - lw + np.log(M): lw = lw2
                    else: c.G.pop(0); c.y0 = oy
                else:                                 # delete first hop, origin -> y1
                    if q == 0: continue
                    oy = c.y0; k = c.G.pop(0); c.y0 = int(oy ^ MASKS[k]); lw2 = logW(c, la, lb, beta)
                    if np.log(rng.random() + 1e-300) < lw2 - lw - np.log(M): lw = lw2
                    else: c.G.insert(0, k); c.y0 = oy
                continue
        u = rng.random()
        if u < 0.4:                                   # insert
            k = rng.integers(M); p = rng.integers(q + 1)
            c.G.insert(p, int(k)); lw2 = logW(c, la, lb, beta)
            if np.log(rng.random() + 1e-300) < lw2 - lw + np.log(M): lw = lw2
            else: c.G.pop(p)
        elif u < 0.8:                                 # remove
            if q == 0: continue
            p = rng.integers(q); k = c.G.pop(p); lw2 = logW(c, la, lb, beta)
            if np.log(rng.random() + 1e-300) < lw2 - lw - np.log(M): lw = lw2
            else: c.G.insert(p, k)
        else:                                         # move the walk origin
            old = c.y0; c.y0 = int(old ^ (1 << rng.integers(N)))
            lw2 = logW(c, la, lb, beta)
            if np.log(rng.random() + 1e-300) < lw2 - lw: lw = lw2
            else: c.y0 = old
    return lw

def run_pt(ladder, beta, nsweep, nburn, stride, seed=0):
    rng = np.random.default_rng(seed)
    K = len(ladder)
    cfgs = [Cfg(int(rng.integers(DIM)), []) for _ in range(K)]
    lw = [logW(cfgs[i], *ladder[i], beta) for i in range(K)]
    swap_ok = np.zeros(K - 1); swap_try = np.zeros(K - 1)
    samples = []
    walker = list(range(K))            # walker[slot] = walker id
    wdir = [0] * K                     # +1 = has touched the bottom, -1 = the top
    roundtrips = 0
    qser = []                          # q at slot 0, one entry per storage event
    for t in range(nsweep):
        for i in range(K):
            lw[i] = sweep(cfgs[i], *ladder[i], beta, rng, lw[i])
        for i in range((t % 2), K - 1, 2):            # alternating replica exchange
            swap_try[i] += 1
            a = logW(cfgs[i], *ladder[i + 1], beta) + logW(cfgs[i + 1], *ladder[i], beta)
            b_ = lw[i] + lw[i + 1]
            if np.log(rng.random() + 1e-300) < a - b_:
                cfgs[i], cfgs[i + 1] = cfgs[i + 1], cfgs[i]
                walker[i], walker[i + 1] = walker[i + 1], walker[i]
                lw[i] = logW(cfgs[i], *ladder[i], beta)
                lw[i + 1] = logW(cfgs[i + 1], *ladder[i + 1], beta)
                swap_ok[i] += 1
        if wdir[walker[0]] != 1: wdir[walker[0]] = 1
        if wdir[walker[K - 1]] == 1: wdir[walker[K - 1]] = -1; roundtrips += 1
        if t >= nburn and (t - nburn) % stride == 0:
            qser.append(len(cfgs[0].G))
            for i in range(K):
                samples.append((i, Cfg(cfgs[i].y0, cfgs[i].G)))
    return samples, swap_ok / np.maximum(swap_try, 1), roundtrips, np.array(qser)

# ---------------------------------------------------------------- MBAR
def mbar(u, Ncounts, iters=6000):
    """u[k,n] = -ln W_k(C_n).  Returns free energies f_k up to an additive constant."""
    K, Ntot = u.shape
    f = np.zeros(K); lnN = np.log(Ncounts.astype(float))
    for _ in range(iters):
        denom = np.logaddexp.reduce((lnN + f)[:, None] - u, axis=0)   # over ensembles j
        fnew = -np.logaddexp.reduce(-u - denom, axis=1)
        fnew -= fnew[0]
        if np.max(np.abs(fnew - f)) < 1e-12: f = fnew; break
        f = fnew
    return f

# ---------------------------------------------------------------- driver
if __name__ == "__main__":
    import sys, time
    beta = 4.0
    nsweep = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    ladder = [(0.0, lb) for lb in grid] + [(la, 1.0) for la in grid[1:]]
    i00, i01, i11 = 0, 4, len(ladder) - 1

    plus = np.ones(DIM) / np.sqrt(DIM)
    Aex = {l: expm(-beta * H_dense(l) / 2) for l in grid}
    O = lambda a, b: float(plus @ Aex[a] @ Aex[b] @ plus)
    f_ex = np.array([-np.log(O(a, b) / O(0.0, 0.0)) for a, b in ladder])
    F_ex = O(0.0, 1.0) ** 2 / (O(0.0, 0.0) * O(1.0, 1.0))

    print(f"beta = {beta},  {len(ladder)} ensembles,  {nsweep} sweeps/replica")
    t0 = time.time()
    samples, acc = run_pt(ladder, beta, nsweep=nsweep, nburn=nsweep // 10,
                          stride=50, seed=17)
    idx = np.array([s_[0] for s_ in samples]); cfg = [s_[1] for s_ in samples]
    qs = np.array([len(c.G) for c in cfg])
    print(f"swap acceptance: {np.round(acc, 2)}")
    print(f"{len(cfg)} stored configs,  <q> = {qs.mean():.2f},  max q = {qs.max()}")
    u = np.empty((len(ladder), len(cfg)))
    for n, c in enumerate(cfg):
        y = states(c)
        for k, (la, lb) in enumerate(ladder):
            u[k, n] = -logW(c, la, lb, beta, y=y)
    Nc = np.array([(idx == k).sum() for k in range(len(ladder))])
    f = mbar(u, Nc)

    print("\n ensemble      f (MBAR)    f (exact)     error")
    for k, (la, lb) in enumerate(ladder):
        print(f"  ({la:.2f},{lb:.2f})   {f[k]:+9.4f}   {f_ex[k]:+9.4f}   {f[k]-f_ex[k]:+7.4f}")
    lnF = -(2 * f[i01] - f[i00] - f[i11])
    print(f"\nF_beta  MBAR  = {np.exp(lnF):.5f}")
    print(f"F_beta  exact = {F_ex:.5f}")
    print(f"relative error = {abs(np.exp(lnF)-F_ex)/F_ex:.2e}   [{time.time()-t0:.0f} s]")


def mbar_cov(u, Ncounts, f):
    """Asymptotic covariance Theta of the MBAR free energies (Shirts & Chodera,
    J. Chem. Phys. 129, 124105 (2008)), computed in the K x K SVD form.
    Var(sum_k c_k f_k) = c^T Theta c for any gauge-invariant c (sum_k c_k = 0)."""
    K, Ntot = u.shape
    lnN = np.log(Ncounts.astype(float))
    denom = np.logaddexp.reduce((lnN + f)[:, None] - u, axis=0)
    logW_ = (f[:, None] - u) - denom              # K x N
    W = np.exp(logW_).T                           # N x K  (columns sum to ~1)
    Nd = np.diag(Ncounts.astype(float))
    U_, sv, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T; S = np.diag(sv)
    inner = np.eye(K) - S @ V.T @ Nd @ V @ S
    return V @ S @ np.linalg.pinv(inner) @ S @ V.T

def tau_int(x, cmax=None):
    """Integrated autocorrelation time (in units of the sampling stride) with
    automatic windowing at the first negative autocorrelation."""
    x = np.asarray(x, float) - np.mean(x); n = x.size
    if n < 8 or np.allclose(x, 0): return np.nan
    c0 = np.dot(x, x) / n; tau = 0.5
    for t in range(1, min(n // 4, cmax or n // 4)):
        ct = np.dot(x[:-t], x[t:]) / (n - t) / c0
        if ct <= 0: break
        tau += ct
    return tau
