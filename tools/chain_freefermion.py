#!/usr/bin/env python3
"""Exact sector-resolved ground-state overlaps for the open chain, by free fermions.

Model (paper Eq. XY_H on a chain): H = J sum_j Z_j Z_{j+1} - Gamma sum_j X_j X_{j+1}, open ends.
H1 = full chain of N sites; H2 = the same chain with bond (m-1, m) removed, m = N//2 by default
(0-based sites; the left block is sites 0..m-1).

Jordan-Wigner with a Y string, a_j = (prod_{k<j} Y_k) Z_j, b_j = (prod_{k<j} Y_k) X_j, makes
both Z_j Z_{j+1} (~ b_j a_{j+1}) and X_j X_{j+1} (~ a_j b_{j+1}) Majorana bilinears, so H is
quadratic. The sector selected by |+> (paper Sec. V.H) fixes prod X over each connected
component of the interaction graph; each such operator is, up to a phase, the parity of one
decoupled Majorana sub-chain, so every in-sector ground state is a Gaussian state.

All signs and phases are derived from explicit Pauli-string algebra, not by hand.

Outputs, for each N: F = |<g1|g2>|^2, -ln F, in-sector gaps Delta_1, Delta_2, the trial-state
weights p_a = |<+|g_a>|^2, and the sufficient projection length of paper Eq. (betareqlnF).

Usage: python3 chain_freefermion.py --N 16 32 64 [--J 1 --Gamma 0.8] [--validate 8 12 16 20]
"""
import argparse
import json
import numpy as np


# ---------------------------------------------------------------- Pauli strings
# (x, z, k) represents i^k * prod_j X_j^{x_j} Z_j^{z_j}   (X before Z on each site)
def pmul(p, q):
    x1, z1, k1 = p
    x2, z2, k2 = q
    return (x1 ^ x2, z1 ^ z2, (k1 + k2 + 2 * bin(z1 & x2).count('1')) % 4)


PX = lambda j: (1 << j, 0, 0)
PZ = lambda j: (0, 1 << j, 0)
PY = lambda j: (1 << j, 1 << j, 1)          # Y = i X Z
ID = (0, 0, 0)


def majoranas(N):
    """gamma[2j] = a_j, gamma[2j+1] = b_j as Pauli strings."""
    g = []
    S = ID
    for j in range(N):
        g.append(pmul(S, PZ(j)))
        g.append(pmul(S, PX(j)))
        S = pmul(S, PY(j))
    return g


def as_bilinear(term, coeff, g, k, l):
    """coeff * term == t * gamma_k gamma_l; return t (complex)."""
    P = pmul(g[k], g[l])
    assert (P[0], P[1]) == (term[0], term[1]), "term is not this bilinear"
    # P = i^kP * mask, term = i^kT * mask  ->  term = i^(kT-kP) P
    return coeff * (1j) ** ((term[2] - P[2]) % 4)


def build_A(N, J, G, removed=()):
    """H = (i/4) sum_kl A_kl gamma_k gamma_l, A real antisymmetric (2N x 2N)."""
    g = majoranas(N)
    A = np.zeros((2 * N, 2 * N))
    for j in range(N - 1):
        if j in removed:
            continue
        for term, coeff, k, l in (
            (pmul(PZ(j), PZ(j + 1)), J, 2 * j + 1, 2 * (j + 1)),         # Z Z ~ b_j a_{j+1}
            (pmul(PX(j), PX(j + 1)), -G, 2 * j, 2 * (j + 1) + 1),        # X X ~ a_j b_{j+1}
        ):
            if coeff == 0:
                continue
            t = as_bilinear(term, coeff, g, k, l)
            akl = -2j * t
            assert abs(akl.imag) < 1e-14
            A[k, l] += akl.real
            A[l, k] -= akl.real
    return A, g


def components(A):
    n = A.shape[0]
    seen = -np.ones(n, int)
    comps = []
    for s in range(n):
        if seen[s] >= 0:
            continue
        stack, c = [s], []
        seen[s] = len(comps)
        while stack:
            u = stack.pop()
            c.append(u)
            for v in np.nonzero(A[u])[0]:
                if seen[v] < 0:
                    seen[v] = len(comps)
                    stack.append(v)
        comps.append(sorted(c))
    return comps


# ---------------------------------------------------------------- Pfaffian
def pfaffian(M):
    """Pfaffian of a real antisymmetric matrix (Parlett-Reid with pivoting)."""
    A = np.array(M, dtype=float)
    n = A.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + np.argmax(np.abs(A[k + 1:, k]))
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            pf = -pf
        if A[k + 1, k] == 0.0:
            return 0.0
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            A[k + 2:, k + 2:] += np.outer(tau, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], tau)
    return pf


# ---------------------------------------------------------------- Gaussian states
def gs_cov(Ac):
    """Ground-state covariance of H_c = (i/4) g^T A_c g via the real Schur form A_c = Q T Q^T
    (robust for exponentially small splittings, where complex eigenvectors are ill defined).
    Returns M_c, the ascending quasiparticle energies, and the block data for flip_lowest."""
    from scipy.linalg import schur
    T, Q = schur(Ac, output='real')
    n = T.shape[0]
    blocks, single = [], []
    i = 0
    while i < n:
        if i + 1 < n and (abs(T[i, i + 1]) > 0 or abs(T[i + 1, i]) > 0):
            blocks.append((i, i + 1, T[i, i + 1]))
            i += 2
        else:
            single.append(i)
            i += 1
    assert len(single) % 2 == 0
    for a, b in zip(single[::2], single[1::2]):          # exact zero modes: pair arbitrarily
        blocks.append((a, b, 0.0))
    Mp = np.zeros_like(Ac)
    data = []
    for a, b, t in blocks:
        sgn = -np.sign(t) if t != 0 else 1.0
        Mp[a, b], Mp[b, a] = sgn, -sgn
        data.append([a, b, abs(t), sgn])
    data.sort(key=lambda d: d[2])
    M = Q @ Mp @ Q.T
    return M, np.array([d[2] for d in data]), (Q, data)


def flip_lowest(M, info):
    """Excite the lowest quasiparticle (flip the sign of its Schur block)."""
    Q, data = info
    a, b, _, sgn = data[0]
    qa, qb = Q[:, a], Q[:, b]
    return M - 2 * sgn * (np.outer(qa, qb) - np.outer(qb, qa))


def energy(A, M):
    return 0.25 * np.sum(A * M)


def parity_value(M, S, g, target_mask):
    """<target> where target = prod_{j in sites} X_j == c * prod_{s in S} gamma_s."""
    P = ID
    for s in S:
        P = pmul(P, g[s])
    assert (P[0], P[1]) == target_mask, "sector operator is not this Majorana product"
    c = (1j) ** ((-P[2]) % 4)                    # target = i^{-kP} prod gamma
    m = len(S) // 2
    val = c * (-1j) ** m * pfaffian(M[np.ix_(S, S)])
    assert abs(val.imag) < 1e-8 and abs(abs(val.real) - 1) < 1e-6, val
    return val.real


def sector_ground_state(N, A, g, blocks):
    """blocks: list of site lists, each a component whose prod X must be +1.
    Returns covariance M of the in-sector ground state, its energy, and the in-sector gap."""
    comps = components(A)
    M = np.zeros_like(A)
    E = 0.0
    excit = []      # per component: list of (energy above comp-GS, parity-flip?) low configs
    for c in comps:
        Ac = A[np.ix_(c, c)]
        Mc, eps, info = gs_cov(Ac)
        sites = sorted({s // 2 for s in c})
        blk = [b for b in blocks if set(sites) <= set(b)]
        # is this component the Majorana carrier of a constrained block's parity?
        constrained = None
        for b in blocks:
            mask = (sum(1 << j for j in b), 0)
            P = ID
            for s in c:
                P = pmul(P, g[s])
            if (P[0], P[1]) == mask:
                constrained = b
        Mfull = np.zeros_like(A)
        Mfull[np.ix_(c, c)] = Mc
        flipped = False
        if constrained is not None:
            par = parity_value(Mfull, c, g, (sum(1 << j for j in constrained), 0))
            if par < 0:
                Mc = flip_lowest(Mc, info)
                Mfull[np.ix_(c, c)] = Mc
                flipped = True
                assert parity_value(Mfull, c, g, (sum(1 << j for j in constrained), 0)) > 0
        M[np.ix_(c, c)] = Mc
        E += energy(Ac, Mc)
        # low-lying excitations within the sector, relative to this component's sector GS
        K = min(6, len(eps))
        confs = []
        for sub in range(1 << K):
            occ = [i for i in range(K) if sub >> i & 1]
            if constrained is not None and len(occ) % 2:
                continue                                  # would flip the fixed parity
            # energies measured from the unconstrained GS; sector GS may have mode 0 occupied
            e = sum(eps[i] for i in occ)
            confs.append(e)
        if flipped:
            # sector states are those with odd occupation; recompute relative to eps0
            confs = []
            for sub in range(1 << K):
                occ = [i for i in range(K) if sub >> i & 1]
                if len(occ) % 2 == 1:
                    confs.append(sum(eps[i] for i in occ))
        confs = np.sort(confs)
        excit.append(confs - confs[0])
    # in-sector gap: smallest nonzero excitation over components (others in their GS)
    gaps = [e[1] for e in excit if len(e) > 1]
    return M, E, min(gaps)


def overlap2(M1, M2):
    """|<psi1|psi2>|^2 for pure Gaussian states with Majorana covariances M1, M2."""
    sign, logdet = np.linalg.slogdet((M1 + M2) / 2)
    if sign == 0:
        return 0.0
    return float(np.exp(0.5 * logdet))


def plus_states(N, g):
    """|+> = (|G_+> + |G_->)/sqrt2, G_s Gaussian; returns their covariances."""
    out = []
    for s in (+1, -1):
        M = np.zeros((2 * N, 2 * N))
        for j in range(N - 1):
            k, l = 2 * j, 2 * (j + 1) + 1             # X_j X_{j+1} = c * i a_j b_{j+1}
            P = pmul(g[k], g[l])
            T = pmul(PX(j), PX(j + 1))
            # T = i^{kT-kP} gamma_k gamma_l = c * (i gamma_k gamma_l)  ->  c = i^{kT-kP-1}
            c = (1j) ** ((T[2] - P[2] - 1) % 4)
            assert abs(c.imag) < 1e-12
            M[k, l], M[l, k] = c.real, -c.real        # <i g_k g_l> = M_kl = +1 * c
        k, l = 1, 2 * (N - 1)                          # b_0, a_{N-1}
        M[k, l], M[l, k] = s, -s
        out.append(M)
    return out


def run(N, J, G, cut=None):
    m = N // 2 if cut is None else cut
    if m % 2 or (N - m) % 2:
        raise SystemExit("odd blocks: prod X of a block spans two Majorana sub-chains; "
                         "not handled (use N divisible by 4 with a midline cut)")
    A1, g = build_A(N, J, G)
    A2, _ = build_A(N, J, G, removed=(m - 1,))
    M1, E1, d1 = sector_ground_state(N, A1, g, [list(range(N))])
    M2, E2, d2 = sector_ground_state(N, A2, g, [list(range(m)), list(range(m, N))])
    F = overlap2(M1, M2)
    Gp, Gm = plus_states(N, g)
    p1 = 0.5 * (overlap2(Gp, M1) + overlap2(Gm, M1))
    p2 = 0.5 * (overlap2(Gp, M2) + overlap2(Gm, M2))
    # paper Eq. (betareqlnF), with epsilon = 0.01 nats
    eps = 0.01
    breq = max((np.log(1 / p) + np.log(1 / F) + 2 * np.log(1 / eps) + np.log(32)) / d
               for p, d in ((p1, d1), (p2, d2)))
    return dict(N=N, cut=m, J=J, Gamma=G, E1=E1, E2=E2, F=F, mlnF=-np.log(F),
                gap1=d1, gap2=d2, p1=p1, p2=p2, beta_sufficient_eps001=breq)


# ---------------------------------------------------------------- finite projection length
def evolve(A, M0, tau):
    """phi(tau) = e^{-tau H}|G>: returns the normalized covariance and ln <phi|phi>.
    Riccati flow dM/dtau = -(A + M A M), d ln n/dtau = -2 <H>; checked against ED."""
    from scipy.integrate import solve_ivp
    n = A.shape[0]

    def f(t, y):
        M = y[:-1].reshape(n, n)
        return np.concatenate([(-(A + M @ A @ M)).ravel(), [-0.5 * np.sum(A * M)]])
    sol = solve_ivp(f, (0, tau), np.concatenate([M0.ravel(), [0.0]]), method='DOP853',
                    rtol=1e-11, atol=1e-12)
    y = sol.y[:, -1]
    M = y[:-1].reshape(n, n)
    return 0.5 * (M - M.T), y[-1]


def ladder(N, J, G, beta, rungs=0, cut=None):
    """Exact ln O(lam_a, lam_b) = ln <+|e^{-beta H(lam_a)/2} e^{-beta H(lam_b)/2}|+> on the
    ladder (0,0) -> (0,1) -> (1,1), H(lam) = (1-lam) H1 + lam H2, plus ln F_beta.
    |+> = (|G_+> + |G_->)/sqrt2 with G_s of opposite, conserved prod Z, so O = (O_+ + O_-)/2."""
    m = N // 2 if cut is None else cut
    A1, g = build_A(N, J, G)
    A2, _ = build_A(N, J, G, removed=(m - 1,))
    tau = beta / 2
    K = max(rungs, 3)
    half = (K - 1) // 2
    lams = sorted({k / half for k in range(half + 1)})
    pairs = [(0.0, l) for l in lams] + [(l, 1.0) for l in lams[1:]]
    evo = {}
    for s, M0 in enumerate(plus_states(N, g)):
        for l in lams:
            evo[s, l] = evolve((1 - l) * A1 + l * A2, M0, tau)
    lnO = []
    for la, lb in pairs:
        terms = []
        for s in (0, 1):
            Ma, na = evo[s, la]
            Mb, nb = evo[s, lb]
            terms.append(0.5 * (na + nb + np.log(overlap2(Ma, Mb))))
        lnO.append(np.logaddexp(*terms) - np.log(2))
    lnO = np.array(lnO)
    lnF = 2 * lnO[half] - lnO[0] - lnO[-1]
    return dict(N=N, beta=beta, pairs=pairs, lnO=lnO.tolist(),
                f_gauge0=(lnO[0] - lnO).tolist(), lnF=float(lnF), F=float(np.exp(lnF)))


# ---------------------------------------------------------------- ED validation
def ed(N, J, G, cut=None):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    m = N // 2 if cut is None else cut
    dim = 1 << N
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(N)) & 1
    zs = 1 - 2 * bits

    def H(removed):
        diag = np.zeros(dim)
        rows, cols = [], []
        for j in range(N - 1):
            if j in removed:
                continue
            diag += J * zs[:, j] * zs[:, j + 1]
            rows.append(idx)
            cols.append(idx ^ (3 << j))
        off = sp.csr_matrix((np.full(dim * len(rows), -G), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(dim, dim))
        return sp.diags(diag) + off

    def flip(mask):
        return sp.csr_matrix((np.ones(dim), (idx, idx ^ mask)), shape=(dim, dim))

    full = (1 << N) - 1
    left = (1 << m) - 1
    pen = 4 * N * (abs(J) + abs(G))
    I = sp.identity(dim)
    H1 = H(()) + pen * (I - flip(full)) / 2
    H2 = H((m - 1,)) + pen * (I - flip(left)) / 2 + pen * (I - flip(full ^ left)) / 2
    out = {}
    vecs = []
    for name, Hm in (('1', H1), ('2', H2)):
        w, v = eigsh(Hm.tocsr(), k=3, which='SA', tol=1e-12)
        o = np.argsort(w)
        w, v = w[o], v[:, o]
        out['E' + name] = w[0]
        out['gap' + name] = w[1] - w[0]
        out['p' + name] = (v[:, 0].sum() ** 2) / dim
        vecs.append(v[:, 0])
    out['F'] = float(np.dot(vecs[0], vecs[1]) ** 2)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, nargs='*', default=[])
    ap.add_argument('--J', type=float, default=1.0)
    ap.add_argument('--Gamma', type=float, default=0.8)
    ap.add_argument('--validate', type=int, nargs='*', default=[])
    ap.add_argument('--beta', type=float, nargs='*', default=[])
    ap.add_argument('--rungs', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args()
    for N in a.validate:
        ff = run(N, a.J, a.Gamma)
        ex = ed(N, a.J, a.Gamma)
        print(f"N={N:3d} validate:", '  '.join(
            f"{k}: ff={ff[k]:.10g} ed={ex[k]:.10g} d={ff[k]-ex[k]:+.1e}" for k in ('E1', 'E2', 'gap1', 'gap2', 'p1', 'p2', 'F')))
    res = [run(N, a.J, a.Gamma) for N in a.N]
    for r in res:
        print(f"N={r['N']:4d}  F={r['F']:.10g}  -lnF={r['mlnF']:.6f}  gaps {r['gap1']:.4f} {r['gap2']:.4f}"
              f"  p1={r['p1']:.4e} p2={r['p2']:.4e}  beta_suff(eps=0.01)={r['beta_sufficient_eps001']:.2f}")
    for N in a.N:
        for b in a.beta:
            L = ladder(N, a.J, a.Gamma, b, a.rungs)
            res.append(L)
            print(f"N={N:4d} beta={b:6.2f}  F_beta={L['F']:.10g}  lnF_beta={L['lnF']:.8f}")
            if a.rungs:
                print('   f_k (gauge f_0=0):', ' '.join(f"{x:.6f}" for x in L['f_gauge0']))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=float)
