#!/usr/bin/env python3
"""Write the full and severed lattice Hamiltonians as PMR-QMC input files.

    H = J * sum_<ij> Z_i Z_j  -  Gamma * sum_<ij> X_i X_j        (open boundaries)

H1 (full) uses every nearest-neighbour bond of the Lx x Ly grid.  H2 (severed)
drops every bond crossing the given cuts.  A vertical cut at c removes the bonds
between columns c-1 and c (0-based); a horizontal cut at r removes the bonds
between rows r-1 and r.  For the quadrant severing of an L x L lattice use
--vcut L/2 --hcut L/2.

Qubits are numbered 1..Lx*Ly in the files, site (x,y) -> y*Lx + x + 1, matching
the "coeff i P j P" format read by prepare.cpp.

Example (3x3, one vertical cut between columns 1 and 2):
    python3 tools/make_model.py --Lx 3 --Ly 3 --J 1 --Gamma 0.8 --vcut 2 --out models/3x3_v2
"""
import argparse
import json
import os


def bonds(Lx, Ly):
    out = []
    for y in range(Ly):
        for x in range(Lx):
            s = y * Lx + x
            if x + 1 < Lx:
                out.append((s, s + 1, 'h', x + 1))   # crosses vertical line at column x+1
            if y + 1 < Ly:
                out.append((s, s + Lx, 'v', y + 1))  # crosses horizontal line at row y+1
    return out


def write_txt(path, J, Gamma, bond_list):
    with open(path, 'w') as f:
        for a, b, _, _ in bond_list:
            if J != 0:
                f.write(f"{J!r} {a+1} Z {b+1} Z\n")
        for a, b, _, _ in bond_list:
            if Gamma != 0:
                f.write(f"{-Gamma!r} {a+1} X {b+1} X\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--Lx', type=int, required=True)
    ap.add_argument('--Ly', type=int, required=True)
    ap.add_argument('--J', type=float, default=1.0)
    ap.add_argument('--Gamma', type=float, default=0.8)
    ap.add_argument('--vcut', type=int, action='append', default=[])
    ap.add_argument('--hcut', type=int, action='append', default=[])
    ap.add_argument('--out', required=True, help='output directory')
    a = ap.parse_args()
    if a.Gamma <= 0:
        raise SystemExit('Gamma must be > 0 (stoquastic, sign-free)')

    full = bonds(a.Lx, a.Ly)
    sev = [b for b in full
           if not ((b[2] == 'h' and b[3] in a.vcut) or (b[2] == 'v' and b[3] in a.hcut))]
    if len(sev) == len(full):
        raise SystemExit('the cuts remove no bonds')
    os.makedirs(a.out, exist_ok=True)
    write_txt(os.path.join(a.out, 'H1.txt'), a.J, a.Gamma, full)
    write_txt(os.path.join(a.out, 'H2.txt'), a.J, a.Gamma, sev)
    meta = dict(Lx=a.Lx, Ly=a.Ly, N=a.Lx * a.Ly, J=a.J, Gamma=a.Gamma, vcut=a.vcut, hcut=a.hcut,
                n_bonds_full=len(full), n_bonds_severed=len(sev))
    with open(os.path.join(a.out, 'model.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
