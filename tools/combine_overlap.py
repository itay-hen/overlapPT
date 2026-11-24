#!/usr/bin/env python3
"""Combine the three overlap runs into R(tau) = Z_12 / sqrt(Z_11 Z_22).

Each run writes overlap_lnZ_<pair>.dat (pair tau lnZ stderr ...).  The runs are
independent, so their bootstrap errors add in quadrature:
    ln R = ln Z_12 - (ln Z_11 + ln Z_22)/2,
    err(ln R)^2 = e12^2 + (e11^2 + e22^2)/4.
R(tau) -> |<g1|g2>|^2 as tau -> infinity (unique ground states assumed).

Usage: python3 tools/combine_overlap.py DIR11 DIR12 DIR22 [--ed ED_JSON]
       (DIR* contain overlap_lnZ_11.dat, _12.dat, _22.dat respectively; ED_JSON is
        the output of tools/ed_overlap.py for a side-by-side comparison)
"""
import argparse
import json
import math
import os


def read(d, pair):
    with open(os.path.join(d, f'overlap_lnZ_{pair}.dat')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            t = line.split()
            return dict(pair=int(t[0]), tau=float(t[1]), lnZ=float(t[2]), err=float(t[3]))
    raise SystemExit(f'no data in {d}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('d11'); ap.add_argument('d12'); ap.add_argument('d22')
    ap.add_argument('--ed')
    a = ap.parse_args()
    r = {p: read(d, p) for p, d in (('11', a.d11), ('12', a.d12), ('22', a.d22))}
    taus = {v['tau'] for v in r.values()}
    if len(taus) != 1:
        raise SystemExit(f'runs have different tau: {taus}')
    tau = taus.pop()
    lnR = r['12']['lnZ'] - 0.5 * (r['11']['lnZ'] + r['22']['lnZ'])
    err = math.sqrt(r['12']['err'] ** 2 + 0.25 * (r['11']['err'] ** 2 + r['22']['err'] ** 2))
    R = math.exp(lnR)
    print(f'tau = {tau}')
    for p in ('11', '12', '22'):
        print(f"  ln Z_{p} = {r[p]['lnZ']:.6f} +- {r[p]['err']:.6f}")
    print(f'  ln R    = {lnR:.6f} +- {err:.6f}')
    print(f'  R       = {R:.6f} +- {R * err:.6f}')
    if a.ed:
        ed = json.load(open(a.ed))
        row = min(ed['tau'], key=lambda x: abs(x['tau'] - tau))
        if abs(row['tau'] - tau) > 1e-9:
            print(f"  (ED file has no tau={tau}; nearest is {row['tau']})")
        else:
            print('  ED at this tau:')
            for p in ('11', '12', '22'):
                ex = row[f'lnZ{p}']
                print(f"    ln Z_{p} exact = {ex:.6f}   MC - exact = {r[p]['lnZ'] - ex:+.6f}  "
                      f"({(r[p]['lnZ'] - ex) / r[p]['err']:+.2f} sigma)")
            print(f"    R exact      = {row['R']:.6f}   MC - exact = {R - row['R']:+.6f}  "
                  f"({(R - row['R']) / (R * err):+.2f} sigma)")
        print(f"  |<g1|g2>|^2 (tau -> inf, exact) = {ed['overlap_sq']:.6f}")


if __name__ == '__main__':
    main()
