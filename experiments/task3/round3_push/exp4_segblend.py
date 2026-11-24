"""Exp4: blend weights (simplex grid, step 0.1) chosen by nested CV, globally or per group
(segment 0..3 / missing, or whale vs rest). Components given on the command line as name=path pairs."""
import sys, itertools
import numpy as np
from common3 import *

X, Xt, y = load()
comps = {}
for a in sys.argv[1:]:
    n, pth = a.split('=', 1); comps[n] = np.clip(np.load(pth), 0, None)
names = list(comps); M = np.stack([comps[n] for n in names], 1)
F = folds(y, 42); fid = np.zeros(len(y), int)
for k, (_, va) in enumerate(F): fid[va] = k
step = 0.1 if len(names) <= 4 else 0.2
W = [np.array(w) * step for w in itertools.product(range(int(1 / step) + 1), repeat=len(names)) if sum(w) == int(1 / step)]
seg = X['segment'].fillna(-1).values
groups = {'global': np.zeros(len(y), int), 'segment': seg.astype(int) + 1, 'whale': (seg == 3).astype(int)}
b, _ = baseline()
for n in names: print(f'{n:12s} {nmae_score(y, comps[n]):.5f}')
for gname, g in groups.items():
    oof = np.zeros(len(y)); pick = []
    for k in range(5):
        inn, out = fid != k, fid == k
        for gv in np.unique(g):
            ri, ro = inn & (g == gv), out & (g == gv)
            errs = [np.abs(y[ri] - M[ri] @ w).sum() for w in W]
            w = W[int(np.argmin(errs))]; oof[ro] = M[ro] @ w; pick.append((k, gv, tuple(np.round(w, 1))))
    d, sd = paired_boot(y, b, oof, 500)
    print(f'{gname:8s} nested {nmae_score(y, oof):.5f}  vs baseline {d:+.5f} (boot sd {sd:.5f})')
    if gname != 'segment': print('   picks', sorted(set(p[1:] for p in pick)))
