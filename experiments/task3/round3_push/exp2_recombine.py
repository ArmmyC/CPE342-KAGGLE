"""Exp2 (optional 2nd arg: npz with oof_mu replacing the pieces mu): decision-theoretic recombination of the pieces: prediction = exact median of
P(0) * delta_0 + P(>0) * exp(mu + c * s * Z), with Z either normal or the empirical distribution of
standardised out-of-fold log residuals (taken from the other folds only). Parameters chosen by nested CV."""
import sys
import numpy as np
from scipy.stats import norm
from common3 import *

X, Xt, y = load()
tag = sys.argv[1] if len(sys.argv) > 1 else 'm42_f42'
FS = int(tag.split('_f')[1])
P = np.load(f'{OUT}/pieces_{tag}.npz'); p, mu, s = P['oof_p'], P['oof_mu'], P['oof_s']
MU = sys.argv[2] if len(sys.argv) > 2 else None
if MU: mu = np.load(MU)['oof_mu']; tag = tag + '_' + MU.split('/')[-1].replace('.npz', '')
F = folds(y, FS); fid = np.zeros(len(y), int)
for k, (_, va) in enumerate(F): fid[va] = k
sp = y > 0
ly = np.log(np.clip(y, 1e-3, None))
sc = np.clip(s, 0.05, 2)
z = (ly - mu) / sc


def predict(rows, zsrc, c, mode, pshift=0.0):
    pp = np.clip(p[rows] + pshift, 1e-6, 1)
    q = np.clip((pp - 0.5) / pp, 1e-4, 1 - 1e-4)
    if mode == 'normal':
        zq = norm.ppf(q)
    elif mode == 'emp':
        zq = np.quantile(zsrc, q)
    else:   # emp_raw: unstandardised residual quantiles, scale c only
        zq = np.quantile(zsrc, q) / sc[rows]
    return np.where(pp <= 0.5, 0.0, np.exp(mu[rows] + c * sc[rows] * zq))


GRID = [(mode, c, ps) for mode in ('normal', 'emp', 'emp_raw') for c in ([1.0] if mode == 'emp_raw' else [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]) for ps in (-0.03, -0.015, 0.0, 0.015, 0.03)]
allrows = np.arange(len(y))
# in-sample table (z from all spenders: mildly optimistic, only for orientation)
best = {}
for mode, c, ps in GRID:
    zs = z[sp] if mode != 'emp_raw' else (ly - mu)[sp]
    best[(mode, c, ps)] = nmae_score(y, predict(allrows, zs, c, mode, ps))
for k_ in sorted(best, key=best.get, reverse=True)[:8]: print('in-sample', k_, round(best[k_], 5))
print('lognorm original (normal, c=1, ps=0):', round(best[('normal', 1.0, 0.0)], 5))
# nested CV: for outer fold k, pick the grid point on the other folds (z source from those folds too)
oof = np.zeros(len(y)); picks = []
for k in range(5):
    inner = fid != k; outer = fid == k
    zs_in = {m: (z if m != 'emp_raw' else ly - mu)[sp & inner] for m in ('normal', 'emp', 'emp_raw')}
    sc_ = {g: nmae_score(y[inner], predict(np.where(inner)[0], zs_in[g[0]], g[1], g[0], g[2])) for g in GRID}
    g = max(sc_, key=sc_.get); picks.append(g)
    oof[outer] = predict(np.where(outer)[0], zs_in[g[0]], g[1], g[0], g[2])
print('nested picks', picks)
print(f'nested-CV recombined {tag}: {nmae_score(y, oof):.5f}')
np.save(f'{OUT}/exp2_oof_{tag}.npy', oof)
