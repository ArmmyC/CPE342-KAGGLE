"""Task 5: scan every raw column and per-player aggregate for a hard gap in its
distribution that separates a minority group (generator-style subpopulations)."""
import os, re, numpy as np, pandas as pd
from common import load, PRED
_, t = load(5)
cols = [c for c in t.columns if c != 'player_id']
bases = sorted({re.sub(r'_[1-4]$', '', c) for c in cols if re.search(r'_[1-4]$', c)})
static = [c for c in cols if not re.search(r'_[1-4]$', c)]
views = {c: t[c].values for c in cols}
for b in bases:
    m = t[[f'{b}_{i}' for i in range(1, 5)]].values
    views[b + '|mean'] = np.nanmean(m, 1)
    views[b + '|std'] = np.nanstd(m, 1)
    views[b + '|min'] = np.nanmin(m, 1); views[b + '|max'] = np.nanmax(m, 1)
out = []
for k, v in views.items():
    x = np.sort(v[~np.isnan(v)])
    if len(x) < 1000 or x[0] == x[-1]: continue
    rng = np.quantile(x, .999) - np.quantile(x, .001)
    gaps = np.diff(x); 
    # candidate gaps with at least 40 rows on the smaller side
    idx = np.arange(len(gaps)); ok = (idx >= 40) & (idx <= len(x) - 41)
    g = np.where(ok, gaps, 0); i = g.argmax()
    # relative to local density: expected gap if uniform around the gap = rng/len
    out.append((g[i] / rng, k, i + 1, len(x) - i - 1, x[i], x[i + 1]))
out.sort(reverse=True)
print('largest gaps (gap/robust range), column, rows below, rows above, gap from-to')
for r in out[:25]: print(f'{r[0]:.4f} {r[1]:32s} {r[2]:6d} {r[3]:6d}  {r[4]:.4g} -> {r[5]:.4g}')
