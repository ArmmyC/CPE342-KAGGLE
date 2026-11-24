"""Task 5: local-scale shift z-scores. Within-row noise grows with the row's own level
(counts, amounts), so (p4 - mean(p1..3)) is divided by a robust noise scale estimated in
bins of that level. Saves the z matrix and summary counts."""
import os, re, numpy as np, pandas as pd
from common import load, PRED
_, df = load(5)
cols = [c for c in df.columns if c != 'player_id']
bases = sorted({re.sub(r'_[1-4]$', '', c) for c in cols if re.search(r'_[1-4]$', c)})
NB = 30

def local_z(m):
    mu = np.nanmean(m[:, :3], 1)
    d = m[:, 3] - mu
    ok = ~np.isnan(d)
    z = np.full(len(d), np.nan)
    if ok.sum() < 100: return z
    edges = np.unique(np.nanquantile(mu[ok], np.linspace(0, 1, NB + 1)))
    idx = np.clip(np.searchsorted(edges, mu, side='right') - 1, 0, len(edges) - 2)
    sig = np.full(len(edges) - 1, np.nan); ctr = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        sel = ok & (idx == k)
        if sel.sum() > 30:
            dd = d[sel]; sig[k] = np.median(np.abs(dd - np.median(dd))) * 1.4826
        ctr[k] = np.nanmedian(mu[sel]) if sel.sum() else np.nan
    g = ~np.isnan(sig) & (sig > 0) & ~np.isnan(ctr)
    if g.sum() < 3: return z
    s = np.interp(mu, ctr[g], sig[g])
    return d / np.maximum(s, 1e-9)

Z = pd.DataFrame({b: local_z(df[[f'{b}_{i}' for i in range(1, 5)]].values) for b in bases})
Z.to_pickle(os.path.join(PRED, 't5_localz.pkl'))
A = Z.abs()
print('local-z |z|>4 rate per feature (should be ~6e-5 if Gaussian):')
print(((A > 4).mean()).sort_values(ascending=False).round(4).head(20).to_dict())
for thr in (3.5, 4, 5):
    nb = (A > thr).sum(1)
    print(f'thr {thr}: rows with >=1,2,3,4,5 big features:', [(nb >= k).sum() for k in range(1, 6)])
