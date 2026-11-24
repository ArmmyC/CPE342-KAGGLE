"""Task 5: second gap scan on engineered views (leave-one-period-out shifts, noise ratios,
linear-regression residuals between features) to find the remaining anomaly subpopulations."""
import os, re, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from common import load, PRED
_, t = load(5)
cols = [c for c in t.columns if c != 'player_id']
bases = sorted({re.sub(r'_[1-4]$', '', c) for c in cols if re.search(r'_[1-4]$', c)})
M = {b: t[[f'{b}_{i}' for i in range(1, 5)]].values for b in bases}
NB = 30

def local_scale(d, mu):
    ok = ~np.isnan(d) & ~np.isnan(mu)
    edges = np.unique(np.nanquantile(mu[ok], np.linspace(0, 1, NB + 1)))
    idx = np.clip(np.searchsorted(edges, mu, side='right') - 1, 0, len(edges) - 2)
    sig = np.full(len(edges) - 1, np.nan); ctr = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        s = ok & (idx == k)
        if s.sum() > 30:
            sig[k] = np.median(np.abs(d[s] - np.median(d[s]))) * 1.4826; ctr[k] = np.median(mu[s])
    g = ~np.isnan(sig) & (sig > 0)
    return np.interp(mu, ctr[g], sig[g]) if g.sum() > 2 else np.full(len(d), np.nan)

V = {}
for b in bases:
    m = M[b]
    for p in range(3):  # leave period p out: shift of period p vs the other three
        o = [i for i in range(4) if i != p]
        mu = np.nanmean(m[:, o], 1); d = m[:, p] - mu
        s = local_scale(d, mu)
        V[f'{b}|z{p+1}'] = d / s
    # within-row noise of periods 1-3 relative to typical at that level
    mu = np.nanmean(m, 1); sd = np.nanstd(m, 1, ddof=1)
    s = local_scale(sd - np.nanmedian(sd), mu)
    V[f'{b}|noise'] = (sd - np.nanmedian(sd)) / s
# regression residual of each row-mean feature on all other row-mean features
R = pd.DataFrame({b: np.nanmean(M[b], 1) for b in bases}); R['bs'] = t.behavior_stability.values; R['age'] = t.account_age_days.values
Rf = R.fillna(R.median()); Rz = (Rf - Rf.mean()) / Rf.std()
for c in Rz.columns:
    X = Rz.drop(columns=c).values; y = Rz[c].values
    pred = np.zeros(len(y)); idx = np.arange(len(y)); folds = np.arange(len(y)) % 5
    for f in range(5):
        r = Ridge(alpha=10).fit(X[folds != f], y[folds != f]); pred[folds == f] = r.predict(X[folds == f])
    V[f'{c}|resid'] = y - pred
out = []
for k, v in V.items():
    x = np.sort(v[np.isfinite(v)])
    if len(x) < 1000: continue
    lo = x[40:-40]
    sp = np.diff(x)
    # gap relative to the median spacing within the central 90%
    c = np.median(np.diff(x[int(.05 * len(x)):int(.95 * len(x))]))
    for side in ('hi', 'lo'):
        g = sp[::-1] if side == 'hi' else sp
        # largest gap with >=40 rows beyond it
        gg = g[40:-40]; i = gg.argmax() + 40
        n_out = i + 1
        out.append((gg.max() / (np.quantile(x, .99) - np.quantile(x, .01)), k, side, n_out))
out.sort(reverse=True)
print('gap/robust range, view, side, rows beyond the gap')
for r in out[:30]: print(f'{r[0]:.3f} {r[1]:34s} {r[2]} {r[3]}')
pd.DataFrame(V).to_pickle(os.path.join(PRED, 't5_views2.pkl'))
