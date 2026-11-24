"""Task 3 model components. Every function trains on the training folds only, predicts the held-out fold
(OOF) and averages the 5 fold models for the test set."""
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm

SEED = 42
N_JOBS = 6
BASE = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, verbose=-1)
QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def lgb_fit(params, Xtr, ytr, Xva, yva, init_tr=None, init_va=None, w_tr=None, w_va=None,
            rounds=5000, es=150, seed=SEED, n_jobs=N_JOBS):
    prm = {**BASE, 'n_jobs': n_jobs, 'seed': seed, **params}
    dtr = lgb.Dataset(Xtr, ytr, init_score=init_tr, weight=w_tr)
    dva = lgb.Dataset(Xva, yva, init_score=init_va, weight=w_va)
    return lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(es, verbose=False)])


def fit_hurdle_pieces(X, Xt, y, folds, seed=SEED):
    """P(spend > 0) classifier, L2 model of log(spend) for spenders starting from log_hp (init_score),
    and a spread model on |log residual| * sqrt(pi/2) (the sd of a normal). Returns OOF and test dicts."""
    ly = np.log(np.clip(y, 1e-3, None))
    off, offt = np.nan_to_num(X['log_hp'].values), np.nan_to_num(Xt['log_hp'].values)
    n, nt = len(y), len(Xt)
    R = {k: np.zeros(n) for k in ['p', 'mu', 's']}
    T = {k: np.zeros(nt) for k in ['p', 'mu', 's']}
    iters = []
    for tr, va in folds:
        m = lgb_fit(dict(objective='binary'), X.iloc[tr], (y[tr] > 0).astype(int), X.iloc[va], (y[va] > 0).astype(int), seed=seed)
        R['p'][va] = m.predict(X.iloc[va]); T['p'] += m.predict(Xt) / 5
        ptr, pva = tr[y[tr] > 0], va[y[va] > 0]
        mm = lgb_fit(dict(objective='l2'), X.iloc[ptr], ly[ptr], X.iloc[pva], ly[pva], off[ptr], off[pva], seed=seed)
        R['mu'][va] = mm.predict(X.iloc[va]) + off[va]; T['mu'] += (mm.predict(Xt) + offt) / 5
        res_tr = np.abs(ly[ptr] - (mm.predict(X.iloc[ptr]) + off[ptr]))
        res_va = np.abs(ly[pva] - (mm.predict(X.iloc[pva]) + off[pva]))
        ms = lgb_fit(dict(objective='l2', learning_rate=0.05, num_leaves=15), X.iloc[ptr], res_tr, X.iloc[pva], res_va, seed=seed)
        R['s'][va] = ms.predict(X.iloc[va]) * np.sqrt(np.pi / 2); T['s'] += ms.predict(Xt) * np.sqrt(np.pi / 2) / 5
        iters.append((m.best_iteration, mm.best_iteration, ms.best_iteration))
    return R, T, iters


def fit_quantile_models(X, Xt, y, folds, seed=SEED):
    """11 quantile LightGBM models of log1p(spend) trained on spenders only."""
    oof_q = np.zeros((len(y), len(QS))); test_q = np.zeros((len(Xt), len(QS)))
    for tr, va in folds:
        ptr, pva = tr[y[tr] > 0], va[y[va] > 0]
        for j, qa in enumerate(QS):
            m = lgb_fit(dict(objective='quantile', alpha=qa), X.iloc[ptr], np.log1p(y[ptr]), X.iloc[pva], np.log1p(y[pva]), seed=seed)
            oof_q[va, j] = m.predict(X.iloc[va]); test_q[:, j] += m.predict(Xt) / 5
    return oof_q, test_q


def combine_quantile_hurdle(p_pos, qpred):
    """Median of the mixture: zero with prob 1 - p, quantile curve of spenders with prob p."""
    p0 = 1 - p_pos
    q = np.clip((0.5 - p0) / np.clip(p_pos, 1e-6, None), QS[0], QS[-1])
    logv = np.array([np.interp(q[i], QS, np.sort(qpred[i])) for i in range(len(q))])
    return np.where(p0 >= 0.5, 0.0, np.expm1(logv))


def combine_lognormal(p_pos, mu, s):
    """Median of the zero / log-normal mixture with normal quantiles."""
    q = np.clip((p_pos - 0.5) / np.clip(p_pos, 1e-6, None), 1e-4, 1 - 1e-4)
    return np.where(p_pos <= 0.5, 0.0, np.exp(mu + np.clip(s, 0.05, 2) * norm.ppf(q)))


def design_fill_values(X):
    """Fill values for the linear design matrix, computed on train only."""
    return {'log_hp': float(X['log_hp'].median()), 'games_played': float(X['games_played'].median())}


def design(d, fill):
    """Design matrix for the log-linear amount model: log scale anchors, missing flags and segment x vip dummies."""
    L = lambda c, f: np.log(d[c].clip(lower=1e-3)).fillna(f)
    A = pd.DataFrame({'lhp': d['log_hp'].fillna(fill['log_hp']),
                      'lh_miss': d['historical_spending'].isna().astype(float),
                      'lp_miss': d['prev_month_spending'].isna().astype(float),
                      'le': L('event_participation_rate', np.log(0.5)), 'le_m': d['event_participation_rate'].isna().astype(float),
                      'lf': L('purchase_frequency', 0.0), 'lf_m': d['purchase_frequency'].isna().astype(float),
                      'gp': d['games_played'].fillna(fill['games_played']), 'prem': d['is_premium_member'].fillna(0.5),
                      'ld': np.log1p(d['days_since_last_purchase'].clip(lower=0)).fillna(3.0)})
    seg = d['segment'].fillna(-1).astype(int); vip = d['vip_status'].fillna(-1).astype(int)
    for s_ in range(-1, 4):
        for v_ in range(-1, 6):
            A[f'sv{s_}_{v_}'] = ((seg == s_) & (vip == v_)).astype(float)
    return A.values


def fit_linear_offset_amount(X, Xt, y, folds, D, Dt, seed=SEED):
    """Log-amount model for spenders: trimmed least squares on the design matrix (refit on |resid| < 0.6)
    gives the starting point, then an L2 LightGBM learns the correction."""
    ly = np.log(np.clip(y, 1e-3, None)); sp = y > 0
    mu = np.zeros(len(y)); mut = np.zeros(len(Xt)); iters = []
    for tr, va in folds:
        ptr, pva = tr[sp[tr]], va[sp[va]]
        b = np.linalg.lstsq(D[ptr], ly[ptr], rcond=None)[0]
        keep = np.abs(ly[ptr] - D[ptr] @ b) < 0.6
        b = np.linalg.lstsq(D[ptr][keep], ly[ptr][keep], rcond=None)[0]
        o_all, o_t = D @ b, Dt @ b
        m = lgb_fit(dict(objective='l2'), X.iloc[ptr], ly[ptr], X.iloc[pva], ly[pva], o_all[ptr], o_all[pva],
                    rounds=20000, es=200, seed=seed)
        mu[va] = m.predict(X.iloc[va]) + o_all[va]; mut += (m.predict(Xt) + o_t) / 5
        iters.append(m.best_iteration)
    return mu, mut, iters


def fit_ratio_l1(X, Xt, y, folds, seed=SEED):
    """Direct conditional median: L1 LightGBM on y / H with weight H, H = exp(log_hp),
    so the weighted L1 loss equals sum |y - H * r|."""
    fill = np.log(np.nanmedian(np.exp(X['log_hp'])))          # train-only fill for missing log_hp
    H = np.exp(np.nan_to_num(X['log_hp'].values, nan=fill)); Ht = np.exp(np.nan_to_num(Xt['log_hp'].values, nan=fill))
    tgt = y / H
    prm = dict(objective='l1', learning_rate=0.05, num_leaves=31)
    oof = np.zeros(len(y)); tp = np.zeros(len(Xt)); iters = []
    for tr, va in folds:
        m = lgb_fit(prm, X.iloc[tr], tgt[tr], X.iloc[va], tgt[va], w_tr=H[tr], w_va=H[va], rounds=20000, es=200, seed=seed)
        oof[va] = m.predict(X.iloc[va]) * H[va]; tp += m.predict(Xt) * Ht / 5
        iters.append(m.best_iteration)
    return np.clip(oof, 0, None), np.clip(tp, 0, None), iters


def lnmed(pp, mu, s, zsrc, c, ps):
    """Median of the zero / log-normal-like mixture. The standardized residual quantile is taken from the
    empirical distribution zsrc (train OOF spenders), the spread is scaled by c and P(spend > 0) shifted by ps."""
    pp = np.clip(pp + ps, 1e-6, 1)
    q = np.clip((pp - 0.5) / pp, 1e-4, 1 - 1e-4)
    return np.where(pp <= 0.5, 0.0, np.exp(mu + c * s * np.quantile(zsrc, q)))


C_GRID = (1.0, 1.1, 1.2, 1.3, 1.4)
SHIFT_GRID = (-0.03, -0.015, 0.0, 0.015)
CS_GRID = [(c, ps) for c in C_GRID for ps in SHIFT_GRID]
WEIGHT_GRID = [np.array(w) / 10 for w in itertools.product(range(11), repeat=3) if sum(w) == 10]


def fit_select(rows, y, sp, p, mu, s, z, hu, r1, whale):
    """Choose (c, shift) for the log-normal-median component and per-group blend weights [hu, lnL, r1],
    using only the OOF rows in `rows`."""
    zs = z[sp & rows]
    g = max(CS_GRID, key=lambda g_: -np.abs(y[rows] - lnmed(p[rows], mu[rows], s[rows], zs, *g_)).sum())
    M = np.stack([hu, lnmed(p, mu, s, zs, *g), r1], 1)
    ws = {}
    for gv in (0, 1):
        r = rows & (whale == gv)
        ws[gv] = WEIGHT_GRID[int(np.argmin([np.abs(y[r] - M[r] @ w).sum() for w in WEIGHT_GRID]))]
    return g, zs, ws, M
