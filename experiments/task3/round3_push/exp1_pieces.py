"""Exp1: rebuild the struct pieces (P(spend>0), log-amount mu with log_hp offset, per-row spread) on the
production folds (StratifiedKFold seed 42 by spend bins), n_jobs=6, and save them for re-combination.
Usage: python exp1_pieces.py [model_seed] [fold_seed]"""
import sys, time
import numpy as np
import lightgbm as lgb
from common3 import *

MS = int(sys.argv[1]) if len(sys.argv) > 1 else 42
FS = int(sys.argv[2]) if len(sys.argv) > 2 else 42
X, Xt, y = load()
base = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, verbose=-1, n_jobs=NJ, seed=MS)


def fit(params, Xtr, ytr, Xva, yva, init_tr=None, init_va=None):
    return lgb.train({**base, **params}, lgb.Dataset(Xtr, ytr, init_score=init_tr), 5000,
                     valid_sets=[lgb.Dataset(Xva, yva, init_score=init_va)], callbacks=[lgb.early_stopping(150, verbose=False)])


ly = np.log(np.clip(y, 1e-3, None))
off, offt = np.nan_to_num(X['log_hp'].values), np.nan_to_num(Xt['log_hp'].values)
n, nt = len(y), len(Xt)
R = {k: np.zeros(n) for k in ['p', 'mu', 's']}; T = {k: np.zeros(nt) for k in ['p', 'mu', 's']}
t0 = time.time()
for k, (tr, va) in enumerate(folds(y, FS)):
    m = fit(dict(objective='binary'), X.iloc[tr], (y[tr] > 0).astype(int), X.iloc[va], (y[va] > 0).astype(int))
    R['p'][va] = m.predict(X.iloc[va]); T['p'] += m.predict(Xt) / 5
    ptr, pva = tr[y[tr] > 0], va[y[va] > 0]
    m = fit(dict(objective='l2'), X.iloc[ptr], ly[ptr], X.iloc[pva], ly[pva], off[ptr], off[pva])
    R['mu'][va] = m.predict(X.iloc[va]) + off[va]; T['mu'] += (m.predict(Xt) + offt) / 5
    res_tr = np.abs(ly[ptr] - (m.predict(X.iloc[ptr]) + off[ptr])); res_va = np.abs(ly[pva] - (m.predict(X.iloc[pva]) + off[pva]))
    ms = fit(dict(objective='l2', learning_rate=0.05, num_leaves=15), X.iloc[ptr], res_tr, X.iloc[pva], res_va)
    R['s'][va] = ms.predict(X.iloc[va]) * np.sqrt(np.pi / 2); T['s'] += ms.predict(Xt) * np.sqrt(np.pi / 2) / 5
    print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
np.savez(f'{OUT}/pieces_m{MS}_f{FS}.npz', **{f'oof_{k}': v for k, v in R.items()}, **{f'test_{k}': v for k, v in T.items()})
from scipy.stats import norm
q = np.clip((R['p'] - 0.5) / np.clip(R['p'], 1e-6, None), 1e-4, 1 - 1e-4)
oof = np.where(R['p'] <= 0.5, 0.0, np.exp(R['mu'] + np.clip(R['s'], 0.05, 2) * norm.ppf(q)))
print(f'lognorm rebuild m{MS} f{FS}: {nmae_score(y, oof):.5f}')
