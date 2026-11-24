"""Exp3: direct L1 (conditional-median) GBDT.
 ratio: target y / H with weight H, H = exp(log_hp) (the log-linear scale), so the weighted L1 equals sum|y - H*r|.
 raw:   plain L1 on y.
Usage: python exp3_direct_l1.py variant [model_seed] [fold_seed]"""
import sys, time
import numpy as np
import lightgbm as lgb
from common3 import *

V = sys.argv[1]; MS = int(sys.argv[2]) if len(sys.argv) > 2 else 42; FS = int(sys.argv[3]) if len(sys.argv) > 3 else 42
X, Xt, y = load()
prm = dict(objective='l1', learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.6, verbose=-1, n_jobs=NJ, seed=MS)
H = np.exp(np.nan_to_num(X['log_hp'].values, nan=np.log(np.nanmedian(np.exp(X['log_hp'])))))
Ht = np.exp(np.nan_to_num(Xt['log_hp'].values, nan=np.log(np.nanmedian(np.exp(X['log_hp'])))))
if V == 'ratio':
    tgt, w, mul, mult = y / H, H, H, Ht
else:
    tgt, w, mul, mult = y, None, np.ones(len(y)), np.ones(len(Xt))
oof = np.zeros(len(y)); tp = np.zeros(len(Xt)); t0 = time.time()
for k, (tr, va) in enumerate(folds(y, FS)):
    m = lgb.train(prm, lgb.Dataset(X.iloc[tr], tgt[tr], weight=None if w is None else w[tr]), 20000,
                  valid_sets=[lgb.Dataset(X.iloc[va], tgt[va], weight=None if w is None else w[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict(X.iloc[va]) * mul[va]; tp += m.predict(Xt) * mult / 5
    print(f'fold {k} it {m.best_iteration} {time.time()-t0:.0f}s', flush=True)
oof, tp = np.clip(oof, 0, None), np.clip(tp, 0, None)
print(f'direct L1 {V} m{MS} f{FS}: {nmae_score(y, oof):.5f}')
np.save(f'{OUT}/exp3_{V}_m{MS}_f{FS}_oof.npy', oof); np.save(f'{OUT}/exp3_{V}_m{MS}_f{FS}_test.npy', tp)
