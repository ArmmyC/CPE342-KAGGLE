"""Task 3: 30-day spending regression. Metric: NMAE = 1 / (1 + sum|y-p| / sum y)."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from common import load, save_pred, nmae_score

train, test = load(3)
y = train['spending_30d'].values
feats = [c for c in train.columns if c not in ('player_id', 'spending_30d')]
X, Xt = train[feats], test[feats]

print('predict-all-zero NMAE:', round(nmae_score(y, np.zeros_like(y)), 4),
      '| predict median:', round(nmae_score(y, np.full_like(y, np.median(y))), 4))

base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=16)
kf = KFold(5, shuffle=True, random_state=42)
results = {}
for name, extra in [('l1', dict(objective='l1')), ('l2log', dict(objective='l2'))]:
    params = {**base, **extra}
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tr_i, va_i in kf.split(X):
        yt = np.log1p(y[tr_i]) if name == 'l2log' else y[tr_i]
        yv = np.log1p(y[va_i]) if name == 'l2log' else y[va_i]
        m = lgb.train(params, lgb.Dataset(X.iloc[tr_i], yt), 5000,
                      valid_sets=[lgb.Dataset(X.iloc[va_i], yv)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p, pt = m.predict(X.iloc[va_i]), m.predict(Xt)
        if name == 'l2log':
            p, pt = np.expm1(p), np.expm1(pt)
        oof[va_i] = np.clip(p, 0, None); tp += np.clip(pt, 0, None) / 5
    print(name, 'NMAE:', round(nmae_score(y, oof), 4))
    results[name] = (oof, tp)
    save_pred(3, name, tp, oof)
