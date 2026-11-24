"""Task 1: anti-cheat binary classification. Metric: macro F2."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
from common import load, save_pred

train, test = load(1)
train = train.dropna(subset=['is_cheater']).reset_index(drop=True)
y = train['is_cheater'].astype(int).values
feats = [c for c in train.columns if c not in ('player_id', 'is_cheater')]
X, Xt = train[feats], test[feats]

params = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=16)
oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr_i, va_i in skf.split(X, y):
    m = lgb.train(params, lgb.Dataset(X.iloc[tr_i], y[tr_i]), 3000,
                  valid_sets=[lgb.Dataset(X.iloc[va_i], y[va_i])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va_i] = m.predict(X.iloc[va_i]); tp += m.predict(Xt) / 5
    print('best iter', m.best_iteration)

for avg in ('binary', 'macro'):
    ths = np.arange(0.05, 0.95, 0.01)
    sc = [fbeta_score(y, oof > t, beta=2, average=avg) for t in ths]
    print(avg, 'F2 @0.5 =', round(fbeta_score(y, oof > 0.5, beta=2, average=avg), 4),
          '| best', round(max(sc), 4), 'at th', round(ths[int(np.argmax(sc))], 2))
save_pred(1, 'lgb', tp, oof)
