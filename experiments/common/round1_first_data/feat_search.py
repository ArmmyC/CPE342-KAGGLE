"""LLM-in-the-loop feature search: candidate feature groups are written by the LLM in
feat_ideas.py (one function per idea, with the reasoning in its docstring); this script
scores each group by 5-fold LightGBM CV against the same model without it.

Usage: python feat_search.py TASK [group ...]
Results are appended to work/feat_search_log.csv so later rounds can build on them.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, f1_score
from common import load, nmae_score, ROOT
import feat_ideas

TASK = int(sys.argv[1])
TARGET = {1: 'is_cheater', 2: 'segment', 3: 'spending_30d'}[TASK]
train, test = load(TASK)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = train[TARGET].values if TASK == 3 else train[TARGET].astype(int).values
raw = train[[c for c in train.columns if c not in ('player_id', TARGET) and not c.startswith('random_metric')]]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(
    raw, y if TASK != 3 else np.digitize(y, [0.01, 100, 1000, 10000, 50000])))
obj = {1: dict(objective='binary'), 2: dict(objective='multiclass', num_class=4), 3: dict(objective='l1')}[TASK]
params = dict(**obj, learning_rate=0.06, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.6, verbose=-1, n_jobs=16, seed=42)


def score(X):
    oof = np.zeros((len(y), 4)) if TASK == 2 else np.zeros(len(y))
    for tr, va in folds:
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), 5000, valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va])
    if TASK == 1:
        return max(fbeta_score(y, oof > t, beta=2) for t in np.arange(0.1, 0.4, 0.01))
    if TASK == 2:
        return f1_score(y, oof.argmax(1), average='macro')
    return nmae_score(y, np.clip(oof, 0, None))


def prep(df):
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype('category')
    return df


groups = sys.argv[2:] or [n for n in dir(feat_ideas) if n.startswith(f't{TASK}_')]
base = score(prep(raw)); print(f'TASK {TASK} baseline {base:.4f}', flush=True)
rows = []
for g in groups:
    t0 = time.time()
    extra = getattr(feat_ideas, g)(raw)
    s = score(prep(pd.concat([raw, extra], axis=1)))
    print(f'  {g:34s} {s:.4f}  delta {s - base:+.4f}  ({extra.shape[1]} cols, {time.time() - t0:.0f}s)', flush=True)
    rows.append(dict(task=TASK, group=g, n_cols=extra.shape[1], score=s, baseline=base, delta=s - base))
log = os.path.join(ROOT, 'work', 'feat_search_log.csv')
pd.DataFrame(rows).to_csv(log, mode='a', header=not os.path.exists(log), index=False)
