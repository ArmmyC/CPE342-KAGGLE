"""Cross-validated models for Task 2: LightGBM, XGBoost, CatBoost and a PyTorch MLP.

All models use the same StratifiedKFold(5, shuffle=True, random_state=42) folds. Each fold's model
predicts its validation rows (out-of-fold, OOF) and the test rows; the test prediction is the mean of
the 5 fold models.
"""
import time

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

N_CLASSES = 4
SEED = 42
N_THREADS = 8          # LightGBM / CatBoost threads (8 gives the same models as the submitted run)
XGB_THREADS = 16       # XGBoost 'hist' results change slightly with nthread; 16 reproduces the submission
MAX_ROUNDS = 10000     # upper bound; early stopping decides the real number of trees
NN_MAX_EPOCHS = 60


def make_folds(y, seed=SEED):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ----------------------------------------------------------------------------- GBDTs
def fit_lgb(Xtr, ytr, Xva, yva, Xte, cats):
    p = dict(objective='multiclass', num_class=N_CLASSES, learning_rate=0.03, num_leaves=63,
             min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
             verbose=-1, n_jobs=N_THREADS, seed=SEED)
    m = lgb.train(p, lgb.Dataset(Xtr, ytr), MAX_ROUNDS, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva), m.predict(Xte), m.best_iteration


def fit_xgb(Xtr, ytr, Xva, yva, Xte, cats):
    p = dict(objective='multi:softprob', num_class=N_CLASSES, eta=0.03, max_depth=7, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.6, tree_method='hist', max_cat_to_onehot=8,
             nthread=XGB_THREADS, seed=SEED)
    dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
    dva = xgb.DMatrix(Xva, yva, enable_categorical=True)
    m = xgb.train(p, dtr, MAX_ROUNDS, evals=[(dva, 'va')], early_stopping_rounds=200, verbose_eval=False)
    it = (0, m.best_iteration + 1)
    pte = m.predict(xgb.DMatrix(Xte, enable_categorical=True), iteration_range=it)
    return m.predict(dva, iteration_range=it), pte, m.best_iteration


def fit_cat(Xtr, ytr, Xva, yva, Xte, cats):
    cat_idx = [Xtr.columns.get_loc(c) for c in cats]

    def conv(d):  # CatBoost wants strings for categorical columns; NaN becomes its own level 'NA'
        return d.assign(**{c: d[c].astype(object).fillna('NA').astype(str) for c in cats})

    m = CatBoostClassifier(loss_function='MultiClass', iterations=MAX_ROUNDS, learning_rate=0.05, depth=7,
                           random_seed=SEED, thread_count=N_THREADS, od_type='Iter', od_wait=200,
                           verbose=False, allow_writing_files=False)
    m.fit(conv(Xtr), ytr, eval_set=(conv(Xva), yva), cat_features=cat_idx)
    return m.predict_proba(conv(Xva)), m.predict_proba(conv(Xte)), m.best_iteration_


GBDT_FITTERS = {'lgb': fit_lgb, 'xgb': fit_xgb, 'cat': fit_cat}


def run_gbdt_cv(name, X, y, X_test, cats, folds):
    """Returns (oof, test, info) where info holds the best iteration per fold and the run time."""
    fit = GBDT_FITTERS[name]
    oof = np.zeros((len(X), N_CLASSES))
    test = np.zeros((len(X_test), N_CLASSES))
    iters, t0 = [], time.time()
    for tr_i, va_i in folds:
        pv, pt, it = fit(X.iloc[tr_i], y[tr_i], X.iloc[va_i], y[va_i], X_test, cats)
        oof[va_i] = pv
        test += pt / len(folds)
        iters.append(int(it))
    return oof, test, {'iters': iters, 'minutes': (time.time() - t0) / 60}


# ----------------------------------------------------------------------------- MLP
def _fold_inputs(NUM, ONE, tr_idx):
    """Median fill and quantile-to-normal transform fitted on the training fold, applied to all rows."""
    med = np.nanmedian(NUM[tr_idx], 0)
    num = np.where(np.isnan(NUM), med, NUM)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=200000, random_state=0)
    qt.fit(num[tr_idx])
    return torch.tensor(np.hstack([qt.transform(num), ONE]), dtype=torch.float32)


def _mlp(d):
    layers, w = [], d
    for h in (512, 256, 128):
        layers += [nn.Linear(w, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.25)]
        w = h
    return nn.Sequential(*layers, nn.Linear(w, N_CLASSES))


def _fit_mlp(X, y_t, tr, va, n_tr, seed):
    torch.manual_seed(seed)
    model = _mlp(X.shape[1])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)
    best, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(NN_MAX_EPOCHS):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(512):
            idx = tr[b.numpy()]
            loss = loss_fn(model(X[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X[va]), y_t[va]).item()
        sched.step(vl)
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 6:          # early stopping on validation loss
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X[va]).softmax(1).numpy(), model(X[n_tr:]).softmax(1).numpy(), ep + 1


def run_nn_cv(NUM, ONE, y, folds, n_seeds=3, threads=N_THREADS):
    """MLP averaged over n_seeds per fold (seed = 100 * fold + s). Returns (oof, test, info)."""
    torch.set_num_threads(threads)
    n_tr = len(y)
    y_t = torch.tensor(y, dtype=torch.long)
    oof = np.zeros((n_tr, N_CLASSES))
    test = np.zeros((len(NUM) - n_tr, N_CLASSES))
    epochs, t0 = [], time.time()
    for k, (tr, va) in enumerate(folds):
        X = _fold_inputs(NUM, ONE, tr)
        for s in range(n_seeds):
            pv, pt, ep = _fit_mlp(X, y_t, tr, va, n_tr, seed=100 * k + s)
            oof[va] += pv / n_seeds
            test += pt / n_seeds / len(folds)
            epochs.append(ep)
    return oof, test, {'epochs': epochs, 'minutes': (time.time() - t0) / 60}
