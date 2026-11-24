"""Shared loading / imputation / folds for push/t3 experiments (copied logic from MLKaggleChallenge/src/t3_struct.py)."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

D = r'<project_root>/MLTest/cpe342-kaggle-main/03_regression'
PRED = r'<project_root>/MLKaggleChallenge/preds'
OUT = os.path.dirname(os.path.abspath(__file__))
NJ = 6


def nmae_score(y, p):
    return 1.0 / (1.0 + np.abs(y - p).sum() / y.sum())


def impute(d):
    d = d.copy()
    h, atv, tx, pod, dr = (d[c] for c in ['historical_spending', 'avg_transaction_value', 'total_transactions',
                                          'purchases_on_discount', 'discount_rate_used'])
    for _ in range(3):
        tx_new = (pod / dr.where(dr > 0)).round()
        tx = tx.fillna(tx_new.where(np.isfinite(tx_new)))
        tx_new = (h / atv.where(atv > 0)).round()
        tx = tx.fillna(tx_new.where(np.isfinite(tx_new)))
        h = h.fillna(atv.where(tx > 0) * tx)
        atv = atv.fillna((h / tx.where(tx > 0)))
        atv = atv.fillna(h.where(tx == 0))
        h = h.fillna(atv.where(tx == 0))
        pod = pod.fillna((dr * tx).round())
        dr = dr.fillna(pod / tx.where(tx > 0))
    d['historical_spending'], d['avg_transaction_value'], d['total_transactions'] = h, atv, tx
    d['purchases_on_discount'], d['discount_rate_used'] = pod, dr
    return d


def load():
    tr = pd.read_csv(os.path.join(D, 'train.csv')); te = pd.read_csv(os.path.join(D, 'test.csv'))
    y = tr['spending_30d'].values
    feats = [c for c in tr.columns if c not in ('player_id', 'spending_30d')]
    # row-wise deterministic rebuild only (each row uses only its own columns)
    X = impute(tr[feats]); Xt = impute(te[feats])
    for d in (X, Xt):
        L = lambda s: np.log(s.clip(lower=1e-3))
        d['log_hist'] = L(d['historical_spending']); d['log_prev'] = L(d['prev_month_spending'])
        hp = 0.63 * d['log_hist'] + 0.37 * d['log_prev']
        d['log_hp'] = hp.fillna(d['log_hist'] + 0.37 * np.log(0.35)).fillna(d['log_prev'] - 0.63 * np.log(0.35))
        d['hist_x_prev'] = d['historical_spending'] + d['prev_month_spending']
        d['tx_value_x_freq'] = d['avg_transaction_value'] * d['purchase_frequency']
    return X.reset_index(drop=True), Xt.reset_index(drop=True), y


def folds(y, seed=42):
    strat = np.digitize(y, [0.01, 100, 1000, 10000, 50000])
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat))


def baseline():
    return np.clip(np.load(os.path.join(PRED, 't3_struct3_oof.npy')), 0, None), np.clip(np.load(os.path.join(PRED, 't3_struct3_test.npy')), 0, None)


def paired_boot(y, a, b, n=2000, seed=0):
    """score(b) - score(a) and bootstrap sd of that difference."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(y - a), np.abs(y - b)
    d0 = nmae_score(y, b) - nmae_score(y, a)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        s = y[i].sum()
        ds.append(1 / (1 + eb[i].sum() / s) - 1 / (1 + ea[i].sum() / s))
    return d0, float(np.std(ds))
