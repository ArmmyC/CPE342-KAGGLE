"""Task 3 data helpers: metric, row-wise rebuild of the purchase columns, engineered features, CV folds."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = 'spending_30d'
SPEND_BINS = [0.01, 100, 1000, 10000, 50000]
PURCHASE_COLS = ['historical_spending', 'avg_transaction_value', 'total_transactions',
                 'purchases_on_discount', 'discount_rate_used']


def nmae_score(y, p):
    """Competition score 1 / (1 + NMAE), NMAE = sum|y - p| / sum(y)."""
    return 1.0 / (1.0 + np.abs(y - p).sum() / y.sum())


def impute(d):
    """Rebuild missing purchase columns from the exact identities found in the data:
    historical_spending = avg_transaction_value * total_transactions (tx = 0 rows store atv = hist),
    discount_rate_used = purchases_on_discount / total_transactions.
    Each row only uses its own columns, so this is a row-wise transform."""
    d = d.copy()
    h, atv, tx, pod, dr = (d[c] for c in PURCHASE_COLS)
    for _ in range(3):
        tx_new = (pod / dr.where(dr > 0)).round()
        tx = tx.fillna(tx_new.where(np.isfinite(tx_new)))
        tx_new = (h / atv.where(atv > 0)).round()
        tx = tx.fillna(tx_new.where(np.isfinite(tx_new)))
        h = h.fillna(atv.where(tx > 0) * tx)
        atv = atv.fillna(h / tx.where(tx > 0))
        atv = atv.fillna(h.where(tx == 0))
        h = h.fillna(atv.where(tx == 0))
        pod = pod.fillna((dr * tx).round())
        dr = dr.fillna(pod / tx.where(tx > 0))
    d['historical_spending'], d['avg_transaction_value'], d['total_transactions'] = h, atv, tx
    d['purchases_on_discount'], d['discount_rate_used'] = pod, dr
    return d


def add_features(d):
    """Row-wise engineered features (log scale of past spending and the log-linear anchor log_hp)."""
    d = d.copy()
    L = lambda s: np.log(s.clip(lower=1e-3))
    d['log_hist'] = L(d['historical_spending'])
    d['log_prev'] = L(d['prev_month_spending'])
    hp = 0.63 * d['log_hist'] + 0.37 * d['log_prev']
    # when one of hist / prev is missing, use the other (prev is about 0.35 x hist)
    d['log_hp'] = hp.fillna(d['log_hist'] + 0.37 * np.log(0.35)).fillna(d['log_prev'] - 0.63 * np.log(0.35))
    d['hist_x_prev'] = d['historical_spending'] + d['prev_month_spending']
    d['tx_value_x_freq'] = d['avg_transaction_value'] * d['purchase_frequency']
    return d


def build_features(frame, feats):
    """Full row-wise pipeline for one frame (train or test separately)."""
    return add_features(impute(frame[feats])).reset_index(drop=True)


def make_folds(y, seed=42):
    """StratifiedKFold(5) on spending bins, same split for every component."""
    strat = np.digitize(y, SPEND_BINS)
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat))
