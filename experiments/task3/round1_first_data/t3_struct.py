"""Task 3 built on the data's structure.

Found in the data (work/deep/t3_ident.txt, t3_target.txt):
- historical_spending = avg_transaction_value * total_transactions (exact, except tx = 0 rows)
- discount_rate_used = purchases_on_discount / total_transactions (exact)
  -> missing values of these five columns can be rebuilt from the others.
- Among spenders, log(spend) is close to linear in log(historical_spending), log(prev_month_spending),
  segment, vip_status, log(event_participation_rate), log(purchase_frequency), ... (R2 0.99, residual sd 0.22).

Model: P(spend > 0) classifier + model of log(spend) for spenders + spread of the log residual;
prediction = median of the zero / log-normal mixture (the MAE-optimal point).

Usage: python t3_struct.py [variant]   variant in {impute_hurdle, lognorm}
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from common import load, save_pred, nmae_score

SEED = 42
train, test = load(3)
train = train.dropna(subset=['spending_30d']).reset_index(drop=True)
y = train['spending_30d'].values
feats = [c for c in train.columns if c not in ('player_id', 'spending_30d')]
full = pd.concat([train[feats], test[feats]], ignore_index=True)


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
        atv = atv.fillna(h.where(tx == 0))          # tx = 0 rows store atv = hist
        h = h.fillna(atv.where(tx == 0))
        pod = pod.fillna((dr * tx).round())
        dr = dr.fillna(pod / tx.where(tx > 0))
    d['historical_spending'], d['avg_transaction_value'], d['total_transactions'] = h, atv, tx
    d['purchases_on_discount'], d['discount_rate_used'] = pod, dr
    return d


full = impute(full)
L = lambda s: np.log(s.clip(lower=1e-3))
full['log_hist'] = L(full['historical_spending'])
full['log_prev'] = L(full['prev_month_spending'])
full['log_hp'] = 0.63 * full['log_hist'] + 0.37 * full['log_prev']
# when one of hist / prev is missing, use the other (prev ~ 0.35 x hist)
full['log_hp'] = full['log_hp'].fillna(full['log_hist'] + 0.37 * np.log(0.35)).fillna(full['log_prev'] - 0.63 * np.log(0.35))
full['hist_x_prev'] = full['historical_spending'] + full['prev_month_spending']
full['tx_value_x_freq'] = full['avg_transaction_value'] * full['purchase_frequency']
X, Xt = full.iloc[:len(train)].reset_index(drop=True), full.iloc[len(train):].reset_index(drop=True)
base = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.6, verbose=-1, n_jobs=16, seed=SEED)


def fit(params, Xtr, ytr, Xva, yva, init_tr=None, init_va=None):
    dtr = lgb.Dataset(Xtr, ytr, init_score=init_tr); dva = lgb.Dataset(Xva, yva, init_score=init_va)
    return lgb.train({**base, **params}, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])


variant = sys.argv[1] if len(sys.argv) > 1 else 'lognorm'
strat = np.digitize(y, [0.01, 100, 1000, 10000, 50000])
oof_p = np.zeros(len(y)); test_p = np.zeros(len(Xt))
oof_mu = np.zeros(len(y)); test_mu = np.zeros(len(Xt)); oof_s = np.zeros(len(y)); test_s = np.zeros(len(Xt))
QS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
oof_q = np.zeros((len(y), len(QS))); test_q = np.zeros((len(Xt), len(QS)))
ly = np.log(np.clip(y, 1e-3, None))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat)):
    m = fit(dict(objective='binary'), X.iloc[tr], (y[tr] > 0).astype(int), X.iloc[va], (y[va] > 0).astype(int))
    oof_p[va] = m.predict(X.iloc[va]); test_p += m.predict(Xt) / 5
    ptr, pva = tr[y[tr] > 0], va[y[va] > 0]
    if variant == 'impute_hurdle':
        for j, qa in enumerate(QS):
            m = fit(dict(objective='quantile', alpha=qa), X.iloc[ptr], np.log1p(y[ptr]), X.iloc[pva], np.log1p(y[pva]))
            oof_q[va, j] = m.predict(X.iloc[va]); test_q[:, j] += m.predict(Xt) / 5
    else:
        # log-spend model with the log-linear offset as starting point (helps extrapolation for whales)
        off_tr, off_va = X['log_hp'].values[ptr], X['log_hp'].values[pva]
        m = fit(dict(objective='l2'), X.iloc[ptr], ly[ptr], X.iloc[pva], ly[pva], np.nan_to_num(off_tr), np.nan_to_num(off_va))
        oof_mu[va] = m.predict(X.iloc[va]) + np.nan_to_num(X['log_hp'].values[va])
        test_mu += (m.predict(Xt) + np.nan_to_num(Xt['log_hp'].values)) / 5
        # spread of the log residual, modelled per row (abs residual * sqrt(pi/2) = sd for a normal)
        res_tr = np.abs(ly[ptr] - (m.predict(X.iloc[ptr]) + np.nan_to_num(off_tr)))
        res_va = np.abs(ly[pva] - (m.predict(X.iloc[pva]) + np.nan_to_num(off_va)))
        ms = fit(dict(objective='l2', learning_rate=0.05, num_leaves=15), X.iloc[ptr], res_tr, X.iloc[pva], res_va)
        oof_s[va] = ms.predict(X.iloc[va]) * np.sqrt(np.pi / 2); test_s += ms.predict(Xt) * np.sqrt(np.pi / 2) / 5
    print(f'fold {k} done', flush=True)


def combine_q(p_pos, qpred):
    p0 = 1 - p_pos
    q = np.clip((0.5 - p0) / np.clip(p_pos, 1e-6, None), QS[0], QS[-1])
    logv = np.array([np.interp(q[i], QS, np.sort(qpred[i])) for i in range(len(q))])
    return np.where(p0 >= 0.5, 0.0, np.expm1(logv))


def combine_ln(p_pos, mu, s):
    q = np.clip((p_pos - 0.5) / np.clip(p_pos, 1e-6, None), 1e-4, 1 - 1e-4)
    return np.where(p_pos <= 0.5, 0.0, np.exp(mu + np.clip(s, 0.05, 2) * norm.ppf(q)))


if variant == 'impute_hurdle':
    oof, tp = combine_q(oof_p, oof_q), combine_q(test_p, test_q)
else:
    oof, tp = combine_ln(oof_p, oof_mu, oof_s), combine_ln(test_p, test_mu, test_s)
    sp = y > 0
    print(f'spender log residual sd {np.std(ly[sp] - oof_mu[sp]):.4f}, mean abs {np.mean(np.abs(ly[sp] - oof_mu[sp])):.4f}')
    for c in (0.8, 0.9, 1.0, 1.1, 1.2):
        print(f'  spread x{c}: NMAE score {nmae_score(y, combine_ln(oof_p, oof_mu, oof_s * c)):.5f}')
print(f'TASK 3 struct {variant} NMAE score {nmae_score(y, oof):.5f}')
save_pred(3, f'struct_{variant}', tp, oof)
