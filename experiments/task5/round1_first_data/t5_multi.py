"""Task 5: one unsupervised detector per anomaly type, then a union.

  shift : >=3 base features whose period-4 value jumps vs periods 1-3 (t5_shift.py)
  bot   : near-constant behaviour, i.e. very low IP-hash entropy + login-time variance;
          the score has a wide natural gap, so a fixed cut-off of 5 sd is used
  drift : Mahalanobis distance of the per-player 4-period means (catches mild,
          many-feature shifts that neither detector above sees)
Rows flagged by shift or bot come first; drift fills up to the target rate.
Saves preds/t5_multi_test.npy as a score where higher = more anomalous.
"""
import numpy as np
import pandas as pd
from common import load, save_pred, PRED

test = load(5)[1]
cols = [c for c in test.columns if c != 'player_id']
base = sorted({c.rsplit('_', 1)[0] for c in cols if c[-2:] in ('_1', '_2', '_3', '_4')})
M = pd.DataFrame({b: test[[f'{b}_{i}' for i in range(1, 5)]].mean(axis=1) for b in base})
M['behavior_stability'] = test['behavior_stability']
M['account_age_days'] = test['account_age_days']
M = M.fillna(M.median())
Z = (M - M.mean()) / M.std()

drift = np.einsum('ij,jk,ik->i', Z.values, np.linalg.inv(np.cov(Z.values.T)), Z.values)
shift = np.floor(np.load(f'{PRED}/t5_shift_nbig_test.npy')) >= 3
bot = -(Z['ip_hash_entropy'] + Z['login_time_variance']).values > 5

score = np.where(shift | bot, 1e6, 0) + drift
save_pred(5, 'multi', score)
print(f'shift {shift.sum()}  bot {bot.sum()}  union {(shift | bot).sum()}  '
      f'(10% target = {int(0.10 * len(test))} rows, rest filled by drift)')
