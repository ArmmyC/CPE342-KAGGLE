"""Task 1 box rule: every training cheater lies inside fixed ranges on six features
(account_age_days 15-1000, level 25-96, level_progression_speed 0.0044-6.26, friend_network_size 10-269,
reports_received 4-19, crosshair_placement 0.80-0.94). Rows outside the box are set to probability 0.
Applied to 't1_seeds'; saves 't1_seedsbox'. Ranges come from the training cheaters only."""
import numpy as np
import os
from sklearn.metrics import fbeta_score
from common import load, save_pred, PRED

BOX = ['account_age_days', 'level', 'level_progression_speed', 'friend_network_size', 'reports_received', 'crosshair_placement']
train, test = load(1)
lab = train.dropna(subset=['is_cheater']).reset_index(drop=True)
y = lab['is_cheater'].astype(int).values
ch = lab[y == 1]
lo, hi = ch[BOX].min(), ch[BOX].max()


def outside(d):
    o = np.zeros(len(d), bool)
    for c in BOX:
        o |= ((d[c] < lo[c]) | (d[c] > hi[c])).fillna(False).values
    return o


oof = np.load(os.path.join(PRED, 't1_seeds_oof.npy')); tp = np.load(os.path.join(PRED, 't1_seeds_test.npy'))
oo, ot = outside(lab), outside(test)
print(f'outside box: train {oo.mean():.4f} (cheaters outside: {oo[y == 1].sum()}), test {ot.mean():.4f}')
for t in (0.37, 0.38, 0.39):
    a = fbeta_score(y, oof > t, beta=2, average='macro'); b = fbeta_score(y, np.where(oo, 0, oof) > t, beta=2, average='macro')
    print(f'th {t}: seeds {a:.4f} -> with box {b:.4f}')
print(f'test rows changed at 0.37: {np.sum((tp > 0.37) & ot)}')
save_pred(1, 'seedsbox', np.where(ot, 0, tp), np.where(oo, 0, oof))
