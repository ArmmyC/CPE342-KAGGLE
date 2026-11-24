"""Mix the neural net into the current best prediction of a task, choosing its weight by OOF score.

Usage: python nn_blend.py TASK BASE_NAME      e.g. python nn_blend.py 3 hurdleblend
Saves preds/t{task}_{BASE_NAME}nn_*.npy (Task 2 also gets refitted class weights *_w.npy).
"""
import sys
import numpy as np
from sklearn.metrics import fbeta_score, f1_score
from common import load, save_pred, nmae_score, PRED

TASK, BASE = int(sys.argv[1]), sys.argv[2]
TARGET = {1: 'is_cheater', 2: 'segment', 3: 'spending_30d'}[TASK]
train = load(TASK)[0].dropna(subset=[TARGET]).reset_index(drop=True)
y = train[TARGET].values if TASK == 3 else train[TARGET].astype(int).values
ld = lambda n, s: np.load(f'{PRED}/t{TASK}_{n}_{s}.npy')
base_oof, nn_oof = ld(BASE, 'oof'), ld('nn', 'oof')
if TASK == 3:
    base_oof = np.clip(base_oof, 0, None)
TH = np.arange(0.05, 0.95, 0.01)


def class_weights(p):
    w = np.ones(4)
    for _ in range(3):
        for k in range(4):
            cand = [(f1_score(y, (p * np.r_[w[:k], v, w[k + 1:]]).argmax(1), average='macro'), v)
                    for v in np.arange(0.5, 2.5, 0.05)]
            w[k] = max(cand)[1]
    return w


def score(p):
    if TASK == 1:
        s = [fbeta_score(y, p > t, beta=2) for t in TH]
        return max(s), TH[int(np.argmax(s))]
    if TASK == 2:
        return f1_score(y, p.argmax(1), average='macro'), None
    return nmae_score(y, p), None


res = []
for a in np.arange(0, 1.01, 0.1):
    s, th = score((1 - a) * base_oof + a * nn_oof)
    res.append((s, a, th))
    print(f'  nn weight {a:.1f}: {s:.4f}' + (f' (threshold {th:.2f})' if th is not None else ''))
best_s, a, th = max(res)
print(f'TASK {TASK}: base {BASE} {res[0][0]:.4f} -> best nn weight {a:.1f}: {best_s:.4f}')
oof = (1 - a) * base_oof + a * nn_oof
save_pred(TASK, f'{BASE}nn', (1 - a) * ld(BASE, 'test') + a * ld('nn', 'test'), oof)
if TASK == 2:
    w = class_weights(oof)
    print('class weights', w.round(2), f'macro F1 with weights {f1_score(y, (oof * w).argmax(1), average="macro"):.4f}')
    np.save(f'{PRED}/t2_{BASE}nn_w.npy', w)
