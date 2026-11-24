"""Pack the per-fold outputs of train.py (preds_raw/) into one compact file per model (preds/<tag>.npz).

Each npz holds: fold (validation fold id per train row), oof / oofc / oofg (clean, corrupted-proxy and
geometric-proxy OOF probabilities, 31546 x 5) and test (5 folds x 25889 x 5 test probabilities).
Usage (inside task4):  python pack_preds.py el5 cl5 eb5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path('preds_raw'), Path('preds')
OUT.mkdir(exist_ok=True)
n = len(pd.read_csv('../data/04_image_classification/train.csv'))
for tag in sys.argv[1:]:
    fold = np.full(n, -1, np.int8)
    arrs = {k: np.full((n, 5), np.nan, np.float32) for k in ('oof', 'oofc', 'oofg')}
    test = []
    for k in range(5):
        i = np.load(RAW / f'{tag}_f{k}_idx.npy')
        fold[i] = k
        for name, a in arrs.items():
            a[i] = np.load(RAW / f'{tag}_f{k}_{name}.npy')[i]
        test.append(np.load(RAW / f'{tag}_f{k}_test.npy').astype(np.float32))
    pd.concat([pd.read_csv(p) for p in sorted(RAW.glob(f'{tag}_log_*.csv'))]).to_csv(OUT / f'{tag}_log.csv', index=False)
    np.savez_compressed(OUT / f'{tag}.npz', fold=fold, test=np.stack(test), **arrs)
    print(tag, 'packed')
