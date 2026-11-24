"""Combine the five task outputs (taskN/output/taskN_pred.csv) into one Kaggle submission file.

Usage (from the repo root, after running the five task notebooks):
    python make_submission.py            -> submission.csv
"""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
sub = None
for k in range(1, 6):
    part = pd.read_csv(root / f'task{k}' / 'output' / f'task{k}_pred.csv')
    assert list(part.columns) == ['id', f'task{k}'], part.columns
    sub = part if sub is None else sub.merge(part, on='id', how='outer', validate='one_to_one')
assert len(sub) == 25889 and sub.notna().all().all()
sub = sub.sort_values('id').reset_index(drop=True)
sub.to_csv(root / 'submission.csv', index=False)
print(sub.head())
print('rows:', len(sub))
