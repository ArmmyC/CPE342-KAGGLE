import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('CPE_DATA') or os.path.join(ROOT, 'data', 'cpe342-kaggle-main')  # CPE_DATA: e.g. data/v2_nov11/cpe342-kaggle-main
PRED = os.environ.get('CPE_PRED') or os.path.join(ROOT, 'preds')  # CPE_PRED keeps predictions for new data apart
os.makedirs(PRED, exist_ok=True)

TASK_DIRS = {
    1: '01_binary_classification',
    2: '02_multiclass_classification',
    3: '03_regression',
    5: '05_anomaly_detection',
}


def load(task):
    d = os.path.join(DATA, TASK_DIRS[task])
    test = pd.read_csv(os.path.join(d, 'test.csv'))
    train = pd.read_csv(os.path.join(d, 'train.csv')) if task != 5 else None
    return train, test


def save_pred(task, name, test_pred, oof=None):
    """Save test predictions (row-aligned with test.csv) and optional OOF preds."""
    np.save(os.path.join(PRED, f't{task}_{name}_test.npy'), np.asarray(test_pred))
    if oof is not None:
        np.save(os.path.join(PRED, f't{task}_{name}_oof.npy'), np.asarray(oof))


def nmae_score(y, p):
    return 1.0 / (1.0 + np.abs(y - p).sum() / y.sum())
