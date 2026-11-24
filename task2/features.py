"""Data loading and feature construction for Task 2 (player segment, 4 classes).

Every fitted quantity (category levels, percentile-rank references, one-hot levels, which columns get
missing flags) is learned from the training rows only. Test rows are only transformed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 'segment'
ID_COL = 'player_id'
TASK_DIR = '02_multiclass_classification'

SOCIAL_COLS = ['friend_count', 'chat_activity_score', 'friend_invites_sent', 'gifts_sent_received']
COMP_COLS = ['ranked_participation_rate', 'tournament_entries', 'competitive_rank']


def load_data(data_dir):
    d = Path(data_dir) / TASK_DIR
    train = pd.read_csv(d / 'train.csv')
    test = pd.read_csv(d / 'test.csv')
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    return train, test


def base_features(train):
    """All columns except the id, the target and the pure-noise random_metric_* columns."""
    return [c for c in train.columns if c not in (ID_COL, TARGET) and not c.startswith('random_metric')]


def text_columns(train, feats):
    return [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]


def _train_rank_sum(df, train, cols):
    """Sum of percentile ranks, where each rank is measured against the TRAIN distribution of that column."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        ref = np.sort(train[c].dropna().values.astype(float))
        v = df[c].values.astype(float)
        out[c] = np.where(np.isnan(v), np.nan, np.searchsorted(ref, v, side='right') / len(ref))
    return out.sum(1)


def gbdt_features(train, test):
    """Feature frames for LightGBM / XGBoost / CatBoost. Returns X, X_test, list of categorical columns."""
    feats = base_features(train)
    cats = text_columns(train, feats)
    frames = []
    for df in (train, test):
        X = df[feats].copy()
        X['n_missing'] = X.isna().sum(1)                       # row-wise
        X['spend_per_hour'] = X['total_spending_thb'] / (X['total_playtime_hours'] + 1)
        X['monthly_x_freq'] = X['avg_monthly_spending'] * X['spending_frequency']
        X['social_sum'] = _train_rank_sum(X, train, SOCIAL_COLS)
        X['comp_sum'] = _train_rank_sum(X, train, COMP_COLS)
        frames.append(X)
    X, X_test = frames
    for c in cats:
        dtype = X[c].astype('category').dtype                  # levels from train only
        X[c] = X[c].astype(dtype)
        X_test[c] = X_test[c].astype(dtype)                    # unseen test levels would become NaN
    return X, X_test, cats


def nn_matrices(train, test):
    """Raw numeric matrix and the fixed (non-fitted per fold) block for the MLP.

    NUM: numeric columns (train rows first, then test rows), still with NaN; the median fill and the
    QuantileTransformer are fitted per fold later. ONE: one-hot of text columns (levels from train, NaN
    gives an all-zero row), missing flags for numeric columns that have NaN in train, and n_missing / 10.
    """
    feats = base_features(train)
    cats = text_columns(train, feats)
    nums = [c for c in feats if c not in cats]
    levels = {c: np.array(sorted(train[c].dropna().unique()), dtype=object) for c in cats}
    miss_cols = [c for c in nums if train[c].isna().any()]

    def one_block(df):
        onehot = [(df[c].to_numpy(dtype=object)[:, None] == levels[c][None, :]).astype(np.float32) for c in cats]
        miss = df[miss_cols].isna().values.astype(np.float32)
        return np.hstack(onehot + [miss, miss.sum(1, keepdims=True) / 10])

    NUM = np.vstack([train[nums].values, test[nums].values]).astype(float)
    ONE = np.vstack([one_block(train), one_block(test)]).astype(np.float32)
    return NUM, ONE, nums, cats
