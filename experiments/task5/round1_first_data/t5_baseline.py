"""Task 5: unsupervised account anomaly detection. Metric: macro F3 (no labels available)."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from common import load, save_pred

_, df = load(5)
cols = [c for c in df.columns if c != 'player_id']
bases = sorted({re.sub(r'_[1-4]$', '', c) for c in cols if re.search(r'_[1-4]$', c)})
static = [c for c in cols if not re.search(r'_[1-4]$', c)]

feat = pd.DataFrame(index=df.index)
for b in bases:
    m = df[[f'{b}_{i}' for i in range(1, 5)]]
    feat[f'{b}_mean'] = m.mean(1)
    feat[f'{b}_std'] = m.std(1)
    feat[f'{b}_maxjump'] = m.diff(axis=1).abs().max(1)
    feat[f'{b}_last_vs_first'] = m.iloc[:, 3] - m.iloc[:, 0]
# geographic travel between consecutive periods
lat = df[[f'login_lat_{i}' for i in range(1, 5)]].values; lon = df[[f'login_lon_{i}' for i in range(1, 5)]].values
feat['geo_maxjump'] = np.nanmax(np.sqrt(np.diff(lat, axis=1) ** 2 + np.diff(lon, axis=1) ** 2), axis=1)
for c in static:
    feat[c] = df[c]
feat = feat.fillna(feat.median())

X = RobustScaler().fit_transform(feat)
X = np.clip(X, -20, 20)
iso = IsolationForest(n_estimators=500, max_samples=2048, random_state=0, n_jobs=16).fit(X)
score = -iso.score_samples(X)  # higher = more anomalous

print('score quantiles', np.quantile(score, [.5, .8, .9, .95, .99]).round(3))
hist, edges = np.histogram(score, bins=30)
for h, e in zip(hist, edges):
    print(f'{e:.3f} {h}')
for c in ['vpn_usage', 'password_changed', 'suspicious_login_time', 'mass_item_sale']:
    print(c, 'mean score flag=1:', score[df[c] == 1].mean().round(3), 'flag=0:', score[df[c] == 0].mean().round(3))
save_pred(5, 'iso', score)
