"""Task 5: per-player behaviour-shift score (period 4 vs own periods 1-3) + Isolation Forest blend."""
import re
import numpy as np
import pandas as pd
from common import load, save_pred, PRED
import os

_, df = load(5)
cols = [c for c in df.columns if c != 'player_id']
bases = sorted({re.sub(r'_[1-4]$', '', c) for c in cols if re.search(r'_[1-4]$', c)})
static = [c for c in cols if not re.search(r'_[1-4]$', c)]


def robust_z(v):
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if mad == 0:
        mad = np.nanstd(v) + 1e-9
    return (v - med) / mad


Z = {}
for b in bases:
    m = df[[f'{b}_{i}' for i in range(1, 5)]].values
    base = np.nanmean(m[:, :3], axis=1)
    Z[b] = robust_z(m[:, 3] - base)
# geographic jump in period 4 vs centre of periods 1-3
lat = df[[f'login_lat_{i}' for i in range(1, 5)]].values; lon = df[[f'login_lon_{i}' for i in range(1, 5)]].values
geo = np.sqrt((lat[:, 3] - np.nanmean(lat[:, :3], 1)) ** 2 + (lon[:, 3] - np.nanmean(lon[:, :3], 1)) ** 2)
Z['geo4'] = robust_z(geo)
Z = pd.DataFrame(Z).fillna(0)
A = Z.abs().clip(upper=30)

score_max = A.max(1)
score_ss = np.sqrt((A ** 2).mean(1))
n_big = (A > 4).sum(1)
for name, s in [('max', score_max), ('rms', score_ss), ('n_big', n_big)]:
    print(name, np.quantile(s, [.5, .9, .95, .97, .99]).round(2))
print('n_big distribution:', pd.Series(n_big).value_counts().sort_index().to_dict())

iso = np.load(os.path.join(PRED, 't5_iso_test.npy'))
iso_flag = iso > 0.48
for k in range(0, 6):
    f = n_big >= k
    print(f'n_big>={k}: flagged {f.sum()}  overlap with iso cluster {np.sum(f & iso_flag)} / {iso_flag.sum()}')
print('top contributing features in n_big>=2 rows:', (A[n_big >= 2] > 4).mean().sort_values(ascending=False).head(15).round(3).to_dict())
save_pred(5, 'shift_rms', score_ss.values)
save_pred(5, 'shift_nbig', n_big.values + score_ss.values / 100)

if os.environ.get('T5_DEBUG'):
    for k in (1, 2, 3):
        rows = A[n_big == k] > 4
        combos = rows.apply(lambda r: '+'.join(sorted(r.index[r])), axis=1).value_counts()
        print(f'--- n_big=={k} top combos'); print(combos.head(12).to_string())
    # expected false-positive count under null: fraction of rows with >=1 big shift among clean rows
    clean = n_big < 3
    print('per-feature exceed rate among clean rows:', (A[clean] > 4).mean().sort_values(ascending=False).head(8).round(4).to_dict())
