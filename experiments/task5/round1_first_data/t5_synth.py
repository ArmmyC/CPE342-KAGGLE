"""Task 5: tune the anomaly detector on synthetic (injected) anomalies instead of guessing weights.

There are no labels, so we make some: copy ordinary-looking players and damage them in generic
ways (a jump that starts in some period, a slow drift over many features, robot-like loss of
variability, risky account flags, a location jump). Each detector component is turned into a
"surprise" score (-log of its upper-tail rank among the real players), and the component weights
are chosen to rank the injected players above the real ones. No player_id information is used.

Usage: python t5_synth.py            saves preds/t5_synth_test.npy
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import load, save_pred, PRED

rng = np.random.default_rng(0)
real = load(5)[1].drop(columns='player_id')
BASES = sorted({c.rsplit('_', 1)[0] for c in real.columns if c[-2:] in ('_1', '_2', '_3', '_4')})
FLAGS = ['password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale']
VARIABILITY = ['ip_hash_entropy', 'login_time_variance', 'session_duration_std', 'kda_std',
               'accuracy_std', 'reaction_time_std', 'purchase_amount_std']
SD = {b: np.nanstd(real[[f'{b}_{i}' for i in range(1, 5)]].values) for b in BASES}


def robust_stats(v):
    med = np.nanmedian(v, 0)
    mad = np.nanmedian(np.abs(v - med), 0) * 1.4826
    return med, np.where(mad == 0, np.nanstd(v, 0) + 1e-9, mad)


class Components:
    """Raw detector components; reference statistics come from the real test players only."""

    def __init__(self, ref):
        self.jump_stats = {k: robust_stats(self._jumps(ref, k)) for k in (2, 3, 4)}
        M = self._means(ref)
        self.mean_med = M.median()
        Z = (M.fillna(self.mean_med) - M.mean()) / M.std()
        self.mu, self.sd = M.mean(), M.std()
        self.icov = np.linalg.inv(np.cov(Z.values.T))

    @staticmethod
    def _jumps(df, k):  # value in period k minus the player's own average before it
        P = np.stack([df[[f'{b}_{i}' for i in range(1, 5)]].values for b in BASES], 1)  # n x base x 4
        return P[:, :, k - 1] - np.nanmean(P[:, :, :k - 1], 2)

    @staticmethod
    def _means(df):
        M = pd.DataFrame({b: df[[f'{b}_{i}' for i in range(1, 5)]].mean(axis=1) for b in BASES})
        M['behavior_stability'] = df['behavior_stability'].values
        M['account_age_days'] = df['account_age_days'].values
        return M

    def __call__(self, df):
        rms, nbig = [], []
        for k in (2, 3, 4):
            med, mad = self.jump_stats[k]
            A = np.nan_to_num(np.abs((self._jumps(df, k) - med) / mad)).clip(max=30)
            rms.append(np.sqrt((A ** 2).mean(1))); nbig.append((A > 4).sum(1))
        Z = (self._means(df).fillna(self.mean_med) - self.mu) / self.sd
        return pd.DataFrame({
            'jump_rms': np.max(rms, 0),
            'jump_count': np.max(nbig, 0) + np.max(rms, 0) / 100,  # tie-break counts by size
            'robot': -(Z['ip_hash_entropy'] + Z['login_time_variance']).values,
            'drift': np.einsum('ij,jk,ik->i', Z.values, self.icov, Z.values),
            'flags': df[FLAGS].fillna(0).sum(1).values + rng.uniform(0, 1e-3, len(df)),
        })


def surprise(ref_vals, vals):
    """-log of the upper-tail rank among real players: 0 for ordinary, large for extreme."""
    srt = np.sort(ref_vals)
    upper = (len(srt) - np.searchsorted(srt, vals, side='left') + 1) / (len(srt) + 1)
    return -np.log(upper)


def inject(rows):
    """Damage each copied player with one generic anomaly type; returns (rows, type names)."""
    rows = rows.copy().reset_index(drop=True)
    kinds = rng.choice(['jump', 'drift', 'robot', 'flags', 'geo'], len(rows))
    for i, kind in enumerate(kinds):
        if kind == 'jump':
            start = rng.integers(2, 5)
            for b in rng.choice(BASES, rng.integers(2, 9), replace=False):
                cols = [f'{b}_{p}' for p in range(start, 5)]
                rows.loc[i, cols] += rng.choice([-1, 1]) * rng.uniform(1, 4) * SD[b]
        elif kind == 'drift':
            for b in rng.choice(BASES, rng.integers(4, 16), replace=False):
                rows.loc[i, [f'{b}_{p}' for p in range(1, 5)]] += rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * SD[b]
            rows.loc[i, 'behavior_stability'] -= rng.uniform(0, 0.1)
        elif kind == 'robot':
            for b in rng.choice(VARIABILITY, rng.integers(1, 5), replace=False):
                rows.loc[i, [f'{b}_{p}' for p in range(1, 5)]] *= rng.uniform(0.02, 0.3)
        elif kind == 'flags':
            rows.loc[i, list(rng.choice(FLAGS, rng.integers(2, 4), replace=False))] = 1
        else:  # sudden move far from the usual login location
            for c in ('login_lat_4', 'login_lon_4'):
                rows.loc[i, c] += rng.choice([-1, 1]) * rng.uniform(5, 40)
    for c in rows.columns:  # keep values inside the range seen in real data
        rows[c] = rows[c].clip(np.nanmin(real[c].values), np.nanmax(real[c].values))
    return rows, kinds


comp = Components(real)
C_real = comp(real)
old = np.load(f'{PRED}/t5_multi_test.npy')
pool = real[pd.Series(old).rank(pct=True).values < 0.85]  # ordinary-looking players to damage
fake, kinds = inject(pool.sample(4000, random_state=0))
C_fake = comp(fake)
S_real = pd.DataFrame({c: surprise(C_real[c].values, C_real[c].values) for c in C_real})
S_fake = pd.DataFrame({c: surprise(C_real[c].values, C_fake[c].values) for c in C_real})

y = np.r_[np.zeros(len(S_real)), np.ones(len(S_fake))]
S = pd.concat([S_real, S_fake], ignore_index=True)
print('average precision of each component alone:')
for c in S:
    print(f'  {c:11s} {average_precision_score(y, S[c]):.3f}  '
          + ' '.join(f'{k}={average_precision_score(y[kinds_mask], S[c][kinds_mask]):.2f}'
                     for k in ('jump', 'drift', 'robot', 'flags', 'geo')
                     for kinds_mask in [np.r_[np.ones(len(S_real), bool), kinds == k]]))

# weight search: weighted sum of surprises and weighted max of surprises
best = (-1, None, None)
for _ in range(3000):
    w = rng.dirichlet(np.ones(S.shape[1]))
    for how in ('sum', 'max'):
        sc = (S.values * w).sum(1) if how == 'sum' else (S.values * w).max(1)
        ap = average_precision_score(y, sc)
        if ap > best[0]:
            best = (ap, w, how)
ap, w, how = best
print(f'best combination: {how} of surprises, AP {ap:.3f}, weights',
      dict(zip(S.columns, w.round(3))))
final = (S_real.values * w).sum(1) if how == 'sum' else (S_real.values * w).max(1)
save_pred(5, 'synth', final)
