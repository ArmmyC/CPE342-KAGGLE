"""Naive Bayes with binned per-feature densities and an EM label-noise model (NB2).

Why naive Bayes: inside each class the features are almost uncorrelated, so the class-conditional
density factorises well and a careful per-feature density estimate is close to the Bayes classifier.

Density per feature and class:
  * text columns: one level per category value (levels from the training data);
  * numeric columns with few distinct values: one level per value;
  * other numeric columns: 'atoms' for frequent exact values (for example clipping bounds) plus
    equal-mass bins (edges from the training data) whose class histograms are smoothed with a
    Gaussian filter.  NaN is simply skipped (it contributes nothing to the likelihood).
Label noise: the observed label is modelled as a noisy copy of a latent clean class with a 4x4
transition matrix T (row = clean class, column = observed label).  EM alternates between the posterior
over the clean class and refitting the densities and T.  Predictions are p_clean @ T, i.e. the
probability of the observed label, which is what the metric scores.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# random_metric_* are pure noise; NOINFO columns showed no class dependence in the EDA
DROP0 = ['random_metric_1', 'random_metric_2', 'random_metric_3']
NOINFO = ['region', 'platform', 'device_type', 'payment_method', 'language', 'account_status', 'player_type_tag',
          'skill_tier', 'account_age_days', 'responds_to_discounts', 'preferred_game_mode', 'avg_match_length',
          'item_type_preference_social']


def nb_features(train, target='segment', id_col='player_id'):
    return [c for c in train.columns if c not in (id_col, target) and c not in DROP0 + NOINFO]


class NB2:
    def __init__(self, nbins=60, smooth=1.5, alpha=0.5, atom_frac=0.002, disc_max=25, em=0):
        self.nbins = nbins          # equal-mass bins for continuous columns
        self.smooth = smooth        # Gaussian smoothing (in bins) of the class histograms
        self.alpha = alpha          # additive (Laplace) smoothing of counts
        self.atom_frac = atom_frac  # a value is an atom if it holds at least this fraction of the rows
        self.disc_max = disc_max    # columns with at most this many distinct values are treated as discrete
        self.em = em                # number of EM iterations for the label-noise model (0 = plain NB)

    # -- encoding: learned on the training data in fit, reused for prediction
    def _codes(self, X, fit):
        if fit:
            self.enc = {}
        out = []
        for c in self.cols:
            v = X[c]
            if not pd.api.types.is_numeric_dtype(v):
                if fit:
                    self.enc[c] = ('cat', {k: i for i, k in enumerate(pd.Series(v.dropna().unique()))})
                mp = self.enc[c][1]
                code = v.map(mp).fillna(-1).astype(int).values   # NaN and unseen levels -> -1 (skipped)
                out.append((code, len(mp), 'cat', 0))
                continue
            x = v.values.astype(float)
            m = ~np.isnan(x)
            if fit:
                u, cnt = np.unique(x[m], return_counts=True)
                if len(u) <= self.disc_max:
                    self.enc[c] = ('disc', u)
                else:
                    atoms = u[cnt >= self.atom_frac * m.sum()]
                    xr = x[m][~np.isin(x[m], atoms)]
                    e = np.unique(np.quantile(xr, np.linspace(0, 1, self.nbins + 1)))
                    e[0], e[-1] = -np.inf, np.inf
                    self.enc[c] = ('bin', atoms, e)
            spec = self.enc[c]
            code = np.full(len(x), -1)
            if spec[0] == 'disc':
                u = spec[1]
                code[m] = np.clip(np.searchsorted(u, x[m]), 0, len(u) - 1)
                out.append((code, len(u), 'disc', 0))
            else:
                atoms, e = spec[1], spec[2]
                nb = len(e) - 1
                xm = x[m]
                i = np.clip(np.searchsorted(e, xm, side='right') - 1, 0, nb - 1)
                is_atom = np.isin(xm, atoms)
                i[is_atom] = nb + np.searchsorted(atoms, xm[is_atom])
                code[m] = i
                out.append((code, nb + len(atoms), 'bin', nb))
        return out

    def _fit_densities(self, codes, R):
        """R: (n, K) soft class responsibilities. Stores log p(level | class) per column."""
        K = R.shape[1]
        self.logp = []
        for code, L, kind, nb in codes:
            m = code >= 0
            cnt = np.zeros((L, K))
            np.add.at(cnt, code[m], R[m])
            cnt = cnt.T
            if kind == 'bin' and self.smooth > 0:
                cnt[:, :nb] = gaussian_filter1d(cnt[:, :nb], self.smooth, axis=1, mode='nearest')
            self.logp.append(np.log((cnt + self.alpha) / (cnt.sum(1, keepdims=True) + self.alpha * L)))

    def _loglik(self, codes):
        z = np.tile(np.log(self.pi), (len(codes[0][0]), 1))
        for (code, L, kind, nb), lp in zip(codes, self.logp):
            m = code >= 0
            z[m] += lp[:, code[m]].T
        return z

    def fit(self, X, y, K=4):
        self.cols = list(X.columns)
        codes = self._codes(X, True)
        Y = np.eye(K)[y]
        R = Y
        self.pi = R.mean(0)
        self.T = np.eye(K)
        self._fit_densities(codes, R)
        for it in range(self.em):
            T = self.T if it > 0 else np.full((K, K), 0.01) + np.eye(K) * 0.96   # start: 4% noise
            z = self._loglik(codes) + np.log(T[:, y].T)        # log p(clean=k) + log p(observed y | clean=k)
            z -= z.max(1, keepdims=True)
            R = np.exp(z)
            R /= R.sum(1, keepdims=True)
            self.pi = R.mean(0)
            self._fit_densities(codes, R)
            T = R.T @ Y
            self.T = T / T.sum(1, keepdims=True)
        return self

    def predict_proba(self, X, clean=False):
        z = self._loglik(self._codes(X, False))
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return p if clean else p @ self.T


def run_nb_cv(train, y, test, cols, folds, **params):
    """OOF probabilities and test probabilities (geometric mean over the fold models, renormalised)."""
    oof = np.zeros((len(train), 4))
    log_test = np.zeros((len(test), 4))
    models = []
    for tr_i, va_i in folds:
        m = NB2(**params).fit(train[cols].iloc[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(train[cols].iloc[va_i])
        log_test += np.log(m.predict_proba(test[cols])) / len(folds)
        models.append(m)
    test_p = np.exp(log_test)
    return oof, test_p / test_p.sum(1, keepdims=True), models
