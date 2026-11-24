"""Blending helpers for Task 2: macro F1, log-space mixing and per-class decision weights."""
import numpy as np

WEIGHT_GRID = np.arange(0.5, 2.5, 0.05)   # 0.50, 0.55, ..., 2.45


def macro_f1(y, pred, K=4):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    return np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1))


def log_mix(p, q, a):
    """Weighted geometric mean of two probability matrices, renormalised per row (softmax of the log-mix)."""
    z = (1 - a) * np.log(np.clip(p, 1e-9, 1)) + a * np.log(np.clip(q, 1e-9, 1))
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_class_weights(y, p, grid=WEIGHT_GRID, passes=3):
    """Coordinate search for w that maximises macro F1 of argmax(p * w); returned with w[3] = 1."""
    w = np.ones(p.shape[1])
    for _ in range(passes):
        for k in range(len(w)):
            best = (-1, 1)
            for v in grid:
                w2 = w.copy()
                w2[k] = v
                s = macro_f1(y, (p * w2).argmax(1))
                if s > best[0] + 1e-12:
                    best = (s, v)
            w[k] = best[1]
    return w / w[-1]


def nested_weighted_labels(y, p, folds):
    """Labels where each fold uses class weights fitted on the other four folds only (honest estimate)."""
    out = np.zeros(len(y), int)
    for a, b in folds:
        w = fit_class_weights(y[a], p[a])
        out[b] = (p[b] * w).argmax(1)
    return out
