"""Data-drift metrics, built from scratch (no Evidently) so the math is explicit and testable.

- PSI  (Population Stability Index): binned reference vs current; 0.1 warn / 0.25 alert.
- KS   (Kolmogorov–Smirnov statistic): distribution-free, continuous features.
- JSD  (Jensen–Shannon distance): symmetric, bounded [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

EPS = 1e-6

WARN, ALERT = 0.1, 0.25  # PSI thresholds


def psi(ref, cur, bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant / very low cardinality -> bin by midpoints of values
        vals = np.unique(ref)
        edges = (np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]])
                 if len(vals) > 1 else np.array([-np.inf, np.inf]))
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    e = np.clip(e, EPS, None)
    a = np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(ref, cur) -> float:
    return float(ks_2samp(np.asarray(ref), np.asarray(cur)).statistic)


def jsd(ref, cur, bins: int = 20) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(ref, edges)[0] + EPS
    q = np.histogram(cur, edges)[0] + EPS
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def feature_drift(ref_df, cur_df, features) -> dict:
    """{feature: {psi, ks, jsd}} plus ('_max_psi', best_feature)."""
    out = {}
    for f in features:
        out[f] = {"psi": psi(ref_df[f], cur_df[f]),
                  "ks": ks(ref_df[f], cur_df[f]),
                  "jsd": jsd(ref_df[f], cur_df[f])}
    worst = max(features, key=lambda f: out[f]["psi"])
    out["_max_psi"] = out[worst]["psi"]
    out["_worst_feature"] = worst
    return out
