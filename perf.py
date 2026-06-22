"""Performance + calibration when (delayed) labels arrive."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def performance(y, proba) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba)
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    default_rate = float(y.mean())
    mean_pred = float(proba.mean())
    return {
        "auc": auc,                          # ranking quality — robust to covariate shift
        "default_rate": default_rate,        # observed positive rate (needs labels)
        "mean_pred": mean_pred,              # average predicted P(default)
        "cal_gap": mean_pred - default_rate, # calibration gap — breaks under covariate shift
    }


def prediction_drift(ref_proba, cur_proba) -> float:
    """PSI on the score distribution — available with no labels at all (early warning)."""
    from drift import psi
    return psi(ref_proba, cur_proba)
