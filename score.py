"""Score batches with P2's committed XGBoost booster (no retraining)."""
from __future__ import annotations

from pathlib import Path

import xgboost as xgb

from stream import FEATURES

MODEL = Path(__file__).resolve().parent / "model" / "model.json"
_BOOSTER = None


def booster() -> xgb.Booster:
    global _BOOSTER
    if _BOOSTER is None:
        b = xgb.Booster()
        b.load_model(str(MODEL))
        _BOOSTER = b
    return _BOOSTER


def score(df) -> "object":
    """Return P(default) for a frame containing the 11 feature columns."""
    X = df[FEATURES]
    return booster().predict(xgb.DMatrix(X, feature_names=FEATURES))
