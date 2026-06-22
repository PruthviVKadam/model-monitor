"""Build a time-ordered production stream for the P2 credit-risk model.

The UCI data has no timestamp, so we simulate production traffic: a fixed **reference** window
(the model's training-time distribution) plus N ordered **batches**. Early batches are drawn
i.i.d. from the same pool (stable); from `drift_start` on we inject a **covariate shift** —
a progressively larger share of delinquent customers (PAY_0 ≥ 2), i.e. a simulated downturn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# P2's exact feature mapping (UCI code -> friendly name), in canonical model order.
FEAT = [("X1", "LIMIT_BAL"), ("X5", "AGE"), ("X6", "PAY_0"), ("X7", "PAY_2"), ("X8", "PAY_3"),
        ("X12", "BILL_AMT1"), ("X18", "PAY_AMT1"), ("X19", "PAY_AMT2"),
        ("X3", "EDUCATION"), ("X4", "MARRIAGE"), ("X2", "SEX")]
FEATURES = [name for _, name in FEAT]


def _ensure_data() -> pd.DataFrame:
    """Load the cached 11-feature + DEFAULT table; fetch from UCI once if absent."""
    csv = DATA / "credit_default.csv"
    if csv.exists():
        return pd.read_csv(csv)
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    df = ds.data.features[[u for u, _ in FEAT]].rename(columns=dict(FEAT))
    df["DEFAULT"] = ds.data.targets.iloc[:, 0].astype(int).values
    DATA.mkdir(exist_ok=True)
    df.to_csv(csv, index=False)
    return df


def make_stream(n_batches: int = 12, drift_start: int = 6, batch_size: int = 1500,
                ref_size: int = 6000, seed: int = 42):
    """Return (reference_df, [batch_df, ...], drift_start). DEFAULT column carried through."""
    df = _ensure_data()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    ref = df.iloc[idx[:ref_size]].reset_index(drop=True)
    pool = df.iloc[idx[ref_size:]].reset_index(drop=True)
    delinquent = (pool["PAY_0"].to_numpy() >= 2)

    batches = []
    for t in range(n_batches):
        if t < drift_start:
            sel = rng.choice(len(pool), size=batch_size, replace=True)        # stable
        else:
            intensity = (t - drift_start + 1) / (n_batches - drift_start)     # 0<..<=1
            w = np.where(delinquent, 1.0 + 8.0 * intensity, 1.0)
            sel = rng.choice(len(pool), size=batch_size, replace=True, p=w / w.sum())
        batches.append(pool.iloc[sel].reset_index(drop=True))
    return ref, batches, drift_start
