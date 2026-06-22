"""Tests: drift-metric correctness (synthetic) + a golden integration run pinning the findings."""
import numpy as np
import pytest

import drift
from monitor import run


# --- drift metric unit tests (fast, synthetic) ---

def test_psi_zero_on_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert drift.psi(x, x) < 1e-6


def test_psi_flags_a_real_shift():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(0, 1, 5000), rng.normal(1.0, 1, 5000)) > drift.ALERT


def test_ks_and_jsd_zero_then_shift():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=5000), rng.normal(2, 1, 5000)
    assert drift.ks(x, x) < 1e-6 and drift.jsd(x, x) < 1e-6
    assert drift.ks(x, y) > 0.3 and drift.jsd(x, y) > 0.1


# --- golden integration run (pins eval/results.md) ---

@pytest.fixture(scope="module")
def res():
    return run(seed=42)


def test_zero_false_alerts_in_stable_phase(res):
    assert res["false_alert_rate"] == 0.0


def test_detection_latency_is_two_batches(res):
    assert res["detection_latency"] == 2


def test_auc_is_blind_but_population_gets_riskier(res):
    assert res["auc_drift"] >= res["auc_stable"] - 0.02   # AUC does not fall under covariate shift
    assert res["dr_drift"] > res["dr_stable"] + 0.10       # default rate clearly rises


def test_worst_feature_under_drift_is_pay0(res):
    last = res["records"][-1]
    assert last["worst"] == "PAY_0" and last["alert"]
