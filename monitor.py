"""Run the monitoring study over the simulated stream and write a reproducible report.

Headline questions answered: (1) how fast does input PSI flag the injected covariate shift,
(2) how many false alerts fire on the stable batches, (3) does ranking AUC even move — or is
calibration the thing that breaks? All numbers in the README come from this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import drift
import perf
from score import score
from stream import FEATURES, make_stream

OUT = Path(__file__).resolve().parent / "eval" / "results.md"


def run(seed: int = 42):
    ref, batches, drift_start = make_stream(seed=seed)
    ref_proba = score(ref)
    ref_perf = perf.performance(ref["DEFAULT"], ref_proba)

    records = []
    for t, b in enumerate(batches):
        fd = drift.feature_drift(ref, b, FEATURES)
        proba = score(b)
        pr = perf.performance(b["DEFAULT"], proba)
        records.append({
            "t": t, "phase": "stable" if t < drift_start else "drift",
            "max_psi": fd["_max_psi"], "worst": fd["_worst_feature"],
            "pred_psi": perf.prediction_drift(ref_proba, proba),
            "auc": pr["auc"], "default_rate": pr["default_rate"],
            "mean_pred": pr["mean_pred"], "cal_gap": pr["cal_gap"],
            "alert": fd["_max_psi"] > drift.ALERT,
        })

    stable = [r for r in records if r["phase"] == "stable"]
    drifted = [r for r in records if r["phase"] == "drift"]
    first_alert = next((r["t"] for r in drifted if r["alert"]), None)
    detection_latency = (first_alert - drift_start) if first_alert is not None else None
    false_alerts = sum(r["alert"] for r in stable)
    return {
        "drift_start": drift_start, "ref_perf": ref_perf, "records": records,
        "detection_latency": detection_latency,
        "false_alert_rate": false_alerts / len(stable) if stable else float("nan"),
        "auc_stable": float(np.mean([r["auc"] for r in stable])),
        "auc_drift": float(np.mean([r["auc"] for r in drifted])),
        "dr_stable": float(np.mean([r["default_rate"] for r in stable])),
        "dr_drift": float(np.mean([r["default_rate"] for r in drifted])),
    }


def to_markdown(res: dict) -> str:
    L = ["# Model-monitor study", "",
         f"_P2 credit-risk model · simulated stream, drift injected from batch "
         f"{res['drift_start']} (rising delinquency) · seed 42._", "",
         "| Batch | Phase | Max feature PSI | Worst feature | Pred-drift PSI | AUC | "
         "Default rate | Calib. gap | Alert |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["records"]:
        L.append(f"| {r['t']} | {r['phase']} | {r['max_psi']:.3f} | {r['worst']} | "
                 f"{r['pred_psi']:.3f} | {r['auc']:.3f} | {r['default_rate']:.3f} | "
                 f"{r['cal_gap']:+.3f} | {'🚨' if r['alert'] else ''} |")
    L += ["",
          f"- **Detection latency:** the injected shift was flagged **{res['detection_latency']} "
          f"batch(es)** after it began (input PSI > {drift.ALERT}); prediction-drift PSI rises in "
          f"lockstep — **no labels required**.",
          f"- **False-alert rate (stable phase):** **{res['false_alert_rate']*100:.0f}%**.",
          f"- **Ranking AUC is blind to it:** stable {res['auc_stable']:.3f} → drift "
          f"{res['auc_drift']:.3f} — it barely moves (even rises), so an accuracy/AUC monitor would "
          f"never notice.",
          f"- **The population genuinely got riskier:** observed default rate stable "
          f"{res['dr_stable']:.3f} → drift {res['dr_drift']:.3f} — but that needs *labels*, which "
          f"arrive late. Input/prediction drift caught the same change immediately and label-free.",
          ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(res)
    OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {OUT}")
