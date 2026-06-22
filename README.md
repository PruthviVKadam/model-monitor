---
title: Model Monitor
emoji: 📡
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 📡 Model Monitor — catch the model rotting *before* the business does

A monitoring layer for a deployed tabular model that tracks **data drift** (PSI / KS / JSD),
**prediction drift**, and **performance** over a simulated production stream — and demonstrates the
uncomfortable truth that **accuracy is the last thing to move when the world changes.**

> Monitors the live model from [Credit Risk Scorer](https://github.com/PruthviVKadam/credit-risk)
> (P2) — no retraining, just its committed booster. Every number below comes from `monitor.py`
> (`eval/results.md`).

## Problem → Approach → Result

- **Problem:** models don't fail loudly — they rot. Inputs drift, the scored population changes, and
  accuracy decays. Teams usually find out from a downstream KPI weeks later. "*Is my model still OK
  right now?*" rarely has an answer.
- **Approach:** replay a time-ordered stream of scoring batches against P2's model. Early batches are
  stable; from batch 6 we inject a **covariate shift** (rising delinquency). Per batch we compute
  **PSI / KS / Jensen–Shannon** per feature, **prediction-drift PSI** on the score distribution, and
  **AUC + default rate** once labels arrive — with tiered alerts (warn 0.10 / alert 0.25).
- **Result (real, from `monitor.py`):**
  - **Detection latency: 2 batches** after the shift began (input PSI > 0.25), **0% false alerts** on
    the stable phase.
  - **Ranking AUC is blind to it:** 0.826 (stable) → **0.854** (drift) — it *rose*. An AUC dashboard
    would stay green.
  - **The population genuinely got riskier:** default rate 0.221 → **0.379** — but that needs labels,
    which arrive late. Input + prediction drift caught the same change **immediately and label-free**
    (PSI on `PAY_0` ramped 0.09 → 0.23 → 0.47 → 0.96).

## Insights

- **Covariate drift is invisible to AUC.** The injected shift *doubled* the real default rate, yet
  ranking AUC actually went **up** (0.826 → 0.854) — because oversampling clearly-delinquent
  customers makes ranking *easier*. An accuracy/AUC monitor would show all-green while the scored
  population fundamentally changed.
- **Input + prediction drift are your label-free early warning.** Default labels lag weeks; PSI on the
  inputs and on the score distribution flagged the shift within **2 batches at 0% false alarms** —
  the only signal available *before* the outcomes are known.
- **A raw metric isn't a monitor — a policy is.** PSI ramps smoothly (0.09 → 0.96); the warn/alert
  tiers plus **worst-feature attribution** (always `PAY_0` here) turn a number into an actionable,
  debuggable alert that points at *what* drifted.

## What it monitors

Reuses P2's committed XGBoost booster + the 11-feature spec (no retraining). Reference window =
6,000 rows of the training-time distribution; production = 12 batches of 1,500, with a covariate
shift injected from batch 6. The UCI data is cached to `data/credit_default.csv` (self-built on first
run), so the app is offline and instant.

## Stack

Python 3.14 · XGBoost (loads P2's `model.json`) · scipy (KS) · scikit-learn (AUC) · pandas · Plotly ·
Streamlit. Drift metrics are **built from scratch** (no Evidently) so the math is explicit and tested.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python monitor.py        # runs the study -> eval/results.md (the table above)
python -m pytest         # 7 tests: PSI/KS/JSD correctness + golden detection (latency=2, 0 false alerts)
streamlit run app.py     # dashboard: batch slider, PSI heatmap, score-dist overlay, drift-vs-AUC
```

## Files

```text
stream.py    reference + time-ordered batches with injected covariate drift
drift.py     PSI / KS / Jensen–Shannon, from scratch
score.py     load P2 booster, score a batch
perf.py      AUC / default rate / calibration gap + prediction-drift PSI
monitor.py   run the study -> eval/results.md
app.py       Streamlit dashboard
model/       P2's committed model.json + feature_spec.json (reused)
data/        cached UCI extract (self-built)
tests/       pytest: metric correctness + golden detection run
```

## Honesty guardrail

The headline was **rewritten to match the data**: an early hypothesis that "calibration degrades" was
**false** (the gap stayed ~0.19) — the real story is that AUC is blind while inputs/predictions move.
Detection latency (2) and the 0% false-alert rate are pinned by a golden test so they can't drift.
