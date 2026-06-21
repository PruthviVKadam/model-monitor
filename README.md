# 📡 Model Monitor — catch the model rotting *before* the business does

A monitoring layer for a deployed tabular model that tracks **data drift**, **prediction drift**,
and **performance decay** over time, raises tiered alerts, and produces a drift report — so a model
silently degrading is caught from its inputs, not from a business metric tanking weeks later.

> **Status:** scaffolding (plan + schedule below). Results table is a template filled only from real
> output. Monitors the model from [Credit Risk Scorer](https://github.com/PruthviVKadam/credit-risk)
> (P2) — reusing a real served model is the point.

## Problem → Approach → Result

- **Problem:** models don't fail loudly — they rot. Input distributions shift (covariate drift), the
  input→target relationship changes (concept drift), and accuracy quietly decays. Most teams find out
  when a downstream KPI drops. The question "*is my model still OK right now?*" usually has no answer.
- **Approach:** replay batches of scoring data across time and compute, per batch: **PSI**, **KS**,
  and **Jensen–Shannon** divergence per feature (data drift); the shift in the **score distribution**
  (prediction drift); and **AUC/precision** once labels arrive (delayed-label performance). Tiered
  thresholds (warn/alert) + an HTML drift report. A Streamlit dashboard with a time slider lets you
  **inject a drift event and watch it get flagged** before performance drops.
- **Result:** _(filled from real run output — no numbers until measured)_

| Metric | Value |
| --- | --- |
| Batches until injected covariate shift detected (PSI > 0.25) | _TBD_ |
| Lead time vs. AUC dropping below threshold | _TBD_ |
| False-alert rate on stable (no-drift) batches | _TBD_ |

## What it monitors
Reuses **P2's committed XGBoost model + feature spec**. A reference window (training/validation
distribution) vs. rolling production batches synthesized from the held-out data, with optional
injected shifts (e.g. income distribution drift, label-prior change).

## Stack

Python 3.14 · pandas · numpy · **scipy** (KS) · scikit-learn (AUC) · Plotly · Streamlit.
Reference project: **Evidently** — the drift metrics here are built from scratch to show the math is
understood (same stance as P3 using hand-rolled retrieval eval instead of RAGAS).

## Reproduce (once built)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python make_stream.py     # build reference + time-ordered batches (+ optional injected drift)
python monitor.py         # per-batch PSI/KS/JS, prediction drift, perf -> runs/ + reports/
streamlit run app.py      # time-slider dashboard; inject drift and watch it flag
```

## Honesty guardrail

Drift thresholds (PSI 0.1 warn / 0.25 alert) are stated as the **chosen policy**, not as results.
Detection latency and false-alert rate are measured on a labeled synthetic stream and copied
verbatim — never asserted without the run that produced them.
