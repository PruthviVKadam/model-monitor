# Build Plan — Model Monitor

## Interview thesis
"This is the guide's whole thesis — *what happens after deployment* — applied to a tabular model.
I monitor my own credit-risk model's inputs, predictions, and performance, and I can show the
monitor catching an injected shift N batches before accuracy drops." The clearest "I own a model in
production" signal on the resume.

## Architecture
```
make_stream.py  reference window + time-ordered production batches; optional injected drift
drift.py        PSI, KS-statistic, Jensen-Shannon divergence (per feature)
perf.py         AUC / precision / recall when (delayed) labels arrive
monitor.py      run all batches -> per-batch metrics + alert tiers -> runs/, reports/*.html
app.py          Streamlit: time slider, per-feature drift heatmap, score-dist overlay, alert log
```

## Drift metrics (defend each)
- **PSI (Population Stability Index):** binned reference vs. current; rule-of-thumb 0.1 warn / 0.25
  alert. Good for monotonic shifts; sensitive to binning — document the choice.
- **KS statistic:** distribution-free, continuous features; complements PSI.
- **Jensen–Shannon divergence:** symmetric, bounded [0,1]; robust for categorical/score dists.
- **Prediction drift:** monitor the *output* score distribution — moves even when you have no labels
  yet (labels are usually delayed). This is the early-warning signal.
- **Performance-on-arrival:** when labels land, recompute AUC/precision per cohort; the gap between
  prediction-drift onset and performance drop is your headline "lead time".

## Key decisions
- **Reference choice:** training/validation distribution as the baseline; rolling vs. fixed window.
- **Concept vs. covariate drift:** input drift can exist without performance loss (and vice-versa) —
  showing you distinguish them is the senior nuance.
- **Alert hygiene:** a rolling average / consecutive-breach rule to avoid single-batch false alarms.

## Phases (≈11 days, 2–3 h/day)
1. **Stream builder** — load P2 model + held-out data; emit reference + ordered batches.
2. **Drift module** — PSI/KS/JS with unit tests (zero drift ⇒ ~0; known shift ⇒ flagged).
3. **Performance + alerting** — delayed-label AUC; warn/alert tiers + consecutive-breach rule.
4. **Drift injection + detection study** — inject shifts; measure detection latency & false alerts.
5. **Dashboard + report + README** — Streamlit time slider; HTML report; real numbers in README.

## Py3.14 de-risk
Pure pandas/numpy/scipy — no heavy ML deps, no torch. Load P2's `model.json` with the same
`xgboost` already verified in the credit-risk venv. Build core metrics by hand (don't depend on
Evidently installing on 3.14).

## Deploy
Streamlit Cloud or HF Spaces (Streamlit SDK). Bundle the synthetic stream so it runs without P2 live.
Optional: point at P2's live `/score` endpoint for a true end-to-end demo (see `ManualSteps.md`).
