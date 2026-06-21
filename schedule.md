# Build Schedule — Model Monitor (~11 days, 2–3 h/day)

| Day | Focus | Done when |
| --- | --- | --- |
| 1 | Repo, venv; load P2 model.json + held-out data | model scores a batch locally |
| 2 | `make_stream.py`: reference window + time-ordered batches | batches reproducible from a seed |
| 3 | PSI implementation + unit test | zero-drift ≈ 0; shifted feature flags |
| 4 | KS + Jensen–Shannon + per-feature drift table | three metrics agree on a known shift |
| 5 | Prediction (score) drift tracking | score-dist divergence plotted over time |
| 6 | `perf.py`: delayed-label AUC/precision per cohort | performance curve over batches |
| 7 | Alert tiers (warn/alert) + consecutive-breach rule | no single-batch false alarms |
| 8 | Drift-injection harness (covariate + label-prior) | injected shift visible in metrics |
| 9 | Detection study: latency + false-alert rate | numbers written to `runs/` |
| 10 | Streamlit dashboard (time slider, heatmap, alert log) + HTML report | app + report render |
| 11 | pytest + README with **real** detection numbers; deploy | tests green; live URL |

**Lead-with-the-number (resume line, once real):**
"Detected an injected covariate shift (PSI > 0.25) __ batches before held-out AUC fell below
0.74 — with a __% false-alert rate on stable batches."

**Cut scope if behind:** drop the live-endpoint integration and label-prior drift; PSI/KS/JS +
prediction drift + the injection study + dashboard is a complete project.
