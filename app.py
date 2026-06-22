"""Model Monitor — drift & performance dashboard over the P2 credit-risk model (Streamlit)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import drift
import perf
from score import score
from stream import FEATURES, make_stream

st.set_page_config(page_title="Model Monitor", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="Building stream + scoring batches…")
def load():
    ref, batches, drift_start = make_stream(seed=42)
    ref_proba = score(ref)
    scored, recs = [], []
    for t, b in enumerate(batches):
        proba = score(b)
        scored.append(proba)
        fd = drift.feature_drift(ref, b, FEATURES)
        pr = perf.performance(b["DEFAULT"], proba)
        recs.append({"t": t, "phase": "stable" if t < drift_start else "drift",
                     "max_psi": fd["_max_psi"], "worst": fd["_worst_feature"],
                     "pred_psi": perf.prediction_drift(ref_proba, proba),
                     "auc": pr["auc"], "default_rate": pr["default_rate"],
                     "alert": fd["_max_psi"] > drift.ALERT,
                     "per_feat": {f: fd[f]["psi"] for f in FEATURES}})
    return ref, batches, ref_proba, scored, recs, drift_start


ref, batches, ref_proba, scored, recs, drift_start = load()

st.title("📡 Model Monitor")
st.caption("Drift + performance monitoring over the **Credit Risk Scorer (P2)** model. A covariate "
           "shift (rising delinquency) is injected from batch %d — watch what catches it." % drift_start)

first_alert = next((r["t"] for r in recs if r["phase"] == "drift" and r["alert"]), None)
latency = (first_alert - drift_start) if first_alert is not None else None
false_alerts = sum(r["alert"] for r in recs if r["phase"] == "stable")
auc_stable = np.mean([r["auc"] for r in recs if r["phase"] == "stable"])
auc_drift = np.mean([r["auc"] for r in recs if r["phase"] == "drift"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Detection latency", f"{latency} batches", help="batches after drift began until PSI>0.25")
c2.metric("False alerts (stable)", f"{false_alerts}")
c3.metric("AUC stable → drift", f"{auc_stable:.3f} → {auc_drift:.3f}", help="AUC is blind to the shift")
c4.metric("Default rate stable → drift",
          f"{np.mean([r['default_rate'] for r in recs if r['phase']=='stable']):.2f}"
          f" → {np.mean([r['default_rate'] for r in recs if r['phase']=='drift']):.2f}")

t = st.slider("Inspect batch", 0, len(batches) - 1, drift_start)
rec = recs[t]

# --- per-feature PSI for the selected batch ---
left, right = st.columns(2)
with left:
    feats = FEATURES
    vals = [rec["per_feat"][f] for f in feats]
    colors = ["#dc2626" if v > drift.ALERT else "#f59e0b" if v > drift.WARN else "#9ca3af" for v in vals]
    bar = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=colors))
    bar.add_vline(x=drift.ALERT, line_dash="dot", line_color="#dc2626", annotation_text="alert 0.25")
    bar.add_vline(x=drift.WARN, line_dash="dot", line_color="#f59e0b", annotation_text="warn 0.10")
    bar.update_layout(title=f"Feature PSI — batch {t} ({rec['phase']})", height=380, margin=dict(t=40),
                      xaxis_title="PSI vs reference")
    st.plotly_chart(bar, use_container_width=True)
with right:
    lo = min(ref_proba.min(), scored[t].min())
    hi = max(ref_proba.max(), scored[t].max())
    edges = np.linspace(lo, hi, 30)
    h = go.Figure()
    h.add_trace(go.Histogram(x=ref_proba, xbins=dict(start=lo, end=hi, size=(hi - lo) / 30),
                             name="reference", opacity=0.6, marker_color="#9ca3af", histnorm="probability"))
    h.add_trace(go.Histogram(x=scored[t], xbins=dict(start=lo, end=hi, size=(hi - lo) / 30),
                             name=f"batch {t}", opacity=0.6, marker_color="#4f46e5", histnorm="probability"))
    h.update_layout(barmode="overlay", title=f"Score distribution — pred-drift PSI {rec['pred_psi']:.3f}",
                    height=380, margin=dict(t=40), xaxis_title="P(default)",
                    legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(h, use_container_width=True)

# --- metrics over time ---
ts = [r["t"] for r in recs]
line = go.Figure()
line.add_trace(go.Scatter(x=ts, y=[r["max_psi"] for r in recs], name="max feature PSI",
                          line=dict(color="#dc2626", width=3)))
line.add_trace(go.Scatter(x=ts, y=[r["pred_psi"] for r in recs], name="prediction-drift PSI",
                          line=dict(color="#4f46e5")))
line.add_trace(go.Scatter(x=ts, y=[r["auc"] for r in recs], name="AUC", yaxis="y2",
                          line=dict(color="#16a34a", dash="dash")))
line.add_hline(y=drift.ALERT, line_dash="dot", line_color="#dc2626")
line.add_vline(x=drift_start - 0.5, line_dash="dot", line_color="black",
               annotation_text="drift injected")
line.update_layout(title="Drift vs. AUC over time — PSI climbs, AUC stays flat",
                   xaxis_title="batch", yaxis_title="PSI",
                   yaxis2=dict(title="AUC", overlaying="y", side="right", range=[0.5, 1.0]),
                   height=420, margin=dict(t=50), legend=dict(orientation="h", y=-0.2))
st.plotly_chart(line, use_container_width=True)

st.caption("Educational demo. The point: input/prediction drift flag the shift label-free and "
           "immediately, while AUC — which needs labels that arrive late — never moves.")
