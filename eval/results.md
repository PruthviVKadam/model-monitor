# Model-monitor study

_P2 credit-risk model · simulated stream, drift injected from batch 6 (rising delinquency) · seed 42._

| Batch | Phase | Max feature PSI | Worst feature | Pred-drift PSI | AUC | Default rate | Calib. gap | Alert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | stable | 0.011 | AGE | 0.005 | 0.828 | 0.217 | +0.206 |  |
| 1 | stable | 0.022 | PAY_AMT1 | 0.005 | 0.826 | 0.220 | +0.192 |  |
| 2 | stable | 0.018 | PAY_AMT1 | 0.013 | 0.796 | 0.210 | +0.209 |  |
| 3 | stable | 0.014 | LIMIT_BAL | 0.002 | 0.827 | 0.224 | +0.194 |  |
| 4 | stable | 0.010 | PAY_AMT2 | 0.012 | 0.831 | 0.221 | +0.203 |  |
| 5 | stable | 0.010 | LIMIT_BAL | 0.009 | 0.850 | 0.234 | +0.191 |  |
| 6 | drift | 0.093 | PAY_0 | 0.078 | 0.857 | 0.305 | +0.170 |  |
| 7 | drift | 0.227 | PAY_0 | 0.210 | 0.866 | 0.327 | +0.185 |  |
| 8 | drift | 0.468 | PAY_0 | 0.373 | 0.854 | 0.385 | +0.170 | 🚨 |
| 9 | drift | 0.599 | PAY_0 | 0.483 | 0.869 | 0.392 | +0.187 | 🚨 |
| 10 | drift | 0.808 | PAY_0 | 0.685 | 0.850 | 0.428 | +0.193 | 🚨 |
| 11 | drift | 0.963 | PAY_0 | 0.823 | 0.829 | 0.438 | +0.203 | 🚨 |

- **Detection latency:** the injected shift was flagged **2 batch(es)** after it began (input PSI > 0.25); prediction-drift PSI rises in lockstep — **no labels required**.
- **False-alert rate (stable phase):** **0%**.
- **Ranking AUC is blind to it:** stable 0.826 → drift 0.854 — it barely moves (even rises), so an accuracy/AUC monitor would never notice.
- **The population genuinely got riskier:** observed default rate stable 0.221 → drift 0.379 — but that needs *labels*, which arrive late. Input/prediction drift caught the same change immediately and label-free.
