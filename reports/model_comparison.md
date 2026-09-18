# Model Comparison: Rule-Based vs Isolation Forest vs XGBoost

All models evaluated against the `is_suspicious` label.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based fraud_score (top 10%) | 1.0 | 0.994 | 0.997 |
| Isolation Forest (unsupervised) | 0.25 | 0.244 | 0.247 |
| XGBoost (supervised) | 0.955 | 0.943 | 0.949 |

XGBoost ROC-AUC: **0.9983**

## Top features driving XGBoost predictions

- velocity_count: 0.8142
- attempt_count: 0.0447
- amount: 0.0406
- switching_pct: 0.0353
- hour_of_day: 0.0206
- bank_enc: 0.0128
- upi_app_enc: 0.0127
- amount_slab_enc: 0.012
