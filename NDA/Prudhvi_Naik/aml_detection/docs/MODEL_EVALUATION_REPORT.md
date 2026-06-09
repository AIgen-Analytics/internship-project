# Model Evaluation Report
## Temporal Baseline (Post-Leakage)
- **Algorithm:** LightGBM
- **ROC-AUC:** ~0.925
- **PR-AUC:** ~0.770
- **Threshold Optimization:** At an operational threshold of 0.70, the system achieves >80% Precision with less than 3% False Positive Rate.

## Typology Evaluation
- **Algorithm:** CatBoost
- Outputs a normalized probability array mapped to a custom dictionary containing the exact distribution of all 12 potential typologies for the investigator.