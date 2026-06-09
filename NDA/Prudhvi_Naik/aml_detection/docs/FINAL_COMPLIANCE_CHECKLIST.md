# Final Compliance Checklist
| Requirement | Status | Verification |
| :--- | :--- | :--- |
| Output Transaction ID | **COMPLETE** | Handled by `run_pipeline.py` |
| Output Fraud Risk Score | **COMPLETE** | LightGBM `.predict_proba()` mapped to 0-100 |
| Output Risk Category | **COMPLETE** | Mapped via mathematical thresholding |
| Output Predicted Typology | **COMPLETE** | CatBoost `.predict()` |
| Output Typology Probabilities | **COMPLETE** | Custom dictionary extraction from `.predict_proba()` |
| Output Key Risk Drivers | **COMPLETE** | SHAP TreeExplainer mapped to Top 3 absolute attributions |
| Graph Analytics (PageRank) | **COMPLETE** | NetworkX integration in `build_features.py` |
| Strict Temporal Validation | **COMPLETE** | 80/20 Chronological Split implemented |
| Fix Explainer Mismatch | **COMPLETE** | Decoupled Stacking Ensemble; shifted to single LightGBM |