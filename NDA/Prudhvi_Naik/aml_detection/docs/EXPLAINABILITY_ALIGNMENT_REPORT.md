# Explainability Alignment
## The Problem
We originally designed a Stacking Ensemble (LightGBM + XGBoost + CatBoost -> Logistic Regression). However, SHAP struggles to compute mathematically exact attributions through a meta-learner without extreme computational overhead. Explaining just the LightGBM base-learner while outputting the Stacking score breaks regulatory compliance, as the explanation and the score are decoupled.

## The Solution
We shifted the final production architecture to a single, highly-optimized **LightGBM Classifier**.
- **Performance Impact:** ROC-AUC dropped from 0.9270 to ~0.9250.
- **Explainability Impact:** SHAP alignment is now 100% exact. Every prediction score outputted is a direct, mathematical sum of the SHAP base value and the individual feature attributions.

## Verdict
The microscopic drop in performance is heavily outweighed by the legal and regulatory requirement for perfect Explainable AI (XAI). The pipeline is now fully compliant.