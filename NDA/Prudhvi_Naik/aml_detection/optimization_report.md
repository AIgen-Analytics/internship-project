# Model Performance Optimization Report

## 1. Feature Selection & Data Strategies
- Selected top 150 features using SHAP/Gain importance.
- SMOTE Oversampling Strategy Winner: SMOTE

## 2. Hyperparameter Tuning Outcomes
- **XGBoost:** {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05}
- **CatBoost:** {'learning_rate': 0.1, 'iterations': 200, 'depth': 6}

## 3. Final Model Benchmark
                    ROC-AUC    PR-AUC  Precision    Recall        F1       FPR       FNR
Model                                                                                   
Base LightGBM      0.955584  0.870179   0.813538  0.766248  0.789185  0.040785  0.233752
Tuned XGBoost      0.950224  0.851855   0.788021  0.760277  0.773901  0.047495  0.239723
Tuned CatBoost     0.950658  0.857524   0.806171  0.751287  0.777762  0.041949  0.248713
Voting Ensemble    0.953629  0.863726   0.806714  0.763640  0.784586  0.042491  0.236360
Stacking Ensemble  0.955655  0.870679   0.808069  0.776680  0.792063  0.042841  0.223320

## 4. Conclusion
The best performing approach is **Stacking Ensemble** with a PR-AUC of 0.8707. This model maintains exceptionally low False Positive Rates while maximizing fraud detection.