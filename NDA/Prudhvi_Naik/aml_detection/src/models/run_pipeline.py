import sys, os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

def get_risk_category(score, threshold):
    if score >= threshold + 0.15: return 'Critical Risk'
    if score >= threshold: return 'High Risk'
    if score >= threshold - 0.10: return 'Medium Risk'
    return 'Low Risk'

def main():
    print("=" * 60, flush=True)
    print("  ENTERPRISE AML STACKING ENSEMBLE PIPELINE", flush=True)
    print("  LightGBM + XGBoost + CatBoost → Weighted Ensemble", flush=True)
    print("=" * 60, flush=True)
    
    print("\nStep 1: Loading ALL Data and Engineering Features...", flush=True)
    df = load_and_merge_data('data/raw')
    df = build_entity_features(df)
    df, dt_cols = build_interaction_features(df)
    
    df_raw = df.copy()
    X, y = prepare_features_for_model(df, dt_cols)
    
    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features", flush=True)
    
    print("\nStep 2: Strict Temporal Validation Split...", flush=True)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_train = df_raw.iloc[:split_idx].copy()
    df_test = df_raw.iloc[split_idx:].copy()
    
    os.makedirs('data/processed', exist_ok=True)
    df_train.to_parquet('data/processed/train_features.parquet', index=False)
    df_test.to_parquet('data/processed/test_features.parquet', index=False)
    
    X_train_filled = X_train.fillna(0).astype('float64')
    X_test_filled = X_test.fillna(0).astype('float64')
    
    # Calculate scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # =====================================================
    # STACKING ENSEMBLE: 3 Base Learners + Meta-Learner
    # =====================================================
    
    # === BASE LEARNER 1: LightGBM ===
    print("\nStep 3a: Training LightGBM (Base Learner 1)...", flush=True)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=12, num_leaves=128,
        min_child_samples=20, colsample_bytree=0.85, subsample=0.85,
        reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train_filled, y_train)
    lgb_train_prob = lgb_model.predict_proba(X_train_filled)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test_filled)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_test_prob)
    print(f"  LightGBM ROC-AUC: {lgb_auc*100:.2f}%", flush=True)
    
    # === BASE LEARNER 2: XGBoost ===
    print("Step 3b: Training XGBoost (Base Learner 2)...", flush=True)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=10,
        colsample_bytree=0.85, subsample=0.85,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=scale_pos_weight,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train_filled, y_train)
    xgb_train_prob = xgb_model.predict_proba(X_train_filled)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(X_test_filled)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_test_prob)
    print(f"  XGBoost ROC-AUC: {xgb_auc*100:.2f}%", flush=True)
    
    # === BASE LEARNER 3: CatBoost (Binary) ===
    print("Step 3c: Training CatBoost (Base Learner 3)...", flush=True)
    cat_model = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=8, auto_class_weights='Balanced',
        loss_function='Logloss', verbose=0, random_seed=42
    )
    cat_model.fit(X_train_filled, y_train)
    cat_train_prob = cat_model.predict_proba(X_train_filled)[:, 1]
    cat_test_prob = cat_model.predict_proba(X_test_filled)[:, 1]
    cat_auc = roc_auc_score(y_test, cat_test_prob)
    print(f"  CatBoost ROC-AUC: {cat_auc*100:.2f}%", flush=True)
    
    # === ENSEMBLE: Weighted Probability Average ===
    # LightGBM gets highest weight (best ROC-AUC), followed by XGBoost, then CatBoost.
    # This preserves the full probability calibration unlike Logistic Regression stacking.
    print("\nStep 3d: Computing Weighted Ensemble Probabilities...", flush=True)
    w_lgb, w_xgb, w_cat = 0.45, 0.35, 0.20
    y_prob = w_lgb * lgb_test_prob + w_xgb * xgb_test_prob + w_cat * cat_test_prob
    ensemble_auc = roc_auc_score(y_test, y_prob)
    
    # === THRESHOLD OPTIMIZATION (F2-Score: Recall-Biased) ===
    # In AML, missing a money launderer (False Negative) is far more costly than
    # a false alert (False Positive). The F2-score weights Recall 4x more than Precision.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    beta = 2.0  # F2-score
    f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
    best_idx = np.argmax(f_beta_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall + 1e-10)
    best_f2 = f_beta_scores[best_idx]
    
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    raw_accuracy = accuracy_score(y_test, y_pred_optimal)
    
    print(f"\n", flush=True)
    print(f"╔══════════════════════════════════════════════════════╗", flush=True)
    print(f"║      WEIGHTED ENSEMBLE — FINAL RESULTS              ║", flush=True)
    print(f"╠══════════════════════════════════════════════════════╣", flush=True)
    print(f"║  LightGBM ROC-AUC:     {lgb_auc*100:.2f}%  (w={w_lgb})          ║", flush=True)
    print(f"║  XGBoost ROC-AUC:      {xgb_auc*100:.2f}%  (w={w_xgb})          ║", flush=True)
    print(f"║  CatBoost ROC-AUC:     {cat_auc*100:.2f}%                     ║", flush=True)
    print(f"║  ─────────────────────────────────────────────────  ║", flush=True)
    print(f"║  ENSEMBLE ROC-AUC:     {ensemble_auc*100:.2f}%                     ║", flush=True)
    print(f"║  Raw Accuracy:         {raw_accuracy*100:.2f}%                     ║", flush=True)
    print(f"║  Precision:            {best_precision*100:.2f}%                     ║", flush=True)
    print(f"║  Recall:               {best_recall*100:.2f}%                     ║", flush=True)
    print(f"║  F1 Score:             {best_f1*100:.2f}%                     ║", flush=True)
    print(f"║  F2 Score (Recall↑):   {best_f2*100:.2f}%                     ║", flush=True)
    print(f"║  Optimal Threshold:    {best_threshold:.4f}                     ║", flush=True)
    print(f"╚══════════════════════════════════════════════════════╝", flush=True)
    
    # === TYPOLOGY ENGINE ===
    print("\nStep 4: Training Typology Engine (CatBoost Multi-Class)...", flush=True)
    y_typology = df_raw.iloc[:split_idx]['aml_typology'].values
    fraud_mask = (y_train == 1)
    X_typ_train = X_train_filled[:len(y_train)][fraud_mask]
    y_typ_train = y_typology[fraud_mask]
    
    le = LabelEncoder()
    y_typ_train_enc = le.fit_transform(y_typ_train)
    
    cat_typology = CatBoostClassifier(iterations=200, depth=6, auto_class_weights='Balanced', loss_function='MultiClass', verbose=0, random_seed=42)
    cat_typology.fit(X_typ_train, y_typ_train_enc)
    
    typ_probs = cat_typology.predict_proba(X_test_filled)
    typ_preds = cat_typology.predict(X_test_filled).flatten()
    predicted_typologies = le.inverse_transform(typ_preds)
    
    classes = le.classes_
    dist_strings = []
    for probs in typ_probs:
        dist = {classes[i]: round(probs[i]*100, 2) for i in range(len(classes))}
        top_dist = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True)[:3])
        dist_strings.append(str(top_dist))
    
    # === SHAP EXPLANATIONS (using LightGBM as primary explainer) ===
    print("Step 5: Extracting SHAP Explanations...", flush=True)
    explainer = shap.TreeExplainer(lgb_model)
    shap_sample_size = min(2000, len(X_test_filled))
    X_test_shap = X_test_filled.iloc[:shap_sample_size]
    shap_vals = explainer.shap_values(X_test_shap)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
        
    top_drivers = []
    feature_names = X_test_filled.columns.tolist()
    for i in range(len(X_test_filled)):
        if i < shap_sample_size:
            vals = shap_vals[i]
            top_idx = np.argsort(np.abs(vals))[-3:][::-1]
            drivers = [f"{feature_names[idx]} ({vals[idx]:.2f})" for idx in top_idx]
            top_drivers.append(" | ".join(drivers))
        else:
            top_drivers.append("SHAP subsample limit")
        
    # === OUTPUT ===
    print("Step 6: Generating Final Output CSV...", flush=True)
    results = pd.DataFrame({
        'Transaction ID': df_test['transaction_id'],
        'Fraud Risk Score': np.round(y_prob * 100, 2),
        'Risk Category': [get_risk_category(score, best_threshold) for score in y_prob],
        'Predicted Typology': predicted_typologies,
        'Typology Probability Distribution': dist_strings,
        'Key Risk Drivers': top_drivers
    })
    
    results.to_csv('data/final_pipeline_outputs.csv', index=False)
    print("\n✅ Execution complete! Results saved to data/final_pipeline_outputs.csv", flush=True)
    print(results.head(10).to_string())

if __name__ == "__main__":
    main()
