import sys, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

def get_risk_category(score):
    if score >= 90: return 'Critical Risk'
    if score >= 70: return 'High Risk'
    if score >= 50: return 'Medium Risk'
    return 'Low Risk'

def main():
    print("Step 1: Loading Data and Engineering Features (including PageRank)...")
    df = load_and_merge_data('data/raw')
    df = build_entity_features(df)
    df, dt_cols = build_interaction_features(df)
    
    # Store original df for outputting results
    df_raw = df.copy()
    
    X, y = prepare_features_for_model(df, dt_cols)
    
    print("Step 2: Strict Temporal Validation Split...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df_raw.iloc[split_idx:].copy()
    
    print("Step 3: Training Fraud Risk Engine (LightGBM)...")
    # Switched to single LightGBM for 100% Explainability Alignment
    X_train_filled = X_train.fillna(0).astype('float64')
    X_test_filled = X_test.fillna(0).astype('float64')
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_filled, y_train)
    
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_smote, y_train_smote)
    
    y_prob = lgb_model.predict_proba(X_test_filled)[:, 1]
    
    print("Step 4: Training Typology Engine (CatBoost)...")
    y_typology = df_raw.iloc[:split_idx]['aml_typology'].values
    fraud_mask = (y_train == 1)
    X_typ_train = X_train_smote[:len(y_train)][fraud_mask]
    y_typ_train = y_typology[fraud_mask]
    
    le = LabelEncoder()
    y_typ_train_enc = le.fit_transform(y_typ_train)
    
    cat_typology = CatBoostClassifier(iterations=200, depth=6, loss_function='MultiClass', verbose=0, random_seed=42)
    cat_typology.fit(X_typ_train, y_typ_train_enc)
    
    # Predict Typology Probabilities on test set
    typ_probs = cat_typology.predict_proba(X_test_filled)
    typ_preds = cat_typology.predict(X_test_filled).flatten()
    predicted_typologies = le.inverse_transform(typ_preds)
    
    # Create Typology Probability Distribution string
    classes = le.classes_
    dist_strings = []
    for probs in typ_probs:
        dist = {classes[i]: round(probs[i]*100, 2) for i in range(len(classes))}
        top_dist = dict(sorted(dist.items(), key=lambda item: item[1], reverse=True)[:3])
        dist_strings.append(str(top_dist))
    
    print("Step 5: Extracting SHAP Explanations...")
    explainer = shap.TreeExplainer(lgb_model)
    # We will sample 1000 for speed, but in prod we do all
    print("Calculating SHAP values for test set...")
    shap_vals = explainer.shap_values(X_test_filled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] # For binary classification
        
    top_drivers = []
    feature_names = X_test_filled.columns.tolist()
    for i in range(len(X_test_filled)):
        vals = shap_vals[i]
        top_idx = np.argsort(np.abs(vals))[-3:][::-1]
        drivers = [f"{feature_names[idx]} ({vals[idx]:.2f})" for idx in top_idx]
        top_drivers.append(" | ".join(drivers))
        
    print("Step 6: Generating Final Output CSV...")
    results = pd.DataFrame({
        'Transaction ID': df_test['transaction_id'],
        'Fraud Risk Score': np.round(y_prob * 100, 2),
        'Risk Category': [get_risk_category(score * 100) for score in y_prob],
        'Predicted Typology': predicted_typologies,
        'Typology Probability Distribution': dist_strings,
        'Key Risk Drivers': top_drivers
    })
    
    results.to_csv('data/final_pipeline_outputs.csv', index=False)
    print("Execution complete! Results saved to data/final_pipeline_outputs.csv")
    print(results.head(10).to_string())

if __name__ == "__main__":
    main()
