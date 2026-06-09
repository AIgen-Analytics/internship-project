import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

def train_multi_model():
    print("Loading data...")
    df = load_and_merge_data('data/raw')
    
    print("Building features...")
    df = build_entity_features(df)
    df, dt_cols = build_interaction_features(df)
    
    # We want to save transaction ID for the demo DB before dropping
    txn_ids = df['transaction_id'].values
    actual_typology = df['aml_typology'].values
    
    X, y = prepare_features_for_model(df, dt_cols)
    training_cols = list(X.columns)
    
    # Save demo db
    demo_df = X.copy()
    demo_df['transaction_id'] = txn_ids
    demo_df['actual_is_aml'] = y
    demo_df['actual_typology'] = actual_typology
    demo_sample = demo_df.sample(1000, random_state=42)
    os.makedirs('backend/assets', exist_ok=True)
    demo_sample.to_parquet('backend/assets/demo_db.parquet', index=False)
    print("Demo DB saved.")
    
    print(f"Dataset shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print("Training LightGBM...")
    clf_lgb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    clf_lgb.fit(X_train, y_train)
    
    print("Training XGBoost...")
    clf_xgb = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    clf_xgb.fit(X_train, y_train)
    
    print("Training CatBoost...")
    clf_cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=-1)
    clf_cat.fit(X_train, y_train)
    
    print("Training Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train.fillna(0), y_train) # RF requires no NaNs
    
    print("Training Ensemble (Voting Classifier)...")
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', clf_lgb),
            ('xgb', clf_xgb),
            ('cat', clf_cat)
        ],
        voting='soft'
    )
    # We skip RF in soft voting because probability calibration of RF differs significantly
    ensemble.fit(X_train, y_train)
    
    models = {'LightGBM': clf_lgb, 'XGBoost': clf_xgb, 'CatBoost': clf_cat, 'RandomForest': clf_rf, 'Ensemble': ensemble}
    
    print("\n--- Model Evaluation ---")
    best_auc = 0
    best_model = None
    for name, m in models.items():
        if name == 'RandomForest':
            preds = m.predict(X_test.fillna(0))
            probs = m.predict_proba(X_test.fillna(0))[:, 1]
        else:
            preds = m.predict(X_test)
            probs = m.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        print(f"{name:15} | Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        
        if auc > best_auc and name == 'Ensemble':
            best_auc = auc
            best_model = m
            
    print("\nSaving best ensemble risk model and training features list...")
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(ensemble, 'outputs/models/lgbm_risk_model.pkl') # Keep name so backend doesn't break
    joblib.dump(training_cols, 'outputs/models/training_cols.pkl')
    
    # Also save LightGBM separately for SHAP (SHAP doesn't support generic VotingClassifier easily)
    joblib.dump(clf_lgb, 'outputs/models/lgbm_shap_model.pkl')
    
    print("Done! Multi-model training complete.")

if __name__ == '__main__':
    train_multi_model()
