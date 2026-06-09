import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model
from src.evaluation.evaluate import evaluate_risk_model

def train_risk_model(data_dir='data/raw', model_out='outputs/models/lgbm_risk_model.pkl'):
    print("Loading data...")
    df = load_and_merge_data(data_dir)
    print("Building features...")
    df = build_entity_features(df)
    df, dt_cols = build_interaction_features(df)
    X, y = prepare_features_for_model(df, dt_cols)
    
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=20, num_leaves=1024,
        min_child_samples=5, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(100)])
    
    print("Evaluating...")
    evaluate_risk_model(clf, X_test, y_test)
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")

if __name__ == '__main__':
    train_risk_model()
