import sys, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

df = load_and_merge_data('data/raw')
df = build_entity_features(df)
df, dt_cols = build_interaction_features(df)
X, y = prepare_features_for_model(df, dt_cols)

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_filled = X_train.fillna(0).astype('float64')
X_test_filled = X_test.fillna(0).astype('float64')

lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=1.0)
lgb_model.fit(X_train_filled, y_train)
y_prob = lgb_model.predict_proba(X_test_filled)[:, 1]

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"No SMOTE - Threshold: {thresh:.2f} | Accuracy: {acc:.4f}")

