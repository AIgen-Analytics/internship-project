import sys, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

print("Loading data...")
df = load_and_merge_data('data/raw')
df = build_entity_features(df)
df, dt_cols = build_interaction_features(df)
X, y = prepare_features_for_model(df, dt_cols)

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_filled = X_train.fillna(0).astype('float64')
X_test_filled = X_test.fillna(0).astype('float64')

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_filled, y_train)

print("Training model...")
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train_smote, y_train_smote)
y_prob = lgb_model.predict_proba(X_test_filled)[:, 1]

best_acc = 0
best_thresh = 0
for thresh in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Threshold: {thresh:.2f} | Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_thresh:.2f}")
