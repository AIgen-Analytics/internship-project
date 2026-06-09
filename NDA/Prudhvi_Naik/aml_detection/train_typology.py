import sys, os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.preprocessing.data_loader import load_and_merge_data
from src.features.build_features import build_entity_features, build_interaction_features, prepare_features_for_model

print("Loading data...")
df = load_and_merge_data('data/raw')
df = build_entity_features(df)
df, dt_cols = build_interaction_features(df)
X, y = prepare_features_for_model(df, dt_cols)
training_cols = list(X.columns)

print(f"X shape: {X.shape}")

fraud_mask = (y == 1)
X_fraud = X[fraud_mask]
y_typology = df.loc[X_fraud.index, 'aml_typology']

le = LabelEncoder()
y_typ_encoded = le.fit_transform(y_typology)

print("Training Typology Model on", X_fraud.shape[1], "features")
clf_typology = lgb.LGBMClassifier(n_estimators=100, random_state=42)
clf_typology.fit(X_fraud, y_typ_encoded)

os.makedirs('outputs/models', exist_ok=True)
joblib.dump(clf_typology, 'outputs/models/lgbm_typology_model.pkl')
joblib.dump(le, 'outputs/models/label_encoder.pkl')

print("Typology model retrained and saved successfully!")
