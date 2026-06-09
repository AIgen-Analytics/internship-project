import nbformat as nbf
import os

NOTEBOOK_DIR = "notebooks"
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# 04_feature_selection
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("# 🎯 Notebook 04 — Feature Selection\n\n**Objective**: Reduce dimensionality and select the most predictive features for the fraud model."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

FEATURE_DIR = os.path.join('..', 'data', 'features')
txn = pd.read_parquet(os.path.join(FEATURE_DIR, 'transactions_features.parquet'))

print(f"Loaded dataset with {txn.shape[1]} features.")"""),
    nbf.v4.new_code_cell("""# We will drop IDs and target for feature selection
drop_cols = ['transaction_id', 'customer_cif_id', 'customer_account_number', 'device_id_fingerprint', 'wallet_account_id', 'is_aml', 'aml_typology', 'fraud_intensity_score', 'fis_band']
features = [c for c in txn.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(txn[c])]

X = txn[features].fillna(0)
y = txn['is_aml']

print(f"Selecting from {len(features)} numerical features...")
# Using a fast method: correlation with target or Random Forest Feature Importance
# Here we will use a small sample to compute RF feature importance for speed
X_sample = X.sample(n=min(50000, len(X)), random_state=42)
y_sample = y.loc[X_sample.index]

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_sample, y_sample)

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
top_50_features = importances.head(50).index.tolist()
print("Top 10 features:")
print(importances.head(10))

# Save selected features dataset
selected_cols = drop_cols + top_50_features
txn[selected_cols].to_parquet(os.path.join(FEATURE_DIR, 'transactions_selected.parquet'), index=False)
print("Saved feature-selected dataset.")""")
]
nbf.write(nb4, os.path.join(NOTEBOOK_DIR, '04_feature_selection.ipynb'))

# 05_fraud_risk_model
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("# 🤖 Notebook 05 — Fraud Risk Model (Binary Classification)\n\n**Objective**: Train an ML model (XGBoost/LightGBM) to predict `is_aml`."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_DIR = os.path.join('..', 'data', 'features')
MODEL_DIR = os.path.join('..', 'outputs', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

txn = pd.read_parquet(os.path.join(FEATURE_DIR, 'transactions_selected.parquet'))
drop_cols = ['transaction_id', 'customer_cif_id', 'customer_account_number', 'device_id_fingerprint', 'wallet_account_id', 'is_aml', 'aml_typology', 'fraud_intensity_score', 'fis_band']
features = [c for c in txn.columns if c not in drop_cols]

X = txn[features]
y = txn['is_aml']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape:", X_train.shape)"""),
    nbf.v4.new_code_cell("""# Train LightGBM Model
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification Report:\\n", classification_report(y_test, y_pred))

# Save Model
joblib.dump(clf, os.path.join(MODEL_DIR, 'lgbm_is_aml.pkl'))
print("Saved Fraud Risk Model.")""")
]
nbf.write(nb5, os.path.join(NOTEBOOK_DIR, '05_fraud_risk_model.ipynb'))

# 06_typology_classification
nb6 = nbf.v4.new_notebook()
nb6.cells = [
    nbf.v4.new_markdown_cell("# 🕵️ Notebook 06 — Typology Classification (Multi-class)\n\n**Objective**: Train a multi-class model to predict `aml_typology` for transactions flagged as AML."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

FEATURE_DIR = os.path.join('..', 'data', 'features')
MODEL_DIR = os.path.join('..', 'outputs', 'models')

txn = pd.read_parquet(os.path.join(FEATURE_DIR, 'transactions_selected.parquet'))
# Only train on AML transactions for typology classification
aml_txn = txn[txn['is_aml'] == 1].copy()

drop_cols = ['transaction_id', 'customer_cif_id', 'customer_account_number', 'device_id_fingerprint', 'wallet_account_id', 'is_aml', 'aml_typology', 'fraud_intensity_score', 'fis_band']
features = [c for c in aml_txn.columns if c not in drop_cols]

X = aml_txn[features]
le = LabelEncoder()
y = le.fit_transform(aml_txn['aml_typology'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape:", X_train.shape)"""),
    nbf.v4.new_code_cell("""# Train LightGBM Multi-class Model
clf_multi = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, objective='multiclass')
clf_multi.fit(X_train, y_train)

# Evaluate
y_pred = clf_multi.predict(X_test)
print("Classification Report:\\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save Model and Label Encoder
joblib.dump(clf_multi, os.path.join(MODEL_DIR, 'lgbm_typology.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder_typology.pkl'))
print("Saved Typology Model.")""")
]
nbf.write(nb6, os.path.join(NOTEBOOK_DIR, '06_typology_classification.ipynb'))

# 07_explainability
nb7 = nbf.v4.new_notebook()
nb7.cells = [
    nbf.v4.new_markdown_cell("# 🔍 Notebook 07 — Explainability (SHAP)\n\n**Objective**: Generate global and local explanations for the Fraud Risk Model using SHAP values."),
    nbf.v4.new_code_cell("""import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt

FEATURE_DIR = os.path.join('..', 'data', 'features')
MODEL_DIR = os.path.join('..', 'outputs', 'models')
OUTPUT_DIR = os.path.join('..', 'outputs')

txn = pd.read_parquet(os.path.join(FEATURE_DIR, 'transactions_selected.parquet'))
clf = joblib.load(os.path.join(MODEL_DIR, 'lgbm_is_aml.pkl'))

drop_cols = ['transaction_id', 'customer_cif_id', 'customer_account_number', 'device_id_fingerprint', 'wallet_account_id', 'is_aml', 'aml_typology', 'fraud_intensity_score', 'fis_band']
features = [c for c in txn.columns if c not in drop_cols]
X = txn[features].sample(n=5000, random_state=42) # Sample for SHAP computation

# Calculate SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

print("SHAP values calculated.")"""),
    nbf.v4.new_code_cell("""# Global Explainability: SHAP Summary Plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reports', 'shap_summary_bar.png'))
plt.show()

# SHAP Feature Impact
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reports', 'shap_summary_impact.png'))
plt.show()""")
]
nbf.write(nb7, os.path.join(NOTEBOOK_DIR, '07_explainability.ipynb'))

# 08_final_inference_pipeline
nb8 = nbf.v4.new_notebook()
nb8.cells = [
    nbf.v4.new_markdown_cell("# 🚀 Notebook 08 — Final Inference Pipeline\n\n**Objective**: Build a single pipeline function that takes raw transaction data, preprocesses it, and outputs Fraud Risk Score + Typologies."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = os.path.join('..', 'outputs', 'models')

# Load models
clf_is_aml = joblib.load(os.path.join(MODEL_DIR, 'lgbm_is_aml.pkl'))
clf_typology = joblib.load(os.path.join(MODEL_DIR, 'lgbm_typology.pkl'))
le_typology = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_typology.pkl'))

# Model features
expected_features = clf_is_aml.feature_name_

def predict_transaction(transaction_df):
    # Preprocessing & Selection
    X = transaction_df.reindex(columns=expected_features).fillna(0)
    
    # 1. Predict AML Probability
    fraud_prob = clf_is_aml.predict_proba(X)[:, 1]
    
    # 2. Predict Typology Probabilities
    typology_probs = clf_typology.predict_proba(X)
    
    results = []
    for i in range(len(transaction_df)):
        txn_res = {
            'transaction_id': transaction_df.iloc[i].get('transaction_id', f'txn_{i}'),
            'fraud_risk_score': round(fraud_prob[i] * 100, 2),
            'risk_category': 'HIGH' if fraud_prob[i] > 0.8 else ('MEDIUM' if fraud_prob[i] > 0.4 else 'LOW'),
            'predicted_typology': le_typology.classes_[np.argmax(typology_probs[i])] if fraud_prob[i] > 0.5 else 'None'
        }
        
        if fraud_prob[i] > 0.5:
            # Top 3 typologies
            top_3_idx = np.argsort(typology_probs[i])[-3:][::-1]
            typologies = {le_typology.classes_[idx]: round(typology_probs[i][idx]*100, 2) for idx in top_3_idx}
            txn_res['typology_probabilities'] = typologies
        else:
            txn_res['typology_probabilities'] = {}
            
        results.append(txn_res)
        
    return results

print("Inference Pipeline Ready.")"""),
    nbf.v4.new_code_cell("""# Test the pipeline
DATA_DIR = os.path.join('..', 'data', 'raw')
sample_txn = pd.read_parquet(os.path.join(DATA_DIR, 'stg_transactions_features.parquet')).sample(5, random_state=42)

predictions = predict_transaction(sample_txn)
for p in predictions:
    print(p)""")
]
nbf.write(nb8, os.path.join(NOTEBOOK_DIR, '08_final_inference_pipeline.ipynb'))

print("Generated Notebooks 4-8.")
