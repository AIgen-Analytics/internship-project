from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import shap

app = FastAPI(title="AML Transaction Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
clf_risk = None
clf_typology = None
le_typology = None
explainer = None
demo_db = None
training_cols = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

@app.on_event("startup")
def load_assets():
    global clf_risk, clf_typology, le_typology, explainer, demo_db, training_cols
    print("Loading models and demo database...")
    try:
        # clf_risk is now an Ensemble VotingClassifier
        clf_risk = joblib.load(os.path.join(MODEL_DIR, 'lgbm_risk_model.pkl'))
        
        # Load single LightGBM specifically for SHAP explanations
        clf_lgbm_only = joblib.load(os.path.join(MODEL_DIR, 'lgbm_shap_model.pkl'))
        explainer = shap.TreeExplainer(clf_lgbm_only)
        
        clf_typology = joblib.load(os.path.join(MODEL_DIR, 'lgbm_typology_model.pkl'))
        le_typology = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        training_cols = joblib.load(os.path.join(MODEL_DIR, 'training_cols.pkl'))
        
        demo_db = pd.read_parquet(os.path.join(ASSETS_DIR, 'demo_db.parquet'))
        print("✅ Models and demo database loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load models or demo DB: {e}")

@app.get("/health")
def health_check():
    return {"status": "online", "models_loaded": clf_risk is not None}

@app.get("/transactions")
def get_transactions():
    if demo_db is None:
        raise HTTPException(status_code=500, detail="Demo DB not loaded")
    
    fraud_txns = demo_db[demo_db['actual_is_aml'] == 1].head(10)[['transaction_id', 'actual_is_aml', 'actual_typology']].to_dict(orient='records')
    legit_txns = demo_db[demo_db['actual_is_aml'] == 0].head(10)[['transaction_id', 'actual_is_aml', 'actual_typology']].to_dict(orient='records')
    
    return {"transactions": fraud_txns + legit_txns}

class PredictRequest(BaseModel):
    transaction_id: str

@app.post("/predict")
def predict_transaction(req: PredictRequest):
    if demo_db is None or clf_risk is None:
        raise HTTPException(status_code=500, detail="System not fully initialized.")
        
    row = demo_db[demo_db['transaction_id'] == req.transaction_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Transaction not found in demo database.")
        
    features_df = row.drop(columns=['transaction_id', 'actual_is_aml', 'actual_typology'])
    
    # Align features exactly to training_cols to prevent any shape mismatch
    if training_cols:
        for col in training_cols:
            if col not in features_df.columns:
                features_df[col] = -999
        features_df = features_df[training_cols]
    
    # 1. Risk Prediction using the Ensemble
    risk_prob = clf_risk.predict_proba(features_df)[0][1] * 100
    if risk_prob >= 80:
        category = "High Risk"
    elif risk_prob >= 40:
        category = "Medium Risk"
    else:
        category = "Low Risk"
        
    # 2. Typology Prediction
    typology_probs = []
    predicted_typology = "None"
    
    if risk_prob >= 40:
        t_probs = clf_typology.predict_proba(features_df)[0]
        top_idx = np.argmax(t_probs)
        predicted_typology = le_typology.inverse_transform([top_idx])[0]
        
        for i, class_name in enumerate(le_typology.classes_):
            typology_probs.append({
                "typology": class_name,
                "probability": float(t_probs[i] * 100)
            })
            
    typology_probs = sorted(typology_probs, key=lambda x: x["probability"], reverse=True)
            
    # 3. SHAP Explainability (using underlying LightGBM)
    shap_vals = explainer.shap_values(features_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
        
    shap_vals_row = shap_vals[0]
    
    feature_impacts = []
    for i, col in enumerate(features_df.columns):
        if abs(shap_vals_row[i]) > 0.05:
            feature_impacts.append({
                "feature": col,
                "impact": float(shap_vals_row[i]),
                "abs_impact": abs(float(shap_vals_row[i])),
                "actual_value": float(features_df[col].iloc[0]) if pd.api.types.is_numeric_dtype(features_df[col]) else str(features_df[col].iloc[0])
            })
            
    feature_impacts = sorted(feature_impacts, key=lambda x: x["abs_impact"], reverse=True)[:5]
    
    return {
        "transaction_id": req.transaction_id,
        "fraud_risk_score": float(risk_prob),
        "risk_category": category,
        "predicted_typology": predicted_typology,
        "typology_probabilities": typology_probs,
        "key_risk_drivers": feature_impacts,
        "actual_is_aml": int(row['actual_is_aml'].iloc[0]),
        "actual_typology": str(row['actual_typology'].iloc[0])
    }
