from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
import zcatalyst_sdk

app = FastAPI(
    title="AML Transaction Monitoring API",
    description="Real-time ML prediction API for Anti-Money Laundering transaction risk scoring.",
    version="1.0.0"
)

# CORS middleware for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL MODEL VARIABLES ---
clf_is_aml = None
clf_typology = None
le_typology = None
expected_features = []

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

@app.on_event("startup")
def load_models():
    global clf_is_aml, clf_typology, le_typology, expected_features
    try:
        risk_model_id = os.environ.get("CATALYST_RISK_MODEL_ID", "111")
        typology_model_id = os.environ.get("CATALYST_TYPOLOGY_MODEL_ID", "222")
        le_model_id = os.environ.get("CATALYST_LE_MODEL_ID", "333")
        
        # For local dev / fallback, we load from disk if Catalyst IDs are not strictly set
        if risk_model_id == "111":
            clf_is_aml = joblib.load(os.path.join(MODEL_DIR, 'lgbm_is_aml.pkl'))
            clf_typology = joblib.load(os.path.join(MODEL_DIR, 'lgbm_typology.pkl'))
            le_typology = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_typology.pkl'))
            print("⚠️ Loaded models from local disk fallback.")
        else:
            # Initialize Catalyst SDK
            catalyst_app = zcatalyst_sdk.initialize()
            filestore = catalyst_app.filestore()
            folder_id = os.environ.get("CATALYST_MODEL_FOLDER_ID", "1234567890")
            folder = filestore.folder(int(folder_id))
            
            print("⬇️ Downloading models from Catalyst File Store...")
            with open("/tmp/lgbm_is_aml.pkl", "wb") as f:
                f.write(folder.file(int(risk_model_id)).download())
            with open("/tmp/lgbm_typology.pkl", "wb") as f:
                f.write(folder.file(int(typology_model_id)).download())
            with open("/tmp/label_encoder_typology.pkl", "wb") as f:
                f.write(folder.file(int(le_model_id)).download())
            
            clf_is_aml = joblib.load("/tmp/lgbm_is_aml.pkl")
            clf_typology = joblib.load("/tmp/lgbm_typology.pkl")
            le_typology = joblib.load("/tmp/label_encoder_typology.pkl")
            print("✅ Models loaded from Catalyst File Store.")

        expected_features = clf_is_aml.feature_name_
        print(f"✅ Models initialized. Expected {len(expected_features)} features.")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")

class TransactionRequest(BaseModel):
    transaction_id: str
    features: Dict[str, Any]

class TypologyProbability(BaseModel):
    typology: str
    probability_percentage: float

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_risk_score: float
    risk_category: str
    predicted_typology: str
    typology_probabilities: List[TypologyProbability]

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": clf_is_aml is not None
    }

@app.post("/train", summary="Train AML Models on Catalyst AppSail")
def train_models_on_catalyst():
    """
    Trains the LightGBM models completely on Zoho Catalyst infrastructure.
    It reads the processed dataset, trains the model, and saves the artifacts
    back to Catalyst File Store.
    """
    try:
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        
        # Load dataset
        data_path = os.path.join(os.path.dirname(__file__), 'assets', 'transactions_preprocessed.parquet')
        df = pd.read_parquet(data_path)
        
        # Prepare Data
        X = df.drop(columns=['transaction_id', 'is_aml', 'typology'])
        y_risk = df['is_aml']
        
        # Train Risk Model
        clf_risk = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
        clf_risk.fit(X, y_risk)
        joblib.dump(clf_risk, '/tmp/lgbm_is_aml.pkl')
        
        # Train Typology Model
        df_fraud = df[df['is_aml'] == 1].copy()
        le = LabelEncoder()
        df_fraud['typology_encoded'] = le.fit_transform(df_fraud['typology'])
        X_typology = df_fraud.drop(columns=['transaction_id', 'is_aml', 'typology', 'typology_encoded'])
        y_typology = df_fraud['typology_encoded']
        
        clf_typ = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
        clf_typ.fit(X_typology, y_typology)
        joblib.dump(clf_typ, '/tmp/lgbm_typology.pkl')
        joblib.dump(le, '/tmp/label_encoder_typology.pkl')
        
        # Upload to Catalyst File Store
        catalyst_app = zcatalyst_sdk.initialize()
        filestore = catalyst_app.filestore()
        folder_id = os.environ.get("CATALYST_MODEL_FOLDER_ID", "1234567890")
        folder = filestore.folder(int(folder_id))
        
        folder.upload_file('/tmp/lgbm_is_aml.pkl')
        folder.upload_file('/tmp/lgbm_typology.pkl')
        folder.upload_file('/tmp/label_encoder_typology.pkl')
        
        # Reload models into active memory
        load_models()
        
        return {"status": "success", "message": "Models trained on Catalyst and saved to File Store."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-risk", response_model=PredictionResponse)
def predict_transaction(req: TransactionRequest):
    if clf_is_aml is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    try:
        input_data = {}
        for f in expected_features:
            input_data[f] = req.features.get(f, 0)
            
        df = pd.DataFrame([input_data])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        risk_prob = clf_is_aml.predict_proba(df)[0][1]
        score = round(risk_prob * 100, 2)
        
        if score >= 80:
            category = "HIGH"
        elif score >= 50:
            category = "MEDIUM"
        else:
            category = "LOW"
            
        predicted_typ = "None"
        typology_probs = []
        
        if score > 50:
            typ_probs = clf_typology.predict_proba(df)[0]
            top_class_idx = np.argmax(typ_probs)
            if le_typology:
                predicted_typ = le_typology.inverse_transform([top_class_idx])[0]
                classes = le_typology.classes_
                for i, prob in enumerate(typ_probs):
                    typology_probs.append(
                        {"typology": classes[i], "probability_percentage": round(prob * 100, 2)}
                    )
                
        # --- Store Prediction in Catalyst Data Store ---
        try:
            if "CATALYST_RISK_MODEL_ID" not in os.environ:
                print("⚠️ Skipping Catalyst Data Store insert (Local Dev Mode)")
            else:
                catalyst_app = zcatalyst_sdk.initialize()
                datastore = catalyst_app.datastore()
                
                # Insert Risk Prediction
                risk_table = datastore.table('risk_predictions')
                risk_table.insert_row({
                    "transaction_id": req.transaction_id,
                    "fraud_risk_score": score,
                    "risk_category": category
                })
                
                # Insert Typology Prediction if high risk
                if score > 50:
                    typology_table = datastore.table('typology_predictions')
                    typology_table.insert_row({
                        "transaction_id": req.transaction_id,
                        "predicted_typology": predicted_typ,
                        "probability_score": score
                    })
        except Exception as ds_error:
            print(f"⚠️ Failed to insert prediction to Data Store: {ds_error}")

        return PredictionResponse(
            transaction_id=req.transaction_id,
            fraud_risk_score=score,
            risk_category=category,
            predicted_typology=predicted_typ,
            typology_probabilities=typologies_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/network-analysis/{transaction_id}")
async def get_network_analysis(transaction_id: str):
    # Dummy logic representing graph generation.
    # In a real scenario, we query graph database (or Catalyst Data Store) for hops.
    import networkx as nx
    from networkx.readwrite import json_graph
    
    G = nx.DiGraph()
    G.add_node("C1", type="customer", risk=80)
    G.add_node("A1", type="account", risk=90)
    G.add_node("C2", type="customer", risk=20)
    G.add_node(transaction_id, type="transaction", amount=50000)
    
    G.add_edge("C1", "A1", relation="OWNS")
    G.add_edge("A1", transaction_id, relation="SENDER")
    G.add_edge(transaction_id, "C2", relation="RECEIVER")
    
    # Mule Ring simulation
    G.add_node("M1", type="account", risk=99)
    G.add_edge("C2", "M1", relation="TRANSFERS")
    G.add_edge("M1", "C1", relation="CIRCULAR_RETURN")
    
    data = json_graph.node_link_data(G)
    return {"status": "success", "graph": data, "detected_patterns": ["Circular Transfer", "Mule Ring Indicator"]}

@app.get("/explainability/{transaction_id}")
async def get_explainability(transaction_id: str):
    # Retrieve pre-computed SHAP explanation from Catalyst File Store
    # Here we mock the top contributing features for the dashboard
    return {
        "transaction_id": transaction_id,
        "shap_values": [
            {"feature": "transfer_amount", "contribution": 3.4},
            {"feature": "velocity_24h", "contribution": 2.1},
            {"feature": "new_device", "contribution": 1.5},
            {"feature": "account_age", "contribution": -0.8}
        ],
        "base_value": -2.1,
        "explanation": "High transfer amount and unusual 24h velocity significantly increased the fraud risk."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
