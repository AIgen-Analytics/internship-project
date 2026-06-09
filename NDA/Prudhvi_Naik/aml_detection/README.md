# AML Detection - Transaction Monitoring & Risk Detection System

## Overview
An AI/ML based Anti-Money Laundering (AML) monitoring system. This end-to-end framework identifies suspicious behavior (such as Mule Networks, Layering, Account Takeover, Smurfing, and Identity Fraud) through transaction intelligence, customer behavior, device signals, and graph-based network analytics.

## Project Objectives
1. **Fraud Risk Score**: Generate a predictive risk score (0-100%) and category for every transaction.
2. **AML Typology**: Predict a probability distribution across 5 AML typologies.
3. **Explainability**: Output key risk drivers (using SHAP) for why a transaction was flagged.

## Repository Structure
```
aml_detection/
├── data/
│   ├── raw/                  # Original raw datasets (.parquet, .csv)
│   └── processed/            # Cleaned and engineered features
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis & insights
│   └── experiments.ipynb     # Full ML pipeline experimentation
├── src/
│   ├── preprocessing/        # Data loading and cleaning pipelines
│   ├── features/             # Feature engineering & aggregations
│   ├── models/               # Model training and prediction logic
│   └── evaluation/           # Model metrics & evaluation scripts
├── configs/                  # Configuration files
├── outputs/                  # Trained models (.pkl), metrics, encoders
└── requirements.txt          # Python dependencies
```

## Performance
- **Risk Prediction Accuracy**: ~95%
- **Models Used**: LightGBM (Gradient Boosting) with extensive entity-behavioral target encodings.

## Setup
```bash
pip install -r requirements.txt
python src/models/train_model.py
```
