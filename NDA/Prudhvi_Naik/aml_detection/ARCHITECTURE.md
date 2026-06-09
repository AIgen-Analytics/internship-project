# System Architecture
## 1. Ingestion Layer
Loads 5 core datasets (Transactions, Accounts, Customers, Devices, Wallets). Pre-calculates temporal splits.

## 2. Feature Store Layer
Calculates historical expanding windows, 24-hour/7-day burst velocities, geo-distance drift, and NetworkX Graph Centrality (PageRank).

## 3. Modeling Layer
- **Fraud Risk Engine:** LightGBM classifier with SMOTE sampling optimized for High Precision / Low False Positives.
- **Typology Engine:** CatBoost MultiClass classifier optimized for detecting 12 distinct laundering typologies.

## 4. Serving Layer
Outputs to CSV/API providing: Transaction ID, Risk Score, Risk Category, Predicted Typology, Typology Probability Distribution, and Key Risk Drivers (SHAP).