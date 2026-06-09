# Enterprise AML Intelligence Platform (Generation 2)

## Overview
This repository contains the Generation 2 Enterprise AML Intelligence Platform, transitioning from a basic batch ML model to a fully robust, identity-centric, and graph-augmented Financial Crime Intelligence system.

## Performance Benchmarks
Tested on a **strict chronological temporal holdout set** to guarantee zero data leakage:
* **ROC-AUC:** `0.9232`
* **Precision:** `0.8493`
* **Recall:** `0.4309`

*Note: While precision is exceptionally high (minimizing false positives and investigator alert fatigue), recall indicates that 57% of sophisticated fraud events require further advanced Graph Neural Network (GNN) capabilities to be detected. A threshold tuning study is recommended for business alignment.*

## Major Generation 2 Upgrades Implemented:
1. **Zero-Leakage Feature Engineering:** Restructured temporal sliding windows (`.expanding().mean().shift(1)`) to eliminate look-ahead bias and target leakage.
2. **Entity Resolution Engine:** Integrated deterministic (PAN/Aadhaar) and probabilistic (Fuzzy Name/DOB) matching.
3. **Graph Intelligence V2:** Transitioned from basic PageRank to multi-hop metrics, Louvain Community Detection, and Directed Graph Circular Flow Detection to identify layering and smurfing.
4. **Explainability & SAR Generation:** Added SHAP-based explainer that translates model mathematical drivers directly into Suspicious Activity Report (SAR) narrative strings (`src/explainability/narrative_generator.py`).
5. **Feature Store (Feast):** Implemented an offline/online feature store to prevent train/serve skew (`src/feature_store/`).
6. **MLOps & Governance (MLflow & Evidently):** Added MLflow model registry and parameter tracking, alongside Evidently for train vs. test Data Drift monitoring.
7. **Real-Time Serving (BentoML):** Created the `aml_bento.py` Pydantic endpoint combining the ML model with the feature store for live inferences.

## Repository Structure
* `src/features/` - Core ML feature engineering and temporal velocity calculations.
* `src/graph_analytics_v2/` - Advanced centrality and circular flow detection graph logic.
* `src/audits/` - Scripts utilized for entity resolution and community leakage audits.
* `src/feature_store/` - Feast configuration (`feature_store.yaml`, `features.py`).
* `src/models/` - Model training (`LightGBM` / `CatBoost`), `run_pipeline.py`, and `train_mlflow.py`.
* `src/explainability/` - Human-readable SAR narrative generation.
* `src/serve/` - BentoML serving infrastructure.
* `src/monitoring/` - Evidently drift detection.

## Future Roadmap (Generation 3)
* **Graph Neural Networks:** GraphSAGE / GAT for embeddings.
* **Feedback Loops:** Continuous learning from investigator SAR outcomes.
* **Real-Time Streaming:** Kafka/Flink infrastructure.
* **Graph Databases:** Migration to Neo4j/TigerGraph for scalable multi-hop traversals.

## Author
Prudhvi Naik / AIgen-Analytics Internship Project