# AML Methodology
## Risk Scoring Framework
- **0-50 (Low Risk):** Straight-Through Processing.
- **50-70 (Medium Risk):** Retrospective periodic review.
- **70-90 (High Risk):** Active alert generation.
- **90-100 (Critical Risk):** Immediate automated account freeze.

## Explainable Output
Regulators mandate transparency. Every score > 70 is accompanied by the Top 3 SHAP drivers (e.g., `cust_pagerank (2.45) | cust_7d_velocity (1.12)`), giving investigators immediate context on why the alert fired.