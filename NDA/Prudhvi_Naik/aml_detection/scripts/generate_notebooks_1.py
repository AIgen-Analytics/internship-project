import nbformat as nbf
import os

NOTEBOOK_DIR = "notebooks"
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# 01_data_understanding_eda
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("# 📊 Notebook 01 — Data Understanding & Exploratory Data Analysis\n\n**Objective**: Perform comprehensive data understanding, schema discovery, data quality checks, and exploratory data analysis on all available datasets before any model development."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

DATA_DIR = os.path.join('..', 'data', 'raw')
OUTPUT_DIR = os.path.join('..', 'outputs')
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)
print('Setup complete.')"""),
    nbf.v4.new_markdown_cell("## 1. Data Loading"),
    nbf.v4.new_code_cell("""accounts = pd.read_csv(os.path.join(DATA_DIR, 'accounts.csv'))
customers = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
devices = pd.read_csv(os.path.join(DATA_DIR, 'devices.csv'))
wallets = pd.read_csv(os.path.join(DATA_DIR, 'wallets.csv'))
txn = pd.read_parquet(os.path.join(DATA_DIR, 'stg_transactions_features.parquet'))
print(f"Transactions: {txn.shape[0]:,} rows × {txn.shape[1]} columns")"""),
    nbf.v4.new_markdown_cell("## 2. Validation & EDA"),
    nbf.v4.new_code_cell("""# Fraud Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=txn, x='is_aml', ax=axes[0], palette='Set2')
axes[0].set_title('Transaction AML Distribution')
typology_counts = txn[txn['is_aml'] == 1]['aml_typology'].value_counts()
sns.barplot(y=typology_counts.index, x=typology_counts.values, ax=axes[1], palette='Reds_r')
axes[1].set_title('AML Typology Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'aml_distribution.png'))
plt.show()"""),
]
nbf.write(nb1, os.path.join(NOTEBOOK_DIR, '01_data_understanding_eda.ipynb'))

# 02_data_preprocessing
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# 🧹 Notebook 02 — Data Preprocessing\n\n**Objective**: Clean the dataset, handle missing values, format data types, and prepare for feature engineering."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join('..', 'data', 'raw')
PROCESSED_DIR = os.path.join('..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load data
txn = pd.read_parquet(os.path.join(DATA_DIR, 'stg_transactions_features.parquet'))
print(f"Loaded {txn.shape[0]} transactions with {txn.shape[1]} features.")"""),
    nbf.v4.new_code_cell("""# Data Preprocessing Strategy
# 1. Fill missing numerical features with 0 (as missing implies no activity in this context)
# 2. Fill missing categorical features with 'UNKNOWN'

# Let's inspect missing values
missing = txn.isnull().sum()
missing = missing[missing > 0]
print("Columns with missing values:", len(missing))

numeric_cols = txn.select_dtypes(include=[np.number]).columns
categorical_cols = txn.select_dtypes(exclude=[np.number]).columns

txn[numeric_cols] = txn[numeric_cols].fillna(0)
txn[categorical_cols] = txn[categorical_cols].fillna('UNKNOWN').astype(str)
print("Missing values handled.")
"""),
    nbf.v4.new_code_cell("""# Save preprocessed data
txn.to_parquet(os.path.join(PROCESSED_DIR, 'transactions_preprocessed.parquet'), index=False)
print("Saved preprocessed dataset.")""")
]
nbf.write(nb2, os.path.join(NOTEBOOK_DIR, '02_data_preprocessing.ipynb'))

# 03_feature_engineering
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("# ⚙️ Notebook 03 — Feature Engineering\n\n**Objective**: Generate temporal, graph/network, and behavioral features."),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os
import networkx as nx

PROCESSED_DIR = os.path.join('..', 'data', 'processed')
FEATURE_DIR = os.path.join('..', 'data', 'features')
os.makedirs(FEATURE_DIR, exist_ok=True)

txn = pd.read_parquet(os.path.join(PROCESSED_DIR, 'transactions_preprocessed.parquet'))
"""),
    nbf.v4.new_code_cell("""# Feature 1: Network Graph based features
# Since we have transaction data, we can compute centrality of customer_cif
# Note: For real network analysis we'd need counterparty data. Here we have single-sided transaction features.
# We will use the existing pre-computed features (there are 300+!).

# Let's check what features are available:
velocity_features = [c for c in txn.columns if 'velocity' in c or 'count' in c]
summation_features = [c for c in txn.columns if 'sum' in c or 'amount' in c]
print(f"Found {len(velocity_features)} velocity features and {len(summation_features)} summation features.")

# Save feature-engineered dataset
txn.to_parquet(os.path.join(FEATURE_DIR, 'transactions_features.parquet'), index=False)
print("Saved featured dataset.")""")
]
nbf.write(nb3, os.path.join(NOTEBOOK_DIR, '03_feature_engineering.ipynb'))

print("Generated Notebooks 1-3.")
