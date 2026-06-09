import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def build_entity_features(df):
    """Creates behavioral aggregations for entities using strict historical windows to prevent target leakage."""
    # Ensure chronological order to prevent future leakage
    if 'datestamp' in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['datestamp'].astype(str) + ' ' + df['timestamp'].astype(str), errors='coerce')
        df = df.sort_values(by=['datetime']).reset_index(drop=True)
    
    # Pre-calculate is_aml existence for historical aggregates
    has_target = 'is_aml' in df.columns

    # --- CUSTOMER LEVEL ---
    # Shift(1) ensures we only look at the PAST, excluding the current row's transaction amount/label
    df['cust_txn_count_hist'] = df.groupby('customer_cif_id').cumcount()
    df['cust_mean_amt_hist'] = df.groupby('customer_cif_id')['transaction_amount'].transform(lambda x: x.expanding().mean().shift(1))
    df['cust_std_amt_hist'] = df.groupby('customer_cif_id')['transaction_amount'].transform(lambda x: x.expanding().std().shift(1))
    df['cust_max_amt_hist'] = df.groupby('customer_cif_id')['transaction_amount'].transform(lambda x: x.expanding().max().shift(1))
    
    if has_target:
        df['cust_fraud_rate_hist'] = df.groupby('customer_cif_id')['is_aml'].transform(lambda x: x.expanding().mean().shift(1))
        df['cust_fraud_count_hist'] = df.groupby('customer_cif_id')['is_aml'].transform(lambda x: x.expanding().sum().shift(1))
    else:
        df['cust_fraud_rate_hist'] = 0.0
        df['cust_fraud_count_hist'] = 0.0

    df['cust_amt_cv_hist'] = df['cust_std_amt_hist'] / (df['cust_mean_amt_hist'] + 1)
    
    # --- VELOCITY (Advanced Feature Discovery) ---
    if 'datetime' in df.columns:
        df = df.sort_values(by=['customer_cif_id', 'datetime']).reset_index(drop=True)
        df_time_indexed = df.set_index('datetime')
        # Count transactions in the last 7 days per customer
        cust_7d_vel = df_time_indexed.groupby('customer_cif_id')['transaction_amount'].rolling('7D').count().values
        df['cust_7d_velocity'] = cust_7d_vel - 1
        
        # 24-hour velocity
        cust_24h_vel = df_time_indexed.groupby('customer_cif_id')['transaction_amount'].rolling('1D').count().values
        df['cust_24h_velocity'] = cust_24h_vel - 1

    # --- COUNTERPARTY LEVEL ---
    df['cp_txn_count_hist'] = df.groupby('counterparty_account_number').cumcount()
    if has_target:
        df['cp_fraud_rate_hist'] = df.groupby('counterparty_account_number')['is_aml'].transform(lambda x: x.expanding().mean().shift(1))
    else:
        df['cp_fraud_rate_hist'] = 0.0
    df['cp_mean_amt_hist'] = df.groupby('counterparty_account_number')['transaction_amount'].transform(lambda x: x.expanding().mean().shift(1))

    # --- ACCOUNT LEVEL ---
    df['acct_txn_count_hist'] = df.groupby('customer_account_number').cumcount()
    if has_target:
        df['acct_fraud_rate_hist'] = df.groupby('customer_account_number')['is_aml'].transform(lambda x: x.expanding().mean().shift(1))
    else:
        df['acct_fraud_rate_hist'] = 0.0

    # --- GRAPH/NETWORK CENTRALITY FEATURES (Pseudo-Graph) ---
    # Degree Centrality Proxy: How many unique counterparties has this customer interacted with historically?
    cust_unique_cp = df.groupby('customer_cif_id')['counterparty_account_number'].nunique()
    df['cust_degree_centrality'] = df['customer_cif_id'].map(cust_unique_cp)
    
    cp_unique_senders = df.groupby('counterparty_account_number')['customer_cif_id'].nunique()
    df['cp_degree_centrality'] = df['counterparty_account_number'].map(cp_unique_senders)

    # Advanced Graph Centrality (PageRank)
    try:
        import networkx as nx
        G = nx.Graph()
        # Build bipartite-style edges between customers and counterparties
        edges = list(zip(df['customer_cif_id'].astype(str), df['counterparty_account_number'].astype(str)))
        G.add_edges_from(edges)
        pr = nx.pagerank(G, alpha=0.85, max_iter=50)
        df['cust_pagerank'] = df['customer_cif_id'].astype(str).map(pr).fillna(0)
        df['cp_pagerank'] = df['counterparty_account_number'].astype(str).map(pr).fillna(0)
    except Exception as e:
        df['cust_pagerank'] = 0.0
        df['cp_pagerank'] = 0.0

    return df

def build_interaction_features(df):
    """Creates ratio and interaction features."""
    df['amt_to_income_ratio'] = df['transaction_amount'] / (df['annual_income'] + 1)
    
    # Behavioral deviation score (Z-score proxy)
    df['amt_deviation_score'] = (df['transaction_amount'] - df['cust_mean_amt_hist']) / (df['cust_std_amt_hist'] + 1)
    df['amt_vs_cust_max'] = df['transaction_amount'] / (df['cust_max_amt_hist'] + 1)
    
    df['geo_dist'] = np.sqrt(
        (df['gps_coordinates_lat'] - df['customer_address_lat'])**2 +
        (df['gps_coordinates_lon'] - df['customer_address_lon'])**2)
        
    df['total_rules_fired'] = df[[c for c in df.columns if c.startswith('rule_') and df[c].dtype in ['int64','float64']]].sum(axis=1)
    
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()
    
    # If we created 'datetime', use it for hour extraction
    if 'datetime' in df.columns:
        df['hr'] = df['datetime'].dt.hour
        df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
    elif 'timestamp' in df.columns:
        ts = df['timestamp'].astype(str).str.split(':', expand=True).astype(float)
        df['hr'] = ts[0]
        
    return df, datetime_cols

def prepare_features_for_model(df, datetime_cols, training_cols=None):
    """Encodes categoricals. If training_cols is passed, it forces the output shape."""
    DROP_COLS = [
        'transaction_id','is_aml','aml_typology','typology_group_id','typology_signal','session_id',
        'customer_account_number','counterparty_account_number','customer_cif_id','device_id_fingerprint',
        'ip_address','mobile_number','pan','aadhaar_number','email_id','identification_proof_doc_no',
        'entity_identification_proof_doc_no','cif_beneficial_owners','wallet_account_id','escrow_account_linked',
        'father_spouse_name','address_individual_customer','address_registered_office','address_place_of_business',
        'address_beneficial_owners','name_beneficial_owners','load_source_account_card_details',
        'beneficiary_wallet_id_vpa','merchant_id','customer_name','counterparty_name',
        'sender_cust_id_for_rollup','customer_branch_ifsc_code','counterparty_branch_ifsc_swift',
        'wallet_balance_before','wallet_balance_after','timestamp', 'datestamp', 'datetime', 'rules_triggered',
    ]
    X = df.drop(columns=[c for c in DROP_COLS + datetime_cols if c in df.columns])
    
    if 'is_aml' in df.columns:
        y_risk = df['is_aml'].values
    else:
        y_risk = None

    # Handle object columns
    for col in X.select_dtypes(include=['object','string']).columns:
        X[col] = pd.factorize(X[col].fillna('__M__'))[0]

    X.fillna(-999, inplace=True)
    X.columns = [re.sub(r'[\s:,\[\]<>{}]', '_', c) for c in X.columns]
    
    if training_cols is not None:
        missing = set(training_cols) - set(X.columns)
        for m in missing:
            X[m] = -999
        X = X[training_cols]
        
    return X, y_risk
