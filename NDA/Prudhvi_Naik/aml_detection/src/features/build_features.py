import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.graph_analytics import compute_graph_features
from src.graph_analytics_v2.centrality_features import compute_v2_centrality
from src.graph_analytics_v2.circular_flow_detector import compute_circular_flow

def build_entity_features(df):
    """Creates behavioral aggregations for entities using strict historical windows to prevent target leakage."""
    # Ensure chronological order to prevent future leakage
    if 'datestamp' in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['datestamp'].astype(str) + ' ' + df['timestamp'].astype(str), errors='coerce')
        df = df.sort_values(by=['datetime']).reset_index(drop=True)
    
    # Pre-calculate is_aml existence for historical aggregates
    has_target = 'is_aml' in df.columns

    # --- SPRINT 1: ENTITY LEVEL (Replacing Customer Level) ---
    # Shift(1) ensures we only look at the PAST, excluding the current row's transaction amount/label
    df['cust_txn_count_hist'] = df.groupby('master_entity_id').cumcount()
    df['cust_mean_amt_hist'] = df.groupby('master_entity_id')['transaction_amount'].transform(lambda x: x.expanding().mean().shift(1))
    df['cust_std_amt_hist'] = df.groupby('master_entity_id')['transaction_amount'].transform(lambda x: x.expanding().std().shift(1))
    df['cust_max_amt_hist'] = df.groupby('master_entity_id')['transaction_amount'].transform(lambda x: x.expanding().max().shift(1))
    
    if has_target:
        df['cust_fraud_rate_hist'] = df.groupby('master_entity_id')['is_aml'].transform(lambda x: x.expanding().mean().shift(1))
        df['cust_fraud_count_hist'] = df.groupby('master_entity_id')['is_aml'].transform(lambda x: x.expanding().sum().shift(1))
    else:
        df['cust_fraud_rate_hist'] = 0.0
        df['cust_fraud_count_hist'] = 0.0

    df['cust_amt_cv_hist'] = df['cust_std_amt_hist'] / (df['cust_mean_amt_hist'] + 1)
    
    # --- VELOCITY (Advanced Feature Discovery) ---
    if 'datetime' in df.columns:
        df = df.sort_values(by=['master_entity_id', 'datetime']).reset_index(drop=True)
        df_time_indexed = df.set_index('datetime')
        # Count transactions in the last 7 days per entity
        cust_7d_vel = df_time_indexed.groupby('master_entity_id')['transaction_amount'].rolling('7D').count().values
        df['cust_7d_velocity'] = cust_7d_vel - 1
        
        # 24-hour velocity
        cust_24h_vel = df_time_indexed.groupby('master_entity_id')['transaction_amount'].rolling('1D').count().values
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

    # --- SPRINT 1: ADVANCED GRAPH ANALYTICS ---
    df = compute_graph_features(df)

    # --- SPRINT 2: GRAPH INTELLIGENCE V2 ---
    df = compute_v2_centrality(df)
    df = compute_circular_flow(df)

    return df

def build_interaction_features(df):
    """Creates ratio and interaction features."""
    if 'annual_income' in df.columns:
        df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce').fillna(0)
        df['amt_to_income_ratio'] = df['transaction_amount'] / (df['annual_income'] + 1)
        # Income percentile bands
        df['income_log'] = np.log1p(df['annual_income'])
    else:
        df['amt_to_income_ratio'] = -1.0
        df['income_log'] = -1.0
    
    # Behavioral deviation score (Z-score proxy)
    df['amt_deviation_score'] = (df['transaction_amount'] - df['cust_mean_amt_hist']) / (df['cust_std_amt_hist'] + 1)
    df['amt_vs_cust_max'] = df['transaction_amount'] / (df['cust_max_amt_hist'] + 1)
    
    # === ADVANCED FEATURE ENGINEERING FOR 95%+ ACCURACY ===
    
    # Transaction amount log transform (reduces skew, improves tree splits)
    df['txn_amt_log'] = np.log1p(df['transaction_amount'])
    
    # Velocity-to-amount cross features
    if 'cust_7d_velocity' in df.columns:
        df['velocity_x_amount'] = df['cust_7d_velocity'] * df['transaction_amount']
        df['velocity_per_amount'] = df['cust_7d_velocity'] / (df['transaction_amount'] + 1)
    if 'cust_24h_velocity' in df.columns:
        df['velocity_24h_x_amount'] = df['cust_24h_velocity'] * df['transaction_amount']
    
    # Risk score cross features
    if 'cust_risk_score_csv' in df.columns:
        df['cust_risk_score_csv'] = pd.to_numeric(df['cust_risk_score_csv'].map(
            {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        ).fillna(0), errors='coerce').fillna(0)
        df['risk_x_amount'] = df['cust_risk_score_csv'] * df['transaction_amount']
        df['risk_x_velocity'] = df['cust_risk_score_csv'] * df.get('cust_7d_velocity', 0)
    
    # PEP / HNI cross features
    for flag in ['pep_flag', 'hni_flag', 'non_face_to_face_flag', 'vkyc_flag']:
        if flag in df.columns:
            df[flag] = df[flag].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}).fillna(0).astype(int)
            df[f'{flag}_x_amount'] = df[flag] * df['transaction_amount']
    
    # Wallet limit utilization
    if 'wallet_per_txn_limit_max' in df.columns:
        df['wallet_limit_utilization'] = df['transaction_amount'] / (df['wallet_per_txn_limit_max'].fillna(999999) + 1)
    
    # Device anomaly composite score
    for col in ['any_vpn', 'any_emulator']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'any_vpn' in df.columns and 'any_emulator' in df.columns:
        df['device_anomaly_score'] = df['any_vpn'] + df['any_emulator'] + df.get('n_devices', 0) / 3
        df['device_risk_x_amount'] = df['device_anomaly_score'] * df['transaction_amount']
    
    # Account age risk (newer accounts = higher risk)
    if 'account_age_days' in df.columns:
        df['account_age_days'] = pd.to_numeric(df['account_age_days'], errors='coerce').fillna(0)
        df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
        df['new_acct_x_amount'] = df['is_new_account'] * df['transaction_amount']
    
    # Counterparty fraud history cross features
    if 'cp_fraud_rate_hist' in df.columns:
        df['cp_risk_x_amount'] = df['cp_fraud_rate_hist'].fillna(0) * df['transaction_amount']
    
    # Community risk cross features
    if 'community_fraud_rate_hist' in df.columns:
        df['community_risk_x_amount'] = df['community_fraud_rate_hist'].fillna(0) * df['transaction_amount']
    
    # Check if geo columns exist
    if 'gps_coordinates_lat' in df.columns and 'customer_address_lat' in df.columns:
        df['geo_dist'] = np.sqrt(
            (df['gps_coordinates_lat'] - df['customer_address_lat'])**2 +
            (df['gps_coordinates_lon'] - df['customer_address_lon'])**2)
    else:
        df['geo_dist'] = -1.0
        
    df['total_rules_fired'] = df[[c for c in df.columns if c.startswith('rule_') and df[c].dtype in ['int64','float64']]].sum(axis=1)
    
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()
    
    # If we created 'datetime', use it for hour extraction
    if 'datetime' in df.columns:
        df['hr'] = df['datetime'].dt.hour
        df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
        df['is_night'] = ((df['hr'] >= 22) | (df['hr'] <= 5)).astype(int)
        df['night_x_amount'] = df['is_night'] * df['transaction_amount']
        df['day_of_week'] = df['datetime'].dt.dayofweek
    elif 'timestamp' in df.columns:
        ts = df['timestamp'].astype(str).str.split(':', expand=True).astype(float)
        df['hr'] = ts[0]
        
    return df, datetime_cols

def prepare_features_for_model(df, datetime_cols, training_cols=None):
    """Encodes categoricals. If training_cols is passed, it forces the output shape."""
    # Drop targets, timestamps, and unique identifiers (NOT features)
    DROP_COLS = [
        # Targets
        'transaction_id','is_aml','aml_typology','typology_group_id','typology_signal',
        # Timestamps (already extracted hr/is_weekend)
        'timestamp', 'datestamp', 'datetime',
        # Unique identifiers (not features — just keys)
        'session_id', 'customer_account_number', 'counterparty_account_number',
        'customer_cif_id', 'master_entity_id', 'customer_cif',
        # Raw unique strings that freeze factorize (millions of unique values)
        'device_id_fingerprint', 'ip_address', 'rules_triggered',
        'load_source_account_card_details', 'beneficiary_wallet_id_vpa',
        'merchant_id', 'sender_cust_id_for_rollup',
        'wallet_account_id',
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
