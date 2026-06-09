import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def build_entity_features(df):
    """Creates behavioral aggregations for entities (customer, counterparty, etc)."""
    # Customer-level
    cust_agg = df.groupby('customer_cif_id').agg(
        cust_txn_count=('transaction_id','count'),
        cust_mean_amt=('transaction_amount','mean'),
        cust_std_amt=('transaction_amount','std'),
        cust_max_amt=('transaction_amount','max'),
        cust_fraud_rate=('is_aml','mean'),
        cust_fraud_count=('is_aml','sum'),
        cust_total_amt=('transaction_amount','sum'),
        cust_unique_cp=('counterparty_account_number','nunique'),
        cust_unique_merch=('merchant_id','nunique'),
    ).reset_index()
    cust_agg['cust_amt_cv'] = cust_agg['cust_std_amt'] / (cust_agg['cust_mean_amt'] + 1)
    df = df.merge(cust_agg, on='customer_cif_id', how='left')

    # Counterparty-level
    cp_agg = df.groupby('counterparty_account_number').agg(
        cp_txn_count=('transaction_id','count'),
        cp_fraud_rate=('is_aml','mean'),
        cp_unique_senders=('customer_cif_id','nunique'),
        cp_mean_amt=('transaction_amount','mean'),
    ).reset_index()
    df = df.merge(cp_agg, on='counterparty_account_number', how='left')

    # Account-level
    acct_agg = df.groupby('customer_account_number').agg(
        acct_txn_count=('transaction_id','count'),
        acct_fraud_rate=('is_aml','mean'),
        acct_mean_amt=('transaction_amount','mean'),
    ).reset_index()
    df = df.merge(acct_agg, on='customer_account_number', how='left')

    return df

def build_interaction_features(df):
    """Creates ratio and interaction features."""
    df['amt_to_income_ratio'] = df['transaction_amount'] / (df['annual_income'] + 1)
    df['amt_vs_cust_mean'] = df['transaction_amount'] / (df['cust_mean_amt'] + 1)
    df['amt_vs_cust_max'] = df['transaction_amount'] / (df['cust_max_amt'] + 1)
    df['geo_dist'] = np.sqrt(
        (df['gps_coordinates_lat'] - df['customer_address_lat'])**2 +
        (df['gps_coordinates_lon'] - df['customer_address_lon'])**2)
    df['total_rules_fired'] = df[[c for c in df.columns if c.startswith('rule_') and df[c].dtype in ['int64','float64']]].sum(axis=1)
    
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()
    for c in datetime_cols:
        df[f'{c}_month'] = df[c].dt.month
    if 'timestamp' in df.columns:
        ts = df['timestamp'].str.split(':', expand=True).astype(float)
        df['hr'] = ts[0]
        
    return df, datetime_cols

def prepare_features_for_model(df, datetime_cols):
    """Encodes categoricals and drops metadata columns."""
    DROP_COLS = [
        'transaction_id','is_aml','aml_typology','typology_group_id','typology_signal','session_id',
        'customer_account_number','counterparty_account_number','customer_cif_id','device_id_fingerprint',
        'ip_address','mobile_number','pan','aadhaar_number','email_id','identification_proof_doc_no',
        'entity_identification_proof_doc_no','cif_beneficial_owners','wallet_account_id','escrow_account_linked',
        'father_spouse_name','address_individual_customer','address_registered_office','address_place_of_business',
        'address_beneficial_owners','name_beneficial_owners','load_source_account_card_details',
        'beneficiary_wallet_id_vpa','merchant_id','customer_name','counterparty_name',
        'sender_cust_id_for_rollup','customer_branch_ifsc_code','counterparty_branch_ifsc_swift',
        'wallet_balance_before','wallet_balance_after','timestamp','rules_triggered',
    ]
    X = df.drop(columns=[c for c in DROP_COLS + datetime_cols if c in df.columns])
    y_risk = df['is_aml'].values

    for col in X.select_dtypes(include=['object','string']).columns:
        if X[col].nunique() <= 100:
            X[col] = LabelEncoder().fit_transform(X[col].fillna('__M__').astype(str))
        else:
            X.drop(columns=[col], inplace=True)

    X.fillna(-999, inplace=True)
    X.columns = [re.sub(r'[\s:,\[\]<>{}]', '_', c) for c in X.columns]
    
    return X, y_risk
