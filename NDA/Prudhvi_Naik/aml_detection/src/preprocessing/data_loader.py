import pandas as pd
import numpy as np
import os

# Add this to allow absolute imports if running from top level
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.entity_resolution import perform_entity_resolution

def load_and_merge_data(data_dir='data/raw'):
    """Loads ALL raw datasets with ALL useful columns, performs Entity Resolution, and merges on master_entity_id."""
    txn = pd.read_parquet(f'{data_dir}/stg_transactions_features.parquet')
    acc = pd.read_csv(f'{data_dir}/accounts.csv', dtype={'account_number': str, 'customer_cif': str})
    cust = pd.read_csv(f'{data_dir}/customers.csv', dtype={'customer_cif': str, 'mobile_number': str})
    dev = pd.read_csv(f'{data_dir}/devices.csv', dtype={'customer_cif': str})
    wal = pd.read_csv(f'{data_dir}/wallets.csv', dtype={'customer_cif': str})

    # --- SPRINT 1: ENTITY RESOLUTION ---
    print("Performing Entity Resolution...", flush=True)
    mapping, confidence = perform_entity_resolution(cust, dev)
    
    acc_to_cif = dict(zip(acc['account_number'], acc['customer_cif']))
    txn['customer_cif'] = txn['customer_account_number'].astype(str).map(acc_to_cif)
    
    txn['master_entity_id'] = txn['customer_cif'].astype(str).map(mapping)
    acc['master_entity_id'] = acc['customer_cif'].astype(str).map(mapping)
    cust['master_entity_id'] = cust['customer_cif'].astype(str).map(mapping)
    dev['master_entity_id'] = dev['customer_cif'].astype(str).map(mapping)
    wal['master_entity_id'] = wal['customer_cif'].astype(str).map(mapping)
    
    txn['entity_resolution_confidence'] = txn['master_entity_id'].map(confidence).fillna(1.0)
    txn = txn.dropna(subset=['master_entity_id'])

    # --- MERGES: KEEP ALL USEFUL COLUMNS ---
    print("Merging features using master_entity_id...", flush=True)

    # ==================== ACCOUNTS (ALL 10 columns used) ====================
    acc2 = acc.rename(columns={'account_status': 'acc_status', 'account_number': 'customer_account_number'})
    # Keep: account_category, account_type, acc_status, credit/debit_summation_period
    # Drop only: customer_cif (already mapped), customer_branch_ifsc (unique branch ID)
    acc2 = acc2.drop(columns=['customer_cif', 'customer_branch_ifsc'], errors='ignore')
    # Convert dates to numeric (days since account opened)
    if 'account_opening_date' in acc2.columns:
        acc2['account_opening_date'] = pd.to_datetime(acc2['account_opening_date'], dayfirst=True, errors='coerce')
        acc2['account_age_days'] = (pd.Timestamp.now() - acc2['account_opening_date']).dt.days
        acc2 = acc2.drop(columns=['account_opening_date'], errors='ignore')
    if 'inoperative_status_date' in acc2.columns:
        acc2['inoperative_status_date'] = pd.to_datetime(acc2['inoperative_status_date'], dayfirst=True, errors='coerce')
        acc2['is_inoperative'] = acc2['inoperative_status_date'].notna().astype(int)
        acc2 = acc2.drop(columns=['inoperative_status_date'], errors='ignore')
    txn = txn.merge(acc2, on=['customer_account_number', 'master_entity_id'], how='left')

    # ==================== CUSTOMERS (ALL 44 columns used) ====================
    cust2 = cust.rename(columns={
        'state':'cust_state', 'city':'cust_city', '_nri_country':'cust_nri',
        'address_lat':'cust_lat', 'address_lon':'cust_lon',
        'customer_risk_score':'cust_risk_score_csv', 'occupation_industry':'cust_occ_csv'
    })
    # KEEP these critical AML columns (previously dropped):
    #   customer_type, customer_entity_type, annual_income, pep_flag, hni_flag,
    #   non_face_to_face_flag, vkyc_flag, source_of_funds, minor_flag,
    #   professional_experience_years, nationality, citizenship, residency,
    #   tax_residency, beneficial_owner_types, passive_nfe
    # 
    # DROP only true unique PII strings that freeze the model:
    cust_drop_pii = [
        'customer_name', 'father_spouse_name', 'pan', 'aadhaar', 'aadhaar_masked',
        'identification_doc_no', 'mobile_number', 'email_id',
        'address_individual', 'address_registered_office', 'address_place_of_business',
        'address_beneficial_owners', 'entity_identification_doc_no',
        'cif_beneficial_owners', 'name_beneficial_owners',
        'customer_cif'
    ]
    cust2 = cust2.drop(columns=[c for c in cust_drop_pii if c in cust2.columns], errors='ignore')
    
    # Convert date columns to numeric features
    for dcol in ['date_of_birth', 'cif_creation_date', 'kyc_update_date', 'date_of_incorporation']:
        if dcol in cust2.columns:
            cust2[dcol] = pd.to_datetime(cust2[dcol], dayfirst=True, errors='coerce')
            cust2[f'{dcol}_days_ago'] = (pd.Timestamp.now() - cust2[dcol]).dt.days
            cust2 = cust2.drop(columns=[dcol], errors='ignore')
    
    # Convert annual_income to numeric
    if 'annual_income' in cust2.columns:
        cust2['annual_income'] = pd.to_numeric(cust2['annual_income'], errors='coerce').fillna(0)
    
    cust_agg = cust2.groupby('master_entity_id').first().reset_index()
    txn = txn.merge(cust_agg, on='master_entity_id', how='left')

    # ==================== DEVICES (ALL 10 columns used) ====================
    # Aggregate per entity: count devices, cities, countries, VPN/emulator flags
    dev_agg = dev.groupby('master_entity_id').agg(
        n_devices=('device_id', 'nunique'),
        n_dev_cities=('geo_city', 'nunique'),
        n_dev_countries=('geo_country', 'nunique'),
        any_vpn=('vpn_flag', 'max'),
        any_emulator=('emulator_flag', 'max'),
        n_unique_ips=('ip_address', 'nunique'),
        n_unique_browsers=('browser_app_info', 'nunique')
    ).reset_index()
    txn = txn.merge(dev_agg, on='master_entity_id', how='left')

    # ==================== WALLETS (ALL 12 columns used) ====================
    wal_agg = wal.groupby('master_entity_id').agg(
        n_wallets=('wallet_id', 'nunique'),
        wallet_kyc_min=('wallet_kyc_category', 'first'),
        wallet_any_active=('wallet_status', lambda x: int((x == 'Active').any())),
        wallet_per_txn_limit_max=('per_txn_limit', 'max'),
        wallet_daily_limit_max=('daily_txn_limit', 'max'),
        wallet_monthly_limit_max=('monthly_txn_limit', 'max'),
        wallet_annual_limit_max=('annual_txn_limit', 'max'),
        wallet_max_balance_limit=('max_balance_limit', 'max'),
        wallet_has_escrow=('escrow_account_linked', lambda x: int(x.notna().any())),
        wallet_has_bank_link=('linked_bank_account', lambda x: int(x.notna().any()))
    ).reset_index()
    txn = txn.merge(wal_agg, on='master_entity_id', how='left')

    print(f"Final merged dataset: {txn.shape[0]} rows x {txn.shape[1]} columns", flush=True)
    return txn
