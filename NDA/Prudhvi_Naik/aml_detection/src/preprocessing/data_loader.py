import pandas as pd
import numpy as np

def load_and_merge_data(data_dir='data/raw'):
    """Loads all 5 raw datasets and performs initial joins."""
    txn = pd.read_parquet(f'{data_dir}/stg_transactions_features.parquet')
    acc = pd.read_csv(f'{data_dir}/accounts.csv', dtype={'account_number': str, 'customer_cif': str})
    cust = pd.read_csv(f'{data_dir}/customers.csv', dtype={'customer_cif': str})
    dev = pd.read_csv(f'{data_dir}/devices.csv', dtype={'customer_cif': str})
    wal = pd.read_csv(f'{data_dir}/wallets.csv', dtype={'customer_cif': str})

    # Accounts
    acc2 = acc.rename(columns={'account_number': 'customer_account_number', 'account_status': 'acc_status'})
    acc2 = acc2.drop(columns=['account_category','account_type','credit_summation_period',
                              'debit_summation_period','customer_cif','account_opening_date',
                              'inoperative_status_date','customer_branch_ifsc'], errors='ignore')
    txn = txn.merge(acc2, on='customer_account_number', how='left')

    # Customers
    cust2 = cust.rename(columns={'customer_cif':'customer_cif_id','state':'cust_state',
                                 'city':'cust_city','_nri_country':'cust_nri',
                                 'address_lat':'cust_lat','address_lon':'cust_lon',
                                 'customer_risk_score':'cust_risk_score_csv','occupation_industry':'cust_occ_csv'})
    cust_drop = ['customer_name','customer_type','customer_entity_type','date_of_birth',
                 'father_spouse_name','nationality','citizenship','residency','tax_residency','pan',
                 'mobile_number','email_id','annual_income','professional_experience_years',
                 'source_of_funds','pep_flag','hni_flag','minor_flag','non_face_to_face_flag',
                 'vkyc_flag','kyc_update_date','date_of_incorporation','place_of_incorporation',
                 'beneficial_owner_types','passive_nfe','address_registered_office','address_place_of_business',
                 'address_beneficial_owners','cif_beneficial_owners','name_beneficial_owners','aadhaar_number',
                 'identification_proof_doc_no','entity_identification_proof_doc_no','aadhaar','aadhaar_masked',
                 'identification_doc_no','entity_identification_doc_no','cif_creation_date','address_individual']
    cust2 = cust2.drop(columns=[c for c in cust_drop if c in cust2.columns], errors='ignore')
    txn = txn.merge(cust2, on='customer_cif_id', how='left')

    # Devices
    dev2 = dev.rename(columns={'customer_cif':'customer_cif_id'})
    dev_agg = dev2.groupby('customer_cif_id').agg(
        n_devices=('device_id','nunique'), n_dev_cities=('geo_city','nunique'),
        n_dev_countries=('geo_country','nunique')).reset_index()
    txn = txn.merge(dev_agg, on='customer_cif_id', how='left')

    # Wallets
    wal2 = wal.rename(columns={'customer_cif':'customer_cif_id'})
    wal_agg = wal2.groupby('customer_cif_id').agg(n_wallets=('wallet_id','nunique')).reset_index()
    txn = txn.merge(wal_agg, on='customer_cif_id', how='left')

    return txn
