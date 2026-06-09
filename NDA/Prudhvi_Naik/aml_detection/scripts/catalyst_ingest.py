import os
import pandas as pd
import zcatalyst_sdk

# Paths to data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Catalyst SDK init
app = zcatalyst_sdk.initialize()
datastore = app.datastore()

def ingest_table(file_name, table_name, is_parquet=False, sample_size=1000):
    print(f"Starting ingestion for {table_name} from {file_name}...")
    file_path = os.path.join(DATA_DIR, file_name)
    
    if is_parquet:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    # Take a sample for ingestion to avoid rate limits initially (can be modified later)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for initial ingestion.")
        
    # Replace NaNs with None for JSON serialization
    df = df.replace({pd.NA: None, float('nan'): None})
    
    # Fill missing string/objects with 'UNKNOWN'
    categorical = df.select_dtypes(include=['object']).columns
    df[categorical] = df[categorical].fillna('UNKNOWN')

    table = datastore.table(table_name)
    
    # Bulk insert is not natively standard in the basic SDK, we will iterate.
    # Alternatively, batch inserts can be used if supported, but loop is safer.
    success = 0
    errors = 0
    
    # To optimize, we group into batches
    batch_size = 100
    records = df.to_dict('records')
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            # Catalyst insert_rows handles a list of rows
            table.insert_rows(batch)
            success += len(batch)
            print(f"Inserted {success}/{len(records)} rows into {table_name}...")
        except Exception as e:
            errors += len(batch)
            print(f"Error inserting batch into {table_name}: {e}")
            
    print(f"✅ Ingestion complete for {table_name}. Success: {success}, Errors: {errors}")

if __name__ == "__main__":
    print("--- Catalyst Data Ingestion Script ---")
    
    # Ensure tables exist in Catalyst Data Store before running this!
    # ingest_table('customers.csv', 'customers', sample_size=3400)
    # ingest_table('accounts.csv', 'accounts', sample_size=5065)
    # ingest_table('devices.csv', 'devices', sample_size=5002)
    # ingest_table('wallets.csv', 'wallets', sample_size=900)
    
    # For transactions, we only sample 5,000 for the initial DB seeding
    # ingest_table('stg_transactions_features.parquet', 'transactions', is_parquet=True, sample_size=5000)
    
    print("Ingestion script ready. Uncomment calls to run once Data Store schemas are manually created in the Catalyst Console.")
