"""Phase 1: Deep Schema Discovery for all AML datasets."""
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

def load_all():
    dfs = {}
    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if f.endswith('.csv'):
            dfs[f] = pd.read_csv(path)
        elif f.endswith('.parquet'):
            dfs[f] = pd.read_parquet(path)
    return dfs

def schema_report(name, df):
    print(f"\n{'='*80}")
    print(f"  FILE: {name}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*80}")
    
    # Dtypes
    print(f"\n--- Column Types ---")
    for col in df.columns:
        nunique = df[col].nunique()
        null_pct = df[col].isnull().mean() * 100
        sample = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else 'ALL NULL'
        if len(sample) > 60:
            sample = sample[:60] + '...'
        print(f"  {col:<50} dtype={str(df[col].dtype):<12} nunique={nunique:<8} null%={null_pct:>6.2f}  sample={sample}")

    # Potential ID columns
    print(f"\n--- Potential ID Columns (high cardinality / unique) ---")
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique == df.shape[0]:
            print(f"  ★ {col} — UNIQUE (primary key candidate)")
        elif nunique > df.shape[0] * 0.8 and df[col].dtype == 'object':
            print(f"  ◆ {col} — near-unique ({nunique}/{df.shape[0]})")
    
    # Potential target/label columns
    print(f"\n--- Potential Target/Label Columns ---")
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique <= 10 and df[col].dtype in ['int64', 'float64', 'object', 'bool']:
            vc = df[col].value_counts()
            print(f"  → {col} ({nunique} unique): {dict(vc.head(10))}")
    
    # Columns with 'fraud', 'risk', 'suspicious', 'label', 'flag', 'aml', 'typology' in name
    keywords = ['fraud', 'risk', 'suspicious', 'label', 'flag', 'aml', 'typology', 'sar', 'alert', 'mule', 'launder']
    matches = [col for col in df.columns if any(k in col.lower() for k in keywords)]
    if matches:
        print(f"\n--- AML/Fraud Keyword Columns ---")
        for col in matches:
            nunique = df[col].nunique()
            vc = df[col].value_counts()
            print(f"  ★ {col} ({nunique} unique): {dict(vc.head(10))}")
    
    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"\n--- Numeric Summary (first 15 cols) ---")
        print(df[num_cols[:15]].describe().round(3).to_string())
    
    print()

def find_foreign_keys(dfs):
    """Find columns that appear across multiple datasets (potential join keys)."""
    print(f"\n{'='*80}")
    print(f"  CROSS-DATASET RELATIONSHIP ANALYSIS")
    print(f"{'='*80}")
    
    col_map = {}
    for name, df in dfs.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower not in col_map:
                col_map[col_lower] = []
            col_map[col_lower].append((name, col, df[col].nunique(), df[col].dtype))
    
    print(f"\n--- Shared Column Names (appear in 2+ files) ---")
    for col_lower, occurrences in sorted(col_map.items()):
        if len(occurrences) >= 2:
            print(f"\n  Column: '{col_lower}'")
            for fname, orig_col, nunique, dtype in occurrences:
                print(f"    {fname:<45} nunique={nunique:<8} dtype={dtype}")

    # ID-style columns
    print(f"\n--- ID-style Columns (contain 'id' or '_id') ---")
    for name, df in dfs.items():
        id_cols = [c for c in df.columns if 'id' in c.lower()]
        if id_cols:
            print(f"\n  {name}:")
            for col in id_cols:
                print(f"    {col}: nunique={df[col].nunique()}, sample={df[col].dropna().iloc[0] if df[col].dropna().shape[0]>0 else 'NULL'}")

def main():
    print("Loading all datasets...")
    dfs = load_all()
    
    for name, df in dfs.items():
        schema_report(name, df)
    
    find_foreign_keys(dfs)
    
    # Print full column lists for the big parquet file
    for name, df in dfs.items():
        if name.endswith('.parquet'):
            print(f"\n{'='*80}")
            print(f"  FULL COLUMN LIST: {name} ({len(df.columns)} columns)")
            print(f"{'='*80}")
            for i, col in enumerate(df.columns):
                print(f"  {i:>3}. {col}")

if __name__ == '__main__':
    main()
