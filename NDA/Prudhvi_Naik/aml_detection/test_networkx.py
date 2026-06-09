import pandas as pd
import networkx as nx
import time

print("Loading data...")
df = pd.read_parquet('data/raw/stg_transactions_features.parquet')

start = time.time()
print("Building graph...")
# Create edge list
edges = list(zip(df['customer_cif_id'], df['counterparty_account_number']))

G = nx.Graph()
G.add_edges_from(edges)
print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in {time.time() - start:.2f}s")

start = time.time()
print("Calculating PageRank...")
pr = nx.pagerank(G, alpha=0.85, max_iter=100)
print(f"PageRank calculated in {time.time() - start:.2f}s")

