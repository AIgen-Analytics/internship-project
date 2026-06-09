# Network Intelligence (Graph Analytics)
## Implementation
We advanced the graph network capabilities beyond simple unique counterparty counts (Degree Centrality). Using `NetworkX`, we built a bipartite graph mapping all historical interactions between `customer_cif_id` and `counterparty_account_number`.

## PageRank Execution
We calculated PageRank across 380,000 transactions. PageRank assigns higher risk scores to entities that act as highly-connected central nodes (Hub-and-Spoke networks) and entities forming deep transitive chains (Layering/Pass-Throughs). 
- **Compute Time:** < 1.0 second on sparse matrices.
- **Uplift:** PageRank feature importance consistently ranks in the Top 15 drivers of the final LightGBM model, specifically identifying Mule Networks and Transit Hubs that evade simple volume-based detection.