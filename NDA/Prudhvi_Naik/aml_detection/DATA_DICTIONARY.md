# Data Dictionary
## Engineered Features
- `cust_mean_amt_hist`: The historical average transaction amount for the customer, strictly prior to the current timestamp.
- `cust_7d_velocity`: The count of transactions by the customer in the prior 7 days.
- `cust_pagerank`: The NetworkX PageRank centrality score for the customer in the bipartite transaction graph.
- `geo_dist`: The calculated distance between the customer's registered address and the transaction's GPS coordinates.
- `total_rules_fired`: The summation of traditional deterministic rules triggered by the transaction.