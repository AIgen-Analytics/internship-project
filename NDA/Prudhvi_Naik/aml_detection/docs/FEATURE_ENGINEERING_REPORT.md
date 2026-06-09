# Feature Engineering Report
## Temporal Leakage Removal
Global pandas aggregations (`mean()`) originally caused the model to peek into the future, inflating performance. We replaced all aggregations with `expanding().mean().shift(1)` after strict chronological sorting.

## Burst Velocity
We utilized Pandas `.rolling()` windows mapped against chronological datetimes to capture 'smurfing'—rapid succession of small transactions that evade standard volume thresholds.

## Graph Centrality
By treating customers and counterparties as nodes, we extracted `Degree Centrality` and `PageRank` to map structural network risk.