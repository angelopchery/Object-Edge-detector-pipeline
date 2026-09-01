# Scene clustering threshold scan

Produced by `scripts/cluster_scenes.py` (method: colorhist, single-linkage
union-find, timestamp prior: +0.05 distance within 5.0s).

| threshold | clusters |
|-----------|----------|
| 0.05 | 84 |
| 0.1 | 40 |
| 0.15 | 16 |
| 0.2 | 7 |
| 0.25 | 5 |
| 0.3 | 4 |
| 0.35 | 3 |
| 0.4 | 3 |
| 0.45 | 1 |
| 0.5 | 1 |
| 0.55 | 1 |
| 0.6 | 1 |
| 0.65 | 1 |
| 0.7 | 1 |
| 0.75 | 1 |
| 0.8 | 1 |

Chosen threshold: **0.1** (auto: knee of the curve, max second difference)
Clusters at chosen threshold: **40**

Cluster sizes: [59, 25, 16, 10, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
