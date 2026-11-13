**Continuous, 100 variables, 20,000 rows** (true graph: 191 edges)

| Method                          |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:--------------------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                           |         36 |     205 |          0.58 |          0.31 |   220 |                  84.7 |
| DAGMA + hill climbing           |         39 |     295 |          0.78 |          0.66 |   136 |                 105.8 |
| DAGMA + hill climbing (tabu 10) |         40 |     291 |          0.78 |          0.68 |   128 |                 105.8 |
| fGES                            |          9 |     252 |          0.8  |          0.72 |   108 |                 103   |

**Continuous, 500 variables, 50,000 rows** (true graph: 987 edges)

| Method                          |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:--------------------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                           |         54 |     720 |          0.6  |          0.34 |   906 |                 379.6 |
| DAGMA + hill climbing           |         64 |    1235 |          0.88 |          0.76 |   400 |                 600.6 |
| DAGMA + hill climbing (tabu 10) |         67 |    1220 |          0.88 |          0.77 |   383 |                 600.6 |
| fGES                            |        312 |    1180 |          0.89 |          0.84 |   282 |                 595   |

**Categorical, 50 variables, 20,000 rows** (true graph: 90 edges)

| Method                |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:----------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                 |         11 |      13 |          0.25 |          0.16 |    82 |                   2.5 |
| DAGMA + hill climbing |         12 |      92 |          0.97 |          0.84 |    18 |                  10.6 |

