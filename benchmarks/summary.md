**Continuous, 100 variables, 20,000 rows** (true graph: 191 edges)

| Method                          |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:--------------------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                           |         36 |     205 |          0.58 |          0.31 |   220 |                  84.7 |
| DAGMA + hill climbing           |         39 |     295 |          0.78 |          0.66 |   136 |                 105.8 |
| DAGMA + hill climbing (tabu 10) |         39 |     291 |          0.78 |          0.68 |   128 |                 105.8 |
| fGES                            |         10 |     252 |          0.8  |          0.72 |   108 |                 103   |
| fGES (bound 20)                 |          3 |     266 |          0.78 |          0.64 |   133 |                  99.1 |

**Continuous, 500 variables, 50,000 rows** (true graph: 987 edges)

| Method                          |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:--------------------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                           |         54 |     720 |          0.6  |          0.34 |   906 |                 379.6 |
| DAGMA + hill climbing           |         64 |    1235 |          0.88 |          0.76 |   400 |                 600.6 |
| DAGMA + hill climbing (tabu 10) |         67 |    1220 |          0.88 |          0.77 |   383 |                 600.6 |
| fGES                            |        330 |    1189 |          0.89 |          0.84 |   295 |                 594.5 |
| fGES (bound 20)                 |         16 |    1064 |          0.95 |          0.89 |   163 |                 586.6 |

**Continuous, 2,500 variables, 100,000 rows** (true graph: 4956 edges)

| Method                |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:----------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                 |        772 |    3779 |          0.62 |          0.38 |  4388 |                1861.7 |
| DAGMA + hill climbing |        901 |    6109 |          0.9  |          0.78 |  1812 |                2898.1 |
| fGES (bound 20)       |        167 |    5083 |          0.98 |          0.94 |   402 |                2880.8 |

**Categorical, 50 variables, 20,000 rows** (true graph: 90 edges)

| Method                |   Time (s) |   Edges |   Skeleton F1 |   Directed F1 |   SHD |   Held-out gain / row |
|:----------------------|-----------:|--------:|--------------:|--------------:|------:|----------------------:|
| DAGMA                 |         11 |      13 |          0.25 |          0.16 |    82 |                   2.5 |
| DAGMA + hill climbing |         12 |      92 |          0.97 |          0.84 |    18 |                  10.6 |

