# Benchmark: 20000 rows x 50 variables (categorical), split = random, test = 4000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                   -43.24 |        -0.8648 |                   0     |              0       |           90 |                0     |             0     |         0     |                0     |             0     |         0     |          0 |    90 |
| dagma       |      13 | True      |      11.1 |                   -40.74 |        -0.8148 |                   2.502 |              0.05492 |           90 |                0.615 |             0.089 |         0.155 |                1     |             0.144 |         0.252 |          5 |    82 |
| dagma_hc    |      92 | True      |      12.1 |                   -32.67 |        -0.6535 |                  10.57  |              0.1205  |           90 |                0.826 |             0.844 |         0.835 |                0.957 |             0.978 |         0.967 |         12 |    18 |

## Method vs method

| comparison       |   heldout_loglik_diff_per_row |   95ci |   skeleton_jaccard |
|:-----------------|------------------------------:|-------:|-------------------:|
| dagma - dagma_hc |                        -8.065 |  0.108 |              0.141 |
