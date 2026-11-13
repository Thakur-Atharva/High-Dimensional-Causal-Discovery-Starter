# Benchmark: 50000 rows x 500 variables (continuous), split = random, test = 10000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                  -1311   |         -2.621 |                     0   |               0      |          987 |                0     |             0     |         0     |                0     |             0     |         0     |          0 |   987 |
| dagma       |     720 | True      |      53.9 |                   -931.1 |         -1.862 |                   379.6 |               0.6349 |          987 |                0.407 |             0.297 |         0.343 |                0.706 |             0.515 |         0.595 |        215 |   906 |
| dagma_hc    |    1235 | True      |      63.9 |                   -710.1 |         -1.42  |                   600.6 |               0.8504 |          987 |                0.684 |             0.856 |         0.761 |                0.791 |             0.99  |         0.879 |        132 |   400 |
| fges        |    1064 | True      |      16.3 |                   -724.1 |         -1.448 |                   586.6 |               0.8345 |          987 |                0.856 |             0.923 |         0.888 |                0.918 |             0.99  |         0.953 |         66 |   163 |

## Method vs method

| comparison       |   heldout_loglik_diff_per_row |    95ci |   skeleton_jaccard |
|:-----------------|------------------------------:|--------:|-------------------:|
| dagma - dagma_hc |                          -221 | 0.3888  |              0.384 |
| dagma - fges     |                          -207 | 0.3791  |              0.406 |
| dagma_hc - fges  |                            14 | 0.06756 |              0.734 |
