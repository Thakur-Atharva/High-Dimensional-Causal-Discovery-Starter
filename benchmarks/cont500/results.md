# Benchmark: 50000 rows x 500 variables (continuous), split = random, test = 10000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                  -1311   |         -2.621 |                     0   |               0      |          987 |                0     |             0     |         0     |                0     |             0     |         0     |          0 |   987 |
| dagma       |     720 | True      |      53.5 |                   -931.1 |         -1.862 |                   379.6 |               0.6349 |          987 |                0.407 |             0.297 |         0.343 |                0.706 |             0.515 |         0.595 |        215 |   906 |
| dagma_hc    |    1235 | True      |      63.5 |                   -710.1 |         -1.42  |                   600.6 |               0.8504 |          987 |                0.684 |             0.856 |         0.761 |                0.791 |             0.99  |         0.879 |        132 |   400 |
| fges        |    1180 | True      |     311.5 |                   -715.6 |         -1.431 |                   595   |               0.8517 |          987 |                0.776 |             0.928 |         0.845 |                0.821 |             0.982 |         0.894 |         53 |   282 |

## Method vs method

| comparison       |   heldout_loglik_diff_per_row |    95ci |   skeleton_jaccard |
|:-----------------|------------------------------:|--------:|-------------------:|
| dagma - dagma_hc |                       -221    | 0.3888  |              0.384 |
| dagma - fges     |                       -215.5  | 0.3891  |              0.369 |
| dagma_hc - fges  |                          5.54 | 0.05517 |              0.694 |
