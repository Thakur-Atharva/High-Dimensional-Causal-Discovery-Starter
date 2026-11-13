# Benchmark: 50000 rows x 500 variables (continuous), split = random, test = 10000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                  -1311   |         -2.621 |                     0   |               0      |          987 |                0     |             0     |         0     |                0     |              0    |         0     |          0 |   987 |
| dagma_hc    |    1220 | True      |      66.8 |                   -710.1 |         -1.42  |                   600.6 |               0.8504 |          987 |                0.694 |             0.858 |         0.768 |                0.801 |              0.99 |         0.885 |        130 |   383 |

## Method vs method

(one method only)
