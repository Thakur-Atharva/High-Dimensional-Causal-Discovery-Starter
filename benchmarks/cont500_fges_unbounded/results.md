# Benchmark: 50000 rows x 500 variables (continuous), split = random, test = 10000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                  -1311   |         -2.621 |                     0   |               0      |          987 |                0     |             0     |          0    |                0     |              0    |         0     |          0 |   987 |
| fges        |    1189 | True      |     330.5 |                   -716.1 |         -1.432 |                   594.5 |               0.8518 |          987 |                0.769 |             0.926 |          0.84 |                0.813 |              0.98 |         0.889 |         53 |   295 |

## Method vs method

(one method only)
