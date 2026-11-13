# Benchmark: 20000 rows x 100 variables (continuous), split = random, test = 4000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                   -247.8 |         -2.478 |                     0   |               0      |          191 |                0     |             0     |         0     |                0     |              0    |         0     |          0 |   191 |
| dagma_hc    |     291 | True      |      39.1 |                   -142.1 |         -1.421 |                   105.8 |               0.5088 |          191 |                0.567 |             0.864 |         0.685 |                0.649 |              0.99 |         0.784 |         24 |   128 |

## Method vs method

(one method only)
