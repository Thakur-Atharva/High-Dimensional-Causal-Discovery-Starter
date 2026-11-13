# Benchmark: 20000 rows x 100 variables (continuous), split = random, test = 4000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                   -247.8 |         -2.478 |                    0    |               0      |          191 |                0     |             0     |         0     |                0     |             0     |         0     |          0 |   191 |
| dagma       |     205 | True      |      36.3 |                   -163.1 |         -1.631 |                   84.74 |               0.4502 |          191 |                0.302 |             0.325 |         0.313 |                0.556 |             0.597 |         0.576 |         52 |   220 |
| dagma_hc    |     295 | True      |      38.7 |                   -142.1 |         -1.421 |                  105.8  |               0.5088 |          191 |                0.546 |             0.843 |         0.663 |                0.641 |             0.99  |         0.778 |         28 |   136 |
| fges        |     252 | True      |       9.4 |                   -144.9 |         -1.449 |                  103    |               0.5049 |          191 |                0.631 |             0.832 |         0.718 |                0.698 |             0.921 |         0.795 |         17 |   108 |

## Method vs method

| comparison       |   heldout_loglik_diff_per_row |    95ci |   skeleton_jaccard |
|:-----------------|------------------------------:|--------:|-------------------:|
| dagma - dagma_hc |                       -21.02  | 0.1735  |              0.359 |
| dagma - fges     |                       -18.22  | 0.1754  |              0.344 |
| dagma_hc - fges  |                         2.804 | 0.06309 |              0.541 |
