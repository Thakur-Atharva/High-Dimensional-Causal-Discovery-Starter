# Benchmark: 100000 rows x 2500 variables (continuous), split = random, test = 20000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                    -6448 |         -2.579 |                       0 |                0     |         4956 |                0     |             0     |         0     |                0     |             0     |         0     |          0 |  4956 |
| dagma       |    3779 | False     |     771.9 |                    -4586 |         -1.834 |                    1862 |                1.165 |         4956 |                0.44  |             0.335 |         0.381 |                0.711 |             0.542 |         0.615 |       1023 |  4388 |
| dagma_hc    |    6109 | True      |     900.6 |                    -3549 |         -1.42  |                    2898 |                1.541 |         4956 |                0.704 |             0.868 |         0.778 |                0.81  |             0.999 |         0.895 |        649 |  1812 |
| fges        |    5083 | True      |     167.3 |                    -3567 |         -1.427 |                    2881 |                1.535 |         4956 |                0.925 |             0.949 |         0.937 |                0.971 |             0.996 |         0.983 |        233 |   402 |

## Method vs method

| comparison       |   heldout_loglik_diff_per_row |    95ci |   skeleton_jaccard |
|:-----------------|------------------------------:|--------:|-------------------:|
| dagma - dagma_hc |                      -1036    | 0.6154  |              0.415 |
| dagma - fges     |                      -1019    | 0.6107  |              0.437 |
| dagma_hc - fges  |                         17.34 | 0.06111 |              0.803 |
