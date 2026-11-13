# Benchmark: 20000 rows x 100 variables (continuous), split = random, test = 4000 rows

Held-out log-likelihood: average over test rows of the log-probability of the whole row under the graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as independent; '95ci' is the half-width of its 95% interval.

| method      |   edges | acyclic   |   seconds |   heldout_loglik_per_row |   per_variable |   gain_vs_empty_per_row |   gain_vs_empty_95ci |   true_edges |   directed_precision |   directed_recall |   directed_f1 |   skeleton_precision |   skeleton_recall |   skeleton_f1 |   reversed |   shd |
|:------------|--------:|:----------|----------:|-------------------------:|---------------:|------------------------:|---------------------:|-------------:|---------------------:|------------------:|--------------:|---------------------:|------------------:|--------------:|-----------:|------:|
| empty graph |       0 | True      |       0   |                   -247.8 |         -2.478 |                    0    |               0      |          191 |                0     |              0    |         0     |                0     |             0     |         0     |          0 |   191 |
| fges        |     266 | True      |       2.9 |                   -148.7 |         -1.487 |                   99.11 |               0.4918 |          191 |                0.553 |              0.77 |         0.643 |                0.665 |             0.927 |         0.775 |         30 |   133 |

## Method vs method

(one method only)
