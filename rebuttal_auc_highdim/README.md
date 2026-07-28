# High-dimensional AUC M ablation

This suite adds Madelon, Gisette, Arcene, and Dexter to the ZOFOGDA AUC experiment.
The public validation split is frozen as the evaluation split; hidden challenge test
sets are never used.

| Dataset | Input dimension | Train / validation | floor(sqrt(d)) |
|---|---:|---:|---:|
| Madelon | 500 | 2000 / 600 | 22 |
| Gisette | 5000 | 6000 / 1000 | 70 |
| Arcene | 10000 | 100 / 100 | 100 |
| Dexter | 20000 | 300 / 300 | 141 |

Formal runs use 500 epochs, batch size 128, seeds 14/42/81, delta 1e-3,
scheduled paper momentum, and M in {1, 5, 10, 15, floor(sqrt(d))}. M=5 is
the original-configuration row.

Short stability pilots freeze gamma=lambda for every M and seed of a dataset:
Madelon 0.01, Gisette 0.003, Arcene 0.005, and Dexter 0.003.

```bash
python rebuttal_auc_highdim/prepare_data.py
python rebuttal_auc_highdim/manage.py build
python rebuttal_auc_highdim/manage.py launch
python rebuttal_auc_highdim/manage.py status
python rebuttal_auc_highdim/manage.py refresh
```

Workers require an empty assigned GPU, audit GPU process exclusivity, record
per-epoch wall time, and restart a contaminated run from epoch 1.
