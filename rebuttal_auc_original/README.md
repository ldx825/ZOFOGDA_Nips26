# Original-code AUC rebuttal experiments

This directory runs only the AUC experiments through the paper's three independent training entries:

- `26.1.29.ZOFOGDA_AUC/ldx_zofogda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttgda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttsgda_auc.py`

The formal matrix has 216 unique runs: six datasets, seeds 14/42/81, 500 epochs, original-configuration baselines, `M={1,5,10,15,floor(sqrt(d))}`, `delta={1e-4,1e-3,1e-2}`, and full/no-x/no-y/no-both momentum. Here `d` is the loaded input feature dimension. Baseline conditions are deduplicated.

Each worker waits for an empty assigned GPU. During training it audits the number of GPU compute contexts once per second; if another context appears, the run is terminated and restarted from epoch 1. Each successful run has an original training CSV, `epoch_times.csv`, `gpu_audit_attempt*.csv`, and `summary.json`.

Historical `result_plot` values are not used in formal tables or default plots. Formal outputs require independently rerun, GPU-exclusive results.

## Commands

```bash
python rebuttal_auc_original/manage.py status
python rebuttal_auc_original/manage.py refresh
python rebuttal_auc_original/manage.py launch
```

The auto-refreshed workbook is `AUC_rebuttal_results.xlsx`. Plot examples:

```bash
python rebuttal_auc_original/plot_results.py --comparison baseline --xaxis time
python rebuttal_auc_original/plot_results.py --comparison M --xaxis query
python rebuttal_auc_original/plot_results.py --comparison momentum --xaxis epoch
```

Runtime artifacts are excluded from Git while jobs are active. Add completed results in a separate, immutable results commit after all 216 runs finish.
