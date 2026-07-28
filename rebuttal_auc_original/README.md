# Original-code AUC rebuttal experiments

This directory runs only the AUC experiments through the paper's three independent training entries:

- `26.1.29.ZOFOGDA_AUC/ldx_zofogda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttgda_auc.py`
- `26.1.29.ZOFOGDA_AUC/ldx_ttsgda_auc.py`

The formal matrix has 378 unique runs: the previous 288 runs plus 72 fixed-momentum ablations and 18 independent ZOFOGDA paper-setting reproductions. All use six datasets, seeds 14/42/81, and 500 epochs. The fixed ablation uses `w=0.1*w+0.9*g` independently for x and y, with full/no-x/no-y/no-both variants. The paper reproduction uses the original scheduled momentum, `M=5`, `delta=1e-3`, batch size 128, and the recorded per-dataset gamma/lambda settings.

For `M=floor(sqrt(d))`, `d` is the complete optimized primal parameter count (network parameters plus AUC scalars `a` and `b`), not the raw input-feature count. With the current one-hidden-layer width-128 model, `d = 128 * input_dim + 259`. Increasing input dimension is the scaling route used here; a five-layer network would be a separate architecture-scaling experiment and is not mixed into this matrix.

For TTGDA and TTSGDA, the workbook keeps the original paper-parameter reruns and the learning-rate-corrected (`lr_x=0.01`, `lr_y=0.1`) reruns as separate rows. They are never mixed.

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
python rebuttal_auc_original/plot_results.py --comparison fixed_momentum --xaxis epoch
```

Runtime artifacts are excluded from Git while jobs are active. Add completed results in a separate, immutable results commit after all 378 runs finish.
