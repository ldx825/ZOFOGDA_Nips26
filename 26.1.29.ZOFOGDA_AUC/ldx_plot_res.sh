#!/bin/bash
set -e

LOG_DIR=./result
IMG_DIR=./img

DATASETS=(climate german svmguide1 spambase a9a ijcnn1 covtype connect acoustic)

for dataset in "${DATASETS[@]}"; do
  echo "=== Plotting ${dataset} ==="
  python plot_res_cli.py \
    --dataset "${dataset}" \
    --log_dir "${LOG_DIR}" \
    --img_dir "${IMG_DIR}" \
    --auc_as_percent \
    --skip_missing
done
