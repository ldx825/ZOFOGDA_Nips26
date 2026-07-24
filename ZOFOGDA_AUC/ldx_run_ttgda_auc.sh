#!/bin/bash
set -e

DATA_DIR=./data
LOG_DIR=./result
DEVICE=cuda
SEED=42

# 所有数据集列表
DATASETS=(climate german svmguide1 spambase a9a ijcnn1 covtype connect acoustic)

# TTGDA 超参数搜索
for dataset in "${DATASETS[@]}"; do
  for lr_x in 1e-2 5e-3 1e-3; do
    for lr_y in 1e-1 5e-2 1e-2; do
      for y_steps in 1 2; do

        RUN_NAME="ttgda_${dataset}_lx${lr_x}_ly${lr_y}_ys${y_steps}"
        echo "=== Running ${RUN_NAME} ==="

        python ldx_ttgda_auc.py \
          --dataset "${dataset}" \
          --data_dir "${DATA_DIR}" \
          --log_dir "${LOG_DIR}/${dataset}" \
          --log_prefix "ttgda_auc" \
          --device "${DEVICE}" \
          --seed "${SEED}" \
          --epochs 200 \
          --batch_size 128 \
          --hidden 128 \
          --lr_x "${lr_x}" \
          --lr_y "${lr_y}" \
          --y_steps_per_batch "${y_steps}" \
          --v_clip 10.0 \
          --standardize

      done
    done
  done
done