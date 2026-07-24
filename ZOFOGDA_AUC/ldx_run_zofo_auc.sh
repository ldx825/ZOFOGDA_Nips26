#!/bin/bash
set -e

DATA_DIR=./data
LOG_DIR=./result

# 例：只改 dataset 和几个超参，就能批量跑
# climate german svmguide1 spambase a9a ijcnn1 covtype connect acoustic
for dataset in a9a
do
  python ldx_zofogda_auc.py \
    --dataset ${dataset} \
    --data_dir ${DATA_DIR} \
    --log_dir ${LOG_DIR}/${dataset} \
    --epochs 200 \
    --batch_size 128 \
    --hidden 128 \
    --eta_k 10.0 --eta_m 1000.0 --c1 3.0 --c2 3.0 \
    --gamma 0.01 --lam 0.01 \
    --zo_delta 1e-4 --zo_q 5 \
    --v_clip 10.0 \
    --seed 42
done
