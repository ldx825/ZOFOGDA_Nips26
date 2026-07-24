#!/bin/bash
set -e

DATA_DIR=./data
LOG_DIR=./result
DEVICE=cuda:0

# 例：只改 dataset 和几个超参，就能批量跑
# climate german svmguide1 spambase a9a ijcnn1 covtype connect acoustic
for dataset in spambase
do
  python ldx_zofogda_auc.py \
    --dataset ${dataset} \
    --data_dir ${DATA_DIR} \
    --log_dir ${LOG_DIR}/${dataset} \
    --epochs 500 \
    --batch_size 256 \
    --hidden 128 \
    --eta_k 10.0 --eta_m 1000.0 --c1 3.0 --c2 3.0 \
    --gamma 0.01 --lam 0.01 \
    --zo_delta 1e-3 --zo_q 5 \
    --v_clip 10.0 \
    --device $DEVICE \
    --seed 42
done


# #!/bin/bash
# set -e

# DATA_DIR=./data
# LOG_DIR=./result
# DEVICE=cuda:4
# SEED=14


# # 所有数据集列表
# # DATASETS=(climate german svmguide1 spambase a9a ijcnn1 covtype connect acoustic)
# DATASETS=(connect)

# # TTGDA 超参数搜索
# for dataset in "${DATASETS[@]}"; do
#   for lr_x in 1e-4; do
#     for lr_y in 1e-6; do
#       for y_steps in 1; do

#         RUN_NAME="ttgda_${dataset}_lx${lr_x}_ly${lr_y}_ys${y_steps}"
#         echo "=== Running ${RUN_NAME} ==="

#         python ldx_ttgda_auc.py \
#           --dataset "${dataset}" \
#           --data_dir "${DATA_DIR}" \
#           --log_dir "${LOG_DIR}/${dataset}" \
#           --log_prefix "ttgda_auc" \
#           --device "${DEVICE}" \
#           --seed "${SEED}" \
#           --epochs 500 \
#           --batch_size 128 \
#           --hidden 128 \
#           --lr_x "${lr_x}" \
#           --lr_y "${lr_y}" \
#           --y_steps_per_batch "${y_steps}" \
#           --v_clip 10.0 \
#           --standardize

#       done
#     done
#   done
# done
