#!/bin/bash
# # 找到你的 CUDA 安装路径，通常是 /usr/local/cuda 或 /usr/local/cuda-版本号
# export CUDA_HOME=/usr/local/cuda  # 或者你的实际 CUDA 路径
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 然后重新运行你的程序

# 任务列表：SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP
TASK=SST2

# 训练方法：none, regular, zo_sgd, zo_conserv, zo_adam, zo_sign_opt, forward_grad
METHOD=zo_sgd

# 训练参数
MODEL_NAME=facebook/opt-125m
STEPS=1000
LEARNING_RATE=1e-5
BATCH_SIZE=8
# GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN=100
NUM_DEV=50
NUM_EVAL=100
SEED=42
TRAIN_SET_SEED=0

# 优化器参数
OPTIMIZER=adamw
MOMENTUM=0.0
ZO_EPS=1e-3
Q=1
PERTURBATION_MODE=two_side

# 提示调优参数（启用提示调优）
PROMPT_TUNING=true
NUM_VIRTUAL_TOKENS=10
PROMPT_INIT_BY_REAL_TOKENS=false

# LoRA参数（禁用）
LORA=false
LORA_ALPHA=16
LORA_R=8

# 前缀调优参数（禁用）
PREFIX_TUNING=false
NUM_PREFIX=5
NO_REPARAM=true
PREFIX_INIT_BY_REAL_ACT=true

# 模型量化参数
LOAD_FLOAT16=false
LOAD_BFLOAT16=false
LOAD_INT8=false

# 训练配置
TEMPLATE_VER=1 # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.
MAX_LENGTH=2048
TRAIN_AS_CLASSIFICATION=false
ONLY_TRAIN_OPTION=true
SAVE_MODEL=true
NO_EVAL=false

# 输出目录
TAG="${METHOD}-${TASK}-${TEMPLATE_VER}-${MODEL_NAME##*/}-PROMPT-STEP${STEPS}-${OPTIMIZER}-LR${LEARNING_RATE}-ZOEPS${ZO_EPS}-Q${Q}"
OUTPUT_DIR="./log"
LOGGING_DIR="${OUTPUT_DIR}/logs"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGGING_DIR}"

# 设置CUDA设备
# export CUDA_VISIBLE_DEVICES=0

echo "Starting training with configuration:"
echo "Task: ${TASK}"
echo "Method: ${METHOD}"
echo "Model: ${MODEL_NAME}"
echo "Steps: ${STEPS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Tag: ${TAG}"

# --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
# 运行训练
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 2026 --wait-for-client run.py \
  --task_name "${TASK}" \
  --trainer "${METHOD}" \
  --model_name "${MODEL_NAME}" \
  --max_steps "${STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --num_train "${NUM_TRAIN}" \
  --num_dev "${NUM_DEV}" \
  --num_eval "${NUM_EVAL}" \
  --seed "${SEED}" \
  --train_set_seed "${TRAIN_SET_SEED}" \
  \
  --optimizer "${OPTIMIZER}" \
  --momentum "${MOMENTUM}" \
  --zo_eps "${ZO_EPS}" \
  --q "${Q}" \
  --perturbation_mode "${PERTURBATION_MODE}" \
  \
  --prompt_tuning "${PROMPT_TUNING}" \
  --num_virtual_tokens "${NUM_VIRTUAL_TOKENS}" \
  --prompt_init_by_real_tokens "${PROMPT_INIT_BY_REAL_TOKENS}" \
  \
  --lora "${LORA}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_r "${LORA_R}" \
  \
  --prefix_tuning "${PREFIX_TUNING}" \
  --num_prefix "${NUM_PREFIX}" \
  --no_reparam "${NO_REPARAM}" \
  --prefix_init_by_real_act "${PREFIX_INIT_BY_REAL_ACT}" \
  \
  --load_float16 "${LOAD_FLOAT16}" \
  --load_bfloat16 "${LOAD_BFLOAT16}" \
  --load_int8 "${LOAD_INT8}" \
  \
  --template_ver "${TEMPLATE_VER}" \
  --max_length "${MAX_LENGTH}" \
  --train_as_classification "${TRAIN_AS_CLASSIFICATION}" \
  --only_train_option "${ONLY_TRAIN_OPTION}" \
  --save_model "${SAVE_MODEL}" \
  --no_eval "${NO_EVAL}" \
  \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  --run_name "${TAG}" \
  --tag "${TAG}" \
  --overwrite_output_dir

echo "Training completed. Results saved to: ${OUTPUT_DIR}"