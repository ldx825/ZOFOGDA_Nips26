# export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
# CUDA_VISIBLE_DEVICES=2 python -m debugpy --listen 2026 --wait-for-client run_bilevel.py \
#   --output_dir "./log" \
#   --training_tasks SST2 \
#   --test_tasks SST2 \
#   --num_tasks_per_iteration 1 \
#   --prefix_tuning True \
#   --trainer bilevel_minimax2 \
#   --model_name facebook/opt-125m \
#   --train_as_classification True \
#   --Lambda 10000 \
#   --max_steps 5000 \
#   --learning_rate 1e-7 \
#   --save_model True \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train 1000 \
#   --num_dev 500 \
#   --num_eval 1000 \
#   --seed 42 \
#   --train_set_seed 0 \
#   --verbose True \
#   --optimizer adamw \
#   --momentum 0.5 \
#   --zo_eps 1e-2 \
#   --q 1 \
#   --perturbation_mode two_side \
#   --lr_scheduler_type constant \
#   --lower_level_num_train_steps 10 \
#   --upper_optimizer "adamw" \
#   --lower_level_per_device_train_batch_size 8 \
#   --lower_level_per_device_eval_batch_size 8 \

#   # 不传递 meta_icl_train_k 等参数，让它们使用默认值None
# # -m debugpy --listen 2026 --wait-for-client

# facebook/opt-125m
# --prefix_tuning  prompt_tuning True \   --zo_eps 1e-4
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python run_bilevel.py \
  --output_dir "./log_bilevel" \
  --training_tasks SST2 \
  --test_tasks SST2 \
  --num_tasks_per_iteration 1 \
  --prompt_tuning True \
  --trainer bilevel_minimax2 \
  --model_name facebook/opt-1.3b \
  --train_as_classification True \
  --Lambda 10000 \
  --max_steps 2500 \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train 1000 \
  --num_dev 500 \
  --num_eval 1000 \
  --seed 42 \
  --train_set_seed 0 \
  --optimizer "sgd" \
  --momentum 0.0 \
  --upper_optimizer "sgd" \
  --upper_momentum 0.0 \
  --zo_eps 1e-3 \
  --q 1 \
  --perturbation_mode two_side \
  --lr_scheduler_type constant \
  --lower_level_num_train_steps 10 \
  --lower_level_per_device_train_batch_size 16 \
  --lower_level_per_device_eval_batch_size 16 \
  --upper_learning_rate 5e-4 \
  --load_bfloat16 True \

  # 不传递 meta_icl_train_k 等参数，让它们使用默认值None
# -m debugpy --listen 2026 --wait-for-client
# -m debugpy --listen 2026 --wait-for-client 