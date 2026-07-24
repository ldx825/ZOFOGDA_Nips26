# zofo_adagda
# prompt_tuning  prefix_tuning  lora
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
CUDA_VISIBLE_DEVICES=0 python run_nonsmooth.py \
  --output_dir "./log" \
  --training_tasks SST2 \
  --test_tasks SST2 \
  --num_tasks_per_iteration 1 \
  --prompt_tuning True \
  --trainer zofo_adagda \
  --model_name facebook/opt-1.3b \
  --train_as_classification True \
  --Lambda 10000 \
  --max_steps 2201 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --upper_learning_rate 1e-5 \
  --lower_level_num_train_steps 1 \
  --num_train 500 \
  --num_dev 1000 \
  --num_eval 1000 \
  --seed 42 \
  --train_set_seed 0 \
  --zo_eps 1e-4 \
  --perturbation_mode two_side \
  --optimizer "adamw" \
  --upper_optimizer "adamw" \
  --ada_mu 0.9 \
  --lower_level_per_device_train_batch_size 8 \
  --lower_level_per_device_eval_batch_size 8 \
  --ada_k 1.0 \
  --ada_m 1.0 \
  --ada_c1 0.9 \
  --ada_c2 0.9 \
  --ada_gamma 1e-4 \
  --ada_lambd 1e-4 \
  --q 1 \
  --q_p 5 \

  # 不传递 meta_icl_train_k 等参数，让它们使用默认值None
# -m debugpy --listen 2026 --wait-for-client

    # # ldx:add:Nonsmooth arguments
    # mu: float = 0.9
    # rho: float = 1e-8
    # gamma: float = 1e-2
    # eta: float = 1e-2
    # lambd: float = 2.0
    # beta: float = 0.9
    # alpha: float = 0.7