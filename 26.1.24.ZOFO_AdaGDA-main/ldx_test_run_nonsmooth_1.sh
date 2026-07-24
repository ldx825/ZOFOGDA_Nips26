#!/bin/bash


# ada_gamma_values=(1e-4 1e-5 1e-6)
# ada_lambd_values=(1e-2 1e-3 1e-4 1e-5)
# 设置环境变量
# 设置环境变量来禁用W&B
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export HF_ENDPOINT=https://hf-mirror.com

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 定义要遍历的参数数组
Lambda_values=(10000)
zo_eps_values=(1e-3)
ada_mu_values=(0.999)
ada_gamma_values=(2e-4)
ada_lambd_values=(2e-4)
seed_values=(81)

task=WSC
lora=True
prefix=False
prompt=False

ada_k=1.0
ada_m=1.0
ada_c1=1.386
ada_c2=1.386

batch_size=16
max_steps=2000

# 循环计数
counter=1

# lora prefix_tuning prompt_tuning

# 嵌套循环遍历所有参数组合
for Lambda in "${Lambda_values[@]}"; do
  for zo_eps in "${zo_eps_values[@]}"; do
    for ada_mu in "${ada_mu_values[@]}"; do
      for ada_gamma in "${ada_gamma_values[@]}"; do
        for ada_lambd in "${ada_lambd_values[@]}"; do
          for seed in "${seed_values[@]}"; do
            echo "====================================================="
            echo "执行第 $counter 次实验:"
            echo "Lambda=$Lambda, zo_eps=$zo_eps, ada_mu=$ada_mu"
            echo "ada_gamma=$ada_gamma, ada_lambd=$ada_lambd, seed=$seed"
            echo "====================================================="
            
            # 创建唯一的输出目录
            output_dir="./log_new_2"
            
            python run_nonsmooth.py \
              --output_dir "$output_dir" \
              --training_tasks $task \
              --test_tasks $task \
              --num_tasks_per_iteration 1 \
              --lora $lora \
              --prefix_tuning $prefix \
              --prompt_tuning $prompt \
              --trainer zofo_adagda \
              --model_name facebook/opt-1.3b \
              --train_as_classification True \
              --Lambda "$Lambda" \
              --max_steps $max_steps \
              --per_device_train_batch_size $batch_size \
              --per_device_eval_batch_size $batch_size \
              --learning_rate 1e-5 \
              --upper_learning_rate 1e-5 \
              --lower_level_num_train_steps 1 \
              --num_train 1000 \
              --num_dev 500 \
              --num_eval 1000 \
              --seed "$seed" \
              --train_set_seed 0 \
              --zo_eps "$zo_eps" \
              --perturbation_mode two_side \
              --optimizer "adamw" \
              --upper_optimizer "adamw" \
              --ada_mu "$ada_mu" \
              --lower_level_per_device_train_batch_size $batch_size \
              --lower_level_per_device_eval_batch_size $batch_size \
              --ada_k $ada_k \
              --ada_m $ada_m \
              --ada_c1 $ada_c1 \
              --ada_c2 $ada_c2 \
              --ada_gamma "$ada_gamma" \
              --ada_lambd "$ada_lambd" \
              --q 1 \
              --q_p 5 \
              --load_bfloat16 True
            
            # 检查命令执行是否成功
            if [ $? -eq 0 ]; then
              echo "第 $counter 次实验执行成功！"
              echo "结果保存在: $output_dir"
            else
              echo "第 $counter 次实验执行失败！"
            fi
            
            echo ""
            ((counter++))
            
            # 可选：每次运行后等待一段时间，避免资源冲突
            # sleep 10
          done
        done
      done
    done
  done
done

echo "所有实验执行完成！总共执行了 $((counter-1)) 次实验。"