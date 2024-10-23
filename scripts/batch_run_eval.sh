#!/bin/bash

for run_name in raw_autoui
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_testset_webshop
    CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_1 train_algorithm=distrl task_set=webshop task_split=test +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_1 +wandb_run_name=${wandb_run_name}
    sleep 5
done

for run_name in raw_autoui
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_trainset_webshop
    CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_1 train_algorithm=distrl task_set=webshop task_split=train +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_1 +wandb_run_name=${wandb_run_name}
    sleep 5
done


for run_name in digirl_async_warmup_1 distrl_warmup_penalty_ret_3_2 raw_autoui
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_trainset
    CUDA_VISIBLE_DEVICES=4,5,6,7
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_2 task_split=train +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_2 +wandb_run_name=${wandb_run_name}
    sleep 5
done

