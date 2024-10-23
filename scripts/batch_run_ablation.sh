#!/bin/bash

for wandb_run_name in distrl_warmup_penalty_3
do
    echo "start training job for $wandb_run_name"
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/multimachine --config-name host train_algorithm=distrl use_retrace=False use_entropy=False use_dper=False +wandb_run_name=${wandb_run_name}
    sleep 5
    echo "start clearning for $wandb_run_name"
    python clear_worker.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
done

for run_name in distrl_warmup_penalty_3
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_testset_general
    CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_1 train_algorithm=distrl task_set=general task_split=test +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_1 +wandb_run_name=${wandb_run_name}
    sleep 5
done

for wandb_run_name in distrl_warmup_penalty_dperw1_50_ret_3
do
    echo "start training job for $wandb_run_name"
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/multimachine --config-name host train_algorithm=distrl use_retrace=True use_entropy=False use_dper=True dper_w1=50 +wandb_run_name=${wandb_run_name}
    sleep 5
    echo "start clearning for $wandb_run_name"
    python clear_worker.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
done

for run_name in distrl_warmup_penalty_dperw1_50_ret_3
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_testset_general
    CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_1 train_algorithm=distrl task_set=general task_split=test +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_1 +wandb_run_name=${wandb_run_name}
    sleep 5
done

for wandb_run_name in distrl_warmup_penalty_dperw1_50_entropy_ret_3
do
    echo "start training job for $wandb_run_name"
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/multimachine --config-name host train_algorithm=distrl use_retrace=True use_entropy=True use_dper=True dper_w1=50 +wandb_run_name=${wandb_run_name}
    sleep 5
    echo "start clearning for $wandb_run_name"
    python clear_worker.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
done

for run_name in distrl_warmup_penalty_dperw1_50_entropy_ret_3
do
    local_path=/home/distrl/logs/host_${run_name}
    wandb_run_name=test_${run_name}_testset_general
    CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "Start evaluaton job for $wandb_run_name, local path: $local_path"
    python run.py --config-path config/evaluate --config-name host_1 train_algorithm=distrl task_set=general task_split=test +wandb_run_name=${wandb_run_name} +save_path=${local_path}
    sleep 5
    echo "Start clearning for $wandb_run_name"
    python clear_worker_eval.py --config-path config/evaluate --config-name host_1 +wandb_run_name=${wandb_run_name}
    sleep 5
done
