#!/bin/bash


for wandb_run_name in distrl_webshop_warmup_penalty_ret_3_1
do
    echo "start training job for $wandb_run_name"
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
    echo "start clearning for $wandb_run_name"
    python clear_worker.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
done
