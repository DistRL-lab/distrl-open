#!/bin/bash

for num_threads in 8
do
    echo "start async collecting job with $num_threads collecing thread per machine"
    wandb_run_name=collect_webshop_1_async_$num_threads
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/multimachine --config-name host task_mode=collect sync_mode=async num_threads=$num_threads +wandb_run_name=${wandb_run_name}
    sleep 5
    echo "start clearning for $wandb_run_name"
    python clear_worker.py --config-path config/multimachine --config-name host +wandb_run_name=${wandb_run_name}
    sleep 5
done
