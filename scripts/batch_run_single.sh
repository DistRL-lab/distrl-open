#!/bin/bash

for wandb_run_name in digirl_single_warmup_1 digirl_single_warmup_2
do
    accelerate launch --config_file config/accelerate_config/default_config.yaml run.py --config-path config/singlemachine --config-name main +wandb_run_name=${wandb_run_name}
    sleep 10
    mv /home/<usrname>/logs/host /home/<usrname>/logs/${wandb_run_name}
    sleep 10
done