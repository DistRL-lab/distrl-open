import os
import hydra
import wandb
import logging
import asyncio
import asyncssh
import transformers
import asyncssh.logging
import accelerate.logging

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from queue import Queue
from datetime import timedelta
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator #Parallel GPU training framework
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
# transformers.logging.set_verbosity_error()

from distrl.misc import colorful_print
from distrl.models import AutoUIAgent
from distrl.algorithms import offpolicy_train_loop, eval_loop, worker_collect_loop
from distrl.environment import BatchedAndroidEnv
from distrl.environment.android import EndResultEvaluator
from distrl.environment.android import autoui_translate_action


logging.basicConfig(level=logging.WARNING)
asyncssh.logging.set_log_level(logging.WARNING)
# TODO: reduce accelerator logging


def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> [!] Huggingface token not found.")

    # Set up distribution utilities etc.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_value = timedelta(seconds=2400)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timeout_value)
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs, ddp_kwargs], project_dir=config.save_path)
    device = accelerator.device
    decode_f = lambda x:x

    # Make agent
    agent = AutoUIAgent(device=device, accelerator=accelerator, 
                        temperature=config.temperature, do_sample=config.do_sample, 
                        policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                        cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                        use_lora=config.use_lora)
    tokenizer = agent.tokenizer

    # Set up environtment parameters
    if config.parallel == "worker" or config.parallel == "single" and accelerator.is_main_process:
        all_tasks = load_task_file(config.assets_path, config.task_set, config.task_split)
        bsize = config.bsize
        thread_id = int(config.thread_id) if "thread_id" in config else 0
        base_port = 5554 + 2 * thread_id * bsize
        evaluators = [EndResultEvaluator(config.gemini_key, config.task_set)] * bsize
        assert len(evaluators) == bsize, ">>> [!] Failed creating enough evaluators!"
        translate_action = autoui_translate_action
        use_feature_extractor = True

    # Set up WandB
    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        run_name = config.parallel if config.parallel == "host" or config.parallel == "single" else f"thread-{thread_id}"
        group_name = config.parallel if config.parallel == "host" or config.parallel == "single" else config.worker_name
        project_name = config.wandb_run_name if "wandb_run_name" in config else config.train_algorithm
        wandb.init(project=project_name, group=group_name, name=run_name, config=dict(config))

    def construct_env(sample_mode):
        """Method for constructing environment"""
        if config.sync_mode == "async":
            avd_suffix = f"_{thread_id}_" if "thread_id" in config else ""
            avd_home_suffix = f"{thread_id}" if "thread_id" in config else ""
            cache_avd_names = [f"test{avd_suffix}{i}" for i in range(1, 1+bsize)]
            android_avd_homes = [config.android_avd_home + avd_home_suffix for i in range(1, 1+bsize)]
        else:
            cache_avd_names = [f"test_{(i-1)%8}_{(i-1)//8+1}" for i in range(1, 1+bsize)]
            android_avd_homes = [config.android_avd_home + f"{(i-1)%8}" for i in range(1, 1+bsize)]
        env = BatchedAndroidEnv(
            avd_name=config.avd_name, 
            cache_avd_names=cache_avd_names, 
            android_avd_homes=android_avd_homes,
            emulator_path=config.emulator_path, 
            adb_path=config.adb_path, 
            udids=[f"emulator-{base_port+2*i}" for i in range(bsize)],
            max_steps=config.max_steps-1, # will have 1 dangling step after stop signal is triggered
            run_headless=True, 
            use_feature_extractor=use_feature_extractor, 
            device=accelerator.device,
            translate_action=translate_action,
            evaluators=evaluators,
            temp_path=os.path.join(config.save_path, "images"),
            save_images=config.save_images,
            all_tasks=all_tasks,
            task_split=config.task_split,
            sample_mode=sample_mode
        )
        return env
    
    env = None
    # autoui will be trained first then evaluated
    if config.parallel == "host":
        if config.task_mode == "evaluate":
            eval_loop(
                agent=agent,
                tokenizer=tokenizer,
                accelerator=accelerator,
                decode_f=decode_f,
                **config
            )
        elif config.task_mode == "train" or config.task_mode == "collect":
            offpolicy_train_loop(
                tokenizer=tokenizer,
                agent=agent,
                accelerator=accelerator,
                decode_f=decode_f,
                **config
            )
    elif config.parallel == "single":
        if accelerator.is_main_process:
            env = construct_env(sample_mode=config.eval_sample_mode)
        offpolicy_train_loop(
            tokenizer=tokenizer,
            agent=agent,
            accelerator=accelerator,
            decode_f=decode_f,
            env=env,
            **config
        )
    elif config.parallel == "worker":
        if config.task_mode == "evaluate":
            if accelerator.is_main_process:
                env = construct_env(sample_mode=config.eval_sample_mode)
            worker_collect_loop(
                env=env,
                agent=agent,
                tokenizer=tokenizer,
                accelerator=accelerator,
                decode_f=decode_f,
                **config
            )
        else:
            if accelerator.is_main_process:
                env = construct_env(sample_mode="random")
            worker_collect_loop(
                env=env,
                agent=agent,
                tokenizer=tokenizer,
                accelerator=accelerator,
                decode_f=decode_f,
                **config
            )

if __name__ == "__main__":
    main()
