import os
import time
import json
import wandb
import torch
import numpy as np

from distrl.misc import colorful_print
from distrl.environment import batch_interact_environment
from distrl.algorithms.digirl import DigiRLTrainer
from distrl.algorithms.distrl import DistRLTrainer
from distrl.algorithms.filteredbc import BCTrainer


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def add_trainer_index(trainer_index):
    def func(trajectory):
        for d in trajectory:
            d.update({"trainer_time_index": trainer_index})
        return trajectory
    return func


def worker_collect_loop(env,\
                agent,\
                tokenizer,\
                accelerator,\
                sequence_length,\
                clip_rho_threshold,\
                clip_pg_rho_threshold,\
                rollout_size: int = 50,\
                collect_iterations: int = 1,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                train_algorithm: str = "digirl",
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                offline_actor_iterations: int = 20,
                offline_critic_iterations: int = 20,
                offline_trajectory_critic_iterations: int = 20,
                trajectory_critic_epochs: int = 5,
                offset = 0,
                **kwargs):
    if train_algorithm == "digirl":
        trainer = DigiRLTrainer(agent=agent,\
                                accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm,
                                trajectory_critic_epochs = trajectory_critic_epochs)
    elif train_algorithm == "filteredbc":
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    elif train_algorithm == "distrl":
        print(">>> Using DistRL trainer")
        trainer = DistRLTrainer(agent=agent,\
                                accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                sequence_length = sequence_length,\
                                clip_rho_threshold = clip_rho_threshold,\
                                clip_pg_rho_threshold = clip_pg_rho_threshold,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm,
                                trajectory_critic_epochs = trajectory_critic_epochs)
    #prepare the model and optimizers
    agent.prepare()
    trainer.prepare()

    colorful_print(">>> [.] Loading Current Trainer from Host...", fg='blue')
    trainer.load_policy(os.path.join(save_path, 'trainer_current_policy.pt'))

    colorful_print(f">>> [.] Worker Collecting Online Data, Offset: {offset}...", fg='blue')
    for i in range(collect_iterations):
        # TODO: check if using the barch collection, i.e. one thread holds multiple emulators
        trajectories = batch_interact_environment(
            agent=agent,
            env=env,
            num_trajectories=rollout_size,
            accelerator=accelerator,
            use_tqdm=False,
            decode_f=decode_f,
            gamma=gamma,
            iter=offset,
            post_f=add_trainer_index(trainer.time_index)
        )
        if not trajectories:
            continue
        
        if use_wandb and accelerator.is_main_process:
            print(">>> Logging WandB...")
            info = {
                "finish_walltime": time.time(),
                "trajectory_amount": len(trajectories),
                "trainer_time_index": trainer.time_index,
                "data_offset": offset,
                "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories])
            }
            data = sum(trajectories, [])
            info.update({
                "rollout.reward.mean": np.mean([d["reward"] for d in data]),
                "rollout.reward.max": np.max([d["reward"] for d in data]),
                "rollout.reward.min": np.min([d["reward"] for d in data])
            })
            wandb.log(info)

            print(">>> Saving local logs...")
            thread_id = kwargs["thread_id"]
            log_path = os.path.join(save_path, f"traj_log_{thread_id}.jsonl")
            info.update({"trajectory_rewards": [d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]})
            prev_logs = []
            try:
                with open(log_path, "r", encoding="utf8") as f_out:
                    for idx, line in enumerate(f_out):
                        try:
                            line = line.strip()
                            line_info = json.loads(line)
                            prev_logs.append(line_info)
                        except Exception as err:
                            print(f">>> Failed reading line {idx} in previuos log file: {err}")
            except Exception as err:
                print(f">>> Failed loading previuos log file: {err}")
            prev_logs.append(info)
            with open(log_path, "w", encoding="utf8") as f_out:
                for log_line in prev_logs:
                    f_out.write(json.dumps(log_line, cls=NpEncoder) + "\n")

        # TODO: add thread-id in saved file
        suffix = f'_{kwargs["thread_id"]}' if 'thread_id' in kwargs.keys() else ''
        torch.save(trajectories, os.path.join(save_path, f'trajectories{suffix}.pt'))

            