import os
import sys
import time
import json
import copy
import wandb
import torch
import asyncio
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from filelock import FileLock
from multiprocessing import Process
from accelerate.utils import broadcast

from distrl.misc import colorful_print
from distrl.data import ReplayBuffer, PriorityReplayBuffer
from distrl.environment import batch_interact_environment
from distrl.algorithms.digirl import DigiRLTrainer
from distrl.algorithms.distrl import DistRLTrainer
from distrl.algorithms.filteredbc import BCTrainer
from distrl.algorithms.worker_collect_loop import add_trainer_index
from distrl.environment.env_utils import add_mc_return
from distrl.algorithms.Eval_parallel import remote_collect_trajectories_sync


USE_DPER = False # Using DPER in replay buffer
MIN_BUFFER_SIZE = 0


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def framestack(all_trajectories):
    """Stack the adjacent frames in a trajectory"""
    new_trajectories = copy.deepcopy(all_trajectories)
    for trajectory, new_trajectory in zip(all_trajectories, new_trajectories):
        for i,(t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i  == 0:
                nt["image_features"] = np.concatenate([t["image_features"], t["image_features"]], axis = -1)
            else:
                nt["image_features"] = np.concatenate([trajectory[i-1]["image_features"], t["image_features"]], axis = -1)
            nt["next_image_features"] = np.concatenate([t["image_features"], t["next_image_features"]], axis = -1)
    return new_trajectories


def eval_loop(agent,
                tokenizer,
                accelerator,
                warmup_iter: int = 20,
                rollout_size: int = 50,
                batch_size: int = 2,
                capacity: int = 500000,
                train_iterations: int = 10,
                epochs: int = 3,
                grad_accum_steps: int = 1,
                critic_lr: float = 1e-3,
                lm_lr: float = 1e-5,
                gamma: float = 0.9,
                tau: float = 0.1,
                sequence_length: int = 5,
                clip_rho_threshold: float = 1.0,
                clip_pg_rho_threshold: float = 1.0,
                use_wandb: bool = False,
                actor_epochs: int = 3,
                train_mode: str = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                train_algorithm: str = "digirl",
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                start_checkpoint_path: str = None,
                offline_actor_iterations: int = 20,
                offline_critic_iterations: int = 20,
                offline_trajectory_critic_iterations: int = 20,
                trajectory_critic_epochs: int = 5,
                parallel: str = 'single',
                wandb_run_name: str = None,
                **kwargs):
    if train_algorithm == "distrl":
        trainer = DistRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            sequence_length=sequence_length,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            trajectory_critic_epochs=trajectory_critic_epochs
        )
    elif train_algorithm == "digirl":
        trainer = DigiRLTrainer(
            agent=agent,
            accelerator=accelerator,
            tokenizer=tokenizer,
            critic_lr=critic_lr,
            lm_lr=lm_lr,
            gamma=gamma,
            tau=tau,
            epochs=epochs,
            actor_epochs=actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            trajectory_critic_epochs=trajectory_critic_epochs
        )
    elif train_algorithm == "filteredbc":
        trainer = BCTrainer(
            agent=agent,
            tokenizer=tokenizer,
            accelerator=accelerator,
            lm_lr = lm_lr,
            epochs = actor_epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm
        )

    # Prepare the models & optimizers
    agent.prepare()
    
    # Initialize arguments
    worker_temp_path = kwargs.get('worker_temp_path')
    worker_run_path = kwargs.get('worker_run_path')
    worker_ips = kwargs.get('worker_ips')
    worker_username = kwargs.get('worker_username')
    num_threads = kwargs.get('num_threads')
    aggregated_save_path = kwargs.get('aggregated_save_path')
    synthetic = kwargs.get('synthetic')
    aggregated_file_path = os.path.join(aggregated_save_path, "aggregated_trajectories.pt")
    sync_mode = kwargs.get('sync_mode')
    task_mode = kwargs.get('task_mode')
    train_time = kwargs.get('train_time')
    evaluation_num = kwargs.get('evaluation_num')
    task_split = kwargs.get('task_split')
    task_set = kwargs.get('task_set')

    # File lock
    aggregate_lock = FileLock(aggregated_file_path + ".lock")
    model_lock = FileLock(os.path.join(save_path, "trainer_current_policy.pt.lock"))

    def process_queue():
        trajectories = []
        if os.path.exists(aggregated_file_path):
            with aggregate_lock:
                trajectories = torch.load(aggregated_file_path)
                os.remove(aggregated_file_path)
        return trajectories

    def evaluation():
        nonlocal evaluation_num
        assert train_algorithm in ['distrl', 'digirl', 'filteredbc'], "Only distrl, digirl and filteredbc are supported"
        if accelerator.is_main_process:
            trajectories = []
            print(">>> Start evaluation...")
        
        iteration_count = 0
        while len(trajectories) < evaluation_num:
            if accelerator.is_main_process:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(
                    asyncio.wait(
                        [remote_collect_trajectories_sync(
                            save_path=save_path,
                            worker_temp_path=worker_temp_path,
                            worker_run_path=worker_run_path,
                            worker_ips=worker_ips,
                            worker_username=worker_username,
                            aggregated_save_path=aggregated_save_path,
                            synthetic=synthetic,
                            aggregate_lock=aggregate_lock,
                            model_lock=model_lock,
                            offset=iteration_count*num_threads,
                            num_threads=num_threads,
                            wandb_run_name=wandb_run_name
                        )]
                    )
                )
                tmp_trajectories = process_queue()
                print(f">>> Process {len(tmp_trajectories)} trajs from queue.")

                # TODO: evaluate and save results
                trajectories.extend(framestack(tmp_trajectories))
                torch.save(trajectories, os.path.join(save_path, "test_trajectories.pt"))       
                info = {"iteration": iteration_count, "iter_start_walltime": time.time()}
                info.update({"rollout.amount": len(tmp_trajectories)})
                info.update({
                    "all.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                    "all.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                    "all.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                    "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in tmp_trajectories]),
                    "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in tmp_trajectories]),
                    "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in tmp_trajectories])
                })
                data_tmp = sum(tmp_trajectories, [])
                data = sum(trajectories, [])
                info.update({
                    "all.reward.mean": np.mean([d["reward"] for d in data]),
                    "all.reward.max": np.max([d["reward"] for d in data]),
                    "all.reward.min": np.min([d["reward"] for d in data]),
                    "rollout.reward.mean": np.mean([d["reward"] for d in data_tmp]),
                    "rollout.reward.max": np.max([d["reward"] for d in data_tmp]),
                    "rollout.reward.min": np.min([d["reward"] for d in data_tmp])
                })
                wandb.log(info)
                
                print(">>> Saving local logs...")
                log_path = os.path.join(save_path, f"test_log_{task_split}.jsonl")
                current_log = {
                    "iteration": info["iteration"], 
                    "iter_start_walltime": info["iter_start_walltime"]
                }
                for check_key in [
                    "rollout.amount", "rollout.mean", "rollout.max", "rollout.min", "rollout.reward.mean", "rollout.reward.max", "rollout.reward.min",
                    "all.mean", "all.max", "all.min", "all.reward.mean", "all.reward.max", "all.reward.min"
                ]:
                    if check_key in info.keys():
                        current_log[check_key] = info[check_key]
                
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
                prev_logs.append(current_log)
                with open(log_path, "w", encoding="utf8") as f_out:
                    for log_line in prev_logs:
                        f_out.write(json.dumps(log_line, cls=NpEncoder) + "\n")
            
            iteration_count += 1
            time.sleep(20)


    def main():
        # Start the remote trajectory collection in the backgroun
        assert parallel == "host" and sync_mode == "sync" and task_mode == "evaluate", "Only support remote and async mode for evaluation!"
        evaluation()

    # Run the main function
    main()
        
