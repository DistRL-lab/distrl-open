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
from distrl.algorithms.Sync_parallel import remote_collect_trajectories_sync
from distrl.algorithms.Async_parallel import remote_collect_trajectories_async


# USE_DPER = False # Using DPER in replay buffer
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


def label_trajectories(trajectories, agent):
    """Label trajectories with critic"""
    baselines = []
    for i in range(0, len(trajectories), 16):
        observations = [t[0]["observation"] for t in trajectories[i:i+16]]
        with agent.accelerator.no_sync(agent.trajectory_critic):
            with torch.no_grad():
                v = agent.trajectory_critic(observations)
                v = torch.nn.Softmax(dim = -1)(v)[:,1]
                baselines.append(v.flatten())
    baselines = torch.cat(baselines, dim = -1)
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)
    return torch.zeros((len(trajectories)))


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


def filterbc_buffer(all_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1):
    """Filter BC buffer"""
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    cutoff = np.quantile(trajectory_rewards, 1 - 0.1)
    filtered_trajectories = []
    for t, b in zip(all_trajectories, trajectory_rewards):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    if not use_dper:
        filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    else:
        filtered_buffer = PriorityReplayBuffer(batch_size=batch_size, capacity=capacity, gamma=gamma, w1=dper_w1)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer


def filter_buffer(all_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1):
    """Filter replay buffer"""
    baselines = label_trajectories(all_trajectories, agent).numpy().flatten()
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    baselines = trajectory_rewards - baselines
    cutoff = np.quantile(baselines, 1 - 0.1)
    filtered_trajectories = []
    for t, b in zip(all_trajectories, baselines):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    if not use_dper:
        filtered_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    else:
        filtered_buffer = PriorityReplayBuffer(batch_size=batch_size, capacity=capacity, gamma=gamma, w1=dper_w1)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer


def offpolicy_train_loop(agent,
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
                         env = None,
                         use_retrace = True,
                         use_entropy = False,
                         use_dper = False,
                         dper_w1 = 50,
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
            trajectory_critic_epochs=trajectory_critic_epochs,
            use_retrace=use_retrace,
            use_entropy=use_entropy
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
    if start_checkpoint_path is not None:
        print(f">>> Load initial checkpoint from {start_checkpoint_path}...")
        trainer.load(os.path.join(start_checkpoint_path, 'trainer.pt'))
    trainer.prepare()

    # Load offline data
    loaded_trajs = False
    if offline_data_path is not None:
        all_trajectories = torch.load(offline_data_path, weights_only=False)
        # all_trajectories = framestack(all_trajectories)
        if accelerator.is_main_process:
            print(f">>> The number of offline trajectories is {len(all_trajectories)}")
        # all_trajectories = [add_mc_return(t, gamma=gamma) for t in all_trajectories]
        train_trajectories = all_trajectories[:int(len(all_trajectories)*0.8)]
        val_trajectories = all_trajectories[int(len(all_trajectories)*0.8):]
        loaded_trajs = 'scratch'
        
    # Resume training from the saved checkpoint
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        assert train_mode != "offline", "Only online/off2on training can be resumed"
        trainer.load(os.path.join(save_path, 'trainer.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'), weights_only=False)
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'), weights_only=False)
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'), weights_only=False)
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'), weights_only=False)
        if accelerator.is_main_process:
            print(f">>> The number of saved online trajectories is {len(all_trajectories)}")
            if use_wandb:
                print(">>> Loading from checkpoint")
        loaded_trajs = 'resume'
            
    if not loaded_trajs:
        train_trajectories = []
        val_trajectories = []
        all_trajectories = []

    # Build and load replay buffer
    if not use_dper:
        replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=capacity)
    else:
        replay_buffer = PriorityReplayBuffer(batch_size=batch_size, capacity=capacity, gamma=gamma, w1=dper_w1)
        validation_buffer = PriorityReplayBuffer(batch_size=batch_size, capacity=capacity, gamma=gamma, w1=dper_w1)
    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])
    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)

    # Offline training
    if not os.path.exists(os.path.join(save_path, 'trainer.pt')):
        #if nothing in the trainer only the offline trainer is saved
        if os.path.exists(os.path.join(save_path, 'trainer_offline.pt')):
            trainer.load(os.path.join(save_path, 'trainer_offline.pt'))
            print(">>> Loading from offline trainer")
        else:
            if offline_data_path is not None and train_mode != "online":
                print(">>> Training Offline")
                info = {}
                # offline training will never use the trajectory-level critic filter, so please use filterbc_buffer
                filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                
                if train_algorithm == "filteredbc":
                    # filtered BC training phase
                    for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update(filtered_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                elif train_algorithm == "digirl" or "distrl":
                    # digirl training phase
                    for i in tqdm(range(offline_trajectory_critic_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                    for i in tqdm(range(offline_critic_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_critic(replay_buffer, validation_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)

                    print(">>> Training Policy")
                    for i in tqdm(range(offline_actor_iterations), disable=not accelerator.is_main_process):
                        info.update(trainer.update_policy(filtered_buffer, filtered_validation_buffer))
                        if use_wandb and accelerator.is_main_process:
                            wandb.log(info)
                if accelerator.is_main_process:
                    trainer.save(os.path.join(save_path, 'trainer_offline.pt'))

    # Start online training
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if accelerator.is_main_process:
        print(">>> Start iterations")
    if loaded_trajs == "resume":
        resume_iter = len(all_trajectories) // rollout_size
    else:
        resume_iter = 0

    # Save the initial trainer
    if accelerator.is_main_process:
        trainer.save_policy(os.path.join(save_path, "trainer_current_policy.pt"))
    
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
    collect_num = kwargs.get('collect_num', None)
    collect_time = kwargs.get('collect_time', None)
    worker_rollout_size = kwargs.get('worker_rollout_size', None)

    # File lock
    aggregate_lock = FileLock(aggregated_file_path + ".lock")
    model_lock = FileLock(os.path.join(save_path, "trainer_current_policy.pt.lock"))

    def start_remote_collection():
        sys.stdout = open(os.path.join(save_path, "collection.log"), 'a')
        sys.stderr = open(os.path.join(save_path, "collection.log"), 'a')
        asyncio.run(
            remote_collect_trajectories_async(
                save_path=save_path,
                worker_temp_path=worker_temp_path,
                worker_run_path=worker_run_path,
                worker_ips=worker_ips,
                worker_username=worker_username,
                trainer=trainer,
                num_threads=num_threads,
                aggregated_save_path=aggregated_save_path,
                synthetic=synthetic,
                aggregate_lock=aggregate_lock,
                model_lock=model_lock,
                wandb_run_name=wandb_run_name
            )
        )

    def process_queue():
        trajectories = []
        if os.path.exists(aggregated_file_path):
            with aggregate_lock:
                trajectories = torch.load(aggregated_file_path)
                os.remove(aggregated_file_path)
        return trajectories

    def train():
        nonlocal train_trajectories, val_trajectories, replay_buffer, validation_buffer, all_trajectories  # Declare these variables as nonlocal
        nonlocal resume_iter
        assert train_mode != "offline", "Only online/off2on need to iteratively train; offline should directly go to eval loop after training"
        assert train_algorithm in ['distrl', 'digirl', 'filteredbc'], "Only distrl, digirl and filteredbc are supported"
        current_iter = resume_iter
        if accelerator.is_main_process:
            progress_bar = tqdm(total=train_iterations, initial=resume_iter, disable=not accelerator.is_main_process)
            trajectories = []
            init_time = time.time()
            print(">>> Start train")
        
        while current_iter < train_iterations:
            buffer_update = torch.zeros(1, dtype=torch.bool, device=accelerator.device)
            if accelerator.is_main_process:
                info = {"iteration": current_iter, "iter_start_walltime": time.time()}
            
            tmp_trajectories = []
            if parallel == 'single':
                assert sync_mode == "sync", "Only support synchronous mode for singlemachine setting!"
                tmp_trajectories = batch_interact_environment(
                    agent=agent,
                    env=env,
                    num_trajectories=rollout_size,
                    accelerator=accelerator,
                    use_tqdm=False,
                    decode_f=decode_f,
                    gamma=gamma,
                    iter=0,
                    post_f=add_trainer_index(trainer.time_index),
                    use_wandb=False
                )
            if parallel == 'host' and accelerator.is_main_process:
                if sync_mode == "sync":
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(
                        asyncio.wait(
                            [remote_collect_trajectories_sync(
                                save_path=save_path,
                                worker_temp_path=worker_temp_path,
                                worker_run_path=worker_run_path,
                                worker_ips=worker_ips,
                                worker_username=worker_username,
                                trainer=trainer,
                                aggregated_save_path=aggregated_save_path,
                                synthetic=synthetic,
                                aggregate_lock=aggregate_lock,
                                model_lock=model_lock,
                                wandb_run_name=wandb_run_name,
                                rollout_size=worker_rollout_size
                            )]
                        )
                    )
                tmp_trajectories = process_queue()
                
            if accelerator.is_main_process:
                print(f">>> Process {len(tmp_trajectories)} trajs from queue.")
                if tmp_trajectories:
                    trajectories.extend(framestack(tmp_trajectories))
                    info.update({"rollout.amount": len(tmp_trajectories)})
                current_train_size = sum([len(traj) for traj in trajectories[:int(len(trajectories) * 0.8)]]) \
                    if trajectories else 0
                
                # Update buffer if there's new collected data
                if trajectories and current_train_size + replay_buffer.all_size >= MIN_BUFFER_SIZE:
                    buffer_update[0] = True
                    colorful_print(f">>> Length of collected trajectories: {len(trajectories)}", fg='green')
                    info.update({
                        "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                        "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                        "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),
                    })

                    # Process new trajectories
                    all_trajectories += trajectories
                    new_train_trajectories = trajectories[:int(len(trajectories) * 0.8)]
                    new_val_trajectories = trajectories[int(len(trajectories) * 0.8):]
                    trajectories = []
                    train_trajectories += new_train_trajectories
                    val_trajectories += new_val_trajectories
                    data = sum(new_train_trajectories, [])
                    val_data = sum(new_val_trajectories, [])
                    for d in data:
                        replay_buffer.insert(**d)
                    for d in val_data:
                        validation_buffer.insert(**d)

                    # Update DPER priorities
                    if use_dper:
                        replay_buffer.update_priorities(agent)
                        validation_buffer.update_priorities(agent)

                    info.update({
                        "rollout.reward.mean": np.mean([d["reward"] for d in data]),
                        "rollout.reward.max": np.max([d["reward"] for d in data]),
                        "rollout.reward.min": np.min([d["reward"] for d in data])
                    })

                    print(">>> Saving Replay Buffer")
                    torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
                    torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
                    torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
                    torch.save(val_trajectories, os.path.join(save_path, 'val_trajectories.pt'))
                    print(">>> Saved Replay Buffer")
                
                info.update({"buffer.all_size": replay_buffer.all_size})
            else:   
                info = {}

            buffer_update = broadcast(buffer_update)
            if buffer_update:
                replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'), weights_only=False)
                all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'), weights_only=False)
                train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'), weights_only=False)
                val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'), weights_only=False)

            if replay_buffer.all_size < MIN_BUFFER_SIZE:
                time.sleep(10)
                continue
            
            if buffer_update or not 'filtered_buffer' in locals() or not 'filtered_validation_buffer' in locals():
                if train_algorithm == "filteredbc":
                    filtered_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                    filtered_validation_buffer = filterbc_buffer(val_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                elif train_algorithm == 'digirl':
                    filtered_buffer = filter_buffer(train_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                    filtered_validation_buffer = filter_buffer(val_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                elif train_algorithm == 'distrl':
                    filtered_buffer = filter_buffer(train_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
                    filtered_validation_buffer = filter_buffer(val_trajectories, batch_size, capacity, agent, gamma, use_dper, dper_w1)
            
            if 'filtered' in train_algorithm:
                info.update(trainer.update(filtered_buffer, no_update_actor=(current_iter < warmup_iter)))
            else:
                info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))
                info.update(trainer.update(replay_buffer, validation_buffer, filtered_buffer, filtered_validation_buffer, no_update_actor=(current_iter < warmup_iter)))
            
            if (current_iter + 1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
                print(">>> Saving")
                trainer.save(os.path.join(save_path, 'trainer.pt'))
                torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
                torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
                torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
                torch.save(val_trajectories, os.path.join(save_path, 'val_trajectories.pt'))
                
            # TODO: mark to avoid redundant trainer distributing operation
            trainer.time_index += 1
            current_iter += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                with model_lock:
                    trainer.save_policy(os.path.join(save_path, "trainer_current_policy.pt"))

            early_stop = torch.zeros(1, dtype=torch.bool, device=accelerator.device)
            if use_wandb and accelerator.is_main_process:
                end_time = time.time()
                info.update({
                    "iter_finish_walltime": end_time,
                    "new_trainer_time_index": trainer.time_index
                })
                wandb.log(info)
                if (end_time - init_time) / 60 > train_time:
                    print(f">>> Early stop at minutes {(end_time - init_time) / 60}...")
                    early_stop[0] = True

            # Save logs to local path
            if accelerator.is_main_process:
                print(">>> Saving local logs...")
                log_path = os.path.join(save_path, f"train_log.jsonl")
                current_log = {
                    "iteration": info["iteration"], 
                    "iter_start_walltime": info["iter_start_walltime"],
                    "iter_finish_walltime": info["iter_finish_walltime"],
                    "new_trainer_time_index": info["new_trainer_time_index"]
                }
                for check_key in ["rollout.amount", "rollout.mean", "rollout.max", "rollout.min", "rollout.reward.mean", "rollout.reward.max", "rollout.reward.min"]:
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
            
            early_stop = broadcast(early_stop)
            if early_stop:
                break

    def collect():
        nonlocal collect_num, collect_time, num_threads, worker_rollout_size
        worker_rollout_size = 16 if worker_rollout_size is None else worker_rollout_size
        assert task_mode == "collect", "Only suppor collect mode."
        assert train_algorithm in ['distrl', 'digirl', 'filteredbc'], "Only distrl, digirl and filteredbc are supported"
        current_iter = resume_iter
        trajectories = []
        if accelerator.is_main_process:
            init_time = time.time()
            if sync_mode == "sync":
                print(f">>> Start collect: {sync_mode}, {worker_rollout_size} for {collect_time}mins")
            elif sync_mode == "async":
                print(f">>> Start collect: {sync_mode}, {num_threads} for {collect_time}mins")
        
        while collect_time is not None or len(trajectories) < collect_num:
            early_stop = torch.zeros(1, dtype=torch.bool, device=accelerator.device)
            if accelerator.is_main_process:
                info = {"iteration": current_iter, "iter_start_walltime": time.time()}
                if parallel == 'single':
                    raise NotImplementedError
                elif parallel == 'host':
                    # Process trajectories from the queue
                    if sync_mode == "sync":
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(
                            asyncio.wait(
                                [remote_collect_trajectories_sync(
                                    save_path=save_path,
                                    worker_temp_path=worker_temp_path,
                                    worker_run_path=worker_run_path,
                                    worker_ips=worker_ips,
                                    worker_username=worker_username,
                                    trainer=trainer,
                                    aggregated_save_path=aggregated_save_path,
                                    synthetic=synthetic,
                                    aggregate_lock=aggregate_lock,
                                    model_lock=model_lock,
                                    wandb_run_name=wandb_run_name,
                                    rollout_size=worker_rollout_size
                                )]
                            )
                        )
                    tmp_trajectories = process_queue()
                
                if tmp_trajectories:
                    print(f">>> Process {len(tmp_trajectories)} trajs from queue.")
                    info.update({
                        "rollout.amount": len(tmp_trajectories),
                        "finish_collect_walltime": time.time()
                    })
                    trajectories.extend(framestack(tmp_trajectories))
                    save_mark = num_threads if sync_mode == "async" else worker_rollout_size
                    torch.save(trajectories, os.path.join(save_path, f"collected_trajectories_{sync_mode}_{save_mark}.pt"))
                
                    if use_wandb:
                        end_time = time.time()
                        wandb.log(info)
                        if collect_time and (end_time - init_time) / 60 > collect_time:
                            print(f">>> Early stop at minutes {(end_time - init_time) / 60}...")
                            early_stop[0] = True

                    print(">>> Saving local logs...")
                    log_path = os.path.join(save_path, f"collect_log_{sync_mode}_{num_threads}.jsonl")
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

                    current_iter += 1

            early_stop = broadcast(early_stop)
            if early_stop:
                break
            time.sleep(20)


    def main():
        # Start the remote trajectory collection in the backgroun
        if sync_mode == "async":
            assert parallel == "host", "Only support remote mode for asynchronous framework!"
            if accelerator.is_main_process:
                collect_process = Process(target=start_remote_collection)
                collect_process.start()
        if task_mode == "collect":
            collect()
        else:
            train()
        if sync_mode == "async" and accelerator.is_main_process:
            collect_process.terminate()

    # Run the main function
    main()
        
