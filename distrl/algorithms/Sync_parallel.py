import asyncio
import asyncssh
import os
import json
import torch
import time
from queue import Queue
import random
import subprocess

# Queue to store aggregated trajectories
trajectory_queue = Queue()
TMP_PATH = '/home/<usrname>/research/LLM_agent/logs/multimachine/tmp'

worker_traj_lock = asyncio.Lock()


def initialize_workers(worker_ips, worker_username):
    """Asdd workers to known hosts and kill existing processes"""
    for worker_ip in worker_ips:
        os.system(f"ssh-keyscan -H {worker_ip} >> ~/.ssh/known_hosts")
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'pkill -U {worker_username}'")
    time.sleep(5)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'skill -u {worker_username}'")
    time.sleep(5)


async def execute_command(host, username, command):
    """Execute command remotely"""
    try:
        async with asyncssh.connect(host, username=username) as conn:
            # Run the command in the background using nohup and &
            result = await conn.run(command, check=True)
            return result.stdout.strip()
    except (OSError, asyncssh.Error) as exc:
        print(f'>>> Error connecting to {host}: {str(exc)}')


async def execute_command_background(host, username, commands, log_path='/dev/null'):
    """Execute command in the background remotely"""
    try:
        async with asyncssh.connect(host, username=username) as conn:
            # Run the command in the background using nohup and &
            commands[-1] = f"nohup {commands[-1]} >> {log_path} 2>&1 &"
            commands.append("echo $!")
            result = await conn.run("\n".join(commands), check=True)
            return result.stdout.strip()
    except (OSError, asyncssh.Error) as exc:
        print(f'>>> Error connecting to {host}: {str(exc)}')


async def copy_file_from_remote(host, username, remote_path, local_path):
    try:
        async with asyncssh.connect(host, username=username) as conn:
            async with conn.start_sftp_client() as sftp:
                await sftp.get(remote_path, local_path, recurse=True)
    except (OSError, asyncssh.Error) as exc:
        print(f'>>> Error copying file from {host}: {str(exc)}')
    

async def copy_file_to_remote(host, username, local_path, remote_temp_path, remote_final_path=None):
    try:
        async with asyncssh.connect(host, username=username) as conn:
            await conn.run(f"rm -rf {remote_temp_path}", check=True)
            copy_command = f"scp -r {local_path} {username}@{host}:{remote_temp_path}"
            result = subprocess.run(copy_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert result.returncode == 0, f"{result.stderr}"
            if remote_final_path is not None:
                await conn.run(f"rm -rf {remote_final_path} && mv {remote_temp_path} {remote_final_path}", check=True)
    except Exception as err:
        print(f'Error copying file to {host}: {str(err)}')


async def setup_worker(worker_ip, worker_username, worker_temp_path):
    # command = f"rm -rf {worker_temp_path} && mkdir -p {worker_temp_path}"
    command = f"mkdir -p {worker_temp_path}"
    await execute_command(worker_ip, worker_username, command)


async def distribute_trainer(worker_ip, worker_username, save_path, worker_temp_path):
    local_path = os.path.join(save_path, "trainer_current_policy.pt")
    
    # Define remote temp path and final path
    remote_temp_path = os.path.join(worker_temp_path, "trainer_current_policy.temp")
    remote_final_path = os.path.join(worker_temp_path, "trainer_current_policy.pt")
    
    # Copy the local trainer to the remote temp path, then rename to final
    await copy_file_to_remote(worker_ip, worker_username, local_path, remote_temp_path, remote_final_path)


async def start_trajectory_collection(worker_ip, worker_username, worker_run_path, worker_temp_path, synthetic, 
    worker_status, worker_pids, worker_finish, wandb_run_name, rollout_size):
    # Check if the thread is idle before starting a new collection
    if not worker_status[worker_ip] and not worker_finish[worker_ip]:  # Only start if the thread is idle and not finished
        if synthetic:
            raise NotImplementedError
        else:
            # TODO: make the worker threads run in background
            print(f">>> Start collecting for worker {worker_ip}")
            worker_name = worker_ip.split(".")[0]
            rollout_override = ""
            if rollout_size:
                rollout_override = f" bsize={rollout_size} rollout_size={rollout_size}"
            command_suffix = ""
            if wandb_run_name:
                command_suffix = command_suffix + f' +wandb_run_name="{wandb_run_name}"'
            commands = ["conda activate distrl", f"cd {worker_run_path}", f"python run.py --config-path config/multimachine --config-name worker_sync{rollout_override} +thread_id=0 +worker_name={worker_name}{command_suffix}"]
            pid = await execute_command_background(worker_ip, worker_username, commands, log_path=f"/home/<usrname>/logs/worker/0.log")
            worker_pids[worker_ip] = int(pid)
        # Mark thread as collecting
        worker_status[worker_ip] = True
        print(f">>> Started collecting for worker {worker_ip}")
    else:
        print(f">>> Worker {worker_ip} is still collecting or finished. Skipping start.") #you can delete this line


async def collect_trajectory(worker_ip, worker_username, worker_temp_path, save_path, synthetic, 
    worker_status, worker_pids, worker_finish):
    if synthetic:
        raise NotImplementedError
    else:
        # check worker running status first
        remote_path = os.path.join(worker_temp_path, f"trajectories_0.pt")
        local_path = os.path.join(save_path, f"{worker_ip}_trajectories_0.pt")
        try:
            worker_pid = worker_pids[worker_ip]
            result = await execute_command(worker_ip, worker_username, f"ps -p {worker_pid}")
            result = [] if result is None else result.split("\n")
            assert len(result) < 2, f"Collect process is still running for {worker_ip}..."
            worker_status[worker_ip] = False
            worker_pids[worker_ip] = 0
            # Attempt to copy the file from the remote server
            async with worker_traj_lock:
                await copy_file_from_remote(worker_ip, worker_username, remote_path, local_path)
            print(f">>> Collected trajectory from {worker_ip}. Deleting remote file.")
            # Delete the file from the remote server after copying
            await execute_command(worker_ip, worker_username, f"rm {remote_path}")
            # check trajectories
            offline_trajectories = torch.load(local_path, map_location=torch.device('cpu'))
            assert len(offline_trajectories) > 0, f"Failed checking retrieved trajectories from {worker_ip}."
            worker_finish[worker_ip] = True
        except FileNotFoundError:
            print(f">>> No trajectory found for {worker_ip}. Restarting...")
        except Exception as err:
            print(err)


async def aggregate_trajectories(worker_ips, save_path, aggregated_save_path, aggregate_lock):
    trajectories_list = []
    for worker_ip in worker_ips:
        local_path = os.path.join(save_path, f"{worker_ip}_trajectories_0.pt")
        if os.path.exists(local_path):
            async with worker_traj_lock:
                trajectories = torch.load(local_path, weights_only=False, map_location=torch.device('cpu'))
                trajectories_list.append(trajectories)
                os.remove(local_path)

    # aggregate
    aggregated_trajectories = [traj for sublist in trajectories_list for traj in sublist]
    if aggregated_trajectories:
        print(f">>> Got aggregated traj: {len(aggregated_trajectories)}")
        aggregated_file_path = os.path.join(aggregated_save_path, "aggregated_trajectories.pt")
        with aggregate_lock:
            if os.path.exists(aggregated_file_path):
                trajectories = torch.load(aggregated_file_path, weights_only=False, map_location=torch.device('cpu'))
                trajectories.extend(aggregated_trajectories)
                aggregated_trajectories = trajectories
            print(f">>> Saving aggregated traj: {len(aggregated_trajectories)}")
            torch.save(aggregated_trajectories, aggregated_file_path)
    return aggregated_trajectories


def all_finish(worker_finish_dict):
    for _, is_finish in worker_finish_dict.items():
        if not is_finish:
            return True
    return False


async def remote_collect_trajectories_sync(
        save_path, 
        worker_temp_path, 
        worker_run_path, 
        worker_ips, 
        worker_username, 
        trainer, 
        aggregated_save_path, 
        synthetic=True, 
        aggregate_lock=None, 
        model_lock=None,
        wandb_run_name=None,
        rollout_size=None
    ):
    # Initialize workers
    print(f">>> In sync collection with worker rollout size {rollout_size}...")
    print(">>> Start initializing workers...")
    if not synthetic: 
        initialize_workers(worker_ips, worker_username)

    # Setup worker environment
    print(">>> Start setting up workers...")
    if not synthetic: 
        setup_tasks = [setup_worker(worker_ip, worker_username, worker_temp_path) for worker_ip in worker_ips]
        await asyncio.gather(*setup_tasks)
    
    # Distribute the trainer to all workers (manually copy the files in advance, so I commented these codes)
    print(">>> Start distributing policy model...")
    if not synthetic: 
        distribute_tasks = [distribute_trainer(worker_ip, worker_username, save_path, worker_temp_path) for worker_ip in worker_ips]
        await asyncio.gather(*distribute_tasks)
    print(">>> Finish distributing policy model...")

    worker_status = {worker_ip: False for worker_ip in worker_ips}
    worker_finish = {worker_ip: False for worker_ip in worker_ips}
    worker_pids = {worker_ip: 0 for worker_ip in worker_ips}

    while all_finish(worker_finish):
        collect_tasks = []
        for worker_ip in worker_ips:
            if not worker_status[worker_ip] and not worker_finish[worker_ip]:  # Check if worker is idle
                collect_tasks.append(start_trajectory_collection(worker_ip, worker_username, worker_run_path, worker_temp_path, synthetic, worker_status, worker_pids, worker_finish, wandb_run_name, rollout_size))
        await asyncio.gather(*collect_tasks)

        trajectory_tasks = []
        for worker_ip in worker_ips:
            if worker_status[worker_ip]:
                trajectory_tasks.append(collect_trajectory(worker_ip, worker_username, worker_temp_path, save_path, synthetic, worker_status, worker_pids, worker_finish))
        await asyncio.gather(*trajectory_tasks)

        # Aggregate all collected trajectories and save to disk (Memory usage for the queue is too high)
        aggregated_qs = await aggregate_trajectories(worker_ips, save_path, aggregated_save_path, aggregate_lock)

        # Perform a delay for the next collection round
        await asyncio.sleep(10)  # Adjust this delay as needed
