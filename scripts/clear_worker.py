import os
import time
import hydra

from omegaconf import DictConfig, OmegaConf


def initialize_workers(worker_ips, worker_username):
    """Kill existing processes on workers"""
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'pkill -U {worker_username}'")
    time.sleep(2)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'skill -u {worker_username}'")
    time.sleep(2)


def move_logs(worker_ips, worker_username, save_path, worker_temp_path, wandb_run_name):
    """Kill existing processes on workers"""
    host_log_home = os.path.split(save_path)[0]
    host_target_log_path = os.path.join(host_log_home, "host_" + wandb_run_name)
    os.system(f"mv {save_path} {host_target_log_path}")
    time.sleep(2)

    worker_log_home = os.path.split(worker_temp_path)[0]
    worker_target_log_path = os.path.join(worker_log_home, "worker_" + wandb_run_name)
    for worker_ip in worker_ips:
        os.system(f"ssh {worker_username}@{worker_ip} 'mv {worker_temp_path} {worker_target_log_path}'")
    time.sleep(2)


def remove_aggregate(aggregated_save_path):
    os.system(f"rm -rf {aggregated_save_path}")
    time.sleep(2)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config: "DictConfig"):
    # colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        initialize_workers(config.worker_ips, config.worker_username)
    except Exception as err:
        print(f"initialize workers failed: {err}")
    try:
        move_logs(config.worker_ips, config.worker_username, config.save_path, config.worker_temp_path, config.wandb_run_name)
    except Exception as err:
        print(f"move logs failed: {err}")
    try:
        remove_aggregate(config.aggregated_save_path)
    except Exception as err:
        print(f"remove aggregate failed: {err}")



if __name__ == "__main__":
    main()
    