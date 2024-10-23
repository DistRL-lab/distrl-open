from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import collections
import itertools
import random
from tqdm import tqdm

def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = np.stack([item[key] for item in batch], axis=0)
    return collated_batch


class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
    
    
class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.data_pointer = 0
        self.all_size = 0
        self.episode_num = 0
        self.batch_size = batch_size
        self.episode_boundaries = [0]  # Track the done indices of each episode

        # Initialize storage for each attribute
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None
        self.log_probs = None
        self.image_features = None
        self.next_image_features = None

    def insert(
        self,
        /,
        observation,
        action,
        image_features,
        next_image_features,
        reward,
        next_observation,
        done,
        mc_return,
        penalty,
        log_prob,
        **kwargs
    ):
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(penalty, (float, int)):
            penalty = np.array(penalty)
        if isinstance(log_prob, (float, int)):
            log_prob = np.array(log_prob)
        if isinstance(done, bool):
            done = np.array(done)
        
        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.image_features = np.empty((self.max_size, *image_features.shape), dtype=image_features.dtype)
            self.next_image_features = np.empty((self.max_size, *next_image_features.shape), dtype=next_image_features.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)
            self.penalties = np.empty((self.max_size, *penalty.shape), dtype=penalty.dtype)
            self.log_probs = np.empty((self.max_size, *log_prob.shape), dtype=log_prob.numpy().dtype)
            
        index = self.data_pointer % self.max_size
        
        if self.data_pointer >= self.max_size and self.dones[index] == True:
            self.episode_boundaries.pop(0)
            self.episode_num -= 1
            
        self.observations[index] = observation
        self.actions[index] = action
        self.image_features[index] = image_features
        self.next_image_features[index] = next_image_features
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = done
        self.mc_returns[index] = mc_return
        self.penalties[index] = penalty
        self.log_probs[index] = log_prob

        self.data_pointer += 1
        self.all_size += 1
        if index > 0 and self.dones[index-1] == True:
            # Add the start index of the next episode
            self.episode_boundaries.append(index)
            self.episode_num += 1
                

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.data_pointer, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "image_features": self.image_features[rand_indices],
            "next_image_features": self.next_image_features[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
            "penalty": self.penalties[rand_indices],
            "log_prob": self.log_probs[rand_indices],
        }
        
    def sample_sequence(self, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        if self.episode_num < 1 or self.data_pointer < sequence_length:
            return None  # Not enough data to sample a sequence

        valid = False
        while not valid:
            episode_index = random.randint(0, self.episode_num - 1)
            start_index = self.episode_boundaries[episode_index]
            end_index = self.episode_boundaries[episode_index + 1] - 1
            
            if end_index < start_index:
                end_index = self.max_size - 1

            if (end_index - start_index) >= sequence_length:
                start = random.randint(start_index, end_index - sequence_length)
                indices = np.arange(start, start + sequence_length)
                batch = {
                    "observation": self.observations[indices],
                    "action": self.actions[indices],
                    "image_features": self.image_features[indices],
                    "next_image_features": self.next_image_features[indices],
                    "reward": self.rewards[indices],
                    "next_observation": self.next_observations[indices],
                    "done": self.dones[indices],
                    "mc_return": self.mc_returns[indices],
                    "penalty": self.penalties[indices],
                    "log_prob": self.log_probs[indices],
                }
                valid = True
                return batch

    def __len__(self):
        return self.data_pointer
    
   
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # sum tree structure
        self.data = np.zeros((capacity, 2), dtype=int)  # store [start_idx, end_idx] of trajectories
        self.write = 0  # write pointer
        self.num_data = 0  # current number of trajectories

    def _propagate(self, idx, change):
        """Propagates the change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """Update the priority of a trajectory and propagate the change up"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, start_idx, end_idx):
        """Add a new trajectory's priority to the tree along with [start_idx, end_idx]"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = [start_idx, end_idx]  # store start_idx and end_idx
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.num_data = min(self.num_data + 1, self.capacity)

    def get(self, s):
        """Retrieve a trajectory based on the sum sampling value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Returns the total priority"""
        return self.tree[0]

    def remove(self, start_idx, end_idx):
        """Removes a trajectory from the tree by setting its priority to 0"""
        idx = np.where((self.data[:, 0] == start_idx) & (self.data[:, 1] == end_idx))[0]
        if len(idx) > 0:
            idx = idx[0] + self.capacity - 1
            self.update(idx, 0)  # set priority to 0


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size=2, capacity=10000, w1=50.0, w2=1.0, w3=0.01, gamma=0.99):
        super().__init__(batch_size, capacity)
        self.sumtree = SumTree(capacity)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.gamma = gamma

    def _get_trajectory_data(self, data_array, start_idx, end_idx):
        if start_idx <= end_idx:
            return data_array[start_idx:end_idx]
        else:
            return np.concatenate((data_array[start_idx:self.max_size], data_array[0:end_idx]))

    def insert(self, **kwargs):
        index = self.data_pointer % self.max_size
        if self.data_pointer >= self.max_size and self.dones[index] == True:
            self.remove_old_trajectory()
        super().insert(**kwargs)

        if kwargs['done']:
            start_idx = self.episode_boundaries[-1] if self.episode_num > 0 else 0
            end_idx = self.data_pointer % self.max_size

            # Add initial priority to SumTree
            initial_priority = 1.0
            self.sumtree.add(initial_priority, start_idx, end_idx)
        
    def remove_old_trajectory(self):
        # Remove the trajectory at the start of the circular buffer when it gets overwritten
        if len(self.episode_boundaries) > 1:
            start_idx = self.episode_boundaries[0]
            end_idx = self.episode_boundaries[1] - 1
            self.sumtree.remove(start_idx, end_idx)
            
    def find_tree_index(self, start_idx, end_idx):
        data_idx_array = np.where((self.sumtree.data[:, 0] == start_idx) & (self.sumtree.data[:, 1] == end_idx))[0]
        if len(data_idx_array) > 0:
            data_idx = data_idx_array[0]
            tree_idx = data_idx + self.sumtree.capacity - 1
            return tree_idx
        else:
            return None

    def update_priorities(self, agent):
        num_episodes = len(self.episode_boundaries) - 1
        print('>>>updating priorities')
        for i in tqdm(range(num_episodes), disable=not agent.accelerator.is_main_process):
            start_idx = self.episode_boundaries[i]
            end_idx = self.episode_boundaries[i + 1] - 1
            
            if end_idx == start_idx:
                continue
            
            # Get data in a trajectory
            observations = self._get_trajectory_data(self.observations, start_idx, end_idx)
            next_observations = self._get_trajectory_data(self.next_observations, start_idx, end_idx)
            actions = self._get_trajectory_data(self.actions, start_idx, end_idx)
            rewards = self._get_trajectory_data(self.rewards, start_idx, end_idx)
            dones = self._get_trajectory_data(self.dones, start_idx, end_idx)
            behavior_log_probs = self._get_trajectory_data(self.log_probs, start_idx, end_idx)
            image_features = self._get_trajectory_data(self.image_features, start_idx, end_idx)
            next_image_features = self._get_trajectory_data(self.next_image_features, start_idx, end_idx)

            flat_observations = [obs for obs in observations]
            flat_next_observations = [obs for obs in next_observations]
            flat_actions = [act for act in actions]

            image_feature_tensors = torch.stack([torch.tensor(feat) for feat in image_features]).to(agent.device)
            next_image_feature_tensors = torch.stack([torch.tensor(feat) for feat in next_image_features]).to(agent.device)

            reward_tensors = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
            done_tensors = torch.tensor(dones, dtype=torch.float32).to(agent.device)
            behavior_log_prob_tensors = torch.tensor(behavior_log_probs, dtype=torch.float32).to(agent.device)

            with torch.no_grad():
                flat_v1, flat_v2 = agent.accelerator.unwrap_model(agent.critic)(
                    flat_observations, image_feature_tensors, flat_actions, detach_model=False)
                flat_nv1, flat_nv2 = agent.accelerator.unwrap_model(agent.critic)(
                    flat_next_observations, next_image_feature_tensors, flat_actions, detach_model=False)

                v1 = torch.nn.functional.softmax(flat_v1, dim=1)[:, 1]
                v2 = torch.nn.functional.softmax(flat_v2, dim=1)[:, 1]
                nv1 = torch.nn.functional.softmax(flat_nv1, dim=1)[:, 1]
                nv2 = torch.nn.functional.softmax(flat_nv2, dim=1)[:, 1]

                # Value estimation
                v_current = torch.minimum(v1, v2).squeeze()
                v_next = torch.minimum(nv1, nv2).squeeze()

                # TD-error
                td_error = reward_tensors + self.gamma * v_next * (1 - done_tensors) - v_current
                td_error_np = td_error.cpu().numpy()

                # Log probabilities
                target_log_probs = agent.get_log_prob(flat_observations, image_feature_tensors, flat_actions).sum(dim=1)
                target_log_prob_tensors = target_log_probs.to(agent.device)

                # Importance sampling weights
                is_weights = torch.exp(target_log_prob_tensors - behavior_log_prob_tensors).cpu().numpy()

                # Policy entropy
                entropy = -target_log_probs.cpu().numpy()

            # Average over the trajectory
            avg_td_error = np.mean(np.abs(td_error_np))
            avg_is_weight = np.mean(is_weights)
            avg_entropy = np.mean(entropy)

            # Priority
            priority = self.w1 * avg_td_error + self.w2 * avg_is_weight + self.w3 * avg_entropy

            # Search tree_idx
            tree_idx = self.find_tree_index(start_idx, end_idx)
            if tree_idx is not None:
                # Update SumTree
                self.sumtree.update(tree_idx, priority)
            else:
                # Add to SumTree
                self.sumtree.add(priority, start_idx, end_idx)
