import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from distrl.data import DummyDataset, collate_fn
from distrl.algorithms.distrl.vtrace import vtrace_from_log_probs
from distrl.algorithms.distrl.retrace import retrace_from_log_probs
import random
from peft import get_peft_model_state_dict, set_peft_model_state_dict


USE_RETRACE=True
REG=False


from validity_checker import ActionValidityChecker

# # Initialize the validity checker with your Gemini API key
# gemini_key = "YOUR_GEMINI_API_KEY"
# validity_checker = ActionValidityChecker(gemini_key=gemini_key)

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class DistRLTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_lr: float = 1e-3,\
                    lm_lr: float = 1e-5,\
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    sequence_length: int = 5,
                    clip_rho_threshold: float = 1.0,
                    clip_pg_rho_threshold: float = 1.0,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3,
                    trajectory_critic_epochs: int = 3,
                    use_retrace = True,
                    use_entropy = False):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr = lm_lr)
        self.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr = critic_lr)
        self.trajectory_critic_optimizer = torch.optim.Adam(agent.trajectory_critic.parameters(), lr = critic_lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.trajectory_critic_epochs = trajectory_critic_epochs
        self.step = 0
        self.tau = tau
        self.sequence_length = sequence_length
        self.clip_rho_threshold = clip_rho_threshold
        self.clip_pg_rho_threshold = clip_pg_rho_threshold
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim = -1)
        self.time_index = 0
        
        self.use_retrace = use_retrace
        self.use_entropy = use_entropy
        
        # self.use_penalty = True  # Set to True to enable the penalty term
        # self.lambda_penalty = 0.1  # Adjust this value based on empirical studies
        # self.gemini_key = 'YOUR_GEMINI_API_KEY'  # Replace with your actual API key


    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)
        self.critic_optimizer = self.accelerator.prepare(self.critic_optimizer)
        self.trajectory_critic_optimizer = self.accelerator.prepare(self.trajectory_critic_optimizer)
    
    def trajectory_critic_loss(self, observation, mc_return, validation = False, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
            v = self.agent.trajectory_critic(observation, detach_model=False)
            regression_target = (mc_return > 0).long()
            v_loss = self.criterion(v, regression_target)
            v_acc = (v.argmax(dim = 1) == regression_target).float().mean()
            if not validation:
                self.accelerator.backward(v_loss)
            v_loss = v_loss.detach().cpu()
            v_acc = v_acc.detach().cpu()
            mc_return = mc_return.detach().cpu()
            v = self.softmax(v)[:, 1]
        info = {"trajectory.v1.loss": v_loss,\
                "trajectory.v1.acc": v_acc,\
                "trajectory.v1.mean": torch.mean(v),\
                "trajectory.v1.min": torch.min(v),\
                "trajectory.v1.max": torch.max(v),\
                "trajectory.v1.std": torch.std(v),\
                "mc_return.mean": torch.mean(mc_return),
                "mc_return.max": torch.max(mc_return),
                "mc_return.min": torch.min(mc_return),
                "mc_return.std": torch.std(mc_return),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info

    def critic_loss(self, observation, image_features, action, reward, next_observation, next_image_features, done, mc_return,
                    validation = False, **kwargs):
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device, dtype = self.accelerator.unwrap_model(self.agent.model).dtype).flatten()
        v1, v2 = self.agent.critic(observation, image_features, action, detach_model=False)
        nv1, nv2 = self.agent.critic(next_observation, next_image_features, action, detach_model=False)

        v1 = v1.reshape(-1, 2)
        v2 = v2.reshape(-1, 2)
        nv1 = nv1.reshape(-1, 2)
        nv2 = nv2.reshape(-1, 2)
        regression_target = (mc_return > 0).long()
        v1_loss = self.criterion(v1, regression_target)
        v1_acc = (v1.argmax(dim = 1) == regression_target).float().mean()
        v2_loss = self.criterion(v2, regression_target)
        v2_acc = (v2.argmax(dim = 1) == regression_target).float().mean()
        nv1_loss = self.criterion(nv1, regression_target)
        nv2_loss = self.criterion(nv2, regression_target)
        if not validation:
            self.accelerator.backward(v1_loss + v2_loss + nv1_loss + nv2_loss)
        v1_loss, v2_loss = v1_loss.detach().cpu(), v2_loss.detach().cpu()
        v1_acc, v2_acc = v1_acc.detach().cpu(), v2_acc.detach().cpu()

        #calculate the probability for logging purpose
        v1 = self.softmax(v1)[:, 1]
        v2 = self.softmax(v2)[:, 1]
        info = {"v1.loss": v1_loss,\
                "v2.loss": v2_loss,\
                "v1.acc": v1_acc,\
                "v2.acc": v2_acc,\
                "v1.mean": torch.mean(v1),\
                "v1.min": torch.min(v1),\
                "v1.max": torch.max(v1),\
                "v1.std": torch.std(v1),
                "v2.mean": torch.mean(v2),
                "v2.max": torch.max(v2),
                "v2.min": torch.min(v2),
                "v2.std": torch.std(v2),
                }
        if validation:
            validation_info = {}
            for k,v in info.items():
                validation_info["validation."+k] = v
            return validation_info
        return info
    
    """
    Generalized Advantage Estimator
     
    Re-trace: This algorithm adjusts the policy gradient and value updates using importance sampling to account for the difference between the behavior policy (which generated the trajectories)
    and the target policy (which is being optimized).

    DPER: Distributed Prioritized Experience Replay
    """    
        
    def actor_loss(self, observation, action, image_features, next_observation, next_image_features, done, mc_return, 
                pi_action, advantage, reward, penalty, log_prob, validation=False, **kwargs):
        # Convert inputs to tensors and move to device
        reward = torch.Tensor(reward).to(self.accelerator.unwrap_model(self.agent.model).device)
        mc_return = torch.Tensor(mc_return).to(self.accelerator.unwrap_model(self.agent.model).device)
        penalty = torch.Tensor(penalty).to(self.accelerator.unwrap_model(self.agent.model).device)
        behavior_log_prob = torch.Tensor(log_prob).to(self.accelerator.unwrap_model(self.agent.model).device)
        done = torch.Tensor(done).to(self.accelerator.unwrap_model(self.agent.model).device)
        
        # Get batch and time step dimensions
        B, T = reward.shape[0], reward.shape[1]
        
        # Flatten image features and move to device
        image_features_tensor = [torch.Tensor(trajectory) for batch in image_features for trajectory in batch]
        flat_image_features = torch.stack(image_features_tensor).to(self.accelerator.unwrap_model(self.agent.model).device)
        
        next_image_features_tensor = [torch.Tensor(trajectory) for batch in next_image_features for trajectory in batch]
        flat_next_image_features = torch.stack(next_image_features_tensor).to(self.accelerator.unwrap_model(self.agent.model).device)
        
        # Flatten observations and actions
        flat_observations = [obs for sublist in observation for obs in sublist]
        flat_actions = [act for sublist in action for act in sublist]
        flat_next_observations = [obs for sublist in next_observation for obs in sublist]
        
        # Get critic value estimates for current and next observations
        with torch.no_grad():
            flat_v1, flat_v2 = self.agent.critic(flat_observations, flat_image_features, flat_actions, detach_model=False)
            flat_nv1, flat_nv2 = self.agent.critic(flat_next_observations, flat_next_image_features, flat_actions, detach_model=False)

        # Assuming v1, v2, nv1, nv2 each have two dimensions, apply softmax and extract the second column (index 1)
        v1 = self.softmax(flat_v1)[:, 1]
        v2 = self.softmax(flat_v2)[:, 1]
        nv1 = self.softmax(flat_nv1)[:, 1]
        nv2 = self.softmax(flat_nv2)[:, 1]

        # Select minimum value estimate to improve stability (soft actor-critic style)
        flat_v = torch.minimum(v1, v2).flatten()
        flat_nv = torch.minimum(nv1, nv2).flatten()
        
        # Reshape to original batch and time dimensions
        v = flat_v.view(B, T)
        nv = flat_nv.view(B, T)
        
        # Compute discount factors
        discounts = ((1 - done) * self.gamma)
        
        # Compute target policy log probabilities
        flat_target_log_prob = self.agent.get_log_prob(flat_observations, flat_image_features, flat_actions).sum(dim=1).flatten()
        target_log_prob = flat_target_log_prob.view(B, T)
        
        # Bootstrapped value for the last time step
        bootstrap_value = nv[:, -1]
        
        # Use either Retrace or V-trace based on the global flag
        if self.use_retrace:
            # Retrace calculation
            retrace_outputs = retrace_from_log_probs(
                behavior_action_log_probs=behavior_log_prob.t(),
                target_action_log_probs=target_log_prob.t(),
                discounts=discounts.t(),
                rewards=(reward.t() + mc_return.t() - 0.05 + penalty.t()),
                values=v.t(),
                bootstrap_value=bootstrap_value
            )
            advantages = retrace_outputs.pg_advantages.t()
            value_targets = retrace_outputs.vs.t()
            clipped_rhos = retrace_outputs.truncated_rhos.t()
        else:
            # V-trace calculation
            vtrace_outputs = vtrace_from_log_probs(
                behavior_action_log_probs=behavior_log_prob.t(),
                target_action_log_probs=target_log_prob.t(),
                discounts=discounts.t(),
                rewards=(reward.t() + mc_return.t() - 0.05 + penalty.t()),
                values=v.t(),
                bootstrap_value=bootstrap_value,
                clip_rho_threshold=self.clip_rho_threshold,
                clip_pg_rho_threshold=self.clip_pg_rho_threshold
            )
            advantages = vtrace_outputs.pg_advantages.t()
            value_targets = vtrace_outputs.vs.t()
            clipped_rhos = vtrace_outputs.clipped_rhos.t()
        
        advantages = torch.clamp(advantages, 0, 1)
        advantages = (advantages > 0).to(dtype=self.accelerator.unwrap_model(self.agent.model).dtype)
        
        # Value loss (critic loss)
        value_loss = (v - value_targets).pow(2).mean()
    
        ### Entropy Regularization ###
        # Entropy encourages exploration, computed as the negative log probability of the target actions
        entropy = -torch.mean(flat_target_log_prob)  # Higher entropy encourages more exploration
        entropy_coeff = 0.05  # Set this in your class to control the strength of entropy regularization

        # Policy gradient loss with corrections from V-trace or Retrace, including entropy regularization
        if self.use_entropy:
            pg_loss = -torch.mean(advantages * target_log_prob) - entropy_coeff * entropy
        else:
            pg_loss = -torch.mean(clipped_rhos * target_log_prob * advantages)

        # Optionally include invalid action penalty term
        # Uncomment the following code to include the penalty term in the loss function
        '''
        # Invalid Action Penalty Term
        # ---------------------------
        # To prevent the generation of nonsensical or invalid commands, we can incorporate an additional penalty term
        # into the actor loss function. This penalty term is computed based on the invalidity of actions, as determined
        # by a pre-trained language model like Gemini-1.5-pro.
        #
        # The penalty term is defined as:
        #   penalty_loss = lambda_penalty * mean(P_invalid(a_t))
        #
        # Where P_invalid(a_t) is 1 if the action a_t is invalid, and 0 otherwise.
        # The function check_action_validity(actions) returns a tensor of P_invalid(a_t) values for each action.
        # The hyperparameter lambda_penalty controls the influence of the penalty term.
        if self.use_penalty:
            # Initialize the ActionValidityChecker if not already done
            if not hasattr(self, 'validity_checker'):
                from validity_checker import ActionValidityChecker
                self.validity_checker = ActionValidityChecker(gemini_key=self.gemini_key)

            # Compute penalty term
            penalty_values = self.validity_checker.check_action_validity(flat_actions)  # Returns tensor of shape [batch_size * T]
            penalty_values = penalty_values.to(device)
            penalty_loss = self.lambda_penalty * penalty_values.mean()
            # Add penalty to the total loss
            total_loss = pg_loss + value_loss + penalty_loss
        else:
            total_loss = pg_loss + value_loss
        '''
        # For now, we just sum pg_loss and value_loss
        total_loss = pg_loss + value_loss

        # Combine and apply gradients if not in validation mode
        if not validation:
            self.accelerator.backward(total_loss)

        # Log information for analysis
        advantages = advantages.detach().cpu()
        info = {
            "pg.loss": pg_loss.detach().cpu().item(),
            "values.loss": value_loss.detach().cpu().item(),
            "values.mean": v.mean(),
            "values.max": torch.max(v),
            "values.min": torch.min(v),
            "values.std": torch.std(v),
            "entropy": entropy.detach().cpu().item(),  # Log entropy
            "advantages.mean": advantages.mean(),
            "advantages.max": torch.max(advantages),
            "advantages.min": torch.min(advantages),
            "advantages.std": torch.std(advantages),
        }
        '''
        if self.use_penalty:
            info["penalty.loss"] = penalty_loss.detach().cpu().item()
            info["penalty.mean"] = penalty_values.mean().item()
            info["penalty.max"] = penalty_values.max().item()
            info["penalty.min"] = penalty_values.min().item()
        '''
        if validation:
            validation_info = {f"validation.{k}": v for k, v in info.items()}
            return validation_info
        
        return info


    def update_trajectory_critic(self, trajectories, validation_trajectories = None):
        info = {}
        info_list = []
        batch_size = 8
        with torch.autograd.set_detect_anomaly(True):
            for _ in tqdm(range(self.trajectory_critic_epochs), disable= not self.accelerator.is_main_process):
                data = [{"observation": traj[0]["observation"], "mc_return": traj[-1]["mc_return"]} for traj in trajectories]
                data = [random.sample(data, 1)[0] for _ in range(self.grad_accum_steps*batch_size)]
                dataloader = DataLoader(DummyDataset(data), batch_size=batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                self.trajectory_critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.trajectory_critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.trajectory_critic_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_trajectories is not None:
            info_list = []
            data = [{"observation": traj[0]["observation"], "mc_return": traj[-1]["mc_return"]} for traj in validation_trajectories]
            data = [random.sample(data, 1)[0] for _ in range(self.grad_accum_steps*batch_size)]
            dataloader = DataLoader(DummyDataset(data), batch_size=batch_size)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.trajectory_critic_loss(validation=True, **batch))
            info.update(dict_mean(info_list))
        return info

    def update_critic(self, replay_buffer, validation_buffer = None):
        self.step += 1
        info = {}
        info_list = []
        with torch.autograd.set_detect_anomaly(True):
            for _ in tqdm(range(self.epochs), disable= not self.accelerator.is_main_process):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                self.critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_buffer is not None:
            info_list = []
            data = [validation_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
            for d in data:
                for k,v in d.items():
                    d[k] = v[0]
            dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.critic_loss(validation=True, **batch))
            info.update(dict_mean(info_list))
        return info
        
    def update_policy(self, replay_buffer, validation_buffer = None, no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
        #update actor
        if not no_update_actor:
            print(">>>updating actor")
            #batchsize for the actor set to 1 for mistral due to memory concern
            # action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
            #action_bsize = replay_buffer.batch_size
            for _ in tqdm(range(self.actor_epochs), disable= not self.accelerator.is_main_process):
                data = [replay_buffer.sample_sequence(self.sequence_length) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False, collate_fn=collate_fn)
                all_pi_actions = []
                all_advantages = []
                # import IPython; IPython.embed()
                dataloader = self.accelerator.prepare(dataloader)
                self.lm_optimizer.zero_grad()
                for batch in dataloader:
                    pi_action = None
                    advantages = None
                    info_list.append(self.actor_loss(**batch, pi_action=pi_action, advantage=advantages))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        if validation_buffer is not None:
            info_list = []
            data = [validation_buffer.sample_sequence(self.sequence_length) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
            dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False, collate_fn=collate_fn)
            dataloader = self.accelerator.prepare(dataloader)
            with torch.no_grad():
                for batch in tqdm(dataloader, disable=True):
                    info_list.append(self.actor_loss(validation=True, pi_action=None, advantage=None, **batch))
            info.update(dict_mean(info_list))
            return info
        return info

    def update(self, replay_buffer, validation_buffer = None, filtered_buffer = None, filtered_validation_buffer = None,no_update_actor=False):
        if filtered_validation_buffer is None:
            filtered_validation_buffer = validation_buffer
        if filtered_buffer is None:
            filtered_buffer = replay_buffer
        info = {}
        info.update(self.update_critic(replay_buffer, validation_buffer))
        info.update(self.update_policy(filtered_buffer, filtered_validation_buffer,no_update_actor=no_update_actor))
        return info

    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)

    def load(self, path):
        self.accelerator.load_state(path)

    def save_policy(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "model.pt")
        info_path = os.path.join(path, "info.json")

        self.accelerator.save_model(self.agent.model, weight_path, safe_serialization=False)
        with open(info_path, "w", encoding="utf8") as f_out:
            json.dump({"time_index": self.time_index}, f_out)

    def load_policy(self, path):
        weight_path = os.path.join(path, "model.pt", "pytorch_model.bin")
        info_path = os.path.join(path, "info.json")

        # load trainer info
        with open(info_path, "r", encoding="utf8") as f_in:
            info_data = json.load(f_in)
            self.time_index = info_data["time_index"]
        
        unwrapped_model = self.accelerator.unwrap_model(self.agent.model)
        unwrapped_model.load_state_dict(torch.load(weight_path))

        # TODO: remove these???
        self.agent.model = unwrapped_model
        self.agent.prepare()
        
    def save_lora(self, path):
        # Extract the LoRA fine-tuned part
        lora_weights = get_peft_model_state_dict(self.accelerator.unwrap_model(self.agent.model), adapter_name="default")

        # set path
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "lora.pt")
        info_path = os.path.join(path, "info.json")

        # Save only the LoRA parameters
        self.accelerator.save(lora_weights, weight_path)
        with open(info_path, "w", encoding="utf8") as f_out:
            json.dump({"time_index": self.time_index}, f_out)

        # unwrapped_model = self.accelerator.unwrap_model(self.agent.model)
        # unwrapped_model.save_pretrained_lora(path)

    def load_lora(self, path):
        # Load the LoRA fine-tuned weights from the file
        weight_path = os.path.join(path, "lora.pt")
        info_path = os.path.join(path, "info.json")
        
        # load trainer info
        with open(info_path, "r", encoding="utf8") as f_in:
            info_data = json.load(f_in)
            self.time_index = info_data["time_index"]
        
        # Load the LoRA weights into the model
        lora_weights = torch.load(weight_path)  # You can also use self.accelerator.load_state_dict()
        set_peft_model_state_dict(self.agent.model, lora_weights)
       