# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Codes adapted from: https://github.com/ray-project/ray/blob/master/rllib/algorithms/impala/vtrace_torch.py

PyTorch version of the functions to compute V-trace off-policy actor critic
targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

In addition to the original paper's code, changes have been made
to support MultiDiscrete action spaces. behaviour_policy_logits,
target_policy_logits and actions parameters in the entry point
multi_from_logits method accepts lists of tensors instead of just
tensors.
"""

import collections
import torch

RetraceReturns = collections.namedtuple("RetraceReturns", "vs pg_advantages truncated_rhos")

def retrace_from_log_probs(
    behavior_action_log_probs,
    target_action_log_probs,
    discounts,
    rewards,
    values,
    bootstrap_value,
):
    """
    Retrace algorithm for softmax policies.
    Reference: "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.

    Args:
        behavior_action_log_probs: A tensor [T, B] containing behavior policy log probabilities.
        target_action_log_probs: A tensor [T, B] containing target policy log probabilities.
        discounts: A tensor [T, B] with the discount factors.
        rewards: A tensor [T, B] with the rewards obtained by following the behavior policy.
        values: A tensor [T, B] with the value function estimates of the target policy.
        bootstrap_value: A tensor [B] with the value function estimate at time T (used for bootstrapping).

    Returns:
        A RetraceReturns namedtuple containing:
        - vs: Target values for value function training.
        - pg_advantages: Policy gradient advantages.
    """
    # Compute log importance sampling ratios
    log_rhos = get_log_rhos(target_action_log_probs, behavior_action_log_probs)

    return from_importance_weights_retrace(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value
    )


def from_importance_weights_retrace(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
):
    """
    Retrace algorithm from importance weights.

    Args:
        log_rhos: Log importance sampling weights [T, B].
        discounts: Discount factors [T, B].
        rewards: Rewards [T, B].
        values: Value function estimates [T, B].
        bootstrap_value: Bootstrap value for the final step [B].

    Returns:
        A RetraceReturns namedtuple containing:
        - vs: Target values for value function training.
        - pg_advantages: Policy gradient advantages.
    """
    rhos = torch.exp(log_rhos)
    truncated_rhos = torch.minimum(torch.ones_like(rhos), rhos)  # Retrace truncation

    # Append the bootstrap value to the values tensor [T+1, B]
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)

    # Calculate deltas for the value function updates
    deltas = truncated_rhos * (rewards + discounts * values_t_plus_1 - values)

    # Calculate vs (target values)
    vs_minus_v_xs = [torch.zeros_like(bootstrap_value)]
    for i in reversed(range(len(discounts))):
        discount_t, delta_t = discounts[i], deltas[i]
        vs_minus_v_xs.append(delta_t + discount_t * truncated_rhos[i] * vs_minus_v_xs[-1])
    vs_minus_v_xs = torch.stack(vs_minus_v_xs[1:])
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])

    # Add the current value estimates to obtain the final targets
    vs = vs_minus_v_xs + values

    # Calculate policy gradient advantages
    vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    pg_advantages = truncated_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return RetraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach(),truncated_rhos=truncated_rhos.detach())


def get_log_rhos(target_action_log_probs, behaviour_action_log_probs):
    """Computes the log importance ratios (log_rhos) for V-trace calculations."""
    # Directly compute the log difference (importance ratios) across the batch for each time step
    log_rhos = target_action_log_probs - behaviour_action_log_probs
    return log_rhos


def convert_to_torch_tensor(x, device=None, pin_memory=False):
    """
    Converts input data to a PyTorch tensor, ensuring it is on the correct device
    and in the appropriate dtype, optionally pinning memory for faster GPU transfer.

    Args:
        x (np.ndarray or torch.Tensor): Input data.
        device (str, optional): The device to place the tensor on ('cpu' or 'cuda').
        pin_memory (bool, optional): Whether to pin memory for faster CPU-to-GPU transfer.

    Returns:
        torch.Tensor: The converted tensor.
    """

    # Check if the input is already a tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    else:  # Assume input is a numpy array
        tensor = torch.from_numpy(x)

    # Convert float64 tensors to float32 for efficiency
    if tensor.dtype == torch.float64:
        tensor = tensor.float()

    # Move tensor to the specified device
    if device is not None:
        tensor = tensor.to(device)

    # Pin memory to speed up GPU transfers
    if pin_memory and device != 'cuda':
        tensor = tensor.pin_memory()

    return tensor

