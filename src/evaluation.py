"""
Policy Evaluation Functions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List


def evaluate_policy(actor: nn.Module, critic: nn.Module, eval_env, device: str, n_episodes: int = 10):
    """
    Evaluate policy with greedy action selection.
    
    Args:
        actor: Policy network
        critic: Value network
        eval_env: Evaluation environment
        device: Device to run on (cpu/cuda/mps)
        n_episodes: Number of episodes to evaluate
    
    Returns:
        returns: List of episode returns
        traj_values: List of value trajectories for each episode
    """
    returns = []
    traj_values = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        ep_values = []

        while True:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(obs_t)
                action = logits.argmax(1).cpu().numpy()[0]
                value = critic(obs_t).squeeze().item()

            obs, reward, term, trunc, _ = eval_env.step(action)
            ep_return += reward
            ep_values.append(value)

            if term or trunc:
                break

        returns.append(ep_return)
        traj_values.append(ep_values)

    return returns, traj_values


def evaluate_policy_vectorenv(actor: nn.Module, critic: nn.Module, eval_envs, device: str, 
                              n_episodes: int = 10, K: int = 6) -> Tuple:
    """
    Evaluate greedy policy with K parallel environments.
    
    Args:
        actor: Policy network
        critic: Value network
        eval_envs: Vectorized environments
        device: Device to run on
        n_episodes: Total number of episodes to evaluate
        K: Number of parallel environments
    
    Returns:
        total_returns: List of episode returns
        traj_values: Empty list (simplified for vectorized env)
    """
    total_returns = []
    episodes_done = 0
    
    obs, _ = eval_envs.reset()
    ep_returns = np.zeros(K)
    
    while episodes_done < n_episodes:
        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            logits = actor(obs_t)
            actions = logits.argmax(1).cpu().numpy()
        
        obs, rewards, terms, truncs, _ = eval_envs.step(actions)
        ep_returns += rewards
        
        # Track completed episodes
        for idx in range(K):
            if (terms[idx] or truncs[idx]) and episodes_done < n_episodes:
                total_returns.append(ep_returns[idx])
                ep_returns[idx] = 0.0
                episodes_done += 1

    traj_values = []  # Simplified
    return total_returns, traj_values
