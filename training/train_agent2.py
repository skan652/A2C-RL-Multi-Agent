"""
Training script for Agent 2: Parallel Workers (K=6, n=1)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from pathlib import Path
from collections import deque
from typing import Dict

from networks import Actor, Critic
from evaluation import evaluate_policy_vectorenv
from advantage import compute_advantages_batch
from config import *


def train_agent2(seed: int, log_dir: Path, K: int = 6) -> Dict:
    """
    Train Agent 2: K parallel workers (K=6, n=1)
    
    Args:
        seed: Random seed for reproducibility
        log_dir: Directory to save training logs
        K: Number of parallel environments
    
    Returns:
        logs: Dictionary containing training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Agent 2 - Seed {seed}, K={K}, Device: {device}")

    # Vectorized environments
    train_envs = SyncVectorEnv([
        lambda: gym.make("CartPole-v1", max_episode_steps=500) for _ in range(K)
    ])
    eval_envs = SyncVectorEnv([
        lambda: gym.make("CartPole-v1", max_episode_steps=500) for _ in range(K)
    ])

    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    critic = Critic(STATE_DIM, HIDDEN_DIM).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    step_count = 0
    episode_returns = deque(maxlen=100)
    eval_returns_history = []
    eval_values_history = []
    actor_losses, critic_losses, entropies = [], [], []
    
    ep_returns_buffer = np.zeros(K)

    obs, _ = train_envs.reset()

    while step_count < MAX_STEPS:
        obs_t = torch.FloatTensor(obs).to(device)

        # Actor: Sample actions for all K environments
        logits = actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Critic: Value estimates for all K states
        values = critic(obs_t).squeeze()

        # Environment steps (parallel)
        next_obs, rewards, terms, truncs, _ = train_envs.step(actions.cpu().numpy())
        step_count += K

        # Track episode returns
        ep_returns_buffer += rewards
        for i in range(K):
            if terms[i] or truncs[i]:
                episode_returns.append(ep_returns_buffer[i])
                ep_returns_buffer[i] = 0.0

        # Bootstrap next values
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(device)
            next_values = critic(next_obs_t).squeeze()

        # Compute batch advantages
        rews_t = torch.FloatTensor(rewards).to(device)
        terms_t = torch.BoolTensor(terms).to(device)
        truncs_t = torch.BoolTensor(truncs).to(device)
        
        advantages = compute_advantages_batch(rews_t, values, next_values, terms_t, truncs_t, GAMMA)

        # Update actor (batch)
        actor_opt.zero_grad()
        actor_loss = -(advantages.detach() * log_probs).mean() - ENT_COEF * dist.entropy().mean()
        actor_loss.backward()
        actor_opt.step()

        # Update critic (batch)
        critic_opt.zero_grad()
        targets = advantages + values
        critic_loss = F.mse_loss(values, targets.detach())
        critic_loss.backward()
        critic_opt.step()

        # Logging
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropies.append(dist.entropy().mean().item())

        obs = next_obs

        # Periodic evaluation
        if step_count % EVAL_INTERVAL == 0:
            eval_returns, eval_values = evaluate_policy_vectorenv(
                actor, critic, eval_envs, device, EVAL_EPS, K
            )
            eval_returns_history.append(np.mean(eval_returns))
            eval_values_history.append(0.0)  # Simplified
            print(f"  Step {step_count}: Eval return {np.mean(eval_returns):.1f}±{np.std(eval_returns):.1f}")

        if step_count % LOG_INTERVAL == 0:
            if episode_returns:
                print(f"  Step {step_count}: Train return {np.mean(episode_returns):.1f}")

    # Final evaluation
    final_returns, final_values = evaluate_policy_vectorenv(actor, critic, eval_envs, device, EVAL_EPS, K)

    logs = {
        "step_count": step_count,
        "train_returns": list(episode_returns),
        "eval_returns": eval_returns_history,
        "eval_values": eval_values_history,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropies": entropies,
        "final_returns": final_returns,
        "final_values": final_values if final_values else [[0]],
        "seed": seed,
        "K": K,
    }

    np.save(log_dir / f"agent2_seed{seed}.npy", logs)
    train_envs.close()
    eval_envs.close()
    
    print(f"✅ Agent 2 Seed {seed} complete: {np.mean(final_returns):.1f}")
    return logs
