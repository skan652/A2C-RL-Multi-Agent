"""
Training script for Agent 1: Stochastic Rewards (K=1, n=1)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from pathlib import Path
from collections import deque
from typing import Dict

from networks import Actor, Critic
from evaluation import evaluate_policy
from advantage import compute_advantage
from wrappers import RewardMaskWrapper
from config import *


def train_agent1(seed: int, log_dir: Path) -> Dict:
    """
    Train Agent 1: Stochastic rewards with 90% masking
    
    Args:
        seed: Random seed for reproducibility
        log_dir: Directory to save training logs
    
    Returns:
        logs: Dictionary containing training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Agent 1 - Seed {seed}, Device: {device}")

    # Wrapped training environment with reward masking
    train_env = RewardMaskWrapper(gym.make("CartPole-v1", max_episode_steps=500), mask_prob=0.9)
    eval_env = gym.make("CartPole-v1", max_episode_steps=500)  # No masking for eval

    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    critic = Critic(STATE_DIM, HIDDEN_DIM).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    step_count = 0
    train_returns = deque(maxlen=100)
    eval_returns_history = []
    eval_values_history = []
    actor_losses, critic_losses, entropies = [], [], []

    while step_count < MAX_STEPS:
        obs, _ = train_env.reset()
        ep_return = 0.0
        done = False

        while not done and step_count < MAX_STEPS:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic(obs_t).squeeze()

            next_obs, reward, term, trunc, _ = train_env.step(action.item())
            done = term or trunc
            ep_return += reward  # This is masked reward
            step_count += 1

            with torch.no_grad():
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                next_value = critic(next_obs_t).squeeze()

            advantage = compute_advantage(reward, value, next_value, term, trunc, GAMMA)

            actor_opt.zero_grad()
            critic_opt.zero_grad()

            actor_loss = -(advantage.detach() * log_prob) - ENT_COEF * dist.entropy()
            actor_loss.backward()
            actor_opt.step()

            target = advantage + value
            critic_loss = F.mse_loss(value, target.detach())
            critic_loss.backward()
            critic_opt.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(dist.entropy().item())

            obs = next_obs

            if done:
                train_returns.append(ep_return)

            if step_count % EVAL_INTERVAL == 0:
                eval_returns, eval_values = evaluate_policy(actor, critic, eval_env, device, EVAL_EPS)
                eval_returns_history.append(np.mean(eval_returns))
                eval_values_history.append(np.mean([np.mean(tv) for tv in eval_values]))
                print(f"  Step {step_count}: Eval return {np.mean(eval_returns):.1f}±{np.std(eval_returns):.1f}")

        if step_count % LOG_INTERVAL == 0:
            if train_returns:
                print(f"  Step {step_count}: Train return {np.mean(train_returns):.1f}")

    final_returns, final_values = evaluate_policy(actor, critic, eval_env, device, EVAL_EPS)

    logs = {
        "step_count": step_count,
        "train_returns": list(train_returns),
        "eval_returns": eval_returns_history,
        "eval_values": eval_values_history,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropies": entropies,
        "final_returns": final_returns,
        "final_values": final_values,
        "seed": seed,
    }

    np.save(log_dir / f"agent1_seed{seed}.npy", logs)
    train_env.close()
    eval_env.close()
    
    print(f"✅ Agent 1 Seed {seed} complete: {np.mean(final_returns):.1f}")
    return logs
