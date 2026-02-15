"""
Training script for Agent 3: N-Step Returns (K=1, n=6)
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
from advantage import compute_nstep_returns
from config import *


def train_agent3(seed: int, log_dir: Path, n_steps: int = 6) -> Dict:
    """
    Train Agent 3: N-step returns (K=1, n=6)
    
    Args:
        seed: Random seed for reproducibility
        log_dir: Directory to save training logs
        n_steps: Number of steps for n-step returns
    
    Returns:
        logs: Dictionary containing training metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Agent 3 - Seed {seed}, n={n_steps}, Device: {device}")

    train_env = gym.make("CartPole-v1", max_episode_steps=500)
    eval_env = gym.make("CartPole-v1", max_episode_steps=500)

    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    critic = Critic(STATE_DIM, HIDDEN_DIM).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    step_count = 0
    episode_returns = deque(maxlen=100)
    eval_returns_history = []
    eval_values_history = []
    actor_losses, critic_losses, entropies = [], [], []

    # Buffers for n-step accumulation
    obs_buffer = []
    action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    value_buffer = []
    done_buffer = []
    trunc_buffer = []

    obs, _ = train_env.reset()
    ep_return = 0.0

    while step_count < MAX_STEPS:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

        logits = actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = critic(obs_t).squeeze()

        next_obs, reward, term, trunc, _ = train_env.step(action.item())
        done = term or trunc
        ep_return += reward
        step_count += 1

        # Store transitions
        obs_buffer.append(obs)
        action_buffer.append(action)
        log_prob_buffer.append(log_prob)
        reward_buffer.append(reward)
        value_buffer.append(value)
        done_buffer.append(term)
        trunc_buffer.append(trunc)

        obs = next_obs

        # Update when buffer reaches n steps or episode ends
        if len(reward_buffer) >= n_steps or done:
            # Get bootstrap value
            if done:
                bootstrap_value = torch.tensor(0.0, device=device)
            else:
                with torch.no_grad():
                    next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                    bootstrap_value = critic(next_obs_t).squeeze()

            # Convert buffers to tensors
            rews_t = torch.FloatTensor(reward_buffer).to(device)
            vals_t = torch.stack(value_buffer)
            log_probs_t = torch.stack(log_prob_buffer)
            
            # Compute n-step returns and advantages
            returns, advantages = compute_nstep_returns(
                rews_t, vals_t, bootstrap_value, GAMMA, len(reward_buffer),
                dones=torch.BoolTensor(done_buffer).to(device),
                truncs=torch.BoolTensor(trunc_buffer).to(device)
            )

            # Update actor
            actor_opt.zero_grad()
            actor_loss = -(advantages.detach() * log_probs_t).mean()
            # Entropy bonus (approximate from last distribution)
            actor_loss -= ENT_COEF * dist.entropy().mean()
            actor_loss.backward()
            actor_opt.step()

            # Update critic
            critic_opt.zero_grad()
            critic_loss = F.mse_loss(vals_t, returns.detach())
            critic_loss.backward()
            critic_opt.step()

            # Logging
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(dist.entropy().item())

            # Clear buffers
            obs_buffer.clear()
            action_buffer.clear()
            log_prob_buffer.clear()
            reward_buffer.clear()
            value_buffer.clear()
            done_buffer.clear()
            trunc_buffer.clear()

        if done:
            episode_returns.append(ep_return)
            obs, _ = train_env.reset()
            ep_return = 0.0

        # Periodic evaluation
        if step_count % EVAL_INTERVAL == 0:
            eval_returns, eval_values = evaluate_policy(actor, critic, eval_env, device, EVAL_EPS)
            eval_returns_history.append(np.mean(eval_returns))
            eval_values_history.append(np.mean([np.mean(tv) for tv in eval_values]))
            print(f"  Step {step_count}: Eval return {np.mean(eval_returns):.1f}±{np.std(eval_returns):.1f}")

        if step_count % LOG_INTERVAL == 0:
            if episode_returns:
                print(f"  Step {step_count}: Train return {np.mean(episode_returns):.1f}")

    # Final evaluation
    final_returns, final_values = evaluate_policy(actor, critic, eval_env, device, EVAL_EPS)

    logs = {
        "step_count": step_count,
        "train_returns": list(episode_returns),
        "eval_returns": eval_returns_history,
        "eval_values": eval_values_history,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropies": entropies,
        "final_returns": final_returns,
        "final_values": final_values,
        "seed": seed,
        "n_steps": n_steps,
    }

    np.save(log_dir / f"agent3_seed{seed}.npy", logs)
    train_env.close()
    eval_env.close()
    
    print(f"✅ Agent 3 Seed {seed} complete: {np.mean(final_returns):.1f}")
    return logs
