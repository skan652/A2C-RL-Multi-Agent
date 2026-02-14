"""
Training script for Agent 4: Combined (K=6, n=6)
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

from networks import Actor4, Critic4
from evaluation import evaluate_policy_vectorenv
from advantage import compute_nstep_returns
from config import Agent4Config


def train_agent4(seed: int, log_dir: Path) -> Dict:
    """
    Train Agent 4: Combined K parallel workers and n-step returns (K=6, n=6)
    
    Args:
        seed: Random seed for reproducibility
        log_dir: Directory to save training logs
    
    Returns:
        logs: Dictionary containing training metrics
    """
    cfg = Agent4Config(seeds=[seed])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Agent 4 - Seed {seed}, K={cfg.K}, n={cfg.n_steps}, Device: {device}")

    # Vectorized environments
    train_envs = SyncVectorEnv([
        lambda: gym.make("CartPole-v1", max_episode_steps=500) for _ in range(cfg.K)
    ])
    eval_envs = SyncVectorEnv([
        lambda: gym.make("CartPole-v1", max_episode_steps=500) for _ in range(cfg.K)
    ])

    actor = Actor4(cfg).to(device)
    critic = Critic4(cfg).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    step_count = 0
    episode_returns = deque(maxlen=100)
    eval_returns_history = []
    eval_values_history = []
    actor_losses, critic_losses, entropies = [], [], []
    
    ep_returns_buffer = np.zeros(cfg.K)

    # Buffers for n-step accumulation (per environment)
    obs_buffers = [[] for _ in range(cfg.K)]
    log_prob_buffers = [[] for _ in range(cfg.K)]
    reward_buffers = [[] for _ in range(cfg.K)]
    value_buffers = [[] for _ in range(cfg.K)]
    done_buffers = [[] for _ in range(cfg.K)]
    trunc_buffers = [[] for _ in range(cfg.K)]

    obs, _ = train_envs.reset()

    while step_count < cfg.max_steps:
        obs_t = torch.FloatTensor(obs).to(device)

        logits = actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = critic(obs_t).squeeze()

        next_obs, rewards, terms, truncs, _ = train_envs.step(actions.cpu().numpy())
        step_count += cfg.K

        # Track episode returns
        ep_returns_buffer += rewards
        for i in range(cfg.K):
            if terms[i] or truncs[i]:
                episode_returns.append(ep_returns_buffer[i])
                ep_returns_buffer[i] = 0.0

        # Store transitions for each environment
        for k in range(cfg.K):
            obs_buffers[k].append(obs[k])
            log_prob_buffers[k].append(log_probs[k])
            reward_buffers[k].append(rewards[k])
            value_buffers[k].append(values[k])
            done_buffers[k].append(terms[k])
            trunc_buffers[k].append(truncs[k])

        obs = next_obs

        # Check if any buffer is ready for update
        update_needed = any(len(rb) >= cfg.n_steps or (len(db) > 0 and db[-1]) 
                          for rb, db in zip(reward_buffers, done_buffers))

        if update_needed:
            all_advantages = []
            all_returns = []
            all_values = []
            all_log_probs = []

            for k in range(cfg.K):
                if len(reward_buffers[k]) == 0:
                    continue

                # Get bootstrap value
                if done_buffers[k] and done_buffers[k][-1]:
                    bootstrap_value = torch.tensor(0.0, device=device)
                else:
                    with torch.no_grad():
                        next_obs_t = torch.FloatTensor(next_obs[k]).unsqueeze(0).to(device)
                        bootstrap_value = critic(next_obs_t).squeeze()

                # Convert to tensors
                rews_t = torch.FloatTensor(reward_buffers[k]).to(device)
                vals_t = torch.stack(value_buffers[k])
                log_probs_t = torch.stack(log_prob_buffers[k])

                # Compute n-step returns
                returns, advantages = compute_nstep_returns(
                    rews_t, vals_t, bootstrap_value, cfg.gamma, len(reward_buffers[k]),
                    dones=torch.BoolTensor(done_buffers[k]).to(device),
                    truncs=torch.BoolTensor(trunc_buffers[k]).to(device)
                )

                all_advantages.append(advantages)
                all_returns.append(returns)
                all_values.append(vals_t)
                all_log_probs.append(log_probs_t)

                # Clear this environment's buffers
                obs_buffers[k].clear()
                log_prob_buffers[k].clear()
                reward_buffers[k].clear()
                value_buffers[k].clear()
                done_buffers[k].clear()
                trunc_buffers[k].clear()

            if all_advantages:
                # Concatenate all data
                all_advantages = torch.cat(all_advantages)
                all_returns = torch.cat(all_returns)
                all_values = torch.cat(all_values)
                all_log_probs = torch.cat(all_log_probs)

                # Update actor
                actor_opt.zero_grad()
                actor_loss = -(all_advantages.detach() * all_log_probs).mean()
                actor_loss -= cfg.ent_coef * dist.entropy().mean()
                actor_loss.backward()
                actor_opt.step()

                # Update critic
                critic_opt.zero_grad()
                critic_loss = F.mse_loss(all_values, all_returns.detach())
                critic_loss.backward()
                critic_opt.step()

                # Logging
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(dist.entropy().mean().item())

        # Periodic evaluation
        if step_count % cfg.eval_interval == 0:
            eval_returns, eval_values = evaluate_policy_vectorenv(
                actor, critic, eval_envs, device, cfg.eval_eps, cfg.K
            )
            eval_returns_history.append(np.mean(eval_returns))
            eval_values_history.append(0.0)  # Simplified
            print(f"  Step {step_count}: Eval return {np.mean(eval_returns):.1f}±{np.std(eval_returns):.1f}")

        if step_count % cfg.log_interval == 0:
            if episode_returns:
                print(f"  Step {step_count}: Train return {np.mean(episode_returns):.1f}")

    # Final evaluation
    final_returns, final_values = evaluate_policy_vectorenv(
        actor, critic, eval_envs, device, cfg.eval_eps, cfg.K
    )

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
        "K": cfg.K,
        "n_steps": cfg.n_steps,
    }

    np.save(log_dir / f"agent4_seed{seed}.npy", logs)
    train_envs.close()
    eval_envs.close()
    
    print(f"✅ Agent 4 Seed {seed} complete: {np.mean(final_returns):.1f}")
    return logs
