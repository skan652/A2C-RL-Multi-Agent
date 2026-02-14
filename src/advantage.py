"""
Advantage Computation Functions
"""

import torch
from typing import Tuple


def compute_advantage(
    reward: float,
    value: torch.Tensor,
    next_value: torch.Tensor,
    term: bool,
    trunc: bool,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute 1-step TD advantage with proper truncation handling.
    
    Args:
        reward: Observed reward
        value: Current state value V(s_t)
        next_value: Next state value V(s_{t+1})
        term: Terminal flag
        trunc: Truncation flag
        gamma: Discount factor
    
    Returns:
        advantage: A(s_t, a_t) = r + γV(s_{t+1}) - V(s_t)
    """
    if term:
        bootstrap = torch.zeros_like(value)
    elif trunc:
        bootstrap = next_value  # Bootstrap for truncation
    else:
        bootstrap = next_value

    reward_t = torch.tensor(reward, device=value.device)
    return reward_t + gamma * bootstrap - value


def compute_advantages_batch(rews: torch.Tensor, vals: torch.Tensor, next_vals: torch.Tensor,
                             terms: torch.Tensor, truncs: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    1-step TD advantages for batch K (vectorized environments).
    
    Args:
        rews: Rewards [K]
        vals: Current values [K]
        next_vals: Next values [K]
        terms: Terminal flags [K]
        truncs: Truncation flags [K]
        gamma: Discount factor
    
    Returns:
        advantages: Normalized advantages [K]
    """
    non_terminal = (~(terms | truncs)).float()  # 1 if not done, 0 if done
    advantages = rews + gamma * next_vals * non_terminal - vals
    
    # Normalize for stability
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def compute_nstep_returns(rews: torch.Tensor, vals: torch.Tensor,
                         bootstrap_value: torch.Tensor, gamma: float, n: int,
                         dones=None, truncs=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute n-step returns shifted: G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V_{t+n}
    Handles episode boundaries: resets accumulation at terminal states.
    
    Args:
        rews: Rewards buffer [n]
        vals: Values buffer [n]
        bootstrap_value: V(s_{t+n}) for bootstrapping
        gamma: Discount factor
        n: Number of steps
        dones: Terminal flags [n]
        truncs: Truncation flags [n]
    
    Returns:
        returns: N-step returns [n]
        advantages: A_t = G_t - V_t [n]
    """
    returns = torch.zeros(n, device=rews.device)
    
    for t in range(n):
        G_t = 0.0
        discount = 1.0
        
        # Accumulate n-step return starting from t
        for k in range(t, n):
            G_t += discount * rews[k]
            discount *= gamma
            
            # Stop accumulation at episode boundary
            if dones is not None and dones[k]:
                break
        else:
            # If no terminal, bootstrap with V(s_{t+n})
            G_t += discount * bootstrap_value.item()
        
        returns[t] = G_t
    
    advantages = returns - vals
    
    # Normalize advantages
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns, advantages
