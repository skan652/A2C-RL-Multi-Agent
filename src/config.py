"""
Configuration and Hyperparameters for A2C Agents
"""

from dataclasses import dataclass
from typing import List


# Shared hyperparameters for Agents 0-3
STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 64
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
ENT_COEF = 0.01
MAX_STEPS = 500_000
EVAL_INTERVAL = 20_000
EVAL_EPS = 10
LOG_INTERVAL = 1_000
SEEDS = [42, 123, 456]


@dataclass
class Agent4Config:
    """Configuration for Agent 4 with advanced settings"""
    state_dim: int = 4
    action_dim: int = 2
    hidden_dim: int = 64
    gamma: float = 0.99
    ent_coef: float = 0.01
    max_steps: int = 500_000
    eval_interval: int = 20_000
    eval_eps: int = 10
    log_interval: int = 1_000
    seeds: List[int] = None
    K: int = 6
    n_steps: int = 6
    lr_actor: float = 3e-5  # Higher for big batch
    lr_critic: float = 1e-3
