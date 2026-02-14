"""
Neural Network Architectures for A2C
"""

import torch
import torch.nn as nn
from config import Agent4Config


class Actor(nn.Module):
    """
    Actor network for discrete action spaces.
    Outputs logits for categorical distribution.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)


class Critic(nn.Module):
    """
    Critic network for value function estimation.
    Outputs V(s) - state value.
    """
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)


class Actor4(nn.Module):
    """Actor network for Agent 4 with config-based initialization"""
    def __init__(self, cfg: Agent4Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)


class Critic4(nn.Module):
    """Critic network for Agent 4 with config-based initialization"""
    def __init__(self, cfg: Agent4Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc_out = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc_out(x)
