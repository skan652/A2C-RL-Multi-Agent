"""
Environment Wrappers
"""

import numpy as np
import gymnasium as gym


class RewardMaskWrapper(gym.Wrapper):
    """
    Masks rewards with probability mask_prob for stochastic reward experiments.
    Used in Agent 1 to study value function estimation under reward uncertainty.
    """
    def __init__(self, env, mask_prob=0.9):
        super().__init__(env)
        self.mask_prob = mask_prob

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        # Keep reward with probability (1 - mask_prob)
        if np.random.rand() > self.mask_prob:
            return obs, r, term, trunc, info
        return obs, 0.0, term, trunc, info
