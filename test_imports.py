#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

import sys
from pathlib import Path

# Setup path (same as notebook)
project_root = Path(__file__).parent
for subdir in ['src', 'training']:
    subdir_path = project_root / subdir
    if subdir_path.exists() and str(subdir_path) not in sys.path:
        sys.path.insert(0, str(subdir_path))

print("Testing imports...")

# Import configuration and utilities
from config import STATE_DIM, ACTION_DIM, SEEDS, MAX_STEPS
from networks import Actor, Critic, Actor4, Critic4
from wrappers import RewardMaskWrapper
from evaluation import evaluate_policy, evaluate_policy_vectorenv
from advantage import compute_advantage, compute_advantages_batch, compute_nstep_returns
from visualization import setup_plots

# Import training functions
from train_agent0 import train_agent0
from train_agent1 import train_agent1
from train_agent2 import train_agent2
from train_agent3 import train_agent3
from train_agent4 import train_agent4

print("✅ All imports successful!")
print(f"📊 Training: {MAX_STEPS:,} steps per agent, {len(SEEDS)} seeds")
print(f"🌱 Seeds: {SEEDS}")
print(f"🎯 STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}")
print("\n✨ Notebook is ready to run on both local and Kaggle environments!")
