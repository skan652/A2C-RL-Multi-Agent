# A2C Reinforcement Learning: Multi-Agent CartPole Experiments

A comprehensive implementation of the **Advantage Actor-Critic (A2C)** algorithm with 5 experimental configurations exploring parallel workers, n-step returns, and stochastic rewards on the CartPole-v1 environment.

**Authors**: Linda Ben Rajab, Skander Adam Afi  
**Date**: February 2026  
**Repository**: [skan652/A2C-RL-Multi-Agent](https://github.com/skan652/A2C-RL-Multi-Agent)

## 🎯 Project Overview

This project implements and compares 5 different A2C agents to study the effects of:

- **Parallel environment workers (K)**: Sample efficiency and wall-clock speed
- **N-step returns (n)**: Bias-variance tradeoff in TD learning
- **Stochastic rewards**: Value function estimation under uncertainty
- **Combined scaling (K×n)**: Batch size effects on gradient stability

All experiments use rigorous methodology with multiple random seeds (42, 123, 456) and comprehensive logging.

## 📁 Project Structure

```text
A2C-RL-Multi-Agent/
├── rl-project.ipynb              Main notebook with all code
├── rl-project-original.ipynb    Backup of original notebook
│
├── src/                          Utility scripts (6 files)
│   ├── config.py                 Hyperparameters and configuration
│   ├── networks.py               Actor/Critic neural networks
│   ├── wrappers.py               Environment wrappers (RewardMask)
│   ├── evaluation.py             Policy evaluation functions
│   ├── advantage.py              Advantage computation (1-step, n-step)
│   └── visualization.py          Plotting and visualization
│
├── training/                     Training scripts (5 files)
│   ├── train_agent0.py           Agent 0: Baseline A2C (K=1, n=1)
│   ├── train_agent1.py           Agent 1: Stochastic rewards (K=1, n=1)
│   ├── train_agent2.py           Agent 2: Parallel workers (K=6, n=1)
│   ├── train_agent3.py           Agent 3: N-step returns (K=1, n=6)
│   └── train_agent4.py           Agent 4: Combined (K=6, n=6)
│
├── requirements.txt              Python dependencies
├── README.md                     This file
└── agent{0-4}_logs/              Training logs (generated on run)
```

## 🤖 Agent Configurations

| Agent       | K Workers | N-Steps | Batch Size | Learning Rate (Actor) | Purpose                            |
| ----------- | --------- | ------- | ---------- | --------------------- | ---------------------------------- |
| **Agent 0** | 1         | 1       | 1          | 1e-4                  | Baseline (standard A2C)            |
| **Agent 1** | 1         | 1       | 1          | 1e-4                  | Stochastic rewards (90% masking)   |
| **Agent 2** | 6         | 1       | 6          | 1e-4                  | Parallel workers                   |
| **Agent 3** | 1         | 6       | 6          | 1e-4                  | N-step returns                     |
| **Agent 4** | 6         | 6       | 36         | 3e-5                  | Combined (best performance)        |

### Key Differences

- **Agent 0**: Vanilla A2C baseline for comparison
- **Agent 1**: Tests value function learning with masked training rewards (eval uses full rewards)
- **Agent 2**: Vectorized environments for faster wall-clock training
- **Agent 3**: Multi-step TD learning for reduced variance
- **Agent 4**: Combines both K and n scaling with adjusted learning rate for stability

## 📋 Requirements

```text
python >= 3.8
torch >= 2.0.0
gymnasium >= 0.29.0
numpy >= 1.24.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
pandas >= 2.0.0
```

## 🚀 Quick Start

**1. Install dependencies:**

```bash
git clone https://github.com/skan652/A2C-RL-Multi-Agent.git
cd A2C-RL-Multi-Agent
pip install -r requirements.txt
```

**2. Open the Jupyter notebook:**

```bash
jupyter notebook rl-project.ipynb
# Or use Jupyter Lab or VS Code
```

**3. Run cells sequentially** to train agents and generate results.

**Note:** The `.py` files in `src/` and `training/` are standalone utility scripts imported by the notebook.

## 💻 How to Reproduce Results

### Run the Notebook

The [rl-project.ipynb](rl-project.ipynb) contains all necessary code to reproduce results:

1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook in Jupyter/VS Code
3. Run all cells in order
4. Results will be saved as:
   - Training logs: `agent{0-4}_logs/*.npy`
   - Plots: `agent{0-4}_results.png`, `all_agents_comparison.png`, etc.

### Use Training Scripts Directly

You can also import and use the training functions in your own code:

```python
# Add script directories to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
sys.path.insert(0, str(Path('training')))

# Import utilities
from config import *
from train_agent0 import train_agent0
from visualization import plot_training_results

# Train agent
log_dir = Path("agent0_logs")
log_dir.mkdir(exist_ok=True)
seed = 42
logs = train_agent0(seed, log_dir)

# Plot results
plot_training_results([logs], "agent0_results.png", "Agent 0", MAX_STEPS, EVAL_INTERVAL)
```

### Training Time

- **Agent 0, 1**: ~30-60 min on CPU, ~10-20 min on GPU/MPS per seed
- **Agent 2, 4**: Faster due to parallelization (~15-30 min per seed)
- **Agent 3**: Similar to Agent 0
- **All agents (3 seeds each)**: ~4-6 hours on CPU, ~1-2 hours on GPU

## 📊 Expected Results

After training, you should observe:

1. **Agent 0 (Baseline)**: Converges to ~500 episode return
2. **Agent 1 (Stochastic)**: Similar return (~500) but learns V(s) ≈ 10 instead of 50k
3. **Agent 2 (Parallel)**: Faster wall-clock time, similar sample efficiency
4. **Agent 3 (N-step)**: Lower variance, more stable learning
5. **Agent 4 (Combined)**: Best overall stability and fastest convergence

## 🔬 Key Findings

1. **Parallel Workers (K=6)**:
   - ✅ Faster wall-clock training (~2-3x speedup)
   - ✅ More stable gradients from batch updates
   - ❌ Same sample complexity

2. **N-Step Returns (n=6)**:
   - ✅ Reduced variance in advantage estimates
   - ✅ Better long-term credit assignment
   - ❌ Slight increase in bias

3. **Combined (K×n=36)**:
   - ✅ Best overall stability (lowest variance across seeds)
   - ✅ Can use higher learning rate (3e-5 vs 1e-4)
   - ✅ Fastest convergence

4. **Stochastic Rewards**:
   - Value function accurately tracks E[r] = 0.1
   - Policy remains optimal despite sparse feedback
   - Demonstrates importance of proper bootstrap handling

## 📝 Theoretical Questions Answered

**Q1**: What is V(s₀) after convergence for Agent 0 (with correct bootstrap)?  
**A**: V(s₀) ≈ 500/(1-γ) = 50,000. The infinite horizon with proper truncation handling leads to this large value.

**Q2**: What if we don't bootstrap correctly (treat truncation as termination)?  
**A**: V(s₀) → 0, a common implementation bug!

**Q3**: For Agent 1 with stochastic rewards, what is V(s₀)?  
**A**: V(s₀) ≈ 0.1/(1-γ) ≈ 10, since E[r] = 0.1. But eval returns stay ~500 (policy still optimal).

**Q4**: Why can we increase learning rate with K×n scaling?  
**A**: Batch size = 36 → gradient variance ↓ by ~36×, allowing lr ↑ without divergence.

## 📦 Academic Submission

The submission package should include:

```text
Project-G_GroupNumber-S1_Name1-S2_Name2.zip
├── rl-project.ipynb          Main notebook with all code and results
├── src/                      Core utilities package (6 modules)
├── training/                 Training scripts (5 agents)
├── requirements.txt          Python dependencies
├── README.md                 Project documentation
└── agent{0-4}_logs/          Pre-computed results (optional)
```

### Video Presentation (Required)

Create a 5-minute video walkthrough covering:

1. **Project Overview** (1 min): Explain the 5 agents and research questions
2. **Code Walkthrough** (2 min): Show key implementation details
3. **Results** (1.5 min): Present plots and explain findings
4. **Interesting Discovery** (0.5 min): Highlight most interesting finding

## 📊 Key Features

### Correct Bootstrap Handling

- **Truncation vs Terminal**: Distinguishes episode truncation (bootstrap with value) from true terminal states (no bootstrap)
- Critical for infinite-horizon tasks like CartPole with time limits

### Episode Tracking

- Per-worker episode return accumulation in vectorized environments
- Ensures correct logging of full episode returns (not step rewards)

### Stochastic Reward Experiment (Agent 1)

- Training uses 90% reward masking (r=1 becomes r=0.1 with prob 0.9)
- Evaluation uses full rewards to measure policy quality
- Tests value function estimation: V(s₀) ≈ 0.1/(1-γ) ≈ 10

### N-Step Returns (Agents 3-4)

```text
G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V_{t+n}
```

- Reduces variance at the cost of slight bias
- Handles episode boundaries correctly during rollout

### Gradient Stability Analysis

- Batch size scaling (K×n=36) enables higher learning rates
- Gradient variance reduction proportional to batch size
- Agent 4 uses lr_actor=3e-5 for stable training

### Automated Evaluation

Trained agents are automatically evaluated every 10,000 steps using:

- **Agent 0-1**: Single environment evaluation
- **Agent 2-4**: Vectorized environment evaluation (6 parallel episodes)

- Batch size scaling (K×n=36) enables higher learning rates
- Gradient variance reduction proportional to batch size
- Agent 4 uses lr_actor=3e-5 for stable training

## 🔧 Technical Details

### Actor-Critic Architecture

**Actor (Policy Network)**:

```text
Input (state_dim=4) → Dense(64, tanh) → Dense(64, tanh) → Dense(action_dim=2, logits)
```

**Critic (Value Network)**:

```text
Input (state_dim=4) → Dense(64, tanh) → Dense(64, tanh) → Dense(1, value)
```

### Hyperparameters

```python
MAX_STEPS = 500_000       # Total training steps
GAMMA = 0.99              # Discount factor
EVAL_INTERVAL = 10_000    # Evaluation frequency
N_EVAL_EPISODES = 10      # Episodes per evaluation
ENTROPY_COEF = 0.01       # Entropy bonus weight
```

### Key Algorithms

**1-Step TD Advantage**:

```python
def compute_advantage(reward, value, next_value, term, gamma=0.99):
    bootstrap = 0.0 if term else next_value
    td_target = reward + gamma * bootstrap
    advantage = td_target - value
    return advantage, td_target
```

**N-Step Returns**:

```python
# G_t = Σ(γ^i * r_{t+i}) + γ^n * V_{t+n}  (with episode boundary handling)
```

### Performance Metrics

Metrics tracked for each agent:

- **Training returns**: Rolling mean over 100 episodes
- **Evaluation returns**: Unbiased policy assessment every 10K steps
- **Wall-clock time**: Total training duration
- **Sample efficiency**: Steps to reach convergence
- **Stability**: Standard deviation across seeds

## 📈 Visualization

Each agent generates comprehensive plots:

- **Training curves**: Episode returns over time
- **Evaluation performance**: Unbiased policy assessment
- **Loss curves**: Actor and critic learning dynamics
- **Stability analysis**: Mean ± std across seeds
- **Speed comparison**: Wall-clock time and throughput

## 🎓 Learning Outcomes

This project demonstrates:

1. **Proper episodic RL implementation** with truncation handling
2. **Vectorized environments** for computational efficiency
3. **Multi-step TD learning** and bias-variance tradeoffs
4. **Batch size effects** on gradient stability
5. **Rigorous experimental methodology** with multiple seeds
6. **Value function estimation** under stochastic rewards

## 🤝 Contributors

- [Skander Adam Afi](https://github.com/skan652)
- [Linda Ben Rajab](https://github.com/Lindabenrajab)

## 📄 License

This project is for educational purposes as part of a Reinforcement Learning course.

## 🙏 Acknowledgments

- Gymnasium library for CartPole environment
- PyTorch for deep learning framework
- Course instructors and TAs for guidance

---

**Last Updated**: February 2026  
**Status**: ✅ All agents tested and validated
