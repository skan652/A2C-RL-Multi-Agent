# A2C Reinforcement Learning: Multi-Agent CartPole Experiments

A comprehensive implementation of the **Advantage Actor-Critic (A2C)** algorithm with 5 experimental configurations exploring parallel workers, n-step returns, and stochastic rewards on the CartPole-v1 environment.

## 🎯 Project Overview

This project implements and compares 5 different A2C agents to study the effects of:

- **Parallel environment workers (K)**: Sample efficiency and wall-clock speed
- **N-step returns (n)**: Bias-variance tradeoff in TD learning
- **Stochastic rewards**: Value function estimation under uncertainty
- **Combined scaling (K×n)**: Batch size effects on gradient stability

All experiments use rigorous methodology with multiple random seeds (42, 123, 456) and comprehensive logging.

## 🤖 Agent Configurations

| Agent | K Workers | N-Steps | Batch Size | Learning Rate (Actor) | Purpose |
| ------- | ----------- | --------- | ------------ | ---------------------- | --------- |
| **Agent 0** | 1 | 1 | 1 | 1e-4 | Baseline (standard A2C) |
| **Agent 1** | 1 | 1 | 1 | 1e-4 | Stochastic rewards (90% masking) |
| **Agent 2** | 6 | 1 | 6 | 1e-4 | Parallel workers |
| **Agent 3** | 1 | 6 | 6 | 1e-4 | N-step returns |
| **Agent 4** | 6 | 6 | 36 | 3e-5 | Combined (best performance) |

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

## 🚀 Installation

**1. Clone the repository:**

```bash
git clone https://github.com/skan652/A2C-RL-Multi-Agent.git
cd A2C-RL-Multi-Agent
```

**2. Install dependencies:**

```bash
pip install torch gymnasium numpy matplotlib seaborn pandas
```

**3. Open the Jupyter notebook:**

```bash
jupyter lab rl-project.ipynb
```

## 💻 Usage

### Training All Agents

Execute the notebook cells sequentially:

1. **Setup** (Cells 1-3): Import libraries and set hyperparameters
2. **Core Components** (Cells 4-11): Define models, environments, and algorithms
3. **Agent Training** (Cells 13-21): Train each agent configuration
4. **Analysis** (Cell 23): Compare results and generate reports

### Training Individual Agents

Each agent can be trained independently:

```python
# Agent 0 (Baseline)
from pathlib import Path
log_dir = Path("agent0_logs")
log_dir.mkdir(exist_ok=True)
all_logs = [train_agent0(seed, log_dir) for seed in [42, 123, 456]]
plot_agent0_results(all_logs, "agent0_results.png")
```

### Evaluation

Trained agents are automatically evaluated every 10,000 steps using:

- **Agent 0-1**: Single environment evaluation
- **Agent 2-4**: Vectorized environment evaluation (6 parallel episodes)

## 📊 Key Features

### 1. Correct Bootstrap Handling

- **Truncation vs Terminal**: Distinguishes episode truncation (bootstrap with value) from true terminal states (no bootstrap)
- Critical for infinite-horizon tasks like CartPole with time limits

### 2. Episode Tracking

- Per-worker episode return accumulation in vectorized environments
- Ensures correct logging of full episode returns (not step rewards)

### 3. Stochastic Reward Experiment (Agent 1)

- Training uses 90% reward masking (r=1 becomes r=0.1 with prob 0.9)
- Evaluation uses full rewards to measure policy quality
- Tests value function estimation: V(s₀) ≈ 0.1/(1-γ) ≈ 10

### 4. N-Step Returns (Agents 3-4)

```text
G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V_{t+n}
```

- Reduces variance at the cost of slight bias
- Handles episode boundaries correctly during rollout

### 5. Gradient Stability Analysis

- Batch size scaling (K×n=36) enables higher learning rates
- Gradient variance reduction proportional to batch size
- Agent 4 uses lr_actor=3e-5 for stable training

## 📁 Project Structure

```text
A2C-RL-Multi-Agent/
├── rl-project.ipynb          # Main notebook with all implementations
├── README.md                  # This file
├── agent0_logs/              # Training logs for baseline agent
├── agent1_logs/              # Training logs for stochastic agent
├── agent2_logs/              # Training logs for K=6 agent
├── agent3_logs/              # Training logs for n=6 agent
├── agent4_logs/              # Training logs for K×n=36 agent
├── agent0_results.png        # Plots for Agent 0
├── agent1_results.png        # Plots for Agent 1
├── agent2_results.png        # Plots for Agent 2
├── agent3_results.png        # Plots for Agent 3
└── agent4_results.png        # Plots for Agent 4
```

## 🧪 Experiments & Results

### Theoretical Questions Answered

**Q1**: What is V(s₀) after convergence for Agent 0?

- **Answer**: V(s₀) ≈ 500/(1-γ) = 50,000 (infinite horizon with correct bootstrap)

**Q2**: What happens without correct bootstrap handling?

- **Answer**: V(s₀) → 0 (false terminal signal at truncation breaks value estimates)

**Q3**: What is V(s₀) for Agent 1 with stochastic rewards?

- **Answer**: V(s₀) ≈ 0.1/(1-γ) = 10 (expected reward is 0.1)
- **Eval returns**: Still ~500 (policy optimal, only training rewards masked)

**Q4**: Why does K×n enable higher learning rates?

- **Answer**: Batch=36 reduces gradient variance by 36× → can increase lr_actor without divergence

### Performance Metrics

Metrics tracked for each agent:

- **Training returns**: Rolling mean over 100 episodes
- **Evaluation returns**: Unbiased policy assessment every 10K steps
- **Wall-clock time**: Total training duration
- **Sample efficiency**: Steps to reach convergence
- **Stability**: Standard deviation across seeds

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
GAMMA = 0.99             # Discount factor
EVAL_INTERVAL = 10_000   # Evaluation frequency
N_EVAL_EPISODES = 10     # Episodes per evaluation
ENTROPY_COEF = 0.01      # Entropy bonus weight
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
G_t = Σ(γ^i * r_{t+i}) + γ^n * V_{t+n}  # With episode boundary handling
```

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

## 🤝 Team Members

- [Skander Adam Afi](https://github.com/skan652)
- [Linda Ben Rajab](https://github.com/Lindabenrajab)

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Last Updated**: February 2026  
**Status**: ✅ All agents tested and validated
