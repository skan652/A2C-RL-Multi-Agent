"""
Visualization and Plotting Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path


def setup_plots():
    """Set up matplotlib style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100


def plot_training_results(all_logs: List[Dict], save_path: str, agent_name: str,
                          max_steps: int = 500_000, eval_interval: int = 20_000):
    """
    Generic plotting function for training results.
    
    Args:
        all_logs: List of log dictionaries from multiple seeds
        save_path: Path to save the plot
        agent_name: Name of the agent (for title)
        max_steps: Maximum training steps
        eval_interval: Evaluation interval
    """
    steps = np.arange(0, max_steps, eval_interval)
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training returns (smoothed)
    train_means = [
        np.convolve(log["train_returns"], np.ones(50) / 50, mode="valid")
        for log in all_logs
    ]
    axes[0, 0].plot(train_means[0])
    if len(train_means) > 1:
        axes[0, 0].fill_between(
            range(len(train_means[0])),
            [min(m[i] for m in train_means) for i in range(len(train_means[0]))],
            [max(m[i] for m in train_means) for i in range(len(train_means[0]))],
            alpha=0.3,
        )
    axes[0, 0].set_title(f"{agent_name}: Training Returns (smoothed)")
    axes[0, 0].set_xlabel("Episodes")
    axes[0, 0].set_ylabel("Return")

    # Evaluation returns
    for log in all_logs:
        axes[0, 1].plot(
            steps[: len(log["eval_returns"])], log["eval_returns"], 
            "o-", alpha=0.7, label=f"Seed {log['seed']}"
        )
    axes[0, 1].set_title(f"{agent_name}: Evaluation Returns")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Return")
    axes[0, 1].legend()

    # Losses
    axes[1, 0].plot(all_logs[0]["actor_losses"][:10000], label="Actor", alpha=0.7)
    axes[1, 0].plot(all_logs[0]["critic_losses"][:10000], label="Critic", alpha=0.7)
    axes[1, 0].set_title("Training Losses")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Value function
    if "final_values" in all_logs[0] and len(all_logs[0]["final_values"]) > 0:
        for log in all_logs:
            if len(log["final_values"]) > 0:
                axes[1, 1].plot(log["final_values"][0], alpha=0.7, label=f"Seed {log['seed']}")
        axes[1, 1].set_title("Value Function (Final Episode)")
        axes[1, 1].set_xlabel("Timesteps")
        axes[1, 1].set_ylabel("V(s)")
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_agents_comparison(all_agent_logs: Dict, save_path: str = "all_agents_comparison.png",
                               max_steps: int = 500_000, eval_interval: int = 20_000):
    """
    Create comprehensive comparison plots across all trained agents.
    
    Args:
        all_agent_logs: Dictionary mapping agent names to their training logs
        save_path: Path to save the comparison plot
        max_steps: Maximum training steps
        eval_interval: Evaluation interval
    """
    if not all_agent_logs:
        print("⚠️  No agents to plot - train some agents first!")
        return
        
    num_agents = len(all_agent_logs)
    cols = min(3, num_agents)
    rows = (num_agents + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if num_agents == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if num_agents > 1 else axes
    
    steps = np.arange(0, max_steps, eval_interval)

    # Plot eval returns for each agent
    for i, (agent_name, logs) in enumerate(all_agent_logs.items()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        
        # Individual seed runs
        for log in logs:
            evals = log["eval_returns"]
            ax.plot(
                steps[: len(evals)], evals, alpha=0.3, linewidth=1
            )

        # Mean across seeds
        max_len = max(len(log["eval_returns"]) for log in logs)
        mean_evals = []
        for step_idx in range(max_len):
            step_vals = [
                log["eval_returns"][step_idx]
                for log in logs
                if step_idx < len(log["eval_returns"])
            ]
            mean_evals.append(np.mean(step_vals))

        ax.plot(steps[: len(mean_evals)], mean_evals, "k-", linewidth=3, label="Mean")
        ax.set_title(f"{agent_name.replace('agent', 'Agent ')}")
        ax.set_ylim(0, 550)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Eval Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_agents, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Comparison of All Agents (Mean ± Individual Runs)", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_stability_comparison(all_agent_logs: Dict, save_path: str = "stability_comparison.png"):
    """
    Plot stability metrics (variance across seeds) for all agents.
    
    Args:
        all_agent_logs: Dictionary of agent logs
        save_path: Path to save the plot
    """
    if not all_agent_logs:
        return
        
    agent_names = []
    means = []
    stds = []
    
    for agent_name, logs in all_agent_logs.items():
        final_returns = [np.mean(log['final_returns']) for log in logs]
        agent_names.append(agent_name.replace('agent', 'Agent '))
        means.append(np.mean(final_returns))
        stds.append(np.std(final_returns))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean returns
    bars1 = ax1.bar(range(len(agent_names)), means, color='skyblue', edgecolor='navy')
    ax1.set_xticks(range(len(agent_names)))
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Return')
    ax1.set_title('Final Performance by Agent')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Standard deviation (stability)
    bars2 = ax2.bar(range(len(agent_names)), stds, color='lightcoral', edgecolor='darkred')
    ax2.set_xticks(range(len(agent_names)))
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.set_ylabel('Std Dev of Returns')
    ax2.set_title('Training Stability (Lower = More Stable)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_value_function_comparison(agent0_logs: List[Dict], agent1_logs: List[Dict], 
                                   save_path: str = "value_comparison.png"):
    """
    Compare value functions between Agent 0 (full rewards) and Agent 1 (stochastic rewards).
    
    Args:
        agent0_logs: Logs from Agent 0
        agent1_logs: Logs from Agent 1
        save_path: Path to save the plot
    """
    if not agent0_logs or not agent1_logs:
        print("⚠️  Need both Agent 0 and Agent 1 data")
        return
        
    if "final_values" not in agent0_logs[0] or "final_values" not in agent1_logs[0]:
        print("⚠️  final_values not found")
        return
    
    if not agent0_logs[0]["final_values"] or not agent1_logs[0]["final_values"]:
        print("⚠️  final_values are empty")
        return
        
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    agent0_final = agent0_logs[0]["final_values"][0]
    agent1_final = agent1_logs[0]["final_values"][0]
    
    min_len = min(len(agent0_final), len(agent1_final))
    traj_steps = np.arange(min_len)

    ax.plot(traj_steps, agent0_final[:min_len], "b-", 
            label="Agent 0: V(s) with full rewards", linewidth=2, alpha=0.8)
    ax.plot(traj_steps, agent1_final[:min_len], "r-", 
            label="Agent 1: V(s) with E[r]=0.1", linewidth=2, alpha=0.8)
    ax.axhline(y=500 / (1 - 0.99), color="b", linestyle="--", 
               alpha=0.5, label="Theoretical V ≈ 500/(1-γ) = 50k")
    ax.axhline(y=0.1 / (1 - 0.99), color="r", linestyle="--", 
               alpha=0.5, label="Theoretical V ≈ 0.1/(1-γ) = 10")
    ax.set_xlabel("Timesteps in Episode")
    ax.set_ylabel("V(s_t)")
    ax.set_title("Value Function: Full Rewards vs Stochastic Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
