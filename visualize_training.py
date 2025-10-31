"""
Training Visualization Dashboard
=================================

Creates interactive visualizations to understand agent behavior:
1. Training curves (win rate, loss, epsilon decay)
2. Q-value heatmaps for different board states
3. Action preferences visualization
4. Performance vs opponent type
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
from collections import defaultdict
import json
import os

# Import game functions from train_improved_agent
import sys
sys.path.append(os.path.dirname(__file__))
from train_improved_agent import getStateKey, _legal_moves, _winner


def load_agent(path='q_agent.pkl'):
    """Load trained agent"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_training_curves(agent, save_path='static/training_curves.png'):
    """Plot training performance curves"""
    if not hasattr(agent, 'rewards') or len(agent.rewards) == 0:
        print("No training data available")
        return

    rewards = agent.rewards
    episodes = len(rewards)

    # Calculate moving averages
    window = min(1000, episodes // 10)
    if window < 1:
        window = 1

    moving_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window)
        moving_avg.append(np.mean(rewards[start:i+1]))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Performance Dashboard', fontsize=16, fontweight='bold')

    # 1. Reward over time
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    axes[0, 0].plot(moving_avg, linewidth=2, label=f'Moving Avg (window={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Win/Draw/Loss distribution over time
    window_size = max(100, episodes // 20)
    win_rate = []
    draw_rate = []
    loss_rate = []

    for i in range(0, episodes, window_size):
        window_rewards = rewards[i:i+window_size]
        wins = sum(1 for r in window_rewards if r > 0.9)
        draws = sum(1 for r in window_rewards if 0.4 < r < 0.6)
        losses = sum(1 for r in window_rewards if r < 0)
        total = len(window_rewards)

        win_rate.append(wins / total * 100)
        draw_rate.append(draws / total * 100)
        loss_rate.append(losses / total * 100)

    x_pos = range(0, episodes, window_size)
    axes[0, 1].plot(x_pos, win_rate, label='Win %', linewidth=2, color='green')
    axes[0, 1].plot(x_pos, draw_rate, label='Draw %', linewidth=2, color='blue')
    axes[0, 1].plot(x_pos, loss_rate, label='Loss %', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].set_title('Win/Draw/Loss Rate Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Cumulative reward
    cumulative = np.cumsum(rewards)
    axes[1, 0].plot(cumulative, linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Epsilon decay (if available)
    if hasattr(agent, 'eps'):
        eps_history = [agent.eps_min + (agent.eps - agent.eps_min) * (agent.eps_decay ** i)
                       for i in range(episodes)]
        axes[1, 1].plot(eps_history, linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate Decay')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Epsilon decay not tracked',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Exploration Rate Decay')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_q_value_heatmap(agent, save_path='static/q_value_heatmap.png'):
    """Visualize Q-values for interesting board states"""
    # Define interesting board states to visualize
    states = {
        'Empty Board': '---------',
        'Center Taken (X)': '----X----',
        'Corner Opening (X)': 'X--------',
        'X Threatens Win': 'XX-OO----',
        'O Threatens Win': 'OO-XX----',
        'Complex Position': 'XOXOX-O--'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Q-Value Heatmaps for Key Board States', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, (state_name, state_str) in enumerate(states.items()):
        board = [[state_str[r*3+c] for c in range(3)] for r in range(3)]
        q_grid = np.zeros((3, 3))

        # Get Q-values for each position
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    action = (i, j)
                    if state_str in agent.Q[action]:
                        q_grid[i, j] = agent.Q[action][state_str]
                    else:
                        q_grid[i, j] = 0
                else:
                    q_grid[i, j] = np.nan  # Occupied positions

        # Plot heatmap
        im = axes[idx].imshow(q_grid, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[idx].set_title(state_name, fontweight='bold')
        axes[idx].set_xticks(range(3))
        axes[idx].set_yticks(range(3))

        # Add text annotations
        for i in range(3):
            for j in range(3):
                if board[i][j] != '-':
                    text = axes[idx].text(j, i, board[i][j], ha="center", va="center",
                                         color="black", fontsize=20, fontweight='bold')
                elif not np.isnan(q_grid[i, j]):
                    text = axes[idx].text(j, i, f'{q_grid[i, j]:.2f}', ha="center", va="center",
                                         color="black", fontsize=10)

        axes[idx].set_xticks(np.arange(3)-.5, minor=True)
        axes[idx].set_yticks(np.arange(3)-.5, minor=True)
        axes[idx].grid(which="minor", color="black", linestyle='-', linewidth=2)
        axes[idx].tick_params(which="minor", size=0)

    plt.colorbar(im, ax=axes, label='Q-Value', shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Q-value heatmap saved to {save_path}")


def plot_state_space_statistics(agent, save_path='static/state_space_stats.png'):
    """Visualize state space coverage and Q-value statistics"""
    # Collect statistics
    total_states_per_action = [len(agent.Q[action]) for action in agent.actions]
    total_unique_states = len(set().union(*[set(agent.Q[action].keys()) for action in agent.actions]))

    # Q-value distribution
    all_q_values = []
    for action in agent.actions:
        all_q_values.extend(agent.Q[action].values())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('State Space Coverage & Q-Value Statistics', fontsize=16, fontweight='bold')

    # 1. States visited per action
    axes[0, 0].bar(range(len(agent.actions)), total_states_per_action, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Action (Position)')
    axes[0, 0].set_ylabel('Number of States')
    axes[0, 0].set_title(f'States Visited per Action (Total: {total_unique_states})')
    axes[0, 0].set_xticks(range(9))
    axes[0, 0].set_xticklabels([f'({a[0]},{a[1]})' for a in agent.actions], rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Q-value distribution
    axes[0, 1].hist(all_q_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Q-Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Q-Value Distribution')
    axes[0, 1].axvline(np.mean(all_q_values), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(all_q_values):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-value statistics per action
    q_means = [np.mean(list(agent.Q[action].values())) if len(agent.Q[action]) > 0 else 0
               for action in agent.actions]
    q_stds = [np.std(list(agent.Q[action].values())) if len(agent.Q[action]) > 0 else 0
              for action in agent.actions]

    x_pos = range(len(agent.actions))
    axes[1, 0].bar(x_pos, q_means, yerr=q_stds, color='orange', alpha=0.7,
                  edgecolor='black', capsize=5)
    axes[1, 0].set_xlabel('Action (Position)')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Mean Q-Value per Action (with Std Dev)')
    axes[1, 0].set_xticks(range(9))
    axes[1, 0].set_xticklabels([f'({a[0]},{a[1]})' for a in agent.actions], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Agent statistics summary
    stats_text = f"""
    Agent Statistics:

    Total Unique States: {total_unique_states:,}
    Total Q-Values: {len(all_q_values):,}

    Q-Value Range: [{min(all_q_values):.3f}, {max(all_q_values):.3f}]
    Q-Value Mean: {np.mean(all_q_values):.3f}
    Q-Value Std: {np.std(all_q_values):.3f}

    Training Episodes: {len(agent.rewards):,}
    Total Reward: {sum(agent.rewards):.1f}

    Current Epsilon: {agent.eps:.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"State space statistics saved to {save_path}")


def generate_all_visualizations(agent_path='q_agent.pkl'):
    """Generate all visualization dashboards"""
    print("=" * 60)
    print("GENERATING TRAINING VISUALIZATIONS")
    print("=" * 60)

    # Load agent
    agent = load_agent(agent_path)
    print(f"Agent loaded: {type(agent).__name__}")

    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    # Generate visualizations
    plot_training_curves(agent)
    plot_q_value_heatmap(agent)
    plot_state_space_statistics(agent)

    print("=" * 60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("View them at:")
    print("  - static/training_curves.png")
    print("  - static/q_value_heatmap.png")
    print("  - static/state_space_stats.png")


if __name__ == "__main__":
    generate_all_visualizations('q_agent.pkl')
