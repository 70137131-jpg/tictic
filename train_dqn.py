"""
Deep Q-Network (DQN) Implementation for Tic-Tac-Toe
====================================================

A neural network-based Q-learning approach that scales to larger games.

Key improvements over tabular Q-learning:
1. Function approximation via neural networks
2. Experience replay buffer
3. Target network for stability
4. Double DQN to reduce overestimation
5. Scalable to larger state spaces
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import time
from collections import deque, defaultdict

# Import game functions
from train_improved_agent import (
    getStateKey, _legal_moves, _winner, best_move_minimax,
    _flatten, get_symmetric_states
)


# --- DQN Network Architecture ---
class DQNetwork(nn.Module):
    """
    Deep Q-Network for Tic-Tac-Toe

    Input: 27 features (9 cells x 3 one-hot encoded [empty, X, O])
    Output: 9 Q-values (one per possible action)
    """

    def __init__(self, hidden_size=128):
        super(DQNetwork, self).__init__()

        # Input: 27 features (9 positions x 3 states)
        self.fc1 = nn.Linear(27, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 9)  # 9 possible actions

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# --- State Encoding ---
def encode_state(board):
    """
    Encode board state as one-hot vector

    Returns: Tensor of shape (27,)
    For each cell: [is_empty, is_X, is_O]
    """
    encoding = []
    for i in range(3):
        for j in range(3):
            cell = board[i][j]
            if cell == '-':
                encoding.extend([1, 0, 0])
            elif cell == 'X':
                encoding.extend([0, 1, 0])
            else:  # 'O'
                encoding.extend([0, 0, 1])

    return torch.FloatTensor(encoding)


def encode_batch(boards):
    """Encode a batch of board states"""
    return torch.stack([encode_state(board) for board in boards])


# --- DQN Agent ---
class DQNAgent:
    """
    Deep Q-Network Agent with:
    - Experience replay
    - Target network
    - Double DQN
    - Epsilon-greedy exploration
    """

    def __init__(self, lr=0.001, gamma=0.99, eps=1.0, eps_decay=0.9995,
                 eps_min=0.01, memory_size=10000, batch_size=64,
                 target_update_freq=100, hidden_size=128):

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(hidden_size).to(self.device)
        self.target_net = DQNetwork(hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = deque(maxlen=memory_size)

        # Statistics
        self.rewards = []
        self.losses = []
        self.steps = 0
        self.episodes = 0

        print(f"DQN Agent initialized on device: {self.device}")

    def get_action(self, board, greedy=False):
        """
        Select action using epsilon-greedy policy
        """
        legal_moves = _legal_moves(board)

        # Epsilon-greedy exploration
        if not greedy and random.random() < self.eps:
            return random.choice(legal_moves)

        # Greedy action selection
        with torch.no_grad():
            state = encode_state(board).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state).cpu().numpy()[0]

            # Mask illegal moves
            legal_mask = np.full(9, -np.inf)
            for (i, j) in legal_moves:
                legal_mask[i * 3 + j] = q_values[i * 3 + j]

            # Select best legal action
            best_action_idx = np.argmax(legal_mask)
            return (best_action_idx // 3, best_action_idx % 3)

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = encode_batch(states).to(self.device)
        actions = torch.LongTensor([[a[0] * 3 + a[1]] for a in actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = encode_batch(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_actions = next_q_policy.argmax(1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)
            next_q_target[dones] = 0.0
            target_q = rewards.unsqueeze(1) + self.gamma * next_q_target

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path='dqn_agent.pkl'):
        """Save agent"""
        checkpoint = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'eps': self.eps,
            'rewards': self.rewards,
            'losses': self.losses,
            'steps': self.steps,
            'episodes': self.episodes
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"DQN agent saved to {path}")

    @staticmethod
    def load(path='dqn_agent.pkl', **kwargs):
        """Load agent from checkpoint"""
        agent = DQNAgent(**kwargs)
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        agent.policy_net.load_state_dict(checkpoint['policy_net_state'])
        agent.target_net.load_state_dict(checkpoint['target_net_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        agent.eps = checkpoint['eps']
        agent.rewards = checkpoint['rewards']
        agent.losses = checkpoint['losses']
        agent.steps = checkpoint['steps']
        agent.episodes = checkpoint['episodes']

        print(f"DQN agent loaded from {path}")
        return agent


# --- Training Functions ---
def play_episode(agent, opponent_policy, agent_first=True, train=True):
    """
    Play one episode against opponent

    Returns: total reward for agent
    """
    board = [['-']*3 for _ in range(3)]
    episode_reward = 0

    if not agent_first:
        # Opponent moves first
        i, j = opponent_policy(board)
        board[i][j] = 'X'

    while True:
        # Agent's turn (plays as 'O')
        state_before = [row[:] for row in board]
        action = agent.get_action(board, greedy=not train)
        board[action[0]][action[1]] = 'O'

        # Check terminal condition
        result = _winner(board)

        if result == 'O':  # Agent wins
            reward = 1.0
            episode_reward += reward
            if train:
                agent.store_transition(state_before, action, reward, board, True)
            return episode_reward
        elif result == 'D':  # Draw
            reward = 0.5
            episode_reward += reward
            if train:
                agent.store_transition(state_before, action, reward, board, True)
            return episode_reward

        # Opponent's turn (plays as 'X')
        try:
            i, j = opponent_policy(board)
            board[i][j] = 'X'
        except:
            # No legal moves
            return episode_reward

        # Check terminal condition
        result = _winner(board)
        state_after = [row[:] for row in board]

        if result == 'X':  # Opponent wins
            reward = -1.0
            episode_reward += reward
            if train:
                agent.store_transition(state_before, action, reward, state_after, True)
            return episode_reward
        elif result == 'D':  # Draw
            reward = 0.5
            episode_reward += reward
            if train:
                agent.store_transition(state_before, action, reward, state_after, True)
            return episode_reward
        else:
            # Game continues
            reward = 0.0
            if train:
                agent.store_transition(state_before, action, reward, state_after, False)


def train_dqn_agent(episodes=50000, save_path='dqn_agent.pkl'):
    """
    Train DQN agent with curriculum learning
    """
    print("=" * 60)
    print("DEEP Q-NETWORK (DQN) TRAINING")
    print("=" * 60)
    print(f"Total Episodes: {episodes:,}")
    print(f"Training Phases:")
    print(f"  Phase 1 (0-30%):   vs Random Opponent")
    print(f"  Phase 2 (30-70%):  vs Self-Play")
    print(f"  Phase 3 (70-100%): vs Minimax Optimal")
    print("=" * 60)

    # Initialize agent
    agent = DQNAgent(
        lr=0.001,
        gamma=0.99,
        eps=1.0,
        eps_decay=0.9995,
        eps_min=0.01,
        memory_size=10000,
        batch_size=64,
        target_update_freq=100,
        hidden_size=128
    )

    # Opponent policies
    def random_opponent(board):
        return random.choice(_legal_moves(board))

    def self_play_opponent(board):
        return agent.get_action(board, greedy=False)

    def minimax_opponent(board):
        return best_move_minimax(board, key='X')

    start_time = time.time()
    wins = draws = losses = 0
    phase = 1

    for ep in range(episodes):
        # Curriculum learning
        if ep < episodes * 0.3:
            opponent = random_opponent
            if phase != 1:
                phase = 1
                print(f"\n>> Phase 1: Training vs Random (Episode {ep+1})")
        elif ep < episodes * 0.7:
            opponent = self_play_opponent
            if phase != 2:
                phase = 2
                print(f"\n>> Phase 2: Training vs Self-Play (Episode {ep+1})")
        else:
            opponent = minimax_opponent
            if phase != 3:
                phase = 3
                print(f"\n>> Phase 3: Training vs Minimax (Episode {ep+1})")

        # Play episode
        agent_first = random.random() < 0.5
        reward = play_episode(agent, opponent, agent_first, train=True)
        agent.rewards.append(reward)

        if reward > 0.9:
            wins += 1
        elif reward > 0.4:
            draws += 1
        else:
            losses += 1

        # Training step
        loss = agent.train_step()

        # Update target network
        agent.steps += 1
        if agent.steps % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()
        agent.episodes += 1

        # Progress updates
        if (ep + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            win_rate = wins / (ep + 1) * 100
            avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
            print(f"Episode {ep+1:,}/{episodes:,} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Wins: {wins} ({win_rate:.1f}%) | "
                  f"Draws: {draws} | Losses: {losses} | "
                  f"Epsilon: {agent.eps:.4f} | "
                  f"Avg Loss: {avg_loss:.4f}")

    # Final stats
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total Time: {total_time:.1f} seconds")
    print(f"Episodes: {episodes:,}")
    print(f"Final Results:")
    print(f"  Wins:   {wins:,} ({wins/episodes*100:.1f}%)")
    print(f"  Draws:  {draws:,} ({draws/episodes*100:.1f}%)")
    print(f"  Losses: {losses:,} ({losses/episodes*100:.1f}%)")
    print(f"Final Epsilon: {agent.eps:.4f}")
    print(f"Avg Loss (last 1000): {np.mean(agent.losses[-1000:]):.4f}")
    print("=" * 60)

    # Save
    agent.save(save_path)
    return agent


# --- Evaluation ---
def evaluate_dqn_agent(agent_path='dqn_agent.pkl', games=1000):
    """Evaluate DQN agent vs minimax"""
    print("\n" + "=" * 60)
    print("EVALUATING DQN AGENT VS MINIMAX OPTIMAL")
    print("=" * 60)

    agent = DQNAgent.load(agent_path)
    agent.policy_net.eval()

    wins = draws = losses = 0

    for _ in range(games):
        board = [['-']*3 for _ in range(3)]
        turn = random.choice(['X', 'O'])

        while True:
            if turn == 'X':
                i, j = best_move_minimax(board, key='X')
                board[i][j] = 'X'
            else:
                i, j = agent.get_action(board, greedy=True)
                board[i][j] = 'O'

            result = _winner(board)
            if result == 'O':
                wins += 1
                break
            elif result == 'X':
                losses += 1
                break
            elif result == 'D':
                draws += 1
                break

            turn = 'O' if turn == 'X' else 'X'

    print(f"Games: {games}")
    print(f"Wins:   {wins} ({wins/games*100:.1f}%)")
    print(f"Draws:  {draws} ({draws/games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/games*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    # Train DQN agent
    agent = train_dqn_agent(episodes=50000, save_path='dqn_agent.pkl')

    # Evaluate
    evaluate_dqn_agent('dqn_agent.pkl', games=1000)
