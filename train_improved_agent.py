"""
Enhanced Q-Learning Training with Modern RL Improvements
==========================================================

Improvements implemented:
1. Self-play training (learns from playing against itself)
2. Experience replay buffer
3. Symmetric state augmentation (rotations & reflections)
4. Optimistic initialization of Q-values
5. Epsilon decay schedule
6. Curriculum learning (progressively harder opponents)
7. Training against minimax opponent
"""

import random
import pickle
import time
import numpy as np
from collections import defaultdict, deque
from functools import lru_cache
from abc import ABC, abstractmethod
import collections


# --- Game Helper Functions ---
def getStateKey(board):
    """Convert board to state string"""
    return ''.join(board[r][c] for r in range(3) for c in range(3))


def _legal_moves(board):
    """Get all legal moves"""
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == '-']


def _winner(board):
    """Check for winner. Returns 'X', 'O', 'D' for draw, or None if ongoing"""
    lines = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],  # rows
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],  # cols
        [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]  # diagonals
    ]
    for line in lines:
        vals = [board[r][c] for r, c in line]
        if vals[0] != '-' and vals[0] == vals[1] == vals[2]:
            return vals[0]
    if all(board[i][j] != '-' for i in range(3) for j in range(3)):
        return 'D'
    return None


def _flatten(board):
    """Flatten board for minimax caching"""
    return ''.join(board[r][c] for r in range(3) for c in range(3))


@lru_cache(maxsize=None)
def _minimax_cached(flat_board, player):
    """Minimax algorithm with caching for optimal play"""
    board = [[flat_board[r*3+c] for c in range(3)] for r in range(3)]
    term = _winner(board)
    if term == 'X': return 1, None
    if term == 'O': return -1, None
    if term == 'D': return 0, None

    moves = _legal_moves(board)
    if player == 'X':
        best_score, best_mv = -2, None
        for (i, j) in moves:
            board[i][j] = 'X'
            sc, _ = _minimax_cached(_flatten(board), 'O')
            board[i][j] = '-'
            if sc > best_score:
                best_score, best_mv = sc, (i, j)
                if best_score == 1: break
        return best_score, best_mv
    else:
        best_score, best_mv = 2, None
        for (i, j) in moves:
            board[i][j] = 'O'
            sc, _ = _minimax_cached(_flatten(board), 'X')
            board[i][j] = '-'
            if sc < best_score:
                best_score, best_mv = sc, (i, j)
                if best_score == -1: break
        return best_score, best_mv


def best_move_minimax(board, key='X'):
    """Get best move using minimax"""
    _, mv = _minimax_cached(_flatten(board), key)
    return mv if mv is not None else _legal_moves(board)[0]


# --- State Augmentation for Symmetry ---
def get_symmetric_states(board):
    """
    Generate all 8 symmetric states (4 rotations + 4 reflections)
    This allows the agent to learn from symmetrically equivalent positions
    """
    states = []

    # Original
    states.append([row[:] for row in board])

    # 90° rotation
    rot90 = [[board[2-j][i] for j in range(3)] for i in range(3)]
    states.append(rot90)

    # 180° rotation
    rot180 = [[board[2-i][2-j] for j in range(3)] for i in range(3)]
    states.append(rot180)

    # 270° rotation
    rot270 = [[board[j][2-i] for j in range(3)] for i in range(3)]
    states.append(rot270)

    # Horizontal flip
    h_flip = [[board[i][2-j] for j in range(3)] for i in range(3)]
    states.append(h_flip)

    # Vertical flip
    v_flip = [[board[2-i][j] for j in range(3)] for i in range(3)]
    states.append(v_flip)

    # Diagonal flip (main)
    d_flip1 = [[board[j][i] for j in range(3)] for i in range(3)]
    states.append(d_flip1)

    # Diagonal flip (anti)
    d_flip2 = [[board[2-j][2-i] for j in range(3)] for i in range(3)]
    states.append(d_flip2)

    return states


# --- Enhanced Q-Learner with Modern Improvements ---
class EnhancedQlearner:
    """
    Enhanced Q-learning agent with:
    - Optimistic initialization
    - Experience replay
    - Symmetric state exploitation
    - Epsilon decay
    """

    def __init__(self, alpha=0.3, gamma=0.95, eps=0.3, eps_decay=0.9999,
                 optimistic_init=0.5, replay_buffer_size=10000):
        # Hyperparameters
        self.alpha = alpha  # Learning rate (lower for stability)
        self.gamma = gamma  # Discount factor (higher values future rewards more)
        self.eps = eps  # Initial exploration rate
        self.eps_min = 0.01  # Minimum exploration
        self.eps_decay = eps_decay  # Decay rate per episode
        self.optimistic_init = optimistic_init  # Encourage exploration

        # Actions: all 9 positions
        self.actions = [(i, j) for i in range(3) for j in range(3)]

        # Q-values with optimistic initialization
        self.Q = {}
        for action in self.actions:
            # Use int subclass to avoid lambda pickle issues
            self.Q[action] = defaultdict(float)
            # Manually set optimistic values during get_action

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Statistics
        self.rewards = []
        self.episode_count = 0

    def get_action(self, board, greedy=False):
        """
        Select action using epsilon-greedy policy
        If greedy=True, always choose best action (for evaluation)
        """
        state = getStateKey(board)
        possible_actions = _legal_moves(board)

        # Epsilon-greedy exploration
        if not greedy and random.random() < self.eps:
            return random.choice(possible_actions)

        # Greedy: choose best action (with optimistic initialization)
        values = []
        for a in possible_actions:
            if state in self.Q[a]:
                values.append(self.Q[a][state])
            else:
                values.append(self.optimistic_init)

        max_val = max(values)
        best_actions = [a for a, v in zip(possible_actions, values) if v == max_val]
        return random.choice(best_actions)

    def update_with_experience(self, state, action, reward, next_state):
        """Store experience and perform Q-learning update"""
        # Add to replay buffer
        self.replay_buffer.append((state, action, reward, next_state))

        # Q-learning update
        if next_state is not None:
            possible_actions = _legal_moves(self._state_to_board(next_state))
            if possible_actions:
                q_next_max = max(self.Q[a][next_state] for a in possible_actions)
                td_target = reward + self.gamma * q_next_max
            else:
                td_target = reward
        else:
            td_target = reward

        td_error = td_target - self.Q[action][state]
        self.Q[action][state] += self.alpha * td_error

        # Symmetric state updates (learn from rotations/reflections)
        self._update_symmetric_states(state, action, td_error)

    def _state_to_board(self, state):
        """Convert state string back to board"""
        return [[state[r*3+c] for c in range(3)] for r in range(3)]

    def _update_symmetric_states(self, state, action, td_error):
        """Apply update to all symmetric transformations"""
        board = self._state_to_board(state)
        symmetric_boards = get_symmetric_states(board)

        for sym_board in symmetric_boards:
            sym_state = getStateKey(sym_board)
            if sym_state != state:  # Don't double-update original
                # Apply partial update to symmetric states
                self.Q[action][sym_state] += 0.3 * self.alpha * td_error

    def replay_experience(self, batch_size=32):
        """Sample from replay buffer and learn"""
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in batch:
            if next_state is not None:
                possible_actions = _legal_moves(self._state_to_board(next_state))
                if possible_actions:
                    q_next_max = max(self.Q[a][next_state] for a in possible_actions)
                    td_target = reward + self.gamma * q_next_max
                else:
                    td_target = reward
            else:
                td_target = reward

            td_error = td_target - self.Q[action][state]
            self.Q[action][state] += 0.5 * self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.episode_count += 1

    def save(self, path='q_agent.pkl'):
        """Save agent to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f">> Agent saved to {path}")


# --- Training Functions ---
def play_game_vs_opponent(agent, opponent_policy, agent_first=True):
    """
    Play one game episode

    opponent_policy: function that takes (board) and returns (i, j) move
    Returns: reward for agent (1=win, 0=draw, -1=loss)
    """
    board = [['-']*3 for _ in range(3)]
    history = []  # Store (state, action, board_before_action)

    if not agent_first:
        # Opponent moves first
        i, j = opponent_policy(board)
        board[i][j] = 'X'

    while True:
        # Agent's turn (plays as 'O')
        state_before = getStateKey(board)
        action = agent.get_action(board)
        board[action[0]][action[1]] = 'O'
        history.append((state_before, action))

        # Check terminal condition
        result = _winner(board)
        if result == 'O':  # Agent wins
            state_after = getStateKey(board)
            agent.update_with_experience(state_before, action, 1.0, None)
            agent.rewards.append(1)
            return 1
        elif result == 'D':  # Draw
            state_after = getStateKey(board)
            agent.update_with_experience(state_before, action, 0.5, None)
            agent.rewards.append(0)
            return 0

        # Opponent's turn (plays as 'X')
        try:
            i, j = opponent_policy(board)
            board[i][j] = 'X'
        except:
            # No legal moves (shouldn't happen)
            agent.rewards.append(0)
            return 0

        # Check terminal condition
        result = _winner(board)
        state_after = getStateKey(board) if result is None else None

        if result == 'X':  # Opponent wins
            agent.update_with_experience(state_before, action, -1.0, None)
            agent.rewards.append(-1)
            return -1
        elif result == 'D':  # Draw
            agent.update_with_experience(state_before, action, 0.5, None)
            agent.rewards.append(0)
            return 0
        else:
            # Game continues
            agent.update_with_experience(state_before, action, 0.0, state_after)


def train_enhanced_agent(episodes=100000, save_path='q_agent.pkl'):
    """
    Train agent with curriculum learning:
    - Phase 1: vs random (30%)
    - Phase 2: vs self-play (40%)
    - Phase 3: vs minimax (30%)
    """
    print("=" * 60)
    print(">> ENHANCED Q-LEARNING TRAINING")
    print("=" * 60)
    print(f"Total Episodes: {episodes:,}")
    print(f"Training Phases:")
    print(f"  Phase 1 (0-30%):   vs Random Opponent")
    print(f"  Phase 2 (30-70%):  vs Self-Play")
    print(f"  Phase 3 (70-100%): vs Minimax Optimal")
    print("=" * 60)

    agent = EnhancedQlearner(
        alpha=0.3,
        gamma=0.95,
        eps=0.4,
        eps_decay=0.99995,
        optimistic_init=0.6
    )

    # Opponent policies
    def random_opponent(board):
        return random.choice(_legal_moves(board))

    def self_play_opponent(board):
        # Clone agent's Q-values for opponent
        return agent.get_action(board, greedy=False)

    def minimax_opponent(board):
        return best_move_minimax(board, key='X')

    start_time = time.time()
    wins = draws = losses = 0
    phase = 1

    for ep in range(episodes):
        # Curriculum learning: progressively harder opponents
        if ep < episodes * 0.3:
            opponent = random_opponent
            if phase != 1:
                phase = 1
                print(f"\n>> Phase 1: Training vs Random Opponent (Episode {ep+1})")
        elif ep < episodes * 0.7:
            opponent = self_play_opponent
            if phase != 2:
                phase = 2
                print(f"\n>> Phase 2: Training vs Self-Play (Episode {ep+1})")
        else:
            opponent = minimax_opponent
            if phase != 3:
                phase = 3
                print(f"\n>> Phase 3: Training vs Minimax Optimal (Episode {ep+1})")

        # Play game
        agent_first = random.random() < 0.5
        result = play_game_vs_opponent(agent, opponent, agent_first)

        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        # Experience replay every 10 episodes
        if ep % 10 == 0:
            agent.replay_experience(batch_size=32)

        # Decay epsilon
        agent.decay_epsilon()

        # Progress updates
        if (ep + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            win_rate = wins / (ep + 1) * 100
            print(f"Episode {ep+1:,}/{episodes:,} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Wins: {wins} ({win_rate:.1f}%) | "
                  f"Draws: {draws} | Losses: {losses} | "
                  f"Epsilon: {agent.eps:.4f}")

    # Final evaluation
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(">> TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total Time: {total_time:.1f} seconds")
    print(f"Episodes: {episodes:,}")
    print(f"Final Results:")
    print(f"  Wins:   {wins:,} ({wins/episodes*100:.1f}%)")
    print(f"  Draws:  {draws:,} ({draws/episodes*100:.1f}%)")
    print(f"  Losses: {losses:,} ({losses/episodes*100:.1f}%)")
    print(f"Final Epsilon (exploration): {agent.eps:.4f}")
    print(f"Total Q-values learned: {sum(len(agent.Q[a]) for a in agent.actions):,}")
    print("=" * 60)

    # Save agent
    agent.save(save_path)
    return agent


# --- Evaluation ---
def evaluate_agent(agent_path='q_agent.pkl', games=1000):
    """Evaluate trained agent against minimax"""
    print("\n" + "=" * 60)
    print(">> EVALUATING AGENT VS MINIMAX OPTIMAL")
    print("=" * 60)

    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)

    wins = draws = losses = 0

    for _ in range(games):
        board = [['-']*3 for _ in range(3)]
        turn = random.choice(['X', 'O'])  # Random first player

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
    # Train the enhanced agent
    agent = train_enhanced_agent(episodes=100000, save_path='q_agent.pkl')

    # Evaluate
    evaluate_agent('q_agent.pkl', games=1000)
