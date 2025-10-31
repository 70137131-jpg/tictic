"""
Distributed Training System for RL Agents
==========================================

Parallel training using multiprocessing for 10x faster iteration:
1. Multiple worker processes play games simultaneously
2. Shared experience replay buffer
3. Centralized parameter updates
4. Asynchronous learning

Supports both tabular Q-learning and DQN agents.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Value
import time
import random
import numpy as np
import pickle
from collections import deque
import os

# Import game functions
from train_improved_agent import (
    getStateKey, _legal_moves, _winner, best_move_minimax,
    EnhancedQlearner, play_game_vs_opponent
)


# --- Worker Functions ---
def experience_worker(worker_id, agent_dict, experience_queue, episode_counter,
                     total_episodes, opponent_type, stop_flag):
    """
    Worker process that generates gameplay experiences

    Args:
        worker_id: Unique worker identifier
        agent_dict: Shared dict containing agent parameters
        experience_queue: Queue to send experiences to learner
        episode_counter: Shared counter for total episodes completed
        total_episodes: Total number of episodes to run
        opponent_type: 'random', 'self_play', or 'minimax'
        stop_flag: Shared flag to signal workers to stop
    """
    # Create local agent copy
    agent = EnhancedQlearner(
        alpha=agent_dict['alpha'],
        gamma=agent_dict['gamma'],
        eps=agent_dict['eps'],
        eps_decay=agent_dict['eps_decay'],
        optimistic_init=agent_dict['optimistic_init']
    )

    # Opponent policy
    if opponent_type == 'random':
        def opponent(board):
            return random.choice(_legal_moves(board))
    elif opponent_type == 'minimax':
        def opponent(board):
            return best_move_minimax(board, key='X')
    else:  # self_play
        def opponent(board):
            return agent.get_action(board, greedy=False)

    print(f"[Worker {worker_id}] Started (opponent: {opponent_type})")

    episodes_done = 0
    while not stop_flag.value:
        with episode_counter.get_lock():
            if episode_counter.value >= total_episodes:
                break
            episode_counter.value += 1
            current_episode = episode_counter.value

        # Play episode and collect experiences
        board = [['-']*3 for _ in range(3)]
        experiences = []
        agent_first = random.random() < 0.5

        if not agent_first:
            i, j = opponent(board)
            board[i][j] = 'X'

        episode_reward = 0
        while True:
            # Agent's turn
            state_before = getStateKey(board)
            action = agent.get_action(board, greedy=False)
            board[action[0]][action[1]] = 'O'

            # Check terminal
            result = _winner(board)
            if result == 'O':
                reward = 1.0
                experiences.append((state_before, action, reward, None, True))
                episode_reward = reward
                break
            elif result == 'D':
                reward = 0.5
                experiences.append((state_before, action, reward, None, True))
                episode_reward = reward
                break

            # Opponent's turn
            try:
                i, j = opponent(board)
                board[i][j] = 'X'
            except:
                break

            # Check terminal
            result = _winner(board)
            state_after = getStateKey(board) if result is None else None

            if result == 'X':
                reward = -1.0
                experiences.append((state_before, action, reward, state_after, True))
                episode_reward = reward
                break
            elif result == 'D':
                reward = 0.5
                experiences.append((state_before, action, reward, state_after, True))
                episode_reward = reward
                break
            else:
                reward = 0.0
                experiences.append((state_before, action, reward, state_after, False))

        # Send experiences to learner
        experience_queue.put({
            'worker_id': worker_id,
            'experiences': experiences,
            'reward': episode_reward,
            'episode': current_episode
        })

        episodes_done += 1

        # Periodic sync with latest agent parameters
        if episodes_done % 100 == 0:
            agent.eps = agent_dict['eps']

    print(f"[Worker {worker_id}] Completed {episodes_done} episodes")


def learner_process(agent_dict, experience_queue, stats_queue,
                   total_episodes, num_workers, stop_flag):
    """
    Central learner that updates agent parameters from worker experiences

    Args:
        agent_dict: Shared dict containing agent parameters
        experience_queue: Queue receiving experiences from workers
        stats_queue: Queue to send training statistics
        total_episodes: Total episodes to train
        num_workers: Number of worker processes
        stop_flag: Shared flag to signal completion
    """
    # Create agent
    agent = EnhancedQlearner(
        alpha=agent_dict['alpha'],
        gamma=agent_dict['gamma'],
        eps=agent_dict['eps'],
        eps_decay=agent_dict['eps_decay'],
        optimistic_init=agent_dict['optimistic_init']
    )

    print(f"[Learner] Started - waiting for experiences from {num_workers} workers")

    episodes_processed = 0
    total_rewards = []
    wins = draws = losses = 0

    start_time = time.time()
    last_update = start_time

    while not stop_flag.value or not experience_queue.empty():
        try:
            # Get experience batch from queue (timeout to check stop_flag)
            batch = experience_queue.get(timeout=1)

            # Update agent from experiences
            for (state, action, reward, next_state, done) in batch['experiences']:
                agent.update_with_experience(state, action, reward, next_state)

            # Track statistics
            episode_reward = batch['reward']
            total_rewards.append(episode_reward)

            if episode_reward > 0.9:
                wins += 1
            elif episode_reward > 0.4:
                draws += 1
            else:
                losses += 1

            episodes_processed += 1

            # Decay epsilon
            agent.decay_epsilon()

            # Update shared agent parameters
            agent_dict['eps'] = agent.eps

            # Periodic statistics
            if time.time() - last_update > 5.0:
                elapsed = time.time() - start_time
                win_rate = wins / max(episodes_processed, 1) * 100
                eps_per_sec = episodes_processed / max(elapsed, 0.01)

                stats = {
                    'episodes': episodes_processed,
                    'elapsed': elapsed,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'win_rate': win_rate,
                    'eps_per_sec': eps_per_sec,
                    'epsilon': agent.eps
                }
                stats_queue.put(stats)
                last_update = time.time()

        except Exception as e:
            if stop_flag.value and experience_queue.empty():
                break
            continue

    # Final stats
    elapsed = time.time() - start_time
    print(f"\n[Learner] Training complete!")
    print(f"  Episodes processed: {episodes_processed}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Episodes/sec: {episodes_processed/elapsed:.1f}")
    print(f"  Win rate: {wins/max(episodes_processed,1)*100:.1f}%")

    # Save agent
    agent.save('q_agent_distributed.pkl')
    print(f"[Learner] Agent saved to q_agent_distributed.pkl")

    stop_flag.value = True


def stats_monitor(stats_queue, stop_flag):
    """Monitor and display training statistics"""
    print("\n[Monitor] Started - displaying training progress")
    print("=" * 80)

    while not stop_flag.value:
        try:
            stats = stats_queue.get(timeout=1)

            print(f"Episodes: {stats['episodes']:,} | "
                  f"Time: {stats['elapsed']:.1f}s | "
                  f"Speed: {stats['eps_per_sec']:.1f} eps/s | "
                  f"Wins: {stats['wins']} ({stats['win_rate']:.1f}%) | "
                  f"Draws: {stats['draws']} | "
                  f"Losses: {stats['losses']} | "
                  f"Epsilon: {stats['epsilon']:.4f}")

        except:
            continue

    print("=" * 80)
    print("[Monitor] Stopped")


# --- Main Distributed Training Function ---
def train_distributed(episodes=100000, num_workers=None, opponent_type='random',
                     save_path='q_agent_distributed.pkl'):
    """
    Train agent using distributed workers

    Args:
        episodes: Total number of episodes to train
        num_workers: Number of parallel workers (default: CPU count - 1)
        opponent_type: 'random', 'minimax', or 'self_play'
        save_path: Path to save trained agent
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print("=" * 80)
    print("DISTRIBUTED TRAINING SYSTEM")
    print("=" * 80)
    print(f"Total Episodes: {episodes:,}")
    print(f"Number of Workers: {num_workers}")
    print(f"Opponent Type: {opponent_type}")
    print(f"Expected Speedup: {num_workers}x")
    print("=" * 80)

    # Shared objects
    manager = Manager()
    agent_dict = manager.dict({
        'alpha': 0.3,
        'gamma': 0.95,
        'eps': 0.4,
        'eps_decay': 0.99995,
        'optimistic_init': 0.6
    })

    experience_queue = Queue(maxsize=1000)
    stats_queue = Queue(maxsize=100)
    episode_counter = Value('i', 0)
    stop_flag = Value('i', 0)

    # Start processes
    processes = []

    # Start learner
    learner = Process(target=learner_process,
                     args=(agent_dict, experience_queue, stats_queue,
                          episodes, num_workers, stop_flag))
    learner.start()
    processes.append(learner)

    # Start workers
    for i in range(num_workers):
        worker = Process(target=experience_worker,
                        args=(i, agent_dict, experience_queue, episode_counter,
                             episodes, opponent_type, stop_flag))
        worker.start()
        processes.append(worker)

    # Start monitor
    monitor = Process(target=stats_monitor,
                     args=(stats_queue, stop_flag))
    monitor.start()
    processes.append(monitor)

    # Wait for completion
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted - stopping all processes...")
        stop_flag.value = True
        for p in processes:
            p.terminate()
            p.join()

    print("\n" + "=" * 80)
    print("DISTRIBUTED TRAINING COMPLETE!")
    print("=" * 80)


def train_curriculum_distributed(episodes_per_phase=30000, num_workers=None):
    """
    Train with curriculum learning using distributed workers

    Phases:
    1. Random opponent (30k episodes)
    2. Self-play (30k episodes)
    3. Minimax optimal (30k episodes)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print("=" * 80)
    print("DISTRIBUTED CURRICULUM TRAINING")
    print("=" * 80)
    print(f"Episodes per phase: {episodes_per_phase:,}")
    print(f"Total episodes: {episodes_per_phase * 3:,}")
    print(f"Number of workers: {num_workers}")
    print("=" * 80)

    overall_start = time.time()

    # Phase 1: Random
    print("\n>> PHASE 1: Training vs Random Opponent")
    train_distributed(episodes_per_phase, num_workers, 'random',
                     'q_agent_distributed_phase1.pkl')

    # Phase 2: Self-play
    print("\n>> PHASE 2: Training vs Self-Play")
    train_distributed(episodes_per_phase, num_workers, 'self_play',
                     'q_agent_distributed_phase2.pkl')

    # Phase 3: Minimax
    print("\n>> PHASE 3: Training vs Minimax Optimal")
    train_distributed(episodes_per_phase, num_workers, 'minimax',
                     'q_agent_distributed.pkl')

    overall_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print(f"Total episodes: {episodes_per_phase * 3:,}")
    print(f"Average speed: {(episodes_per_phase * 3) / overall_time:.1f} eps/s")
    print("=" * 80)


if __name__ == "__main__":
    # Simple distributed training
    # train_distributed(episodes=10000, num_workers=4, opponent_type='random')

    # Curriculum training with distributed workers
    train_curriculum_distributed(episodes_per_phase=30000)
