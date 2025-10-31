from flask import Flask, request, jsonify, render_template
import pickle, os, random
from functools import lru_cache
from typing import Tuple
from abc import ABC, abstractmethod
import collections
import numpy as np

# --- Agent Classes (required for unpickling) ---
from collections import deque

class Learner(ABC):
    """Parent class for Q-learning and SARSA agents."""
    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.actions = []
        for i in range(3):
            for j in range(3):
                self.actions.append((i,j))
        self.Q = {}
        for action in self.actions:
            self.Q[action] = collections.defaultdict(int)
        self.rewards = []

    def get_action(self, s):
        possible_actions = [a for a in self.actions if s[a[0]*3 + a[1]] == '-']
        if random.random() < self.eps:
            action = possible_actions[random.randint(0,len(possible_actions)-1)]
        else:
            values = np.array([self.Q[a][s] for a in possible_actions])
            ix_max = np.where(values == np.max(values))[0]
            if len(ix_max) > 1:
                ix_select = np.random.choice(ix_max, 1)[0]
            else:
                ix_select = ix_max[0]
            action = possible_actions[ix_select]
        self.eps *= (1.-self.eps_decay)
        return action

    @abstractmethod
    def update(self, s, s_, a, a_, r):
        pass

class Qlearner(Learner):
    """Q-learning agent."""
    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        super().__init__(alpha, gamma, eps, eps_decay)

    def update(self, s, s_, a, a_, r):
        if s_ is not None:
            possible_actions = [action for action in self.actions if s_[action[0]*3 + action[1]] == '-']
            Q_options = [self.Q[action][s_] for action in possible_actions]
            self.Q[a][s] += self.alpha*(r + self.gamma*max(Q_options) - self.Q[a][s])
        else:
            self.Q[a][s] += self.alpha*(r - self.Q[a][s])
        self.rewards.append(r)

# Enhanced Q-Learner class (for new model compatibility)
class EnhancedQlearner:
    """Enhanced Q-learning agent with modern improvements"""
    def __init__(self, alpha=0.3, gamma=0.95, eps=0.3, eps_decay=0.9999,
                 optimistic_init=0.5, replay_buffer_size=10000):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = 0.01
        self.eps_decay = eps_decay
        self.optimistic_init = optimistic_init
        self.actions = [(i, j) for i in range(3) for j in range(3)]
        self.Q = {}
        for action in self.actions:
            self.Q[action] = collections.defaultdict(float)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.rewards = []
        self.episode_count = 0

app = Flask(__name__, template_folder="templates")

# Load agent (pickled Qlearner) lazily
_agent_cache = None
_agent_load_time = None

def load_agent(path="q_agent.pkl"):
    global _agent_cache, _agent_load_time

    # Check if file has been modified since last load
    # Resolve path relative to this file for serverless environments (e.g., Vercel)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = path if os.path.isabs(path) else os.path.join(base_dir, path)

    if os.path.isfile(model_path):
        file_mtime = os.path.getmtime(model_path)
        if _agent_load_time is None or file_mtime > _agent_load_time:
            print(f"Loading agent from {model_path}...")
            try:
                with open(model_path, "rb") as f:
                    _agent_cache = pickle.load(f)
                _agent_load_time = file_mtime
                print(f"Agent loaded successfully! Type: {type(_agent_cache).__name__}")
            except Exception as e:
                print(f"Error loading agent: {e}")
                _agent_cache = None
                raise

    if _agent_cache is None:
        raise FileNotFoundError(f"Agent not found or failed to load from {model_path}")

    return _agent_cache

# --- Helpers (adapted, minimal) ---
def getStateKey(board):
    return ''.join(board[r][c] for r in range(3) for c in range(3))

def _legal_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == '-']

def _winner(board):
    lines = [
        [(0,0),(0,1),(0,2)],[(1,0),(1,1),(1,2)],[(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)],[(0,1),(1,1),(2,1)],[(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)],[(0,2),(1,1),(2,0)]
    ]
    for line in lines:
        a,b,c = line
        v1,v2,v3 = board[a[0]][a[1]], board[b[0]][b[1]], board[c[0]][c[1]]
        if v1 != '-' and v1 == v2 == v3:
            return v1
    if any(board[i][j] == '-' for i in range(3) for j in range(3)):
        return None
    return 'D'

def _agent_greedy_move(agent, board) -> Tuple[int,int]:
    # Check if agent is DQN (has get_action method)
    if hasattr(agent, 'get_action'):
        # DQN agent
        return agent.get_action(board, greedy=True)

    # Tabular Q-learning agent
    state = getStateKey(board)
    possible = _legal_moves(board)

    # Handle case where agent hasn't seen this state before
    # Get Q-values for each possible action, default to 0 if not seen
    values = []
    for action in possible:
        if hasattr(agent.Q[action], '__getitem__'):
            # Q is a dict-like object (defaultdict)
            q_value = agent.Q[action].get(state, 0)
        else:
            # Q is an object with get method
            q_value = agent.Q[action][state] if state in agent.Q[action] else 0
        values.append(q_value)

    # If all values are the same (likely all 0), choose randomly
    if len(set(values)) == 1:
        return random.choice(possible)

    max_val = max(values)
    best_idxs = [i for i,v in enumerate(values) if v == max_val]
    choice_idx = random.choice(best_idxs)
    return possible[choice_idx]

# --- Cached minimax (for opponent) ---
def _flatten(board):
    return ''.join(board[r][c] for r in range(3) for c in range(3))

@lru_cache(maxsize=None)
def _minimax_cached(flat_board, player):
    board = [[flat_board[r*3+c] for c in range(3)] for r in range(3)]
    term = _winner(board)
    if term == 'X': return 1, None
    if term == 'O': return -1, None
    if term == 'D': return 0, None
    moves = _legal_moves(board)
    if player == 'X':
        best_score = -2; best_mv = None
        for (i,j) in moves:
            board[i][j] = 'X'
            sc,_ = _minimax_cached(_flatten(board), 'O')
            board[i][j] = '-'
            if sc > best_score:
                best_score, best_mv = sc, (i,j)
                if best_score == 1: break
        return best_score, best_mv
    else:
        best_score = 2; best_mv = None
        for (i,j) in moves:
            board[i][j] = 'O'
            sc,_ = _minimax_cached(_flatten(board), 'X')
            board[i][j] = '-'
            if sc < best_score:
                best_score, best_mv = sc, (i,j)
                if best_score == -1: break
        return best_score, best_mv

def best_move_minimax(board, key='X'):
    _, mv = _minimax_cached(_flatten(board), key)
    moves = _legal_moves(board)
    return mv if mv is not None else moves[0]

# --- Flask routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Training visualization dashboard"""
    import time
    return render_template("dashboard.html", timestamp=int(time.time()))

@app.route("/api/generate_visualizations", methods=["POST"])
def api_generate_visualizations():
    """Generate training visualizations on demand"""
    try:
        from visualize_training import generate_all_visualizations
        generate_all_visualizations('q_agent.pkl')
        return jsonify({"status": "success", "message": "Visualizations generated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/move", methods=["POST"])
def api_move():
    """
    Request JSON: {"board": [[...], [...], [...]]},"difficulty": "easy|medium|hard"}
    Response JSON: {"move": [i, j]}
    """
    data = request.get_json()
    board = data.get("board")
    difficulty = data.get("difficulty", "hard")  # Default to hard

    if board is None:
        return jsonify({"error":"no board provided"}), 400

    agent = load_agent()

    try:
        # Difficulty levels control randomness
        if difficulty == "easy":
            # 40% chance of random move (weak play)
            if random.random() < 0.4:
                mv = random.choice(_legal_moves(board))
            else:
                mv = _agent_greedy_move(agent, board)
        elif difficulty == "medium":
            # 15% chance of random move (moderate play)
            if random.random() < 0.15:
                mv = random.choice(_legal_moves(board))
            else:
                mv = _agent_greedy_move(agent, board)
        else:  # hard
            # Always optimal (perfect play)
            mv = _agent_greedy_move(agent, board)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"move": [mv[0], mv[1]]})

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Request JSON: {"games":100, "opponent":"minimax"|"random"|"teacher", "teacher_ability":0.9}
    Returns {"wins":int,"draws":int,"losses":int}
    """
    payload = request.get_json() or {}
    games = int(payload.get("games", 100))
    opponent = payload.get("opponent", "minimax")
    teacher_ability = float(payload.get("teacher_ability", 0.9))

    agent = load_agent()

    # build opponent move function
    if opponent == "random":
        def opp_move(board): return random.choice(_legal_moves(board))
    elif opponent == "minimax":
        def opp_move(board): return best_move_minimax(board, key='X')
    else:
        # fallback to random for 'teacher' (teacher code not included here)
        def opp_move(board): return random.choice(_legal_moves(board))

    wins = draws = losses = 0
    for _ in range(games):
        board = [['-']*3 for _ in range(3)]
        turn = 'X'
        while True:
            if turn == 'X':
                i,j = opp_move(board)
                board[i][j] = 'X'
            else:
                i,j = _agent_greedy_move(agent, board)
                board[i][j] = 'O'
            term = _winner(board)
            if term is not None:
                if term == 'X': losses += 1
                elif term == 'O': wins += 1
                else: draws += 1
                break
            turn = 'O' if turn == 'X' else 'X'
    return jsonify({"wins": wins, "draws": draws, "losses": losses})

if __name__ == "__main__":
    # default dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
