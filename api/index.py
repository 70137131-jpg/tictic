from flask import Flask, request, jsonify
import pickle
import os
import random
from functools import lru_cache
from typing import Tuple
import collections
import numpy as np
from collections import deque

# --- Agent Classes (required for unpickling) ---
class Learner:
    """Parent class for Q-learning agents."""
    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.actions = [(i,j) for i in range(3) for j in range(3)]
        self.Q = {}
        for action in self.actions:
            self.Q[action] = collections.defaultdict(int)
        self.rewards = []

class Qlearner(Learner):
    """Q-learning agent."""
    def __init__(self, alpha, gamma, eps, eps_decay=0.):
        super().__init__(alpha, gamma, eps, eps_decay)

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

app = Flask(__name__)

# --- Agent Loading ---
_agent_cache = None

def load_agent():
    global _agent_cache
    if _agent_cache is not None:
        return _agent_cache
    
    # Try multiple paths for Vercel
    possible_paths = [
        'q_agent.pkl',
        '../q_agent.pkl',
        os.path.join(os.path.dirname(__file__), '..', 'q_agent.pkl'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'q_agent.pkl'),
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    _agent_cache = pickle.load(f)
                print(f"✅ Agent loaded from: {path}")
                return _agent_cache
            except Exception as e:
                print(f"❌ Failed to load from {path}: {e}")
                continue
    
    # If no agent found, create a dummy one
    print("⚠️ No agent found, creating dummy agent")
    _agent_cache = Qlearner(alpha=0.5, gamma=0.9, eps=0.1)
    return _agent_cache

# --- Helper Functions ---
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
    state = getStateKey(board)
    possible = _legal_moves(board)
    
    if not possible:
        return (0, 0)  # Fallback
    
    try:
        values = []
        for action in possible:
            q_value = agent.Q[action].get(state, 0) if hasattr(agent.Q[action], 'get') else 0
            values.append(q_value)
        
        if len(set(values)) == 1:
            return random.choice(possible)
        
        max_val = max(values)
        best_idxs = [i for i,v in enumerate(values) if v == max_val]
        return possible[random.choice(best_idxs)]
    except:
        return random.choice(possible)

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
            sc,_ = _minimax_cached(''.join(board[r][c] for r in range(3) for c in range(3)), 'O')
            board[i][j] = '-'
            if sc > best_score:
                best_score, best_mv = sc, (i,j)
                if best_score == 1: break
        return best_score, best_mv
    else:
        best_score = 2; best_mv = None
        for (i,j) in moves:
            board[i][j] = 'O'
            sc,_ = _minimax_cached(''.join(board[r][c] for r in range(3) for c in range(3)), 'X')
            board[i][j] = '-'
            if sc < best_score:
                best_score, best_mv = sc, (i,j)
                if best_score == -1: break
        return best_score, best_mv

# --- Routes ---
@app.route("/")
def index():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TicTacToe AI</title>
<style>*{box-sizing:border-box}body{font-family:Arial,sans-serif;background:#FFE951;margin:0;padding:20px;display:flex;justify-content:center;align-items:center;min-height:100vh}.container{background:#FFF;padding:40px;border:6px solid #000;box-shadow:12px 12px 0 #000;max-width:550px;text-align:center}h3{font-size:36px;margin:0 0 20px}#board{border-collapse:separate;border-spacing:8px;margin:30px auto}#board td{width:100px;height:100px;font-size:48px;font-weight:700;border:5px solid #000;background:#FFF;cursor:pointer;text-align:center;box-shadow:4px 4px 0 #000}#board td:hover{transform:translate(-2px,-2px);box-shadow:6px 6px 0 #000}#board td.X{background:#FF6B6B}#board td.O{background:#4ECDC4}button{padding:15px 30px;font-size:18px;font-weight:700;border:5px solid #000;background:#4ECDC4;cursor:pointer;margin:10px;box-shadow:6px 6px 0 #000}button:hover{transform:translate(-2px,-2px);box-shadow:8px 8px 0 #000}#status{margin-top:20px;padding:20px;font-size:18px;border:5px solid #000;background:#FFF;box-shadow:6px 6px 0 #000;font-weight:700}</style>
</head><body><div class="container"><h3>Tic-Tac-Toe AI</h3><p>You are <strong>X</strong>, AI is <strong>O</strong></p>
<table id="board"><tr><td data-pos="0,0">-</td><td data-pos="0,1">-</td><td data-pos="0,2">-</td></tr>
<tr><td data-pos="1,0">-</td><td data-pos="1,1">-</td><td data-pos="1,2">-</td></tr>
<tr><td data-pos="2,0">-</td><td data-pos="2,1">-</td><td data-pos="2,2">-</td></tr></table>
<button id="reset">Reset Game</button><div id="status">Click a cell to start!</div></div>
<script>const board=document.getElementById('board'),status=document.getElementById('status');let gameActive=!0;function readBoard(){const b=[];return document.querySelectorAll('#board tr').forEach(a=>{const c=[];a.querySelectorAll('td').forEach(a=>c.push(a.textContent.trim())),b.push(c)}),b}function writeBoard(b){document.querySelectorAll('#board td').forEach(a=>{const[c,d]=a.dataset.pos.split(',').map(Number);a.textContent=b[c][d],a.className='-'!==b[c][d]?b[c][d]:''})}function checkWinner(b){const a=[[[0,0],[0,1],[0,2]],[[1,0],[1,1],[1,2]],[[2,0],[2,1],[2,2]],[[0,0],[1,0],[2,0]],[[0,1],[1,1],[2,1]],[[0,2],[1,2],[2,2]],[[0,0],[1,1],[2,2]],[[0,2],[1,1],[2,0]]];for(const c of a){const[d,e,f]=c,g=b[d[0]][d[1]],h=b[e[0]][e[1]],i=b[f[0]][f[1]];if('-'!==g&&g===h&&h===i)return g}return b.every(a=>a.every(a=>'-'!==a))?'Draw':null}async function handleMove(a,b){if(!gameActive)return;const c=readBoard();if('-'!==c[a][b])return;c[a][b]='X',writeBoard(c);let d=checkWinner(c);if(d)return void endGame(d);status.textContent='🤖 AI thinking...';try{const a=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({board:c,difficulty:'hard'})}),b=await a.json(),[e,f]=b.move;c[e][f]='O',writeBoard(c),d=checkWinner(c),d?endGame(d):status.textContent='AI placed O at ('+e+','+f+')'}catch(a){status.textContent='❌ Error: '+a.message}}function endGame(a){gameActive=!1,'X'===a?status.textContent='🎉 You Win!':'O'===a?status.textContent='🤖 AI Wins!':status.textContent="Draw!"}function reset(){gameActive=!0,writeBoard([['-','-','-'],['-','-','-'],['-','-','-']]),status.textContent='Click a cell to start!'}board.querySelectorAll('td').forEach(a=>{a.addEventListener('click',()=>{const[b,c]=a.dataset.pos.split(',').map(Number);handleMove(b,c)})}),document.getElementById('reset').addEventListener('click',reset);</script>
</body></html>"""

@app.route("/api/move", methods=["POST"])
def api_move():
    try:
        data = request.get_json()
        board = data.get("board")
        difficulty = data.get("difficulty", "hard")
        
        if board is None:
            return jsonify({"error": "no board provided"}), 400
        
        agent = load_agent()
        
        # Difficulty levels
        if difficulty == "easy" and random.random() < 0.4:
            mv = random.choice(_legal_moves(board))
        elif difficulty == "medium" and random.random() < 0.15:
            mv = random.choice(_legal_moves(board))
        else:
            mv = _agent_greedy_move(agent, board)
        
        return jsonify({"move": [mv[0], mv[1]]})
    
    except Exception as e:
        print(f"Error in api_move: {e}")
        # Return random move as fallback
        try:
            moves = _legal_moves(board)
            mv = random.choice(moves) if moves else (0, 0)
            return jsonify({"move": [mv[0], mv[1]]})
        except:
            return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    try:
        payload = request.get_json() or {}
        games = int(payload.get("games", 100))
        
        agent = load_agent()
        wins = draws = losses = 0
        
        for _ in range(games):
            board = [['-']*3 for _ in range(3)]
            turn = 'X'
            while True:
                if turn == 'X':
                    moves = _legal_moves(board)
                    i, j = random.choice(moves) if moves else (0, 0)
                    board[i][j] = 'X'
                else:
                    i, j = _agent_greedy_move(agent, board)
                    board[i][j] = 'O'
                
                term = _winner(board)
                if term is not None:
                    if term == 'X': losses += 1
                    elif term == 'O': wins += 1
                    else: draws += 1
                    break
                turn = 'O' if turn == 'X' else 'X'
        
        return jsonify({"wins": wins, "draws": draws, "losses": losses})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel handler
app = app