# Tic-Tac-Toe Reinforcement Learning Flask Application

A Q-learning agent that learns to play Tic-Tac-Toe optimally, with a Flask web interface for playing against the trained agent.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (if not already trained):
   Open `Untitled2.ipynb` in Jupyter/VSCode and run the training cells, or run:
   ```python
   # In the notebook, execute the cell with:
   train_agent(episodes=50000, teacher_ability=0.9)
   ```
   
   This will create `q_agent.pkl` containing the trained model.

3. **Verify the trained model exists:**
   ```bash
   # Should see q_agent.pkl in the directory
   ls -la q_agent.pkl
   ```

## Running the Flask Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   Navigate to http://127.0.0.1:5000

3. **Play against the agent:**
   - Click cells to place X (your move)
   - Click "Ask agent for move" to get O (agent's move)
   - Click "Reset" to start a new game

## API Endpoints

### GET /
Returns the HTML UI for playing against the agent.

### POST /api/move
Request the agent's next move for a given board state.

**Request:**
```json
{
  "board": [
    ["-", "X", "-"],
    ["-", "-", "-"],
    ["-", "-", "-"]
  ]
}
```

**Response:**
```json
{
  "move": [1, 1]
}
```

### POST /api/evaluate
Evaluate the agent's performance over multiple games.

**Request:**
```json
{
  "games": 100,
  "opponent": "minimax",
  "teacher_ability": 0.9
}
```

**Response:**
```json
{
  "wins": 45,
  "draws": 50,
  "losses": 5
}
```

## Model Training

The agent uses Q-learning with the following parameters:
- Learning rate (alpha): 0.5
- Discount factor (gamma): 0.9
- Exploration rate (epsilon): 0.1
- Training episodes: 50,000

The trained model is saved as `q_agent.pkl` and loaded automatically by the Flask app.

## Project Structure

```
.
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Web UI
├── requirements.txt      # Python dependencies
├── Untitled2.ipynb       # Training notebook
├── q_agent.pkl          # Trained model (generated after training)
└── README.md            # This file
```
