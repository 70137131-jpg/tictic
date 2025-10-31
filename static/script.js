
document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const statusElement = document.getElementById('status');
    const resetButton = document.getElementById('reset');
    const difficultyButtons = document.querySelectorAll('.difficulty-btn');
    let gameActive = true;
    let agentThinking = false;
    let currentDifficulty = 'easy';  // Default difficulty
    let moveHistory = [];  // Track all moves [{player: 'X'/'O', pos: [i,j], board: [...]}]
    let replayMode = false;
    let replayIndex = 0;

    function readBoard() {
        const board = [];
        const rows = boardElement.querySelectorAll('tr');
        rows.forEach(tr => {
            const row = [];
            tr.querySelectorAll('td').forEach(td => row.push(td.textContent.trim()));
            board.push(row);
        });
        return board;
    }

    function copyBoard(board) {
        return board.map(row => [...row]);
    }

    function writeBoard(board) {
        const tds = boardElement.querySelectorAll('td');
        tds.forEach(td => {
            const [r, c] = td.dataset.pos.split(',').map(Number);
            const symbol = board[r][c];
            td.textContent = symbol;
            td.className = symbol !== '-' ? symbol : '';
            if (!gameActive) {
                td.classList.add('disabled');
            }
        });
    }

    function checkWinner(board) {
        const lines = [
            [[0,0],[0,1],[0,2]], [[1,0],[1,1],[1,2]], [[2,0],[2,1],[2,2]], // rows
            [[0,0],[1,0],[2,0]], [[0,1],[1,1],[2,1]], [[0,2],[1,2],[2,2]], // cols
            [[0,0],[1,1],[2,2]], [[0,2],[1,1],[2,0]]  // diagonals
        ];

        for (const line of lines) {
            const [[r1,c1],[r2,c2],[r3,c3]] = line;
            const v1 = board[r1][c1], v2 = board[r2][c2], v3 = board[r3][c3];
            if (v1 !== '-' && v1 === v2 && v2 === v3) {
                return { winner: v1, line };
            }
        }

        // Check for draw
        const isDraw = board.every(row => row.every(cell => cell !== '-'));
        if (isDraw) {
            return { winner: 'Draw', line: null };
        }

        return null;
    }

    function highlightWinningLine(line) {
        if (!line) return;
        line.forEach(([r, c]) => {
            const cell = boardElement.querySelector(`td[data-pos="${r},${c}"]`);
            cell.classList.add('winning-cell');
        });
    }

    function createConfetti() {
        const colors = ['#FF6B6B', '#4ECDC4', '#FFE951', '#95E1D3', '#F38181'];
        const confettiCount = 100;

        for (let i = 0; i < confettiCount; i++) {
            setTimeout(() => {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.style.left = Math.random() * 100 + '%';
                confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
                confetti.style.opacity = Math.random();
                confetti.style.transform = `rotate(${Math.random() * 360}deg)`;
                document.body.appendChild(confetti);

                setTimeout(() => confetti.remove(), 5000);
            }, i * 30);
        }
    }

    function endGame(winner, line) {
        gameActive = false;
        agentThinking = false;

        const tds = boardElement.querySelectorAll('td');
        tds.forEach(td => td.classList.add('disabled'));

        if (winner === 'Draw') {
            statusElement.textContent = "It's a Draw! 🤝";
            statusElement.className = 'winner';
        } else {
            highlightWinningLine(line);
            if (winner === 'X') {
                statusElement.textContent = '🎉 YOU WIN! 🎉';
                createConfetti();
            } else {
                statusElement.textContent = '🤖 AGENT WINS! 🤖';
                createConfetti();
            }
            statusElement.className = 'winner';
        }
    }

    async function handleCellClick(event) {
        if (!gameActive || agentThinking || replayMode) return;

        const cell = event.target;
        if (cell.textContent.trim() === '-') {
            const [r, c] = cell.dataset.pos.split(',').map(Number);
            cell.textContent = 'X';
            cell.className = 'X';

            let board = readBoard();

            // Track player move
            moveHistory.push({
                player: 'X',
                pos: [r, c],
                board: copyBoard(board)
            });
            updateMoveHistoryUI();

            let result = checkWinner(board);
            if (result) {
                endGame(result.winner, result.line);
                return;
            }

            // Automatically trigger agent move after player move
            await handleAgentMove();
        }
    }

    async function handleAgentMove() {
        if (!gameActive || agentThinking) return;

        agentThinking = true;
        const board = readBoard();
        statusElement.textContent = '🤔 Agent is thinking...';
        statusElement.className = '';

        // Add a small delay to make it feel more natural
        await new Promise(resolve => setTimeout(resolve, 500));

        try {
            const res = await fetch('/api/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ board, difficulty: currentDifficulty })
            });
            const data = await res.json();
            if (data.error) {
                statusElement.textContent = `Error: ${data.error}`;
                agentThinking = false;
                return;
            }
            const [i, j] = data.move;
            board[i][j] = 'O';
            writeBoard(board);

            // Track agent move
            moveHistory.push({
                player: 'O',
                pos: [i, j],
                board: copyBoard(board)
            });
            updateMoveHistoryUI();

            const result = checkWinner(board);
            if (result) {
                endGame(result.winner, result.line);
            } else {
                statusElement.textContent = `Agent placed O at (${i}, ${j}). Your turn!`;
                agentThinking = false;
            }
        } catch (err) {
            statusElement.textContent = '❌ Request failed!';
            console.error(err);
            agentThinking = false;
        }
    }

    function resetGame() {
        gameActive = true;
        agentThinking = false;
        replayMode = false;
        replayIndex = 0;
        moveHistory = [];
        statusElement.className = '';
        writeBoard([['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]);
        statusElement.textContent = '👉 Click a cell to place your X!';

        const tds = boardElement.querySelectorAll('td');
        tds.forEach(td => {
            td.classList.remove('disabled', 'winning-cell');
        });

        updateMoveHistoryUI();
    }

    function updateMoveHistoryUI() {
        const historyList = document.getElementById('move-history-list');
        if (!historyList) return;

        historyList.innerHTML = '';
        moveHistory.forEach((move, idx) => {
            const li = document.createElement('li');
            li.textContent = `Move ${idx + 1}: ${move.player} → (${move.pos[0]}, ${move.pos[1]})`;
            li.classList.add('history-item');
            if (replayMode && idx === replayIndex - 1) {
                li.classList.add('current');
            }
            li.addEventListener('click', () => {
                enterReplayMode(idx + 1);
            });
            historyList.appendChild(li);
        });
    }

    function enterReplayMode(moveIdx) {
        if (moveIdx < 0 || moveIdx > moveHistory.length) return;

        replayMode = true;
        replayIndex = moveIdx;
        gameActive = false;

        if (moveIdx === 0) {
            writeBoard([['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]);
        } else {
            writeBoard(moveHistory[moveIdx - 1].board);
        }

        statusElement.textContent = `📼 Replay Mode - Move ${moveIdx}/${moveHistory.length}`;
        updateMoveHistoryUI();
    }

    function exitReplayMode() {
        if (!replayMode) return;
        replayMode = false;
        gameActive = true;

        // Restore to latest position
        if (moveHistory.length > 0) {
            writeBoard(moveHistory[moveHistory.length - 1].board);
        }

        // Check if game was already over
        const result = checkWinner(readBoard());
        if (result) {
            gameActive = false;
        }

        updateMoveHistoryUI();
    }

    function replayPrevious() {
        if (replayIndex > 0) {
            enterReplayMode(replayIndex - 1);
        }
    }

    function replayNext() {
        if (replayIndex < moveHistory.length) {
            enterReplayMode(replayIndex + 1);
        } else {
            exitReplayMode();
        }
    }

    // Difficulty button event listeners
    difficultyButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            difficultyButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');
            // Update current difficulty
            currentDifficulty = btn.dataset.difficulty;
            // Reset game when difficulty changes
            resetGame();
        });
    });

    boardElement.querySelectorAll('td').forEach(td => {
        td.addEventListener('click', handleCellClick);
    });

    resetButton.addEventListener('click', resetGame);

    // Replay control buttons
    const replayPrevBtn = document.getElementById('replay-prev');
    const replayNextBtn = document.getElementById('replay-next');
    const replayExitBtn = document.getElementById('replay-exit');

    if (replayPrevBtn) replayPrevBtn.addEventListener('click', replayPrevious);
    if (replayNextBtn) replayNextBtn.addEventListener('click', replayNext);
    if (replayExitBtn) replayExitBtn.addEventListener('click', exitReplayMode);

    // Initial setup
    resetGame();
});
