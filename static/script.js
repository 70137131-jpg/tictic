
document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const statusElement = document.getElementById('status');
    const resetButton = document.getElementById('reset');
    let gameActive = true;
    let agentThinking = false;

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
        if (!gameActive || agentThinking) return;

        const cell = event.target;
        if (cell.textContent.trim() === '-') {
            cell.textContent = 'X';
            cell.className = 'X';

            let board = readBoard();
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
                body: JSON.stringify({ board })
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

            const result = checkWinner(board);
            if (result) {
                endGame(result.winner, result.line);
            } else {
                statusElement.textContent = `Agent moved to (${i}, ${j}) ✨ Your turn!`;
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
        statusElement.className = '';
        writeBoard([['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]);
        statusElement.textContent = '👉 Click a cell to place your X!';

        const tds = boardElement.querySelectorAll('td');
        tds.forEach(td => {
            td.classList.remove('disabled', 'winning-cell');
        });
    }

    boardElement.querySelectorAll('td').forEach(td => {
        td.addEventListener('click', handleCellClick);
    });

    resetButton.addEventListener('click', resetGame);

    // Initial setup
    resetGame();
});
