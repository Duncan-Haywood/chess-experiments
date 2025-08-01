// Chess game state
let game = new Chess();
let board = null;
let currentPosition = 0;
let gameHistory = [];

// Initialize the board
function initBoard() {
    const config = {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    
    board = Chessboard('board', config);
    updateStatus();
}

// Check if a move is legal
function onDragStart(source, piece, position, orientation) {
    // Don't allow moves if game is over
    if (game.game_over()) return false;
    
    // Only pick up pieces for the side to move
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

// Handle piece drop
function onDrop(source, target) {
    // See if the move is legal
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Always promote to queen for simplicity
    });
    
    // Illegal move
    if (move === null) return 'snapback';
    
    // Update game history
    gameHistory = game.history();
    currentPosition = gameHistory.length;
    
    updateStatus();
    updateMoveList();
}

// Update board position after move
function onSnapEnd() {
    board.position(game.fen());
}

// Update game status
function updateStatus() {
    let status = '';
    
    const moveColor = game.turn() === 'w' ? 'White' : 'Black';
    
    // Checkmate?
    if (game.in_checkmate()) {
        status = `Game over, ${moveColor} is in checkmate.`;
    }
    // Draw?
    else if (game.in_draw()) {
        status = 'Game over, drawn position';
    }
    // Game still on
    else {
        status = `${moveColor} to move`;
        
        // Check?
        if (game.in_check()) {
            status += ', in check';
        }
    }
    
    $('#status').text(status);
    $('#fen-display').text(game.fen());
}

// Navigation functions
function goToStart() {
    game.reset();
    currentPosition = 0;
    board.position(game.fen());
    updateStatus();
    updateMoveList();
}

function goToPrevious() {
    if (currentPosition > 0) {
        currentPosition--;
        const newGame = new Chess();
        for (let i = 0; i < currentPosition; i++) {
            newGame.move(gameHistory[i]);
        }
        game = newGame;
        board.position(game.fen());
        updateStatus();
        updateMoveList();
    }
}

function goToNext() {
    if (currentPosition < gameHistory.length) {
        game.move(gameHistory[currentPosition]);
        currentPosition++;
        board.position(game.fen());
        updateStatus();
        updateMoveList();
    }
}

function goToEnd() {
    while (currentPosition < gameHistory.length) {
        game.move(gameHistory[currentPosition]);
        currentPosition++;
    }
    board.position(game.fen());
    updateStatus();
    updateMoveList();
}

function flipBoard() {
    board.flip();
}

// Load PGN
function loadPGN() {
    const pgn = $('#pgn-input').val();
    
    $.ajax({
        url: '/api/game/load',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ pgn: pgn }),
        success: function(data) {
            game.load_pgn(pgn);
            gameHistory = game.history();
            currentPosition = gameHistory.length;
            board.position(game.fen());
            updateStatus();
            updateMoveList();
            $('#status').text('Game loaded successfully');
        },
        error: function(xhr) {
            $('#status').text('Error loading PGN: ' + xhr.responseText);
        }
    });
}

// Analyze position
function analyzePosition() {
    $('#engine-status').text('Analyzing...');
    
    // Get selected engine and depth
    const engine = $('#engine-select').val() || 'minimax';
    const depth = parseInt($('#depth-select').val()) || 4;
    
    $.ajax({
        url: '/api/analyze',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 
            fen: game.fen(),
            engine: engine,
            depth: depth
        }),
        success: function(data) {
            const score = data.score;
            $('#eval-score').text(data.evaluation);
            
            // Update evaluation bar
            const percentage = Math.max(0, Math.min(100, 50 + (score * 5)));
            $('#eval-fill').css('width', percentage + '%');
            
            if (score > 0.5) {
                $('#eval-fill').css('background-color', 'white');
            } else if (score < -0.5) {
                $('#eval-fill').css('background-color', '#333');
            } else {
                $('#eval-fill').css('background-color', '#888');
            }
            
            // Show best move
            let statusText = `Analysis complete (${data.engineName}, depth ${data.depth})`;
            if (data.bestMove) {
                statusText += ` - Best move: ${data.bestMove}`;
                highlightMove(data.bestMove);
            }
            $('#engine-status').text(statusText);
        },
        error: function(xhr) {
            $('#engine-status').text('Error: ' + xhr.responseText);
        }
    });
}

// Highlight suggested move
function highlightMove(moveStr) {
    // Remove existing highlights
    $('.highlight-move').removeClass('highlight-move');
    
    // Parse move
    const move = game.move(moveStr);
    if (move) {
        // Highlight squares
        $('#board .square-' + move.from).addClass('highlight-move');
        $('#board .square-' + move.to).addClass('highlight-move');
        
        // Undo the move (we just wanted to parse it)
        game.undo();
    }
}

// Update move list display
function updateMoveList() {
    const moves = game.history({ verbose: true });
    let html = '';
    
    for (let i = 0; i < moves.length; i += 2) {
        html += '<div class="move-pair">';
        html += '<span class="move-number">' + ((i / 2) + 1) + '.</span>';
        
        // White move
        html += '<span class="move' + (i === currentPosition - 1 ? ' active' : '') + 
                '" data-index="' + i + '">' + moves[i].san + '</span>';
        
        // Black move
        if (i + 1 < moves.length) {
            html += '<span class="move' + (i + 1 === currentPosition - 1 ? ' active' : '') + 
                    '" data-index="' + (i + 1) + '">' + moves[i + 1].san + '</span>';
        }
        
        html += '</div>';
    }
    
    $('#move-list').html(html);
}

// Go to specific move
$(document).on('click', '.move', function() {
    const index = parseInt($(this).data('index'));
    
    // Reset to start
    game.reset();
    currentPosition = 0;
    
    // Play moves up to clicked position
    for (let i = 0; i <= index; i++) {
        game.move(gameHistory[i]);
        currentPosition++;
    }
    
    board.position(game.fen());
    updateStatus();
    updateMoveList();
});

// Tab switching
$('.tab').click(function() {
    $('.tab').removeClass('active');
    $(this).addClass('active');
    
    $('.tab-content').hide();
    $('#' + $(this).data('tab') + '-tab').show();
});

// Initialize when document is ready
$(document).ready(function() {
    initBoard();
    
    // Bind button events
    $('#startBtn').click(goToStart);
    $('#prevBtn').click(goToPrevious);
    $('#nextBtn').click(goToNext);
    $('#endBtn').click(goToEnd);
    $('#flipBtn').click(flipBoard);
    $('#loadPgnBtn').click(loadPGN);
    $('#analyzeBtn').click(analyzePosition);
});

// Keyboard shortcuts
$(document).keydown(function(e) {
    if (e.key === 'ArrowLeft') {
        goToPrevious();
    } else if (e.key === 'ArrowRight') {
        goToNext();
    } else if (e.key === 'Home') {
        goToStart();
    } else if (e.key === 'End') {
        goToEnd();
    }
});