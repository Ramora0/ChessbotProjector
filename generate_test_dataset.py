#!/usr/bin/env python3
"""Generate a dataset of chess position reconstruction questions/answers."""

import chess
import random
import json
from pathlib import Path


def generate_random_position(min_moves: int = 5, max_moves: int = 80) -> chess.Board:
    """Generate a random chess position by playing random legal moves."""
    board = chess.Board()
    num_moves = random.randint(min_moves, max_moves)

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves or board.is_game_over():
            break
        move = random.choice(legal_moves)
        board.push(move)

    return board


def format_position_answer(board: chess.Board) -> str:
    """Format a chess position as a piece list with game state info."""
    piece_locations = {
        'White': {
            'King': [],
            'Queen': [],
            'Rooks': [],
            'Bishops': [],
            'Knights': [],
            'Pawns': [],
        },
        'Black': {
            'King': [],
            'Queen': [],
            'Rooks': [],
            'Bishops': [],
            'Knights': [],
            'Pawns': [],
        }
    }

    piece_name_map = {
        chess.KING: 'King',
        chess.QUEEN: 'Queen',
        chess.ROOK: 'Rooks',
        chess.BISHOP: 'Bishops',
        chess.KNIGHT: 'Knights',
        chess.PAWN: 'Pawns',
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'White' if piece.color == chess.WHITE else 'Black'
            piece_name = piece_name_map[piece.piece_type]
            square_name = chess.square_name(square)
            piece_locations[color][piece_name].append(square_name)

    lines = []

    # Piece locations
    for color in ['White', 'Black']:
        for piece_type in ['King', 'Queen', 'Rooks', 'Bishops', 'Knights', 'Pawns']:
            squares = piece_locations[color][piece_type]
            if squares:
                squares_str = ', '.join(sorted(squares))
                lines.append(f"{color} {piece_type}: {squares_str}")

    # Side to move
    side_to_move = "White" if board.turn == chess.WHITE else "Black"
    lines.append(f"Side to move: {side_to_move}")

    # Castling rights
    castling_parts = []
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_parts.append("White kingside")
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_parts.append("White queenside")
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_parts.append("Black kingside")
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_parts.append("Black queenside")

    if castling_parts:
        lines.append(f"Castling rights: {', '.join(castling_parts)}")
    else:
        lines.append("Castling rights: None")

    # En passant square
    if board.ep_square is not None:
        ep_square_name = chess.square_name(board.ep_square)
        lines.append(f"En passant square: {ep_square_name}")
    else:
        lines.append("En passant square: None")

    return '\n'.join(lines)


def generate_question() -> str:
    """Generate a question asking to reconstruct the position."""
    return "Reconstruct the chess position. List every piece and where it goes, plus the side to move, castling rights, and en passant square."


def generate_example(idx: int) -> dict:
    """Generate a single question/answer example."""
    board = generate_random_position()
    question = generate_question()
    answer = format_position_answer(board)

    return {
        'id': idx,
        'question': question,
        'answer': answer,
        'fen': board.fen(),
    }


def main():
    num_examples = 20000
    output_file = Path('position_reconstruction_dataset.json')

    print(f"Generating {num_examples} examples...")

    examples = []
    for i in range(num_examples):
        example = generate_example(i)
        examples.append(example)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_examples} examples")

    print(f"Writing to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"Done! Generated {len(examples)} examples.")

    # Print a sample example
    print("\n--- Sample Example ---")
    sample = examples[0]
    print(f"Question:\n{sample['question']}\n")
    print(f"Answer:\n{sample['answer']}")


if __name__ == '__main__':
    main()
