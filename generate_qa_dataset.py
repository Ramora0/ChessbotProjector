"""
Generate a balanced Q&A dataset from chess positions.

Creates question-answer pairs with natural language answers across categories.
Includes model-based questions for best move, winrate, and move evaluation.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from question_phraser import (
    QuestionPhraser,
    get_piece_name,
    get_color_name,
    format_piece,
    format_piece_list,
    format_square_list,
    format_material_diff,
    format_material_count,
)
from policy_index import policy_index
from tokenizer import process_fen, create_tokenizer


CATEGORIES = [
    "imbalance",
    "check",
    "attacked_defended",
    "hanging",
    "pinned",
    "reconstruction",
    "legal_moves",
    "best_move",
    "position_winrate",
    "move_winrate",
]

# Build move lookup table
MOVE_TO_IDX = {move: idx for idx, move in enumerate(policy_index)}
NUM_VALUE_BINS = 128


def describe_position(winrate: float) -> str:
    """Convert winrate to descriptive term relative to side to move."""
    if winrate >= 0.90:
        return "winning"
    elif winrate >= 0.75:
        return "much better"
    elif winrate >= 0.58:
        return "better"
    elif winrate >= 0.52:
        return "slightly better"
    elif winrate >= 0.48:
        return "equal"
    elif winrate >= 0.42:
        return "slightly worse"
    elif winrate >= 0.25:
        return "worse"
    elif winrate >= 0.10:
        return "much worse"
    else:
        return "losing"


def describe_move_quality(regret: float) -> str:
    """Convert regret (best_winrate - move_winrate) to move quality description."""
    if regret <= 0.01:
        return "best"
    elif regret <= 0.03:
        return "good"
    elif regret <= 0.06:
        return "okay"
    elif regret <= 0.10:
        return "inaccuracy"
    elif regret <= 0.15:
        return "mistake"
    else:
        return "blunder"


@dataclass
class QACollector:
    """Collects Q&A pairs ensuring balanced distribution across categories and scenarios."""

    target_per_category: int
    categories: list[str]
    # Track by category -> scenario -> list of QA pairs
    # Scenarios help balance within category (e.g., "in_check" vs "not_in_check")
    data: dict[str, dict[str, list[dict]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    # Track total per category
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Track current position to enforce one question per position
    _current_fen: str | None = field(default=None, repr=False)
    _current_fen_used: bool = field(default=False, repr=False)

    def start_position(self, fen: str) -> None:
        """Mark the start of processing a new position."""
        self._current_fen = fen
        self._current_fen_used = False

    def add(self, category: str, fen: str, question: str, answer: str, scenario: str = "default") -> bool:
        """Add a Q&A pair if we still need examples for this category (one per position)."""
        if self.counts[category] >= self.target_per_category:
            return False
        if self._current_fen_used:
            return False

        self.data[category][scenario].append({
            "fen": fen,
            "question": question,
            "answer": answer,
            "category": category,
        })
        self.counts[category] += 1
        self._current_fen_used = True
        return True

    def needs(self, category: str) -> bool:
        """Check if we still need examples for this category."""
        return self.counts[category] < self.target_per_category

    def needs_scenario(self, category: str, scenario: str, target: int) -> bool:
        """Check if we need more of a specific scenario."""
        return len(self.data[category][scenario]) < target

    def scenario_count(self, category: str, scenario: str) -> int:
        """Get count for a specific scenario."""
        return len(self.data[category][scenario])

    def is_complete(self) -> bool:
        """Check if all categories have enough examples."""
        return all(self.counts[cat] >= self.target_per_category for cat in self.categories)

    def get_all(self) -> list[dict]:
        """Get all collected Q&A pairs."""
        result = []
        for cat in self.categories:
            for scenario_data in self.data[cat].values():
                result.extend(scenario_data)
        return result

    def status(self) -> dict[str, dict]:
        """Get current collection status."""
        return {
            cat: {
                "total": self.counts[cat],
                "scenarios": {s: len(v) for s, v in self.data[cat].items()}
            }
            for cat in self.categories
        }

    def total_count(self) -> int:
        """Get total number of collected examples."""
        return sum(self.counts.values())


# =============================================================================
# Answer Generators - extract facts and format natural language answers
# =============================================================================

def get_material_counts(board: chess.Board) -> tuple[dict[int, int], dict[int, int]]:
    """Get piece counts for each side (excluding kings)."""
    white = defaultdict(int)
    black = defaultdict(int)

    for piece in board.piece_map().values():
        if piece.piece_type != chess.KING:
            if piece.color == chess.WHITE:
                white[piece.piece_type] += 1
            else:
                black[piece.piece_type] += 1

    return dict(white), dict(black)


def generate_imbalance(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate material imbalance questions with natural answers."""
    cat = "imbalance"
    if not collector.needs(cat):
        return

    white_counts, black_counts = get_material_counts(board)

    # Determine scenario for balancing
    white_val = sum(v * {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}.get(k, 0) for k, v in white_counts.items())
    black_val = sum(v * {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}.get(k, 0) for k, v in black_counts.items())

    if white_val > black_val:
        scenario = "white_ahead"
    elif black_val > white_val:
        scenario = "black_ahead"
    else:
        scenario = "equal"

    target_per_scenario = -(-collector.target_per_category // 3)  # ceiling division

    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    # Generate different question types
    q = phraser.format_question(cat, "material_imbalance")
    a = format_material_diff(white_counts, black_counts)
    collector.add(cat, fen, q, a, scenario)

    q = phraser.format_question(cat, "who_has_more_material")
    if white_val > black_val:
        a = f"White has more material, up by {white_val - black_val} points."
    elif black_val > white_val:
        a = f"Black has more material, up by {black_val - white_val} points."
    else:
        a = "Material is equal."
    collector.add(cat, fen, q, a, scenario)

    q = phraser.format_question(cat, "count_material")
    a = format_material_count(white_counts, black_counts)
    collector.add(cat, fen, q, a, scenario)


def generate_check(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate check questions with natural answers."""
    cat = "check"
    if not collector.needs(cat):
        return

    is_in_check = board.is_check()
    scenario = "in_check" if is_in_check else "not_in_check"
    target_per_scenario = collector.target_per_category // 2

    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    turn_color = get_color_name(board.turn)
    other_color = get_color_name(not board.turn)

    if is_in_check:
        king_sq = board.king(board.turn)
        attackers = list(board.attackers(not board.turn, king_sq))

        attacker_descs = []
        for sq in attackers:
            piece = board.piece_at(sq)
            attacker_descs.append(format_piece(piece.piece_type, piece.color, chess.square_name(sq)))

        if len(attacker_descs) == 1:
            check_desc = f"Yes, {attacker_descs[0]} is giving check."
            status_desc = f"{attacker_descs[0]} is giving check."
        else:
            check_desc = f"Yes, {turn_color} is in double check from {' and '.join(attacker_descs)}."
            status_desc = f"{turn_color.capitalize()} is in double check from {' and '.join(attacker_descs)}."

        q = phraser.format_question(cat, "is_in_check", color=turn_color)
        collector.add(cat, fen, q, check_desc, scenario)

        q = phraser.format_question(cat, "check_status")
        a = f"{turn_color.capitalize()} is in check. {status_desc}"
        collector.add(cat, fen, q, a, scenario)
    else:
        q = phraser.format_question(cat, "is_in_check", color=turn_color)
        a = f"No, {turn_color} is not in check."
        collector.add(cat, fen, q, a, scenario)

        q = phraser.format_question(cat, "check_status")
        a = "Neither side is in check."
        collector.add(cat, fen, q, a, scenario)


def generate_attacked_defended(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate attacked/defended questions with natural answers."""
    cat = "attacked_defended"
    if not collector.needs(cat):
        return

    pieces = list(board.piece_map().items())
    if not pieces:
        return

    # Sample a piece to ask about
    sq, piece = random.choice(pieces)
    sq_name = chess.square_name(sq)
    p_name = get_piece_name(piece.piece_type)

    attackers = list(board.attackers(not piece.color, sq))
    defenders = list(board.attackers(piece.color, sq))

    # Format attacker list
    attacker_pieces = []
    for atk_sq in attackers:
        atk_piece = board.piece_at(atk_sq)
        attacker_pieces.append((atk_piece.piece_type, atk_piece.color, chess.square_name(atk_sq)))

    # Format defender list
    defender_pieces = []
    for def_sq in defenders:
        def_piece = board.piece_at(def_sq)
        defender_pieces.append((def_piece.piece_type, def_piece.color, chess.square_name(def_sq)))

    # Determine safety and scenario
    is_safe = len(attackers) == 0 or len(defenders) >= len(attackers)
    scenario = "safe" if is_safe else "unsafe"
    target_per_scenario = collector.target_per_category // 2

    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    # Piece safety question
    q = phraser.format_question(cat, "piece_safety", piece=p_name, square=sq_name)
    if not attackers:
        a = f"Yes, the {p_name} on {sq_name} is not attacked."
        if defenders:
            a += f" It is defended by {format_piece_list(defender_pieces)}."
    elif not defenders:
        a = f"No, the {p_name} on {sq_name} is attacked by {format_piece_list(attacker_pieces)} but has no defenders."
    elif len(defenders) >= len(attackers):
        a = f"Yes, the {p_name} on {sq_name} has {len(defenders)} defender(s) and {len(attackers)} attacker(s). It is attacked by {format_piece_list(attacker_pieces)} and defended by {format_piece_list(defender_pieces)}."
    else:
        a = f"No, the {p_name} on {sq_name} has only {len(defenders)} defender(s) against {len(attackers)} attacker(s). It is attacked by {format_piece_list(attacker_pieces)} and defended by {format_piece_list(defender_pieces)}."
    collector.add(cat, fen, q, a, scenario)

    # What attacks square question
    q = phraser.format_question(cat, "what_attacks_square", square=sq_name)
    if attackers:
        a = f"{sq_name} is attacked by {format_piece_list(attacker_pieces)}."
    else:
        a = f"Nothing is attacking {sq_name}."
    collector.add(cat, fen, q, a, scenario)

    # What defends square question
    q = phraser.format_question(cat, "what_defends_square", square=sq_name)
    if defenders:
        a = f"{sq_name} is defended by {format_piece_list(defender_pieces)}."
    else:
        a = f"Nothing is defending {sq_name}."
    collector.add(cat, fen, q, a, scenario)


def generate_hanging(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate hanging piece questions with natural answers."""
    cat = "hanging"
    if not collector.needs(cat):
        return

    # Find all hanging pieces
    hanging_pieces = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not piece.color, sq)
        defenders = board.attackers(piece.color, sq)
        if attackers and not defenders:
            hanging_pieces.append((piece.piece_type, piece.color, chess.square_name(sq)))

    has_hanging = len(hanging_pieces) > 0
    scenario = "has_hanging" if has_hanging else "no_hanging"
    target_per_scenario = collector.target_per_category // 2

    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    # Any hanging pieces question
    q = phraser.format_question(cat, "any_hanging")
    if hanging_pieces:
        a = f"Yes, {format_piece_list(hanging_pieces)} {'is' if len(hanging_pieces) == 1 else 'are'} hanging."
    else:
        a = "No, there are no hanging pieces."
    collector.add(cat, fen, q, a, scenario)

    # Color-specific questions
    for color in [chess.WHITE, chess.BLACK]:
        color_hanging = [p for p in hanging_pieces if p[1] == color]
        color_name = get_color_name(color)

        q = phraser.format_question(cat, "hanging_for_color", color=color_name)
        if color_hanging:
            a = f"Yes, {format_piece_list(color_hanging)} {'is' if len(color_hanging) == 1 else 'are'} hanging."
        else:
            a = f"No, {color_name} has no hanging pieces."
        collector.add(cat, fen, q, a, scenario)

    # Specific piece question (if there's a piece to ask about)
    if board.piece_map():
        sq, piece = random.choice(list(board.piece_map().items()))
        if piece.piece_type != chess.KING:
            sq_name = chess.square_name(sq)
            p_name = get_piece_name(piece.piece_type)

            attackers = board.attackers(not piece.color, sq)
            defenders = board.attackers(piece.color, sq)
            is_hanging = bool(attackers) and not bool(defenders)

            q = phraser.format_question(cat, "is_piece_hanging", piece=p_name, square=sq_name)
            if is_hanging:
                a = f"Yes, the {p_name} on {sq_name} is hanging - it is attacked but not defended."
            elif attackers:
                a = f"No, the {p_name} on {sq_name} is attacked but also defended."
            else:
                a = f"No, the {p_name} on {sq_name} is not attacked."
            collector.add(cat, fen, q, a, scenario)


def generate_pinned(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate pinned piece questions with natural answers."""
    cat = "pinned"
    if not collector.needs(cat):
        return

    # Find all pinned pieces
    pinned_pieces = []
    pin_info = {}  # sq -> pinner info

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        if board.is_pinned(piece.color, sq):
            pinned_pieces.append((piece.piece_type, piece.color, chess.square_name(sq)))

            # Find the pinner
            pin_mask = board.pin(piece.color, sq)
            for pin_sq in chess.SQUARES:
                if pin_mask & chess.BB_SQUARES[pin_sq]:
                    pinner = board.piece_at(pin_sq)
                    if pinner and pinner.color != piece.color:
                        pin_info[sq] = (pinner.piece_type, pinner.color, chess.square_name(pin_sq))
                        break

    has_pins = len(pinned_pieces) > 0
    scenario = "has_pins" if has_pins else "no_pins"
    target_per_scenario = collector.target_per_category // 2

    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    # Any pins question
    q = phraser.format_question(cat, "any_pins")
    if pinned_pieces:
        a = f"Yes, {format_piece_list(pinned_pieces)} {'is' if len(pinned_pieces) == 1 else 'are'} pinned."
    else:
        a = "No, there are no pinned pieces."
    collector.add(cat, fen, q, a, scenario)

    # Color-specific questions
    for color in [chess.WHITE, chess.BLACK]:
        color_pinned = [p for p in pinned_pieces if p[1] == color]
        color_name = get_color_name(color)

        q = phraser.format_question(cat, "pins_for_color", color=color_name)
        if color_pinned:
            a = f"Yes, {format_piece_list(color_pinned)} {'is' if len(color_pinned) == 1 else 'are'} pinned."
        else:
            a = f"No, {color_name} has no pinned pieces."
        collector.add(cat, fen, q, a, scenario)

    # Specific piece question
    non_king_pieces = [(sq, p) for sq, p in board.piece_map().items() if p.piece_type != chess.KING]
    if non_king_pieces:
        sq, piece = random.choice(non_king_pieces)
        sq_name = chess.square_name(sq)
        p_name = get_piece_name(piece.piece_type)
        is_pinned = board.is_pinned(piece.color, sq)

        q = phraser.format_question(cat, "is_piece_pinned", piece=p_name, square=sq_name)
        if is_pinned and sq in pin_info:
            pinner = pin_info[sq]
            a = f"Yes, the {p_name} on {sq_name} is pinned by {format_piece(pinner[0], pinner[1], pinner[2])}."
        elif is_pinned:
            a = f"Yes, the {p_name} on {sq_name} is pinned."
        else:
            a = f"No, the {p_name} on {sq_name} is not pinned."
        collector.add(cat, fen, q, a, scenario)


def generate_reconstruction(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate position reconstruction questions with natural answers."""
    cat = "reconstruction"
    if not collector.needs(cat):
        return

    # What's on a square
    sq = random.choice(chess.SQUARES)
    sq_name = chess.square_name(sq)
    piece = board.piece_at(sq)

    q = phraser.format_question(cat, "what_is_on_square", square=sq_name)
    if piece:
        color_name = get_color_name(piece.color)
        piece_name = get_piece_name(piece.piece_type)
        a = f"{color_name.capitalize()}'s {piece_name}."
    else:
        a = f"{sq_name} is empty."
    collector.add(cat, fen, q, a)

    # Where is a piece type
    color = random.choice([chess.WHITE, chess.BLACK])
    color_name = get_color_name(color)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    piece_type = random.choice(piece_types)
    piece_name = get_piece_name(piece_type)

    squares = [chess.square_name(sq) for sq, p in board.piece_map().items()
               if p.piece_type == piece_type and p.color == color]

    q = phraser.format_question(cat, "where_is_piece", color=color_name, piece=piece_name)
    if not squares:
        a = f"{color_name.capitalize()} has no {piece_name}."
    elif len(squares) == 1:
        a = f"On {squares[0]}."
    else:
        a = f"On {format_square_list(squares)}."
    collector.add(cat, fen, q, a)

    # Whose turn
    q = phraser.format_question(cat, "whose_turn")
    turn_color = get_color_name(board.turn)
    a = f"It is {turn_color}'s turn."
    collector.add(cat, fen, q, a)

    # Castling rights
    q = phraser.format_question(cat, "castling_rights")
    rights = []
    if board.has_kingside_castling_rights(chess.WHITE):
        rights.append("white kingside")
    if board.has_queenside_castling_rights(chess.WHITE):
        rights.append("white queenside")
    if board.has_kingside_castling_rights(chess.BLACK):
        rights.append("black kingside")
    if board.has_queenside_castling_rights(chess.BLACK):
        rights.append("black queenside")

    if rights:
        a = f"Yes, castling is available: {', '.join(rights)}."
    else:
        a = "No, castling is no longer available."
    collector.add(cat, fen, q, a)

    # Can color castle
    color = random.choice([chess.WHITE, chess.BLACK])
    color_name = get_color_name(color)
    q = phraser.format_question(cat, "can_color_castle", color=color_name)

    can_ks = board.has_kingside_castling_rights(color)
    can_qs = board.has_queenside_castling_rights(color)
    if can_ks and can_qs:
        a = f"Yes, {color_name} can castle both kingside and queenside."
    elif can_ks:
        a = f"Yes, {color_name} can castle kingside only."
    elif can_qs:
        a = f"Yes, {color_name} can castle queenside only."
    else:
        a = f"No, {color_name} cannot castle."
    collector.add(cat, fen, q, a)

    # En passant
    q = phraser.format_question(cat, "en_passant")
    if board.has_legal_en_passant():
        ep_sq = chess.square_name(board.ep_square)
        a = f"Yes, en passant is available on {ep_sq}."
    else:
        a = "No, en passant is not available."
    collector.add(cat, fen, q, a)


def generate_legal_moves(board: chess.Board, fen: str, collector: QACollector, phraser: QuestionPhraser):
    """Generate legal move questions with natural answers."""
    cat = "legal_moves"
    if not collector.needs(cat):
        return

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return

    # Pick a piece to ask about
    turn_pieces = [(sq, p) for sq, p in board.piece_map().items() if p.color == board.turn]
    if not turn_pieces:
        return

    sq, piece = random.choice(turn_pieces)
    sq_name = chess.square_name(sq)
    p_name = get_piece_name(piece.piece_type)

    piece_moves = [m for m in legal_moves if m.from_square == sq]
    # Deduplicate targets (promotions create multiple moves to same square)
    target_squares = list(dict.fromkeys(chess.square_name(m.to_square) for m in piece_moves))

    # What moves can piece make
    q = phraser.format_question(cat, "piece_legal_moves", piece=p_name, square=sq_name)
    if target_squares:
        a = f"The {p_name} on {sq_name} can move to {format_square_list(target_squares)}."
    else:
        a = f"The {p_name} on {sq_name} has no legal moves."
    collector.add(cat, fen, q, a)

    # Is specific move legal (pick a legal one)
    if piece_moves:
        move = random.choice(piece_moves)
        move_san = board.san(move)
        q = phraser.format_question(cat, "is_move_legal", move=move_san)
        to_sq = chess.square_name(move.to_square)
        a = f"Yes, {move_san} is legal. The {p_name} can move from {sq_name} to {to_sq}."
        collector.add(cat, fen, q, a, "legal")

    # Is specific move legal (pick an illegal one)
    # Also track legal target squares (to avoid false negatives for promotions)
    legal_targets = {(m.from_square, m.to_square) for m in legal_moves}
    for target in random.sample(chess.SQUARES, min(8, 64)):
        if target == sq:
            continue
        # Skip if any move to this target is legal (handles promotions)
        if (sq, target) in legal_targets:
            continue
        potential = chess.Move(sq, target)
        to_sq = chess.square_name(target)
        # For illegal moves, use SAN-style notation (piece + destination)
        piece_symbol = piece.symbol().upper() if piece.piece_type != chess.PAWN else ""
        illegal_move_san = f"{piece_symbol}{to_sq}"
        q = phraser.format_question(cat, "is_move_legal", move=illegal_move_san)

        # Try to explain why
        try:
            # Check if it's a pseudo-legal move
            if potential in board.pseudo_legal_moves:
                # Would leave king in check
                a = f"No, {illegal_move_san} is not legal because it would leave the king in check."
            else:
                a = f"No, {illegal_move_san} is not legal. The {p_name} cannot move to {to_sq}."
        except Exception:
            a = f"No, {illegal_move_san} is not legal."
        collector.add(cat, fen, q, a, "illegal")
        break

    # Can piece move to specific square
    if piece_moves:
        move = random.choice(piece_moves)
        to_sq = chess.square_name(move.to_square)
        q = phraser.format_question(cat, "can_piece_move_to", piece=p_name, from_square=sq_name, to_square=to_sq)
        a = f"Yes, the {p_name} on {sq_name} can move to {to_sq}."
        collector.add(cat, fen, q, a)


# =============================================================================
# Model-based Question Generators
# =============================================================================

@dataclass
class ModelContext:
    """Holds model and related utilities for model-based questions."""
    model: torch.nn.Module
    tokenizer: object
    device: torch.device

    def get_model_outputs(self, fen: str, board: chess.Board) -> dict:
        """Run model inference on a position and return outputs."""
        # Tokenize the FEN
        processed = process_fen(fen)
        encoding = self.tokenizer.encode(processed)
        input_ids = torch.tensor([encoding.ids], device=self.device)

        # Get legal move mask
        legal_mask = torch.zeros(len(policy_index), dtype=torch.bool, device=self.device)
        for move in board.legal_moves:
            uci = move.uci()
            if uci in MOVE_TO_IDX:
                legal_mask[MOVE_TO_IDX[uci]] = True

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        policy_logits = outputs.policy_logits[0]  # [policy_dim]
        winrate_logits = outputs.winrate_logits[0]  # [num_bins]

        # Compute position winrate from winrate head
        bin_centers = torch.linspace(0, 1, NUM_VALUE_BINS, device=self.device)
        winrate_probs = F.softmax(winrate_logits, dim=-1)
        position_winrate = (winrate_probs * bin_centers).sum().item()

        # Compute move winrates from policy head (sigmoid)
        move_winrates = torch.sigmoid(policy_logits)

        # Get best move (argmax over legal moves only)
        masked_logits = policy_logits.clone()
        masked_logits[~legal_mask] = float('-inf')
        best_move_idx = masked_logits.argmax().item()
        best_move_uci = policy_index[best_move_idx]
        best_move_winrate = move_winrates[best_move_idx].item()

        return {
            "policy_logits": policy_logits,
            "move_winrates": move_winrates,
            "position_winrate": position_winrate,
            "best_move_uci": best_move_uci,
            "best_move_winrate": best_move_winrate,
            "legal_mask": legal_mask,
        }


def generate_best_move(
    board: chess.Board,
    fen: str,
    collector: QACollector,
    phraser: QuestionPhraser,
    model_ctx: Optional[ModelContext] = None,
):
    """Generate best move questions using model predictions."""
    cat = "best_move"
    if not collector.needs(cat):
        return
    if model_ctx is None:
        return

    outputs = model_ctx.get_model_outputs(fen, board)
    best_move = outputs["best_move_uci"]
    best_winrate = outputs["best_move_winrate"]
    turn_color = get_color_name(board.turn)

    # Convert UCI to SAN
    try:
        move_obj = chess.Move.from_uci(best_move)
        move_san = board.san(move_obj)
    except Exception:
        move_san = best_move

    # What is the best move
    q = phraser.format_question(cat, "what_is_best_move")
    a = f"The best move is {move_san}."
    collector.add(cat, fen, q, a)

    # Best move with position assessment
    q = phraser.format_question(cat, "best_move_with_eval")
    position_desc = describe_position(best_winrate)
    a = f"The best move is {move_san}. After this move, {turn_color} is {position_desc}."
    collector.add(cat, fen, q, a)


def generate_position_winrate(
    board: chess.Board,
    fen: str,
    collector: QACollector,
    phraser: QuestionPhraser,
    model_ctx: Optional[ModelContext] = None,
):
    """Generate position winrate questions using model predictions."""
    cat = "position_winrate"
    if not collector.needs(cat):
        return
    if model_ctx is None:
        return

    outputs = model_ctx.get_model_outputs(fen, board)
    winrate = outputs["position_winrate"]
    turn_color = get_color_name(board.turn)

    # Determine scenario for balancing
    if winrate > 0.55:
        scenario = "winning"
    elif winrate < 0.45:
        scenario = "losing"
    else:
        scenario = "equal"

    target_per_scenario = -(-collector.target_per_category // 3)  # ceiling division
    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    position_desc = describe_position(winrate)

    # How is the position
    q = phraser.format_question(cat, "how_is_position")
    a = f"{turn_color.capitalize()} is {position_desc}."
    collector.add(cat, fen, q, a, scenario)

    # Who is winning
    q = phraser.format_question(cat, "who_is_winning")
    if winrate >= 0.52:
        a = f"{turn_color.capitalize()} is {position_desc}."
    elif winrate <= 0.48:
        a = f"{turn_color.capitalize()} is {position_desc}."
    else:
        a = "The position is equal."
    collector.add(cat, fen, q, a, scenario)


def generate_move_winrate(
    board: chess.Board,
    fen: str,
    collector: QACollector,
    phraser: QuestionPhraser,
    model_ctx: Optional[ModelContext] = None,
):
    """Generate move winrate questions using model predictions."""
    cat = "move_winrate"
    if not collector.needs(cat):
        return
    if model_ctx is None:
        return

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return

    outputs = model_ctx.get_model_outputs(fen, board)
    move_winrates = outputs["move_winrates"]
    best_winrate = outputs["best_move_winrate"]
    turn_color = get_color_name(board.turn)

    # Pick a random legal move to ask about
    move = random.choice(legal_moves)
    move_uci = move.uci()

    if move_uci not in MOVE_TO_IDX:
        return

    move_idx = MOVE_TO_IDX[move_uci]
    move_wr = move_winrates[move_idx].item()

    # Convert to SAN
    move_san = board.san(move)

    # Determine move quality based on regret
    regret = best_winrate - move_wr
    move_quality = describe_move_quality(regret)

    # Determine scenario for balancing
    if regret <= 0.03:
        scenario = "good_move"
    elif regret <= 0.10:
        scenario = "okay_move"
    else:
        scenario = "bad_move"

    target_per_scenario = -(-collector.target_per_category // 3)  # ceiling division
    if not collector.needs_scenario(cat, scenario, target_per_scenario):
        return

    # Position after move
    q = phraser.format_question(cat, "eval_after_move", move=move_san)
    position_after = describe_position(move_wr)
    a = f"After playing {move_san}, {turn_color} is {position_after}."
    collector.add(cat, fen, q, a, scenario)

    # Is move good
    q = phraser.format_question(cat, "is_move_good", move=move_san)
    if move_quality == "best":
        a = f"{move_san} is the best move."
    elif move_quality == "good":
        a = f"{move_san} is a good move."
    elif move_quality == "okay":
        a = f"{move_san} is an okay move, but there are better options."
    elif move_quality == "inaccuracy":
        a = f"{move_san} is an inaccuracy."
    elif move_quality == "mistake":
        a = f"{move_san} is a mistake."
    else:
        a = f"{move_san} is a blunder."
    collector.add(cat, fen, q, a, scenario)


# Generators that don't need the model
BASIC_GENERATORS = [
    generate_imbalance,
    generate_check,
    generate_attacked_defended,
    generate_hanging,
    generate_pinned,
    generate_reconstruction,
    generate_legal_moves,
]

# Generators that require the model
MODEL_GENERATORS = [
    generate_best_move,
    generate_position_winrate,
    generate_move_winrate,
]


def load_model(model_path: str | Path, device: str = "cuda") -> ModelContext:
    """Load the chess model for model-based questions."""
    from model import ChessPolicyValueModel

    print(f"Loading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ChessPolicyValueModel.from_pretrained_compiled(model_path)
    model.to(device)
    model.eval()

    tokenizer = create_tokenizer()

    print(f"Model loaded on {device}")
    return ModelContext(model=model, tokenizer=tokenizer, device=device)


def generate_qa_dataset(
    source_dataset_path: str | Path,
    output_path: str | Path,
    total_examples: int = 50000,
    seed: int = 42,
    model_path: str | Path | None = None,
    device: str = "cuda",
):
    """Generate a balanced Q&A dataset from chess positions.

    Args:
        source_dataset_path: Path to HuggingFace dataset with FENs
        output_path: Where to save the generated Q&A dataset
        total_examples: Total number of Q&A pairs to generate
        seed: Random seed for reproducibility
        model_path: Optional path to chess model for model-based questions
        device: Device to run model on ("cuda" or "cpu")
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Determine which categories to use
    if model_path is not None:
        categories = CATEGORIES
        model_ctx = load_model(model_path, device)
    else:
        # Only basic categories without model
        categories = [c for c in CATEGORIES if c not in ["best_move", "position_winrate", "move_winrate"]]
        model_ctx = None
        print("No model provided - skipping model-based questions (best_move, position_winrate, move_winrate)")

    target_per_category = total_examples // len(categories)
    print(f"\nGenerating {total_examples} Q&A pairs")
    print(f"Categories: {len(categories)}")
    print(f"Target per category: {target_per_category}")
    print()

    collector = QACollector(target_per_category=target_per_category, categories=categories)
    phraser = QuestionPhraser()

    print(f"Loading source dataset from {source_dataset_path}...")
    source_dataset = load_from_disk(source_dataset_path).shuffle(seed=seed)
    print(f"Loaded {len(source_dataset):,} positions")

    print("\nGenerating Q&A pairs...")
    pbar = tqdm(source_dataset, desc="Positions: 0 Q&A")

    for example in pbar:
        if collector.is_complete():
            break

        fen = example["fen"]
        try:
            board = chess.Board(fen)
        except Exception:
            continue

        collector.start_position(fen)
        count_before = collector.total_count()

        # Try basic generators first
        for generator in BASIC_GENERATORS:
            generator(board, fen, collector, phraser)
            if collector.total_count() > count_before:
                break

        # If no basic question added and model available, try model generators
        if collector.total_count() == count_before and model_ctx is not None:
            for generator in MODEL_GENERATORS:
                generator(board, fen, collector, phraser, model_ctx)
                if collector.total_count() > count_before:
                    break

        pbar.set_description(f"{collector.total_count():,} Q&A")

    print("\nCollection status:")
    for cat_name, info in collector.status().items():
        scenarios = ", ".join(f"{k}:{v}" for k, v in info["scenarios"].items())
        print(f"  {cat_name}: {info['total']} total ({scenarios})")

    all_qa = collector.get_all()
    random.shuffle(all_qa)
    print(f"\nTotal Q&A pairs: {len(all_qa)}")

    qa_dataset = Dataset.from_list(all_qa)
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    qa_dataset.save_to_disk(output_path)
    print(f"Saved to {output_path}")

    return qa_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Q&A dataset from chess positions")
    parser.add_argument("--source", type=str, default="/fs/scratch/PAS2836/lees_stuff/action_value",
                        help="Path to source HuggingFace dataset")
    parser.add_argument("--output", type=str, default="/fs/scratch/PAS2836/lees_stuff/easy_chess_questions",
                        help="Output path")
    parser.add_argument("--total", type=int, default=50000, help="Total Q&A pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to chess model for model-based questions (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference")

    args = parser.parse_args()
    generate_qa_dataset(
        source_dataset_path=args.source,
        output_path=args.output,
        total_examples=args.total,
        seed=args.seed,
        model_path=args.model,
        device=args.device,
    )
