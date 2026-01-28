"""
Question phrasing module for chess Q&A dataset generation.

Loads question templates from JSON and provides utilities for formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


TEMPLATES_PATH = Path(__file__).parent / "question_templates.json"


@dataclass
class QuestionTemplate:
    """A single question template."""
    id: str
    question: str
    variables: list[str]

    def format(self, **kwargs) -> str:
        """Format the question with given variables."""
        return self.question.format(**kwargs)


class QuestionPhraser:
    """Loads templates and generates formatted questions."""

    def __init__(self, templates_path: Path | str | None = None):
        if templates_path is None:
            templates_path = TEMPLATES_PATH

        with open(templates_path) as f:
            raw_templates = json.load(f)

        self.categories: dict[str, list[QuestionTemplate]] = {}

        for category, data in raw_templates.items():
            self.categories[category] = [
                QuestionTemplate(
                    id=t["id"],
                    question=t["question"],
                    variables=t["variables"],
                )
                for t in data["templates"]
            ]

    def get_template(self, category: str, template_id: str) -> QuestionTemplate:
        """Get a specific template by category and ID."""
        for t in self.categories[category]:
            if t.id == template_id:
                return t
        raise ValueError(f"Template '{template_id}' not found in category '{category}'")

    def format_question(self, category: str, template_id: str, **kwargs) -> str:
        """Format a question from a template."""
        template = self.get_template(category, template_id)
        return template.format(**kwargs)

    def list_categories(self) -> list[str]:
        """List all available categories."""
        return list(self.categories.keys())

    def list_templates(self, category: str) -> list[str]:
        """List all template IDs in a category."""
        return [t.id for t in self.categories[category]]


# =============================================================================
# Piece and square naming utilities
# =============================================================================

PIECE_NAMES = {
    1: "pawn",
    2: "knight",
    3: "bishop",
    4: "rook",
    5: "queen",
    6: "king",
}

PIECE_SYMBOLS = {
    1: "",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
}


def get_piece_name(piece_type: int) -> str:
    """Get piece name from type (1=pawn, 2=knight, etc.)."""
    return PIECE_NAMES[piece_type]


def get_color_name(color: bool) -> str:
    """Get color name from boolean (True=white, False=black)."""
    return "white" if color else "black"


def format_piece(piece_type: int, color: bool, square: str) -> str:
    """Format a piece reference like 'white's e4 pawn' or 'black's f6 knight'."""
    color_name = get_color_name(color)
    piece_name = get_piece_name(piece_type)
    return f"{color_name}'s {square} {piece_name}"


def format_piece_short(piece_type: int, square: str) -> str:
    """Format a piece reference like 'pawn on e4' or 'knight on f6'."""
    piece_name = get_piece_name(piece_type)
    return f"{piece_name} on {square}"


def format_piece_list(pieces: list[tuple[int, bool, str]]) -> str:
    """
    Format a list of pieces into natural language.

    Args:
        pieces: List of (piece_type, color, square) tuples

    Returns:
        String like "white's e4 pawn and black's d5 knight"
    """
    if not pieces:
        return "nothing"

    formatted = [format_piece(pt, c, sq) for pt, c, sq in pieces]

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def format_square_list(squares: list[str]) -> str:
    """Format a list of squares into natural language."""
    if not squares:
        return "nowhere"

    if len(squares) == 1:
        return squares[0]
    elif len(squares) == 2:
        return f"{squares[0]} and {squares[1]}"
    else:
        return ", ".join(squares[:-1]) + f", and {squares[-1]}"


def format_material_diff(white_counts: dict[int, int], black_counts: dict[int, int]) -> str:
    """
    Format material difference into natural language.

    Args:
        white_counts: {piece_type: count} for white
        black_counts: {piece_type: count} for black

    Returns:
        String like "White is up a rook. Black has 2 extra pawns."
    """
    # Calculate differences
    diffs = {}
    for pt in [5, 4, 3, 2, 1]:  # Q, R, B, N, P order
        w = white_counts.get(pt, 0)
        b = black_counts.get(pt, 0)
        if w != b:
            diffs[pt] = w - b

    if not diffs:
        return "Material is equal."

    white_up = []
    black_up = []

    for pt, diff in diffs.items():
        name = get_piece_name(pt)
        if abs(diff) == 1:
            piece_str = f"a {name}" if name != "pawn" else "a pawn"
        else:
            piece_str = f"{abs(diff)} {name}s"

        if diff > 0:
            white_up.append(piece_str)
        else:
            black_up.append(piece_str)

    parts = []
    if white_up:
        parts.append(f"White is up {', '.join(white_up)}")
    if black_up:
        parts.append(f"Black is up {', '.join(black_up)}")

    return ". ".join(parts) + "."


def format_material_count(white_counts: dict[int, int], black_counts: dict[int, int]) -> str:
    """
    Format full material count for both sides.

    Returns:
        String like "White has a queen, 2 rooks, and 3 pawns. Black has 2 rooks and 5 pawns."
    """
    def format_side(counts: dict[int, int]) -> str:
        parts = []
        for pt in [5, 4, 3, 2, 1]:  # Q, R, B, N, P
            count = counts.get(pt, 0)
            if count > 0:
                name = PIECE_NAMES[pt]
                if count == 1:
                    parts.append(f"a {name}")
                else:
                    parts.append(f"{count} {name}s")
        if not parts:
            return "no pieces"
        elif len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    white_str = format_side(white_counts)
    black_str = format_side(black_counts)

    return f"White has {white_str}. Black has {black_str}."
