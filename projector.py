"""Chess-to-Qwen projector model."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn


class ChessProjector(nn.Module):
    """Projects chess model hidden states to Qwen embedding space using MLPs.

    Architecture (~10M params):
    - Per-token MLP: 768 -> 2048 -> 4096
    - All 72 tokens preserved

    If fixed_embeddings=True, ignores input and returns learned fixed embeddings
    (useful as ablation test to measure baseline).
    """

    def __init__(
        self,
        chess_hidden_size: int = 768,
        qwen_hidden_size: int = 4096,
        intermediate_size: int = 2048,
        num_tokens: int = 72,
        fixed_embeddings: bool = False,
    ):
        super().__init__()
        self.chess_hidden_size = chess_hidden_size
        self.qwen_hidden_size = qwen_hidden_size
        self.intermediate_size = intermediate_size
        self.num_tokens = num_tokens
        self.fixed_embeddings = fixed_embeddings

        if fixed_embeddings:
            # Learn fixed embeddings that ignore input (ablation baseline)
            self.learned_embeddings = nn.Parameter(
                torch.randn(1, num_tokens, qwen_hidden_size) * 0.02
            )
            self.mlp = None
        else:
            # MLP: 768 -> 2048 -> 4096 (applied per token)
            # No activation/norm after final linear - output must match Qwen's embedding distribution
            self.mlp = nn.Sequential(
                nn.LayerNorm(chess_hidden_size),
                nn.Linear(chess_hidden_size, intermediate_size),
                nn.GELU(),
                nn.LayerNorm(intermediate_size),
                nn.Linear(intermediate_size, qwen_hidden_size),
            )
            self.learned_embeddings = None

    def forward(self, chess_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chess_hidden_states: [batch, 72, 768] from chess model

        Returns:
            [batch, 72, 4096] prefix embeddings for Qwen
        """
        if self.fixed_embeddings:
            # Ignore input, return fixed learned embeddings expanded to batch size
            batch_size = chess_hidden_states.size(0)
            return self.learned_embeddings.expand(batch_size, -1, -1)
        else:
            return self.mlp(chess_hidden_states)

    def save_pretrained(self, save_directory: str) -> None:
        """Save projector weights and config."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "projector.pt"))
        config = {
            "chess_hidden_size": self.chess_hidden_size,
            "qwen_hidden_size": self.qwen_hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_tokens": self.num_tokens,
            "fixed_embeddings": self.fixed_embeddings,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cpu") -> "ChessProjector":
        """Load projector from saved checkpoint."""
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(
            os.path.join(load_directory, "projector.pt"), map_location=device
        )
        model.load_state_dict(state_dict)
        return model
