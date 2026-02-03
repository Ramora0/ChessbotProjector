"""Chess-to-Qwen Q-Former projector model."""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    """Single Q-Former block with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cross_attention_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention to chess hidden states
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn_kv_norm = nn.LayerNorm(cross_attention_dim)
        self.cross_attn_q = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_k = nn.Linear(cross_attention_dim, hidden_size)
        self.cross_attn_v = nn.Linear(cross_attention_dim, hidden_size)
        self.cross_attn_out = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self, queries: torch.Tensor, chess_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [batch, num_queries, hidden_size]
            chess_hidden: [batch, 72, chess_hidden_size]

        Returns:
            [batch, num_queries, hidden_size]
        """
        # Self-attention with pre-norm
        residual = queries
        queries = self.self_attn_norm(queries)
        queries, _ = self.self_attn(queries, queries, queries)
        queries = residual + queries

        # Cross-attention with pre-norm
        residual = queries
        queries_normed = self.cross_attn_norm(queries)
        chess_normed = self.cross_attn_kv_norm(chess_hidden)

        batch_size, num_queries, _ = queries_normed.shape
        _, seq_len, _ = chess_normed.shape

        q = self.cross_attn_q(queries_normed).view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.cross_attn_k(chess_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.cross_attn_v(chess_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.cross_attn_dropout(attn_weights)

        cross_out = torch.matmul(attn_weights, v)
        cross_out = cross_out.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.hidden_size
        )
        cross_out = self.cross_attn_out(cross_out)
        queries = residual + cross_out

        # FFN with pre-norm
        residual = queries
        queries = self.ffn_norm(queries)
        queries = residual + self.ffn(queries)

        return queries


class ChessQFormerProjector(nn.Module):
    """Projects chess model hidden states to Qwen embedding space using Q-Former.

    Architecture:
    - Learnable query tokens that attend to chess hidden states
    - Multiple Q-Former blocks with self-attention + cross-attention
    - Final projection to Qwen embedding dimension

    This allows the model to learn what information to extract from the
    chess representation rather than forcing a fixed mapping.
    """

    def __init__(
        self,
        chess_hidden_size: int = 768,
        qwen_hidden_size: int = 4096,
        num_query_tokens: int = 32,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chess_hidden_size = chess_hidden_size
        self.qwen_hidden_size = qwen_hidden_size
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_size) * 0.02
        )

        # Q-Former blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                cross_attention_dim=chess_hidden_size,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final projection to Qwen hidden size
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, qwen_hidden_size),
        )
        self.output_norm = nn.LayerNorm(qwen_hidden_size)

    def forward(self, chess_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chess_hidden_states: [batch, 72, 768] from chess model

        Returns:
            [batch, num_query_tokens, qwen_hidden_size] prefix embeddings for Qwen
        """
        batch_size = chess_hidden_states.size(0)

        # Expand query tokens to batch size
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # Pass through Q-Former blocks
        for block in self.blocks:
            queries = block(queries, chess_hidden_states)

        # Project to Qwen embedding space
        output = self.output_proj(queries)
        output = self.output_norm(output)

        return output

    def save_pretrained(self, save_directory: str) -> None:
        """Save projector weights and config."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "projector.pt"))
        config = {
            "architecture": "qformer",
            "chess_hidden_size": self.chess_hidden_size,
            "qwen_hidden_size": self.qwen_hidden_size,
            "num_query_tokens": self.num_query_tokens,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: str = "cpu") -> "ChessQFormerProjector":
        """Load projector from saved checkpoint."""
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        # Remove architecture key if present (not a constructor arg)
        config.pop("architecture", None)
        model = cls(**config)
        state_dict = torch.load(
            os.path.join(load_directory, "projector.pt"), map_location=device
        )
        model.load_state_dict(state_dict)
        return model
