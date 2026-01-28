from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


class MultiTaskAttentionPooling(nn.Module):
    """Multi-task attention pooling with shared K/V projections.

    Computes multiple task-specific outputs in a single forward pass by using
    separate learnable queries for each task while sharing the K/V projections.
    """

    def __init__(self, hidden_size: int, task_output_dims: dict[str, int]) -> None:
        """
        Args:
            hidden_size: Dimension of input hidden states
            task_output_dims: Dict mapping task names to output dimensions
                             e.g., {'policy': 1958, 'winrate': 3}
        """
        super().__init__()
        self.task_names = list(task_output_dims.keys())
        self.num_tasks = len(self.task_names)

        # Shared K/V projections across all tasks
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

        # Task-specific queries, norms, and output projections
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, hidden_size))
            for name in self.task_names
        })
        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(hidden_size)
            for name in self.task_names
        })
        self.output_projs = nn.ModuleDict({
            name: nn.Linear(hidden_size, output_dim)
            for name, output_dim in task_output_dims.items()
        })

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            Dict mapping task names to outputs: {task_name: [batch_size, output_dim]}
        """
        # Shared K/V projections (computed once)
        k = self.key_proj(hidden_states)    # [batch, seq_len, hidden]
        v = self.value_proj(hidden_states)  # [batch, seq_len, hidden]

        outputs = {}
        for task_name in self.task_names:
            # Task-specific query
            q = self.queries[task_name].unsqueeze(0)  # [1, 1, hidden]

            # Compute attention weights
            attn_weights = torch.matmul(q, k.transpose(
                1, 2)) * self.scale  # [batch, 1, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Weighted sum of values
            pooled = torch.matmul(attn_weights, v).squeeze(
                1)  # [batch, hidden]

            # Task-specific normalization and projection
            pooled = self.norms[task_name](pooled)
            outputs[task_name] = self.output_projs[task_name](pooled)

        return outputs


@dataclass
class ChessPolicyValueOutput(ModelOutput):
    policy_logits: torch.Tensor = None
    winrate_logits: torch.Tensor = None
    control_logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class ChessPolicyValueModel(LlamaPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.policy_dim = config.policy_dim
        self.transformer = LlamaModel(config)
        self._disable_causal_mask()
        self._disable_rope()
        hidden_size = config.hidden_size

        # Learned positional embeddings for all tokens
        # Tokens 0-63: board positions (a8, b8, ..., h1)
        # Token 64: turn (w/b)
        # Tokens 65-68: castling rights (K, Q, k, q)
        # Token 69: en passant square
        # Token 70: halfmove clock
        # Token 71: fullmove number
        self.position_embeddings = nn.Embedding(72, hidden_size)

        # Multi-task attention pooling (shared K/V, task-specific queries)
        # Winrate head predicts win% in 128 bins (0.0 to 1.0)
        self.num_value_bins = 128
        self.task_head = MultiTaskAttentionPooling(
            hidden_size=hidden_size,
            task_output_dims={
                'policy': self.policy_dim,
                'winrate': self.num_value_bins,
            }
        )

        # Per-square control head: each of the 64 board squares predicts its own attacker counts
        # Applied to hidden states at positions 0-63 (the board tokens)
        # Output: 2 values per square (white attackers, black attackers)
        self.control_head = nn.Linear(hidden_size, 2)

        # Language modeling head for masked token prediction (kept for checkpoint compatibility)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        self.post_init()

    # type: ignore[override]
    def load_state_dict(self, state_dict, strict: bool = False):
        result = super().load_state_dict(state_dict, strict=strict)

        missing = list(getattr(result, "missing_keys", ()))
        unexpected = list(getattr(result, "unexpected_keys", ()))
        if missing:
            print(
                "load_state_dict: missing keys while loading checkpoint:",
                ", ".join(missing),
            )
        if unexpected:
            print(
                "load_state_dict: unexpected keys while loading checkpoint:",
                ", ".join(unexpected),
            )

        return result

    @classmethod
    def from_pretrained_compiled(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a model that was saved with torch.compile() applied.

        This handles the _orig_mod. prefix that torch.compile() adds to state dict keys.
        """
        import os
        from transformers import AutoConfig

        # Load config
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Initialize model
        model = cls(config)

        # Load state dict
        state_dict_path = os.path.join(
            pretrained_model_name_or_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            # Try model.safetensors
            state_dict_path = os.path.join(
                pretrained_model_name_or_path, "model.safetensors")
            if os.path.exists(state_dict_path):
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
            else:
                raise FileNotFoundError(
                    f"Could not find model weights in {pretrained_model_name_or_path}")
        else:
            import torch
            state_dict = torch.load(state_dict_path, map_location="cpu")

        # Strip _orig_mod. prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=False)

        return model

    def _disable_causal_mask(self) -> None:
        for block in self.transformer.layers:
            block.self_attn.is_causal = False

    def _disable_rope(self) -> None:
        """Disable rotary position embeddings - we use learned positional embeddings instead."""
        import transformers.models.llama.modeling_llama as llama_module

        def no_op_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return q, k  # Return unchanged

        llama_module.apply_rotary_pos_emb = no_op_rotary

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> ChessPolicyValueOutput:
        # Convert input_ids to embeddings
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.embed_tokens(input_ids)
        original_seq_len = input_embeds.size(1)

        # Add learned positional embeddings to all tokens
        position_ids = torch.arange(
            original_seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(
            batch_size, -1)  # [batch_size, seq_len]
        position_embeds = self.position_embeddings(
            position_ids)  # [batch_size, seq_len, hidden_size]
        input_embeds = input_embeds + position_embeds

        # Process all tokens through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=input_embeds, **kwargs)
        hidden_states = transformer_outputs.last_hidden_state

        # Multi-task attention pooling (single forward pass)
        task_outputs = self.task_head(hidden_states)
        policy_logits = task_outputs['policy']
        winrate_logits = task_outputs['winrate']

        # Per-square control prediction: apply control head to each board square's hidden state
        board_hidden_states = hidden_states[:, :64, :]  # [batch, 64, hidden]
        control_logits_per_square = self.control_head(board_hidden_states)  # [batch, 64, 2]
        # Reshape to [batch, 128]: first 64 = white counts, last 64 = black counts
        control_logits = torch.cat([
            control_logits_per_square[:, :, 0],  # [batch, 64] white attackers
            control_logits_per_square[:, :, 1],  # [batch, 64] black attackers
        ], dim=1)  # [batch, 128]

        if not return_dict:
            return (
                policy_logits,
                winrate_logits,
                control_logits,
                transformer_outputs.hidden_states,
                transformer_outputs.attentions,
            )

        return ChessPolicyValueOutput(
            policy_logits=policy_logits,
            winrate_logits=winrate_logits,
            control_logits=control_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
