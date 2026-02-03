"""Inference script for testing the chess-to-Qwen projector."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

from model import ChessPolicyValueModel
from tokenizer import create_tokenizer, process_fen
from projector import ChessProjector

DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def get_chess_hidden_states(
    chess_model: ChessPolicyValueModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract hidden states from chess model."""
    batch_size = input_ids.size(0)
    device = input_ids.device

    # Get token embeddings
    input_embeds = chess_model.transformer.embed_tokens(input_ids)
    seq_len = input_embeds.size(1)

    # Add positional embeddings
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    position_embeds = chess_model.position_embeddings(position_ids)
    input_embeds = input_embeds + position_embeds

    # Forward through transformer
    with torch.no_grad():
        outputs = chess_model.transformer(inputs_embeds=input_embeds)

    return outputs.last_hidden_state


def main():
    parser = argparse.ArgumentParser(description="Test chess-to-Qwen projector")
    parser.add_argument(
        "--projector-checkpoint",
        type=str,
        required=True,
        help="Path to saved projector checkpoint directory",
    )
    parser.add_argument(
        "--chess-model-path",
        type=str,
        required=True,
        help="Path to chess model checkpoint",
    )
    parser.add_argument(
        "--qwen-model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Qwen model name or path",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Load Qwen in 8-bit quantization",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Stream generated tokens to stdout",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=DEFAULT_FEN,
        help="FEN position to analyze",
    )
    parser.add_argument(
        "--norm-correct",
        action="store_true",
        help="Normalize projector output to match Qwen's average norm (~1.8)",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"\nFEN: {args.fen}\n")

    # Load chess model
    print(f"Loading chess model from {args.chess_model_path}...")
    chess_model = ChessPolicyValueModel.from_pretrained_compiled(args.chess_model_path)
    chess_model.to(args.device)
    chess_model.eval()

    # Load Qwen model
    print(f"Loading Qwen model {args.qwen_model_name}...")
    qwen_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
    }
    if args.use_8bit:
        qwen_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        qwen_kwargs.pop("device_map")

    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_name,
        **qwen_kwargs,
    )
    qwen_model.eval()

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # Load projector
    print(f"Loading projector from {args.projector_checkpoint}...")
    projector = ChessProjector.from_pretrained(
        args.projector_checkpoint,
        device=args.device,
    )
    projector.to(args.device, dtype=torch.bfloat16)
    projector.eval()

    # Create chess tokenizer
    chess_tokenizer = create_tokenizer()

    # Process FEN
    processed_fen = process_fen(args.fen)
    chess_encoding = chess_tokenizer.encode(processed_fen)
    chess_input_ids = torch.tensor(
        [chess_encoding.ids],
        dtype=torch.long,
        device=args.device,
    )

    print(f"Chess input shape: {chess_input_ids.shape}")

    # Get chess hidden states and project to Qwen space (do this once)
    with torch.no_grad():
        chess_hidden = get_chess_hidden_states(chess_model, chess_input_ids)
        print(f"Chess hidden shape: {chess_hidden.shape}")

        chess_prefix = projector(chess_hidden.to(torch.bfloat16))
        print(f"Chess prefix shape: {chess_prefix.shape}")

        # Print projector output statistics
        chess_prefix_fp32 = chess_prefix.float()
        chess_avg_norm = chess_prefix_fp32.norm(dim=-1).mean().item()
        print(
            "Chess prefix stats: "
            f"mean={chess_prefix_fp32.mean().item():.6f}, "
            f"std={chess_prefix_fp32.std().item():.6f}, "
            f"min={chess_prefix_fp32.min().item():.6f}, "
            f"max={chess_prefix_fp32.max().item():.6f}, "
            f"avg_norm={chess_avg_norm:.6f}"
        )

        # Compare to Qwen's native embeddings
        sample_text = "The chess position shows"
        sample_ids = qwen_tokenizer(sample_text, return_tensors="pt")["input_ids"].to(args.device)
        sample_embeds = qwen_model.get_input_embeddings()(sample_ids).float()
        qwen_avg_norm = sample_embeds.norm(dim=-1).mean().item()
        print(
            "Qwen embed stats:   "
            f"mean={sample_embeds.mean().item():.6f}, "
            f"std={sample_embeds.std().item():.6f}, "
            f"min={sample_embeds.min().item():.6f}, "
            f"max={sample_embeds.max().item():.6f}, "
            f"avg_norm={qwen_avg_norm:.6f}"
        )

        # Optionally normalize projector output to match Qwen's average norm
        if args.norm_correct:
            target_norm = 1.8
            scale_factor = target_norm / chess_avg_norm
            chess_prefix = chess_prefix * scale_factor
            print(f"Norm correction: scaled by {scale_factor:.4f} (target avg_norm={target_norm})")

    # Pre-compute the user prefix embeddings (constant for all questions)
    prefix_text = "<|im_start|>user\n"
    prefix_tokens = qwen_tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    prefix_ids = prefix_tokens["input_ids"].to(args.device)
    with torch.no_grad():
        prefix_embeds = qwen_model.get_input_embeddings()(prefix_ids)

    # Get the end token ID for generation stopping
    im_end_id = qwen_tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    print("\nReady! Enter questions about the position (type 'quit' to exit).\n")
    print("=" * 60)

    # Interactive chat loop
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Build fresh prompt for this question (no history)
        suffix_text = f"{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>"
        suffix_tokens = qwen_tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
        suffix_ids = suffix_tokens["input_ids"].to(args.device)

        with torch.no_grad():
            suffix_embeds = qwen_model.get_input_embeddings()(suffix_ids)

            # Concatenate: [prefix, chess, suffix]
            combined_embeds = torch.cat([prefix_embeds, chess_prefix, suffix_embeds], dim=1)

            attention_mask = torch.ones(
                combined_embeds.size(0),
                combined_embeds.size(1),
                dtype=torch.long,
                device=args.device,
            )

            print("\nAssistant: ", end="", flush=True)

            streamer = None
            if args.stream:
                streamer = TextStreamer(
                    qwen_tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

            outputs = qwen_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=qwen_tokenizer.pad_token_id,
                eos_token_id=im_end_id,
                streamer=streamer,
            )

        # If not streaming, print the response
        if not args.stream:
            generated_ids = outputs[0]
            response = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(response)


if __name__ == "__main__":
    main()
