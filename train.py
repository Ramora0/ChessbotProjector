"""Training script for chess-to-Qwen projector."""

from __future__ import annotations

import argparse
import os
import subprocess

import wandb

# =============================================================================
# Copy dataset to local tmp before training (avoids I/O bottleneck on scratch)
# =============================================================================
DATASET_SRC = "/fs/scratch/PAS2836/lees_stuff/easy_chess_questions"
TMPDIR = os.environ.get("TMPDIR", "/tmp")
LOCAL_DATASET = os.path.join(TMPDIR, "easy_chess_questions")

if not os.path.exists(LOCAL_DATASET):
    print(f"Copying dataset to local tmp: {LOCAL_DATASET}")
    subprocess.run(["cp", "-r", DATASET_SRC, LOCAL_DATASET], check=True)
    print("Dataset copy complete.")
else:
    print(f"Using cached dataset at {LOCAL_DATASET}")
# =============================================================================

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import ChessPolicyValueModel
from projector import ChessProjector
from tokenizer import create_tokenizer, process_fen

# =============================================================================
# TRAINING CONFIGURATION (optimized for 40GB A100)
# =============================================================================
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
NUM_EPOCHS = 3
INTERMEDIATE_SIZE = 2048  # MLP hidden dimension
SAVE_STEPS = 1000
LOG_STEPS = 10
NUM_WORKERS = 4
MAX_GRAD_NORM = 1.0
# =============================================================================


class ChessQADataset(Dataset):
    """Dataset for chess Q&A training."""

    def __init__(self, dataset_path: str, chess_tokenizer, qwen_tokenizer, max_length: int = 256):
        self.dataset = load_from_disk(dataset_path)
        self.chess_tokenizer = chess_tokenizer
        self.qwen_tokenizer = qwen_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        fen = example["fen"]
        question = example["question"]
        answer = example["answer"]

        # Process FEN for chess model
        processed_fen = process_fen(fen)
        chess_encoding = self.chess_tokenizer.encode(processed_fen)
        chess_input_ids = torch.tensor(chess_encoding.ids, dtype=torch.long)

        # Create Q&A text for Qwen (chat format with empty thinking block)
        # Chess tokens go after "<|im_start|>user\n", before the question
        prefix_text = "<|im_start|>user\n"
        question_part = f"{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>"
        answer_part = f"{answer}<|im_end|>"

        # Tokenize prefix (before chess tokens)
        prefix_tokens = self.qwen_tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=False
        )
        prefix_length = prefix_tokens["input_ids"].size(1)

        # Tokenize question part to find answer start (relative to after chess tokens)
        question_tokens = self.qwen_tokenizer(
            question_part, return_tensors="pt", add_special_tokens=False
        )
        question_length = question_tokens["input_ids"].size(1)

        # Tokenize full suffix (question + answer, after chess tokens)
        suffix_text = question_part + answer_part
        suffix_tokens = self.qwen_tokenizer(
            suffix_text, max_length=self.max_length, truncation=True,
            return_tensors="pt", add_special_tokens=False
        )

        return {
            "chess_input_ids": chess_input_ids,
            "prefix_ids": prefix_tokens["input_ids"].squeeze(0),
            "prefix_mask": prefix_tokens["attention_mask"].squeeze(0),
            "suffix_ids": suffix_tokens["input_ids"].squeeze(0),
            "suffix_mask": suffix_tokens["attention_mask"].squeeze(0),
            "answer_start_idx": question_length,  # relative to suffix start
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function with padding."""
    chess_input_ids = torch.stack([item["chess_input_ids"] for item in batch])

    # Prefix is always the same length, no padding needed
    prefix_ids = torch.stack([item["prefix_ids"] for item in batch])
    prefix_mask = torch.stack([item["prefix_mask"] for item in batch])

    # Pad suffix
    max_suffix_len = max(item["suffix_ids"].size(0) for item in batch)
    suffix_ids = []
    suffix_mask = []
    answer_start_indices = []

    for item in batch:
        seq_len = item["suffix_ids"].size(0)
        pad_len = max_suffix_len - seq_len
        suffix_ids.append(F.pad(item["suffix_ids"], (0, pad_len), value=0))
        suffix_mask.append(F.pad(item["suffix_mask"], (0, pad_len), value=0))
        answer_start_indices.append(item["answer_start_idx"])

    return {
        "chess_input_ids": chess_input_ids,
        "prefix_ids": prefix_ids,
        "prefix_mask": prefix_mask,
        "suffix_ids": torch.stack(suffix_ids),
        "suffix_mask": torch.stack(suffix_mask),
        "answer_start_indices": torch.tensor(answer_start_indices, dtype=torch.long),
    }


def get_chess_hidden_states(chess_model: ChessPolicyValueModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract hidden states from chess model."""
    batch_size = input_ids.size(0)
    device = input_ids.device

    input_embeds = chess_model.transformer.embed_tokens(input_ids)
    seq_len = input_embeds.size(1)

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    position_embeds = chess_model.position_embeddings(position_ids)
    input_embeds = input_embeds + position_embeds

    with torch.no_grad():
        outputs = chess_model.transformer(inputs_embeds=input_embeds)

    return outputs.last_hidden_state


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_start_indices: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Compute causal LM loss only on answer tokens."""
    batch_size, seq_len, vocab_size = logits.shape

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_mask = torch.zeros_like(shift_mask, dtype=torch.float)

    for i in range(batch_size):
        answer_start = prefix_length + answer_start_indices[i] - 1
        valid_len = shift_mask[i].sum().item()
        answer_start = min(answer_start, valid_len - 1)
        loss_mask[i, answer_start:] = shift_mask[i, answer_start:].float()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction="none"
    )
    loss = loss.view(batch_size, -1)
    masked_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    return masked_loss


def main():
    parser = argparse.ArgumentParser(description="Train chess-to-Qwen projector")
    parser.add_argument("--name", type=str, required=True, help="Run name for checkpoints and wandb")
    parser.add_argument("--chess-model-path", type=str, default="../Chessbot/final-model/checkpoint-515268")
    parser.add_argument("--qwen-model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-path", type=str, default=LOCAL_DATASET)
    parser.add_argument("--output-dir", type=str, default="./projector_checkpoints")
    parser.add_argument(
        "--fixed-embeddings",
        action="store_true",
        help="Ablation: learn fixed embeddings that ignore chess input (baseline test)",
    )
    args = parser.parse_args()

    # Use name for output directory
    output_dir = os.path.join(args.output_dir, args.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load chess model (frozen)
    print(f"Loading chess model from {args.chess_model_path}...")
    chess_model = ChessPolicyValueModel.from_pretrained_compiled(args.chess_model_path)
    chess_model.to(device)
    chess_model.eval()
    for param in chess_model.parameters():
        param.requires_grad = False

    # Load Qwen model (frozen, bf16)
    print(f"Loading Qwen model {args.qwen_model_name}...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    qwen_model.eval()
    for param in qwen_model.parameters():
        param.requires_grad = False

    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_hidden_size = qwen_model.config.hidden_size
    print(f"Qwen hidden size: {qwen_hidden_size}")

    chess_tokenizer = create_tokenizer()

    # Initialize projector (trainable)
    if args.fixed_embeddings:
        print("Initializing FIXED EMBEDDINGS (ablation baseline)...")
    else:
        print("Initializing projector...")
    projector = ChessProjector(
        chess_hidden_size=768,
        qwen_hidden_size=qwen_hidden_size,
        intermediate_size=INTERMEDIATE_SIZE,
        fixed_embeddings=args.fixed_embeddings,
    )
    projector.to(device, dtype=torch.bfloat16)
    projector.train()

    num_params = sum(p.numel() for p in projector.parameters())
    print(f"Projector parameters: {num_params:,}")

    # Create dataset, train/val split, and dataloaders
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = ChessQADataset(
        dataset_path=args.dataset_path,
        chess_tokenizer=chess_tokenizer,
        qwen_tokenizer=qwen_tokenizer,
    )
    print(f"Dataset size: {len(dataset):,}")

    val_size = max(1, int(0.05 * len(dataset)))
    val_size = min(val_size, max(1, len(dataset) - 1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train size: {len(train_dataset):,} | Val size: {len(val_dataset):,}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    steps_per_epoch = max(1, len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS)
    max_steps = NUM_EPOCHS * steps_per_epoch
    print(f"Training for {NUM_EPOCHS} epochs = {max_steps:,} steps")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(projector.parameters(), lr=LEARNING_RATE)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize wandb
    wandb.init(
        project="chessformer-projector",
        name=args.name,
        config={
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "num_epochs": NUM_EPOCHS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_grad_norm": MAX_GRAD_NORM,
            "projector_params": num_params,
            "dataset_size": len(dataset),
            "chess_model_path": args.chess_model_path,
            "qwen_model_name": args.qwen_model_name,
            "fixed_embeddings": args.fixed_embeddings,
        },
    )

    def evaluate() -> float:
        projector.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                chess_input_ids = batch["chess_input_ids"].to(device)
                prefix_ids = batch["prefix_ids"].to(device)
                prefix_mask = batch["prefix_mask"].to(device)
                suffix_ids = batch["suffix_ids"].to(device)
                suffix_mask = batch["suffix_mask"].to(device)
                answer_start_indices = batch["answer_start_indices"].to(device)

                chess_hidden = get_chess_hidden_states(chess_model, chess_input_ids)
                chess_embeds = projector(chess_hidden.to(torch.bfloat16))
                prefix_embeds = qwen_model.get_input_embeddings()(prefix_ids)
                suffix_embeds = qwen_model.get_input_embeddings()(suffix_ids)

                combined_embeds = torch.cat([prefix_embeds, chess_embeds, suffix_embeds], dim=1)

                chess_mask = torch.ones(
                    chess_embeds.size(0), chess_embeds.size(1),
                    dtype=prefix_mask.dtype, device=device
                )
                combined_attention_mask = torch.cat([prefix_mask, chess_mask, suffix_mask], dim=1)

                outputs = qwen_model(
                    inputs_embeds=combined_embeds, attention_mask=combined_attention_mask
                )

                batch_size = suffix_ids.size(0)
                prefix_len = prefix_ids.size(1)
                chess_len = chess_embeds.size(1)
                labels = torch.full(
                    (batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=device
                )
                labels[:, prefix_len + chess_len:] = suffix_ids

                loss = compute_loss(
                    outputs.logits, labels, combined_attention_mask, answer_start_indices, prefix_len + chess_len
                )
                total_loss += loss.item()
                total_batches += 1

        projector.train()
        return total_loss / max(1, total_batches)

    # Training loop
    print(f"\nStarting training...")
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    accumulated_loss = 0.0
    data_iter = iter(train_dataloader)

    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        optimizer.zero_grad()

        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            chess_input_ids = batch["chess_input_ids"].to(device)
            prefix_ids = batch["prefix_ids"].to(device)
            prefix_mask = batch["prefix_mask"].to(device)
            suffix_ids = batch["suffix_ids"].to(device)
            suffix_mask = batch["suffix_mask"].to(device)
            answer_start_indices = batch["answer_start_indices"].to(device)

            # Get embeddings for each part
            chess_hidden = get_chess_hidden_states(chess_model, chess_input_ids)
            chess_embeds = projector(chess_hidden.to(torch.bfloat16))
            prefix_embeds = qwen_model.get_input_embeddings()(prefix_ids)
            suffix_embeds = qwen_model.get_input_embeddings()(suffix_ids)

            # Concatenate: [prefix, chess, suffix]
            combined_embeds = torch.cat([prefix_embeds, chess_embeds, suffix_embeds], dim=1)

            # Build attention mask
            chess_mask = torch.ones(
                chess_embeds.size(0), chess_embeds.size(1),
                dtype=prefix_mask.dtype, device=device
            )
            combined_attention_mask = torch.cat([prefix_mask, chess_mask, suffix_mask], dim=1)

            outputs = qwen_model(
                inputs_embeds=combined_embeds, attention_mask=combined_attention_mask
            )

            # Build labels: [-100 for prefix, -100 for chess, suffix_ids]
            batch_size = suffix_ids.size(0)
            prefix_len = prefix_ids.size(1)
            chess_len = chess_embeds.size(1)
            labels = torch.full(
                (batch_size, combined_embeds.size(1)), -100, dtype=torch.long, device=device
            )
            labels[:, prefix_len + chess_len:] = suffix_ids

            # answer_start_indices is relative to suffix, so offset = prefix_len + chess_len
            loss = compute_loss(
                outputs.logits, labels, combined_attention_mask, answer_start_indices, prefix_len + chess_len
            )
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(projector.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % LOG_STEPS == 0:
            avg_loss = accumulated_loss / LOG_STEPS
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
            wandb.log({
                "loss": avg_loss,
                "learning_rate": current_lr,
                "step": global_step,
            })
            accumulated_loss = 0.0

        pbar.update(1)

        if global_step % steps_per_epoch == 0:
            epoch = global_step // steps_per_epoch
            val_loss = evaluate()
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch,
                "step": global_step,
            })
            print(f"\nEpoch {epoch} validation loss: {val_loss:.4f}")

        if global_step % SAVE_STEPS == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
            print(f"\nSaving checkpoint to {checkpoint_dir}...")
            projector.save_pretrained(checkpoint_dir)

    pbar.close()

    final_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    print(f"Saving final checkpoint to {final_dir}...")
    projector.save_pretrained(final_dir)
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
