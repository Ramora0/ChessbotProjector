"""Display random Q&A pairs from the chess questions dataset."""

import argparse
import random
from datasets import load_from_disk

DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/easy_chess_questions"


def main():
    parser = argparse.ArgumentParser(description="Display random Q&A pairs from chess dataset")
    parser.add_argument("--category", "-c", type=str, default=None,
                        help="Filter by category (e.g., best_move, check, hanging)")
    args = parser.parse_args()

    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"Loaded {len(dataset):,} questions\n")

    # Get available categories
    categories = set(dataset["category"])
    print(f"Available categories: {', '.join(sorted(categories))}\n")

    # Filter by category if specified
    if args.category:
        if args.category not in categories:
            print(f"Error: '{args.category}' is not a valid category.")
            print(f"Choose from: {', '.join(sorted(categories))}")
            return
        dataset = dataset.filter(lambda x: x["category"] == args.category)
        print(f"Filtered to {len(dataset):,} questions in category '{args.category}'\n")

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    idx_iter = iter(indices)

    print("Press Enter for next question, 'q' to quit\n")
    print("-" * 60)

    while True:
        try:
            idx = next(idx_iter)
        except StopIteration:
            random.shuffle(indices)
            idx_iter = iter(indices)
            idx = next(idx_iter)
            print("\n[Reshuffled - showing questions again]\n")

        example = dataset[idx]
        print(f"\nFEN: {example['fen']}")
        print(f"\nQuestion: {example['question']}")
        print(f"\nAnswer: {example['answer']}")
        print(f"\nCategory: {example['category']}")
        print("\n" + "-" * 60)

        user_input = input()
        if user_input.strip().lower() == 'q':
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
