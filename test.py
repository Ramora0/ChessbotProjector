"""Display random Q&A pairs from the chess questions dataset."""

import random
from datasets import load_from_disk

DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/easy_chess_questions"


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"Loaded {len(dataset):,} questions\n")

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
