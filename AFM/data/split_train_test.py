#!/usr/bin/env python3
"""
Split jsonl dataset into train and test sets.
Usage: python split_train_test.py <input_file> <output_train> <output_test> --test_size <num>
"""
import json
import argparse
import random
from pathlib import Path


def split_jsonl_dataset(input_file, output_train, output_test, test_size=200, seed=42):
    """
    Split a jsonl file into train and test sets.
    
    Args:
        input_file: Path to input jsonl file
        output_train: Path to output training jsonl file
        output_test: Path to output test jsonl file
        test_size: Number of samples for test set
        seed: Random seed for reproducibility
    """
    # Read all lines
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    print(f"Total samples: {total_samples}")
    
    if test_size >= total_samples:
        raise ValueError(f"Test size ({test_size}) must be less than total samples ({total_samples})")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Split indices
    test_indices = set(indices[:test_size])
    train_indices = set(indices[test_size:])
    
    # Write train and test files
    train_samples = []
    test_samples = []
    
    for idx, line in enumerate(lines):
        if idx in test_indices:
            test_samples.append(line)
        else:
            train_samples.append(line)
    
    # Create output directories if they don't exist
    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(output_test).parent.mkdir(parents=True, exist_ok=True)
    
    # Write train file
    print(f"Writing {len(train_samples)} training samples to {output_train}...")
    with open(output_train, 'w', encoding='utf-8') as f:
        f.writelines(train_samples)
    
    # Write test file
    print(f"Writing {len(test_samples)} test samples to {output_test}...")
    with open(output_test, 'w', encoding='utf-8') as f:
        f.writelines(test_samples)
    
    # Verify the split
    print("\n=== Split Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Train ratio: {len(train_samples)/total_samples*100:.2f}%")
    print(f"Test ratio: {len(test_samples)/total_samples*100:.2f}%")
    
    # Validate files can be parsed
    print("\n=== Validating output files ===")
    with open(output_train, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Train line {i} has invalid JSON: {e}")
    
    with open(output_test, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Test line {i} has invalid JSON: {e}")
    
    print("\n✓ Split completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Split jsonl dataset into train and test sets')
    parser.add_argument('input_file', type=str, help='Input jsonl file path')
    parser.add_argument('output_train', type=str, help='Output training jsonl file path')
    parser.add_argument('output_test', type=str, help='Output test jsonl file path')
    parser.add_argument('--test_size', type=int, default=200, help='Number of test samples (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_jsonl_dataset(
        args.input_file,
        args.output_train,
        args.output_test,
        test_size=args.test_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

