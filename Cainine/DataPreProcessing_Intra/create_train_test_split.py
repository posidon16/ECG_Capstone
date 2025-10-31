"""
Create Train/Test Split for Canine ECG Data

Purpose: Split the preprocessed canine data into separate train and test sets
for proper evaluation of the hierarchical classification system.

This creates a stratified split to ensure proper class representation in both sets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: Combined preprocessed data (relative path)
INPUT_FILE = os.path.join(SCRIPT_DIR, "PreProcessedData", "canine_all_enhanced.csv")

# Output: Separate train and test files
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "PreProcessedData")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "canine_train.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "canine_test.csv")

# Split configuration
TEST_SIZE = 0.15  # 15% for testing, 85% for training
RANDOM_STATE = 1337  # For reproducibility

# ============================================================================
# LOAD AND SPLIT
# ============================================================================

def create_train_test_split():
    """
    Load combined canine data and create stratified train/test split
    """
    print("="*70)
    print("CANINE ECG - TRAIN/TEST SPLIT CREATION")
    print("="*70)

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print("Please run enhanced_preprocessing.py first to create the combined dataset.")
        return

    print(f"\nLoading data from: {INPUT_FILE}")

    # Load data (headerless CSV: features + label in last column)
    df = pd.read_csv(INPUT_FILE, header=None, low_memory=False)
    print(f"Loaded: {len(df):,} total samples")

    # Separate features and labels
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)

    # Show class distribution
    print("\nClass distribution in full dataset:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        label_name = {0: 'N', 1: 'S', 2: 'V'}.get(int(label), f'Unknown({label})')
        print(f"  {label_name}: {count:,} samples ({100*count/len(y):.2f}%)")

    # Create stratified split
    print(f"\nCreating stratified split (test_size={TEST_SIZE*100:.1f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Stratified to maintain class proportions
    )

    print(f"\nTrain set: {len(y_train):,} samples")
    for label, count in zip(*np.unique(y_train, return_counts=True)):
        label_name = {0: 'N', 1: 'S', 2: 'V'}.get(int(label), f'Unknown({label})')
        print(f"  {label_name}: {count:,} samples ({100*count/len(y_train):.2f}%)")

    print(f"\nTest set: {len(y_test):,} samples")
    for label, count in zip(*np.unique(y_test, return_counts=True)):
        label_name = {0: 'N', 1: 'S', 2: 'V'}.get(int(label), f'Unknown({label})')
        print(f"  {label_name}: {count:,} samples ({100*count/len(y_test):.2f}%)")

    # Save train set
    print(f"\nSaving train set to: {TRAIN_FILE}")
    train_data = np.column_stack([X_train, y_train])
    pd.DataFrame(train_data).to_csv(TRAIN_FILE, index=False, header=False)
    print(f"✓ Saved {len(y_train):,} training samples")

    # Save test set
    print(f"\nSaving test set to: {TEST_FILE}")
    test_data = np.column_stack([X_test, y_test])
    pd.DataFrame(test_data).to_csv(TEST_FILE, index=False, header=False)
    print(f"✓ Saved {len(y_test):,} test samples")

    print("\n" + "="*70)
    print("SPLIT CREATION COMPLETE")
    print("="*70)

    print("\nNext steps:")
    print("1. Update transfer learning scripts to use canine_train.csv for training")
    print("2. Use canine_test.csv for final evaluation of hierarchical system")
    print("3. Keep test set completely separate - no peeking during development!")

    print("\nFile summary:")
    print(f"  Original: {INPUT_FILE}")
    print(f"  Train:    {TRAIN_FILE} ({100*(1-TEST_SIZE):.1f}% of data)")
    print(f"  Test:     {TEST_FILE} ({100*TEST_SIZE:.1f}% of data)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    create_train_test_split()
