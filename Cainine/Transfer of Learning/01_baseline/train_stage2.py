"""
Canine ECG - S vs V Sub-Classifier Training
Stage 2 of Hierarchical Canine ECG Classification

Purpose: Train arrhythmia sub-classifier (S vs V) for canine ECG
Input: Only S and V samples from canine data
Output: Stage 2 model for hierarchical classification

Note: Canine data has severe S class imbalance (0.65% S vs 6.12% V)
      Heavy augmentation needed for S class
"""

import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Add, Flatten, Activation
from tensorflow.keras.callbacks import (
    LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc, roc_auc_score
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Get script directory for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

    # Paths (relative)
    DATA_FILE = os.path.join(PROJECT_ROOT, "Cainine", "DataPreProcessing_TEST", "PreProcessedData", "canine_train.csv")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

    # Data
    INPUT_LEN = 187  # Padded to match MIT-BIH model input
    LABEL_COL = -1   # Last column

    # Task configuration
    KEEP_CLASSES = [1, 2]  # S and V only (exclude Normal)

    # Training
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    VAL_SPLIT = 0.15
    SEED = 1337

    # Data augmentation (HEAVY for S class)
    AUGMENT_S_CLASS = True
    S_AUGMENTATION_FACTOR = 10  # Multiply S samples by 10 to balance with V

    # Class weights (in case augmentation isn't enough)
    USE_CLASS_WEIGHTS = True


def set_seed(seed=Config.SEED):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================================
# DATA LOADING
# ============================================================================

def adapt_length(X, target_len=187):
    """
    Pad canine data from 186 to 187 features to match MIT-BIH model input

    Args:
        X: Input array (N, L) or (N, L, 1)
        target_len: Target length (default 187)

    Returns:
        Padded array (N, target_len, 1)
    """
    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    N, L, C = X.shape

    if L == target_len:
        return X

    # Pad with zeros at the end
    if L < target_len:
        pad_width = ((0, 0), (0, target_len - L), (0, 0))
        return np.pad(X, pad_width, mode='constant', constant_values=0)

    # Crop if longer (shouldn't happen for canine data)
    return X[:, :target_len, :]


def load_canine_arrhythmia_data(config):
    """
    Load canine data and filter to S and V classes only

    Returns:
        X: ECG signals (N, 186) - will be padded to 187 later
        y: Labels - S=0, V=1
    """
    print("="*70)
    print("LOADING CANINE ARRHYTHMIA DATA (S and V only)")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {config.DATA_FILE}")
    df = pd.read_csv(config.DATA_FILE, header=None, low_memory=False)

    print(f"Total loaded: {len(df):,} samples")

    # Separate features and labels
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_orig = df.iloc[:, -1].to_numpy(dtype=np.int64)

    print(f"\nOriginal label distribution:")
    for label, count in zip(*np.unique(y_orig, return_counts=True)):
        label_name = {0: 'N', 1: 'S', 2: 'V'}.get(int(label), f'Unknown({label})')
        print(f"  {label_name}: {count:,} samples ({100*count/len(y_orig):.2f}%)")

    # Filter to S and V only
    mask = np.isin(y_orig, config.KEEP_CLASSES)
    X = X[mask]
    y_orig = y_orig[mask]

    # Binary remap: S=0, V=1
    y = (y_orig == 2).astype(np.int64)

    print(f"\nAfter filtering to arrhythmia only (S and V):")
    print(f"  Total samples: {len(y):,}")
    print(f"  S (0): {np.sum(y == 0):,} samples ({100*np.mean(y==0):.2f}%)")
    print(f"  V (1): {np.sum(y == 1):,} samples ({100*np.mean(y==1):.2f}%)")

    s_to_v_ratio = np.sum(y == 0) / np.sum(y == 1)
    print(f"\n  S:V ratio = 1:{1/s_to_v_ratio:.1f} (SEVERE IMBALANCE!)")

    return X, y


def augment_s_class(X, y, factor=10):
    """
    Heavily augment S class (minority) to balance with V

    Uses simple augmentation (noise, scaling) to create synthetic samples
    """
    print(f"\nAugmenting S class (factor={factor})...")

    s_indices = np.where(y == 0)[0]
    v_indices = np.where(y == 1)[0]

    print(f"  Original S samples: {len(s_indices):,}")
    print(f"  Original V samples: {len(v_indices):,}")

    # Generate augmented S samples
    augmented_s = []
    augmented_labels = []

    for _ in range(factor):
        for idx in s_indices:
            beat = X[idx].copy()

            # Add small random noise
            noise = np.random.normal(0, 0.02, beat.shape)
            beat = beat + noise

            # Random amplitude scaling (0.95 to 1.05)
            scale = np.random.uniform(0.95, 1.05)
            beat = beat * scale

            augmented_s.append(beat)
            augmented_labels.append(0)  # S class

    # Combine original and augmented
    X_augmented = np.vstack([X, np.array(augmented_s)])
    y_augmented = np.hstack([y, np.array(augmented_labels)])

    print(f"\nAfter augmentation:")
    print(f"  Total samples: {len(y_augmented):,}")
    print(f"  S (0): {np.sum(y_augmented == 0):,} samples ({100*np.mean(y_augmented==0):.2f}%)")
    print(f"  V (1): {np.sum(y_augmented == 1):,} samples ({100*np.mean(y_augmented==1):.2f}%)")

    return X_augmented, y_augmented


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_s_vs_v_model(input_len=187):
    """
    Build S vs V sub-classifier
    Same architecture as MIT-BIH sub-classifier for consistency

    Args:
        input_len: Input signal length (default 187, padded from canine 186)
    """
    print("\n" + "="*70)
    print("BUILDING S vs V SUB-CLASSIFIER MODEL")
    print("="*70)

    inp = Input(shape=(input_len, 1), name='input')

    # Initial conv
    x = Conv1D(filters=32, kernel_size=5, strides=1)(inp)

    # Residual Block 1
    conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    act1 = Activation("relu")(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(act1)
    skip1 = Add()([conv2, x])
    act2 = Activation("relu")(skip1)
    x = MaxPooling1D(pool_size=5, strides=2)(act2)

    # Residual Block 2
    conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    act1 = Activation("relu")(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(act1)
    skip2 = Add()([conv2, x])
    act2 = Activation("relu")(skip2)
    x = MaxPooling1D(pool_size=5, strides=2)(act2)

    # Residual Block 3
    conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    act1 = Activation("relu")(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(act1)
    skip3 = Add()([conv2, x])
    act2 = Activation("relu")(skip3)
    x = MaxPooling1D(pool_size=5, strides=2)(act2)

    # Residual Block 4
    conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    act1 = Activation("relu")(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(act1)
    skip4 = Add()([conv2, x])
    act2 = Activation("relu")(skip4)
    x = MaxPooling1D(pool_size=5, strides=2)(act2)

    # Residual Block 5
    conv1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    act1 = Activation("relu")(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(act1)
    skip5 = Add()([conv2, x])
    act2 = Activation("relu")(skip5)
    x = MaxPooling1D(pool_size=5, strides=2)(act2)

    # Dense layers
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32)(x)

    # Binary output: S vs V
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inp, outputs=output, name='canine_s_vs_v')

    print("\nModel architecture:")
    model.summary(line_length=100)

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config):
    """
    Main training function
    """
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load data
    X, y = load_canine_arrhythmia_data(config)

    # Augment S class
    if config.AUGMENT_S_CLASS:
        X, y = augment_s_class(X, y, factor=config.S_AUGMENTATION_FACTOR)

    # Adapt length from 186 to 187 to match MIT-BIH model input
    X = adapt_length(X, target_len=config.INPUT_LEN)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VAL_SPLIT, random_state=config.SEED,
        stratify=y
    )

    print(f"\nTrain/Val Split:")
    print(f"  Training: {len(y_train):,} samples")
    print(f"    S: {np.sum(y_train==0):,} ({100*np.mean(y_train==0):.1f}%)")
    print(f"    V: {np.sum(y_train==1):,} ({100*np.mean(y_train==1):.1f}%)")
    print(f"  Validation: {len(y_val):,} samples")
    print(f"    S: {np.sum(y_val==0):,} ({100*np.mean(y_val==0):.1f}%)")
    print(f"    V: {np.sum(y_val==1):,} ({100*np.mean(y_val==1):.1f}%)")

    # Build model
    model = build_s_vs_v_model(input_len=config.INPUT_LEN)

    # Compile
    optimizer = Adam(learning_rate=config.LEARNING_RATE)

    # Class weights (additional to augmentation)
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        s_weight = len(y_train) / (2 * np.sum(y_train == 0))
        v_weight = len(y_train) / (2 * np.sum(y_train == 1))
        class_weights = {0: s_weight, 1: v_weight}
        print(f"\nClass weights: S={s_weight:.2f}, V={v_weight:.2f}")

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config.OUTPUT_DIR, 'best_canine_subclass_s_v.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    print("\n" + "="*70)
    print("TRAINING CANINE S vs V SUB-CLASSIFIER")
    print("="*70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION ON VALIDATION SET")
    print("="*70)

    y_val_prob = model.predict(X_val, verbose=0).reshape(-1)
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred,
                                target_names=['Supraventricular (S)', 'Ventricular (V)'],
                                digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print("\nConfusion Matrix:")
    print("             Predicted")
    print("               S      V")
    print(f"Actual S    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       V    {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Metrics
    accuracy = (y_val_pred == y_val).mean()
    roc_auc = roc_auc_score(y_val, y_val_prob)

    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    pr_auc = auc(recall, precision)

    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")

    # Save predictions
    np.save(os.path.join(config.OUTPUT_DIR, 'val_probs.npy'), y_val_prob)
    np.save(os.path.join(config.OUTPUT_DIR, 'val_labels.npy'), y_val)

    print(f"\nModel saved to: {config.OUTPUT_DIR}/best_canine_subclass_s_v.h5")
    print("\n" + "="*70)
    print("TRAINING COMPLETE - CANINE STAGE 2 MODEL READY!")
    print("="*70)

    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Train canine S vs V sub-classifier
    """
    print("\n" + "="*70)
    print("CANINE ECG - S vs V SUB-CLASSIFIER TRAINING")
    print("Stage 2 of Hierarchical Classification")
    print("="*70)
    print("\nNote: Canine data has severe S class imbalance")
    print("      Using heavy augmentation + class weights")
    print("="*70)

    config = Config()
    model, history = train_model(config)

if __name__ == "__main__":
    main()
