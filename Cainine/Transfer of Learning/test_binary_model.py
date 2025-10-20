"""
Test script for binary transfer learning model
Tests the canine_finetuned_binary.h5 model on enhanced preprocessed data
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Configuration (relative paths)
MODEL_PATH = os.path.join(SCRIPT_DIR, "canine_tl_out_v2", "canine_finetuned_binary.h5")
DATA_PATH = os.path.join(PROJECT_ROOT, "Cainine", "DataPreProcessing", "PreProcessedData", "canine_test.csv")

def load_and_remap_data(csv_path: str, include_s_beats: bool = True):
    """
    Load data and apply the same binary remapping as training
    """
    # Load headerless CSV
    df = pd.read_csv(csv_path, header=None, low_memory=False)

    # Separate features and labels
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_orig = df.iloc[:, -1].to_numpy(dtype=np.int64)

    print(f"[data] Loaded {len(X)} samples")
    print(f"[data] Original label distribution: {dict(zip(*np.unique(y_orig, return_counts=True)))}")

    # Apply same binary remapping as training
    if include_s_beats:
        # N=0 stays 0, S=1 and V=2 become 1 (arrhythmia)
        normal_classes = [0]  # N
        arrhythmia_classes = [1, 2]  # S, V
        keep_classes = normal_classes + arrhythmia_classes
    else:
        # Original approach: only N vs V
        keep_classes = [0, 2]  # N, V
        arrhythmia_classes = [2]

    # Filter to keep only relevant classes
    mask = np.isin(y_orig, keep_classes)
    X_filtered = X[mask]
    y_filtered = y_orig[mask]

    # Binary remap: Normal=0, Arrhythmia=1
    if include_s_beats:
        y_binary = np.isin(y_filtered, arrhythmia_classes).astype(np.int64)
    else:
        y_binary = (y_filtered == 2).astype(np.int64)

    print(f"[remap] Filtered to {len(X_filtered)} samples")
    print(f"[remap] Original classes kept: {dict(zip(*np.unique(y_filtered, return_counts=True)))}")
    print(f"[remap] Binary mapping: {dict(zip(*np.unique(y_binary, return_counts=True)))}")

    return X_filtered, y_binary, y_filtered

def prepare_input(X: np.ndarray) -> np.ndarray:
    """Prepare input for model (add channel dimension if needed)"""
    if X.ndim == 2:
        X = X[:, :, np.newaxis]  # Add channel dimension
    return X.astype(np.float32)

def evaluate_binary_model(model_path: str, data_path: str, include_s_beats: bool = True, limit: int = None):
    """
    Evaluate binary transfer learning model
    """
    print("="*60)
    print("BINARY TRANSFER LEARNING MODEL EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Include S-beats as arrhythmia: {include_s_beats}")

    # Load model
    print("\n[step] Loading model...")
    model = load_model(model_path, compile=False)
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Load and prepare data
    print("\n[step] Loading and preparing data...")
    X, y_binary, y_orig = load_and_remap_data(data_path, include_s_beats)

    if limit is not None:
        X = X[:limit]
        y_binary = y_binary[:limit]
        y_orig = y_orig[:limit]
        print(f"[limit] Limited to first {limit} samples")

    # Prepare input
    X_prepared = prepare_input(X)
    print(f"[input] Prepared shape: {X_prepared.shape}")

    # Make predictions
    print("\n[step] Making predictions...")
    y_prob = model.predict(X_prepared, verbose=1)

    # Handle output shape
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)  # Binary sigmoid output
    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]  # Softmax output, take positive class

    y_pred = (y_prob >= 0.5).astype(int)

    # Evaluate
    print("\n" + "="*50)
    print("BINARY CLASSIFICATION RESULTS")
    print("="*50)

    # Basic metrics
    accuracy = (y_pred == y_binary).mean()
    print(f"Accuracy: {accuracy:.3f}")

    # Class distribution
    print(f"\nTrue class distribution: {dict(zip(*np.unique(y_binary, return_counts=True)))}")
    print(f"Pred class distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")

    # Confusion matrix
    cm = confusion_matrix(y_binary, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print("    N    A")  # Normal, Arrhythmia
    for i, row in enumerate(cm):
        label = "N" if i == 0 else "A"
        print(f"{label} {row}")

    # Classification report
    class_names = ["Normal", "Arrhythmia"]
    print(f"\nClassification Report:")
    print(classification_report(y_binary, y_pred, target_names=class_names, digits=3))

    # PR curve analysis
    precision, recall, thresholds = precision_recall_curve(y_binary, y_prob)
    pr_auc = auc(recall, precision)

    # Find best F1 threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"\nThreshold Analysis:")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"Best F1 threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {f1_scores[best_idx]:.3f}")
    print(f"At best threshold - Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}")

    # Show some examples of each class
    print(f"\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)

    # Normal examples
    normal_indices = np.where(y_binary == 0)[0]
    if len(normal_indices) > 0:
        print("\nNormal beats (sample):")
        for i in normal_indices[:5]:
            prob = y_prob[i]
            pred_label = "Normal" if y_pred[i] == 0 else "Arrhythmia"
            orig_label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
            orig_label = orig_label_map.get(y_orig[i], "?")
            print(f"  Row {i} | orig={orig_label} | pred={pred_label} | prob={prob:.3f}")

    # Arrhythmia examples
    arrhythmia_indices = np.where(y_binary == 1)[0]
    if len(arrhythmia_indices) > 0:
        print("\nArrhythmia beats (sample):")
        for i in arrhythmia_indices[:10]:
            prob = y_prob[i]
            pred_label = "Normal" if y_pred[i] == 0 else "Arrhythmia"
            orig_label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
            orig_label = orig_label_map.get(y_orig[i], "?")
            print(f"  Row {i} | orig={orig_label} | pred={pred_label} | prob={prob:.3f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'best_f1': f1_scores[best_idx]
    }

def main():
    """Main evaluation"""
    try:
        results = evaluate_binary_model(
            MODEL_PATH,
            DATA_PATH,
            include_s_beats=True,  # Match training configuration
            limit=None  # Set to e.g. 10000 for faster testing
        )

        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']:.1%}")
        print(f"PR-AUC: {results['pr_auc']:.3f}")
        print(f"Best F1 Score: {results['best_f1']:.3f}")
        print(f"Optimal Threshold: {results['best_threshold']:.3f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()