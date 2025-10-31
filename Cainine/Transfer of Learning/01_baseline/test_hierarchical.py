"""
Canine ECG - Hierarchical Classification Testing
2-Stage System: Stage 1 (N vs Arrhythmia) + Stage 2 (S vs V)

Purpose: Test complete hierarchical canine ECG classification system
Input: Canine test data
Models:
  - Stage 1: canine_finetuned_binary.h5 (N vs Arrhythmia)
  - Stage 2: best_canine_subclass_s_v.h5 (S vs V)

Output: Comprehensive performance metrics and confusion matrices
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Get script directory for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRANSFER_LEARNING_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(TRANSFER_LEARNING_DIR))

    # Paths (relative)
    # Use dedicated test set created by create_train_test_split.py
    TEST_DATA = os.path.join(PROJECT_ROOT, "Cainine", "DataPreProcessing_Inter", "PreProcessedData", "canine_test.csv")

    # Models
    STAGE1_MODEL = os.path.join(SCRIPT_DIR, "models", "canine_binary_original.h5")
    STAGE2_MODEL = os.path.join(TRANSFER_LEARNING_DIR, "stage2_models", "models", "best_canine_subclass_s_v.h5")

    # Data
    INPUT_LEN = 187  # Padded to match MIT-BIH model input

    # Thresholds
    STAGE1_THRESHOLD = 0.0101  # N vs Arrhythmia (optimized for 80% recall)
    STAGE2_THRESHOLD = 0.5  # S vs V

    SEED = 1337


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


def load_canine_test_data(config):
    """
    Load canine test data with proper label mapping

    Returns:
        X: ECG signals (N, 187, 1) - padded from 186 to match MIT-BIH model
        y_true_3class: True labels in 3-class format (N=0, S=1, V=2)
        y_true_binary: True labels in binary format (N=0, Arrhythmia=1)
    """
    print("="*70)
    print("LOADING CANINE TEST DATA")
    print("="*70)

    # Check if test file exists
    if not os.path.exists(config.TEST_DATA):
        print(f"\nERROR: Test file not found: {config.TEST_DATA}")
        print("\nPlease run create_train_test_split.py first to create the test set:")
        print("  python Cainine/create_train_test_split.py")
        raise FileNotFoundError(f"Test file not found: {config.TEST_DATA}")

    print(f"\nLoading data from: {config.TEST_DATA}")
    df = pd.read_csv(config.TEST_DATA, header=None, low_memory=False)

    print(f"Test set size: {len(df):,} samples")

    # Separate features and labels
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_orig = df.iloc[:, -1].to_numpy(dtype=np.int64)

    print(f"\nLabel distribution:")
    for label, count in zip(*np.unique(y_orig, return_counts=True)):
        label_name = {0: 'N', 1: 'S', 2: 'V'}.get(int(label), f'Unknown({label})')
        print(f"  {label_name}: {count:,} samples ({100*count/len(y_orig):.2f}%)")

    # Create binary labels (for Stage 1 evaluation)
    y_true_binary = np.isin(y_orig, [1, 2]).astype(np.int64)  # S or V = 1

    # Use original labels as 3-class (for final evaluation)
    y_true_3class = y_orig.copy()

    # Adapt length from 186 to 187 to match MIT-BIH model input
    X = adapt_length(X, target_len=config.INPUT_LEN)

    print(f"\nReady for testing: {len(X):,} samples with shape {X.shape}")

    return X, y_true_3class, y_true_binary


# ============================================================================
# HIERARCHICAL CLASSIFIER
# ============================================================================

class HierarchicalCanineClassifier:
    """
    2-Stage Hierarchical Classifier for Canine ECG

    Stage 1: Normal vs Arrhythmia (binary)
    Stage 2: S vs V (binary, applied only to arrhythmia cases)

    Final output: 3-class (N=0, S=1, V=2)
    """

    def __init__(self, stage1_path, stage2_path,
                 stage1_threshold=0.5, stage2_threshold=0.5):
        """
        Load both models

        Args:
            stage1_path: Path to Stage 1 model (N vs Arrhythmia)
            stage2_path: Path to Stage 2 model (S vs V)
            stage1_threshold: Threshold for Stage 1 (default 0.5)
            stage2_threshold: Threshold for Stage 2 (default 0.5)
        """
        print("\nLoading hierarchical models...")

        # Stage 1: N vs Arrhythmia
        print(f"  Stage 1: {stage1_path}")
        self.stage1_model = load_model(stage1_path)
        self.stage1_threshold = stage1_threshold

        # Stage 2: S vs V
        print(f"  Stage 2: {stage2_path}")
        self.stage2_model = load_model(stage2_path)
        self.stage2_threshold = stage2_threshold

        print("Models loaded successfully")

    def predict_batch(self, X, verbose=True):
        """
        Predict 3-class labels using hierarchical approach

        Args:
            X: Input data (N, 187, 1)
            verbose: Print progress

        Returns:
            y_pred: Predicted labels (N=0, S=1, V=2)
            stage1_probs: Stage 1 probabilities (arrhythmia)
            stage2_probs: Stage 2 probabilities (V | arrhythmia)
        """
        N = len(X)

        # Stage 1: N vs Arrhythmia
        if verbose:
            print("\n  Stage 1: Classifying Normal vs Arrhythmia...")

        p_arrhythmia = self.stage1_model.predict(X, verbose=0).reshape(-1)
        arrhythmia_mask = p_arrhythmia >= self.stage1_threshold

        n_normal = np.sum(~arrhythmia_mask)
        n_arrhythmia = np.sum(arrhythmia_mask)

        if verbose:
            print(f"    Classified as Normal: {n_normal:,} ({100*n_normal/N:.2f}%)")
            print(f"    Classified as Arrhythmia: {n_arrhythmia:,} ({100*n_arrhythmia/N:.2f}%)")

        # Initialize predictions (all as Normal)
        y_pred = np.zeros(N, dtype=np.int64)

        # Initialize Stage 2 probabilities (NaN for Normal cases)
        stage2_probs = np.full(N, np.nan, dtype=np.float32)

        # Stage 2: S vs V (only for arrhythmia cases)
        if n_arrhythmia > 0:
            if verbose:
                print(f"\n  Stage 2: Classifying {n_arrhythmia:,} arrhythmia cases (S vs V)...")

            X_arrhythmia = X[arrhythmia_mask]
            p_v = self.stage2_model.predict(X_arrhythmia, verbose=0).reshape(-1)

            # Store Stage 2 probabilities
            stage2_probs[arrhythmia_mask] = p_v

            # Classify as S or V
            v_mask = p_v >= self.stage2_threshold
            s_mask = ~v_mask

            n_s = np.sum(s_mask)
            n_v = np.sum(v_mask)

            if verbose:
                print(f"    Classified as S: {n_s:,} ({100*n_s/n_arrhythmia:.2f}% of arrhythmia)")
                print(f"    Classified as V: {n_v:,} ({100*n_v/n_arrhythmia:.2f}% of arrhythmia)")

            # Update predictions
            arrhythmia_indices = np.where(arrhythmia_mask)[0]
            y_pred[arrhythmia_indices[s_mask]] = 1  # S
            y_pred[arrhythmia_indices[v_mask]] = 2  # V

        return y_pred, p_arrhythmia, stage2_probs


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, labels, title, filename, output_dir='.'):
    """
    Plot and save confusion matrix

    Args:
        cm: Confusion matrix array
        labels: List of class labels
        title: Plot title
        filename: Output filename
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()


def plot_all_confusion_matrices(cm_stage1, cm_stage2, cm_final, output_dir='.'):
    """
    Generate all confusion matrix plots for hierarchical system

    Args:
        cm_stage1: Stage 1 confusion matrix (N vs Arrhythmia)
        cm_stage2: Stage 2 confusion matrix (S vs V)
        cm_final: Final 3-class confusion matrix
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
    print("="*70)

    # Stage 1: N vs Arrhythmia
    print("\n1. Stage 1 Confusion Matrix (Normal vs Arrhythmia)...")
    plot_confusion_matrix(
        cm_stage1,
        labels=['Normal', 'Arrhythmia'],
        title='Stage 1: Binary Classification (Normal vs Arrhythmia)',
        filename='canine_stage1_confusion_matrix.png',
        output_dir=output_dir
    )

    # Stage 2: S vs V
    print("\n2. Stage 2 Confusion Matrix (S vs V)...")
    plot_confusion_matrix(
        cm_stage2,
        labels=['S (Supraventricular)', 'V (Ventricular)'],
        title='Stage 2: Arrhythmia Sub-Classification (S vs V)',
        filename='canine_stage2_confusion_matrix.png',
        output_dir=output_dir
    )

    # Final: 3-class
    print("\n3. Final 3-Class Confusion Matrix (N, S, V)...")
    plot_confusion_matrix(
        cm_final,
        labels=['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)'],
        title='Final: Hierarchical 3-Class Classification (Canine ECG)',
        filename='canine_final_confusion_matrix.png',
        output_dir=output_dir
    )

    print("\nAll confusion matrices saved successfully!")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_hierarchical_system(classifier, X_test, y_true_3class, y_true_binary, output_dir='.'):
    """
    Comprehensive evaluation of hierarchical system

    Args:
        classifier: HierarchicalCanineClassifier instance
        X_test: Test data
        y_true_3class: True 3-class labels (N=0, S=1, V=2)
        y_true_binary: True binary labels (N=0, Arrhythmia=1)
    """
    print("\n" + "="*70)
    print("HIERARCHICAL SYSTEM EVALUATION")
    print("="*70)

    # Get predictions
    y_pred_3class, stage1_probs, stage2_probs = classifier.predict_batch(X_test)

    # ========================================================================
    # Stage 1 Evaluation (N vs Arrhythmia)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: Normal vs Arrhythmia Classification")
    print("="*70)

    y_pred_binary = (stage1_probs >= classifier.stage1_threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(
        y_true_binary, y_pred_binary,
        target_names=['Normal (N)', 'Arrhythmia (S+V)'],
        digits=4
    ))

    # Confusion matrix
    cm_stage1 = confusion_matrix(y_true_binary, y_pred_binary)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 N      Arrhythmia")
    print(f"Actual N       {cm_stage1[0,0]:6d}  {cm_stage1[0,1]:6d}")
    print(f"       Arrhyth {cm_stage1[1,0]:6d}  {cm_stage1[1,1]:6d}")

    # Metrics
    acc_stage1 = accuracy_score(y_true_binary, y_pred_binary)
    roc_auc_stage1 = roc_auc_score(y_true_binary, stage1_probs)

    print(f"\nStage 1 Metrics:")
    print(f"  Accuracy: {acc_stage1:.4f}")
    print(f"  ROC-AUC: {roc_auc_stage1:.4f}")

    # ========================================================================
    # Stage 2 Evaluation (S vs V) - Only on arrhythmia cases
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Arrhythmia Sub-classification (S vs V)")
    print("="*70)

    # Filter to only true arrhythmia cases THAT REACHED STAGE 2
    # (i.e., correctly identified as arrhythmia in Stage 1)
    arrhythmia_mask = y_true_binary == 1
    y_true_arrhythmia = y_true_3class[arrhythmia_mask]  # Should be 1 or 2
    y_pred_arrhythmia = y_pred_3class[arrhythmia_mask]  # Final predictions for true arrhythmias
    stage2_probs_arrhythmia = stage2_probs[arrhythmia_mask]

    # Filter out NaN values (cases not sent to Stage 2)
    # This happens when Stage 1 misclassifies arrhythmia as Normal
    valid_mask = ~np.isnan(stage2_probs_arrhythmia)
    y_true_stage2_valid = y_true_arrhythmia[valid_mask]
    y_pred_stage2_valid = y_pred_arrhythmia[valid_mask]
    stage2_probs_valid = stage2_probs_arrhythmia[valid_mask]

    # Binary labels for Stage 2: S=0, V=1
    # ONLY evaluate on cases that actually reached Stage 2
    y_true_stage2_binary = (y_true_stage2_valid == 2).astype(int)  # V=1, S=0
    y_pred_stage2_binary = (y_pred_stage2_valid == 2).astype(int)  # V=1, S=0

    print(f"\nEvaluating on {len(y_true_arrhythmia):,} true arrhythmia cases...")
    if not valid_mask.all():
        n_missing = np.sum(~valid_mask)
        print(f"  Note: {n_missing:,} arrhythmia cases were misclassified as Normal in Stage 1")
        print(f"  Stage 2 metrics calculated on {len(y_true_stage2_valid):,} cases that reached Stage 2")

    print("\nClassification Report (on cases that reached Stage 2):")
    print(classification_report(
        y_true_stage2_binary, y_pred_stage2_binary,
        target_names=['Supraventricular (S)', 'Ventricular (V)'],
        digits=4
    ))

    # Confusion matrix (only on cases that reached Stage 2)
    cm_stage2 = confusion_matrix(y_true_stage2_binary, y_pred_stage2_binary)
    print("\nConfusion Matrix (among cases that reached Stage 2):")
    print("             Predicted")
    print("               S      V")
    print(f"Actual S    {cm_stage2[0,0]:5d}  {cm_stage2[0,1]:5d}")
    print(f"       V    {cm_stage2[1,0]:5d}  {cm_stage2[1,1]:5d}")

    # Metrics
    acc_stage2 = accuracy_score(y_true_stage2_binary, y_pred_stage2_binary) if len(y_true_stage2_binary) > 0 else 0.0
    # FIX #1: Pass probabilities directly to ROC-AUC, not thresholded boolean
    roc_auc_stage2 = roc_auc_score(y_true_stage2_binary, stage2_probs_valid) if len(y_true_stage2_valid) > 0 else 0.0

    print(f"\nStage 2 Metrics:")
    print(f"  Accuracy: {acc_stage2:.4f}")
    print(f"  ROC-AUC: {roc_auc_stage2:.4f}")

    # ========================================================================
    # Final 3-Class Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL: 3-Class Hierarchical Classification (N, S, V)")
    print("="*70)

    print("\nClassification Report:")
    print(classification_report(
        y_true_3class, y_pred_3class,
        target_names=['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)'],
        digits=4
    ))

    # Confusion matrix
    cm_final = confusion_matrix(y_true_3class, y_pred_3class)
    print("\nConfusion Matrix:")
    print("             Predicted")
    print("               N      S      V")
    print(f"Actual N    {cm_final[0,0]:5d}  {cm_final[0,1]:5d}  {cm_final[0,2]:5d}")
    print(f"       S    {cm_final[1,0]:5d}  {cm_final[1,1]:5d}  {cm_final[1,2]:5d}")
    print(f"       V    {cm_final[2,0]:5d}  {cm_final[2,1]:5d}  {cm_final[2,2]:5d}")

    # Overall metrics
    acc_final = accuracy_score(y_true_3class, y_pred_3class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_3class, y_pred_3class, average='macro'
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {acc_final:.4f}")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall: {recall:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")

    # ========================================================================
    # Error Attribution Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("ERROR ATTRIBUTION ANALYSIS")
    print("="*70)

    # Total errors
    n_errors = np.sum(y_pred_3class != y_true_3class)
    print(f"\nTotal errors: {n_errors:,} / {len(y_true_3class):,} ({100*n_errors/len(y_true_3class):.2f}%)")

    # Stage 1 errors (misclassifying N vs Arrhythmia)
    stage1_errors = (y_pred_binary != y_true_binary)
    n_stage1_errors = np.sum(stage1_errors)

    print(f"\nStage 1 errors (N vs Arrhythmia): {n_stage1_errors:,} ({100*n_stage1_errors/len(y_true_binary):.2f}%)")

    # Break down Stage 1 errors
    false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))  # N classified as Arrhythmia
    false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))  # Arrhythmia classified as N

    print(f"  False Positives (N -> Arrhythmia): {false_positives:,}")
    print(f"  False Negatives (Arrhythmia -> N): {false_negatives:,}")

    # Stage 2 errors (among correctly identified arrhythmias)
    correctly_identified_arrhythmia = (y_true_binary == 1) & (y_pred_binary == 1)
    if np.sum(correctly_identified_arrhythmia) > 0:
        y_true_stage2_correct = y_true_3class[correctly_identified_arrhythmia]
        y_pred_stage2_correct = y_pred_3class[correctly_identified_arrhythmia]
        stage2_errors = np.sum(y_true_stage2_correct != y_pred_stage2_correct)

        print(f"\nStage 2 errors (S vs V, among correctly identified arrhythmias): {stage2_errors:,}")
        print(f"  Out of {np.sum(correctly_identified_arrhythmia):,} correctly identified arrhythmias")
        print(f"  Error rate: {100*stage2_errors/np.sum(correctly_identified_arrhythmia):.2f}%")

    # ========================================================================
    # Generate Confusion Matrix Plots
    # ========================================================================
    plot_all_confusion_matrices(cm_stage1, cm_stage2, cm_final, output_dir=output_dir)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    return {
        'stage1_accuracy': acc_stage1,
        'stage1_roc_auc': roc_auc_stage1,
        'stage2_accuracy': acc_stage2,
        'stage2_roc_auc': roc_auc_stage2,
        'final_accuracy': acc_final,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1': f1,
        'confusion_matrix_stage1': cm_stage1,
        'confusion_matrix_stage2': cm_stage2,
        'confusion_matrix_final': cm_final
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Test complete hierarchical canine ECG classification system
    """
    print("\n" + "="*70)
    print("CANINE ECG - HIERARCHICAL CLASSIFICATION TESTING")
    print("2-Stage System: Stage 1 (N vs Arrhythmia) + Stage 2 (S vs V)")
    print("="*70)

    config = Config()

    # Check if models exist
    if not os.path.exists(config.STAGE1_MODEL):
        print(f"\nERROR: Stage 1 model not found at: {config.STAGE1_MODEL}")
        print("Please train the Stage 1 model first.")
        return

    if not os.path.exists(config.STAGE2_MODEL):
        print(f"\nERROR: Stage 2 model not found at: {config.STAGE2_MODEL}")
        print("Please train the Stage 2 model first using train_subclassifier_s_vs_v.py")
        return

    # Load test data
    X_test, y_test_3class, y_test_binary = load_canine_test_data(config)

    # Initialize hierarchical classifier
    classifier = HierarchicalCanineClassifier(
        stage1_path=config.STAGE1_MODEL,
        stage2_path=config.STAGE2_MODEL,
        stage1_threshold=config.STAGE1_THRESHOLD,
        stage2_threshold=config.STAGE2_THRESHOLD
    )

    # Set output directory (same as script directory)
    output_dir = config.SCRIPT_DIR

    # Evaluate
    results = evaluate_hierarchical_system(
        classifier, X_test, y_test_3class, y_test_binary, output_dir=output_dir
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nStage 1 (N vs Arrhythmia):")
    print(f"  Accuracy: {results['stage1_accuracy']:.4f}")
    print(f"  ROC-AUC: {results['stage1_roc_auc']:.4f}")

    print(f"\nStage 2 (S vs V):")
    print(f"  Accuracy: {results['stage2_accuracy']:.4f}")
    print(f"  ROC-AUC: {results['stage2_roc_auc']:.4f}")

    print(f"\nFinal 3-Class System:")
    print(f"  Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Macro Precision: {results['final_precision']:.4f}")
    print(f"  Macro Recall: {results['final_recall']:.4f}")
    print(f"  Macro F1-Score: {results['final_f1']:.4f}")

    print("\n" + "="*70)
    print("GENERATED CONFUSION MATRIX PLOTS")
    print("="*70)
    print(f"\nConfusion matrices saved to: {output_dir}")
    print("  - canine_stage1_confusion_matrix.png (N vs Arrhythmia)")
    print("  - canine_stage2_confusion_matrix.png (S vs V)")
    print("  - canine_final_confusion_matrix.png (3-class N/S/V)")

    print("\n" + "="*70)
    print("TESTING COMPLETE - CANINE HIERARCHICAL SYSTEM EVALUATED!")
    print("="*70)
   
if __name__ == "__main__":
    main()
