"""
Hierarchical ECG Classification Testing Script
Tests 2-stage model (Stage 1: N vs Arr, Stage 2: S vs V) on MIT-BIH test data

Usage:
    python test_hierarchical_model.py
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, auc, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Get script directory for relative paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    # Paths (relative to script location)
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Data")  # MITBIH/Data/
    MODEL_DIR = SCRIPT_DIR

    # Model files
    STAGE1_MODEL = os.path.join(MODEL_DIR, "best_model_binary.h5")          # N vs Arrhythmia
    STAGE2_MODEL = os.path.join(MODEL_DIR, "best_model_subclass_s_v.h5")   # S vs V

    # Data files
    TEST_FILE = os.path.join(DATA_DIR, "mitbih_test.csv")

    # Configuration
    STAGE1_THRESHOLD = 0.5  # Threshold for arrhythmia detection
    STAGE2_THRESHOLD = 0.5  # Threshold for V vs S classification

    # Label mapping
    LABEL_MAP = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
    CLASSES_OF_INTEREST = [0, 1, 2]  # N, S, V (exclude F, Q)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(config):
    """
    Load MIT-BIH test data and filter to N, S, V classes

    Returns:
        X: ECG signals (N, 187)
        y: True labels (N,) - original 0/1/2 for N/S/V
    """
    print("="*70)
    print("LOADING TEST DATA")
    print("="*70)

    # Load CSV
    df = pd.read_csv(config.TEST_FILE, header=None)
    print(f"\nLoaded {len(df)} total samples from test set")

    # Extract features and labels
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_orig = df.iloc[:, -1].values.astype(int)

    print(f"Original label distribution:")
    for label, count in zip(*np.unique(y_orig, return_counts=True)):
        label_name = config.LABEL_MAP.get(label, f"Unknown({label})")
        print(f"  {label_name}: {count} samples ({100*count/len(y_orig):.1f}%)")

    # Filter to N, S, V only
    mask = np.isin(y_orig, config.CLASSES_OF_INTEREST)
    X = X[mask]
    y = y_orig[mask]

    print(f"\nAfter filtering to N, S, V:")
    print(f"  Total samples: {len(y)}")
    for label in config.CLASSES_OF_INTEREST:
        count = np.sum(y == label)
        label_name = config.LABEL_MAP[label]
        print(f"  {label_name}: {count} samples ({100*count/len(y):.1f}%)")

    # Add channel dimension for CNN
    X = X[:, :, np.newaxis]  # (N, 187, 1)

    return X, y


# ============================================================================
# HIERARCHICAL CLASSIFICATION
# ============================================================================

class HierarchicalClassifier:
    """
    Two-stage hierarchical ECG classifier
    Stage 1: Normal vs Arrhythmia
    Stage 2: Supraventricular vs Ventricular (only if arrhythmia)
    """

    def __init__(self, stage1_path, stage2_path, stage1_threshold=0.5, stage2_threshold=0.5):
        """
        Args:
            stage1_path: Path to Stage 1 model (N vs Arrhythmia)
            stage2_path: Path to Stage 2 model (S vs V)
            stage1_threshold: Threshold for arrhythmia detection
            stage2_threshold: Threshold for V classification
        """
        print("\n" + "="*70)
        print("LOADING HIERARCHICAL MODELS")
        print("="*70)

        print(f"\nStage 1: Loading binary classifier (N vs Arrhythmia)...")
        print(f"  Path: {stage1_path}")
        self.stage1_model = load_model(stage1_path, compile=False)
        print(f"  Input shape: {self.stage1_model.input_shape}")
        print(f"  Output shape: {self.stage1_model.output_shape}")

        print(f"\nStage 2: Loading sub-classifier (S vs V)...")
        print(f"  Path: {stage2_path}")
        self.stage2_model = load_model(stage2_path, compile=False)
        print(f"  Input shape: {self.stage2_model.input_shape}")
        print(f"  Output shape: {self.stage2_model.output_shape}")

        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold

        print(f"\nThresholds:")
        print(f"  Stage 1 (Arrhythmia detection): {stage1_threshold}")
        print(f"  Stage 2 (V classification): {stage2_threshold}")

        print("\nModels loaded successfully!")

    def predict_batch(self, X, verbose=True):
        """
        Predict on batch of ECG beats using hierarchical approach

        Args:
            X: ECG signals (N, 187, 1)
            verbose: Print progress

        Returns:
            predictions: Predicted labels (N,) - 0=N, 1=S, 2=V
            probabilities: Dict with all probability information
        """
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=int)

        # Storage for probabilities
        p_arrhythmia = np.zeros(n_samples)
        p_v_given_arr = np.zeros(n_samples)
        stage2_called = np.zeros(n_samples, dtype=bool)

        if verbose:
            print(f"\nPredicting on {n_samples} samples...")

        # Stage 1: Classify all samples (N vs Arrhythmia)
        if verbose:
            print("  Stage 1: Detecting arrhythmias...")
        p_arrhythmia = self.stage1_model.predict(X, verbose=0).reshape(-1)

        # Identify arrhythmia samples
        arrhythmia_mask = p_arrhythmia >= self.stage1_threshold
        n_arrhythmia = np.sum(arrhythmia_mask)

        if verbose:
            print(f"    Detected {n_arrhythmia} arrhythmias ({100*n_arrhythmia/n_samples:.1f}%)")

        # Stage 2: Classify arrhythmia samples (S vs V)
        if n_arrhythmia > 0:
            if verbose:
                print(f"  Stage 2: Classifying arrhythmia types...")

            X_arrhythmia = X[arrhythmia_mask]
            p_v = self.stage2_model.predict(X_arrhythmia, verbose=0).reshape(-1)
            p_v_given_arr[arrhythmia_mask] = p_v
            stage2_called[arrhythmia_mask] = True

            # Classify as S or V
            v_mask_in_arr = p_v >= self.stage2_threshold
            s_mask_in_arr = ~v_mask_in_arr

            # Map back to full array indices
            arr_indices = np.where(arrhythmia_mask)[0]
            predictions[arr_indices[v_mask_in_arr]] = 2  # V
            predictions[arr_indices[s_mask_in_arr]] = 1  # S

            if verbose:
                print(f"    Classified as S: {np.sum(s_mask_in_arr)} ({100*np.sum(s_mask_in_arr)/n_arrhythmia:.1f}%)")
                print(f"    Classified as V: {np.sum(v_mask_in_arr)} ({100*np.sum(v_mask_in_arr)/n_arrhythmia:.1f}%)")
        else:
            if verbose:
                print("  Stage 2: Skipped (no arrhythmias detected)")

        # Normal samples remain 0
        predictions[~arrhythmia_mask] = 0  # N

        if verbose:
            print(f"\nFinal predictions:")
            for label in [0, 1, 2]:
                count = np.sum(predictions == label)
                label_name = {0: 'N', 1: 'S', 2: 'V'}[label]
                print(f"  {label_name}: {count} samples ({100*count/n_samples:.1f}%)")

        # Return predictions and probability details
        probabilities = {
            'p_arrhythmia': p_arrhythmia,
            'p_v_given_arr': p_v_given_arr,
            'stage2_called': stage2_called
        }

        return predictions, probabilities


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_hierarchical(y_true, y_pred, probabilities, config):
    """
    Comprehensive evaluation of hierarchical classifier

    Args:
        y_true: True labels (0=N, 1=S, 2=V)
        y_pred: Predicted labels (0=N, 1=S, 2=V)
        probabilities: Dict with probability information
        config: Configuration object
    """
    print("\n" + "="*70)
    print("HIERARCHICAL MODEL EVALUATION")
    print("="*70)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")

    # Classification report
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT (3-Class: N, S, V)")
    print("-"*70)
    target_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # Confusion matrix
    print("-"*70)
    print("CONFUSION MATRIX (3-Class)")
    print("-"*70)
    cm = confusion_matrix(y_true, y_pred)
    print("\nRows = True, Columns = Predicted")
    print("        N      S      V")
    for i, label in enumerate(['N', 'S', 'V']):
        print(f"  {label}  {cm[i,0]:5d}  {cm[i,1]:5d}  {cm[i,2]:5d}")

    # Per-class metrics
    print("\n" + "-"*70)
    print("PER-CLASS DETAILED METRICS")
    print("-"*70)
    for i, label_name in enumerate(['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)']):
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)

        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        tn = np.sum(~mask_true & ~mask_pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{label_name}:")
        print(f"  True Positives:  {tp:5d}")
        print(f"  False Positives: {fp:5d}")
        print(f"  False Negatives: {fn:5d}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    # Stage-wise evaluation
    print("\n" + "-"*70)
    print("STAGE-WISE PERFORMANCE")
    print("-"*70)

    # Stage 1: N vs (S+V)
    y_true_binary = (y_true > 0).astype(int)  # 0=N, 1=Arrhythmia
    y_pred_stage1 = (probabilities['p_arrhythmia'] >= config.STAGE1_THRESHOLD).astype(int)

    stage1_acc = accuracy_score(y_true_binary, y_pred_stage1)
    print(f"\nStage 1 (N vs Arrhythmia):")
    print(f"  Accuracy: {stage1_acc:.4f} ({100*stage1_acc:.2f}%)")

    cm_stage1 = confusion_matrix(y_true_binary, y_pred_stage1)
    print(f"\n  Confusion Matrix:")
    print(f"            Pred_N  Pred_Arr")
    print(f"  True_N      {cm_stage1[0,0]:5d}    {cm_stage1[0,1]:5d}")
    print(f"  True_Arr    {cm_stage1[1,0]:5d}    {cm_stage1[1,1]:5d}")

    # Stage 2: S vs V (only on actual arrhythmias)
    mask_true_arr = y_true > 0
    if np.sum(mask_true_arr) > 0:
        y_true_s_v = y_true[mask_true_arr]  # 1=S, 2=V
        y_pred_s_v = y_pred[mask_true_arr]

        # Convert to binary for S vs V
        y_true_s_v_binary = (y_true_s_v == 2).astype(int)  # 0=S, 1=V
        y_pred_s_v_binary = (y_pred_s_v == 2).astype(int)

        stage2_acc = accuracy_score(y_true_s_v_binary, y_pred_s_v_binary)
        print(f"\nStage 2 (S vs V, on true arrhythmias only):")
        print(f"  Samples: {len(y_true_s_v)}")
        print(f"  Accuracy: {stage2_acc:.4f} ({100*stage2_acc:.2f}%)")

        cm_stage2 = confusion_matrix(y_true_s_v_binary, y_pred_s_v_binary)
        print(f"\n  Confusion Matrix:")
        print(f"          Pred_S  Pred_V")
        print(f"  True_S    {cm_stage2[0,0]:5d}    {cm_stage2[0,1]:5d}")
        print(f"  True_V    {cm_stage2[1,0]:5d}    {cm_stage2[1,1]:5d}")

    # Error analysis
    print("\n" + "-"*70)
    print("ERROR ANALYSIS")
    print("-"*70)

    errors = y_true != y_pred
    n_errors = np.sum(errors)
    print(f"\nTotal errors: {n_errors} / {len(y_true)} ({100*n_errors/len(y_true):.2f}%)")

    if n_errors > 0:
        print("\nError breakdown:")
        for true_label in [0, 1, 2]:
            for pred_label in [0, 1, 2]:
                if true_label != pred_label:
                    count = np.sum((y_true == true_label) & (y_pred == pred_label))
                    if count > 0:
                        true_name = {0: 'N', 1: 'S', 2: 'V'}[true_label]
                        pred_name = {0: 'N', 1: 'S', 2: 'V'}[pred_label]
                        print(f"  {true_name} -> {pred_name}: {count} samples")

    # Stage 2 calling statistics
    n_stage2_called = np.sum(probabilities['stage2_called'])
    print(f"\nStage 2 invocations: {n_stage2_called} / {len(y_true)} ({100*n_stage2_called/len(y_true):.2f}%)")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'stage1_accuracy': stage1_acc,
        'stage2_accuracy': stage2_acc if np.sum(mask_true_arr) > 0 else None
    }


def plot_confusion_matrix(y_true, y_pred, title='Hierarchical Model Confusion Matrix',
                         filename='hierarchical_confusion_matrix.png',
                         labels=['N', 'S', 'V']):
    """
    Plot confusion matrix with visualization

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: Output filename
        labels: Class labels for axes
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(Config.MODEL_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix plot saved to: {output_path}")

    plt.close()


def plot_stage_confusion_matrices(y_true, y_pred, probabilities, config):
    """
    Plot confusion matrices for each stage of hierarchical classification

    Args:
        y_true: True labels (0=N, 1=S, 2=V)
        y_pred: Predicted labels
        probabilities: Dictionary with probability information
        config: Configuration object
    """
    print("\n" + "="*70)
    print("GENERATING STAGE-WISE CONFUSION MATRIX PLOTS")
    print("="*70)

    # Stage 1: N vs Arrhythmia
    y_true_binary = (y_true > 0).astype(int)  # 0=N, 1=Arrhythmia
    y_pred_stage1 = (probabilities['p_arrhythmia'] >= config.STAGE1_THRESHOLD).astype(int)

    print("\nPlotting Stage 1 confusion matrix (Normal vs Arrhythmia)...")
    plot_confusion_matrix(
        y_true_binary,
        y_pred_stage1,
        title='Stage 1: Binary Classification (Normal vs Arrhythmia)',
        filename='stage1_confusion_matrix.png',
        labels=['Normal', 'Arrhythmia']
    )

    # Stage 2: S vs V (only on actual arrhythmias)
    mask_true_arr = y_true > 0
    if np.sum(mask_true_arr) > 0:
        y_true_s_v = y_true[mask_true_arr]  # 1=S, 2=V
        y_pred_s_v = y_pred[mask_true_arr]

        # Convert to binary for S vs V
        y_true_s_v_binary = (y_true_s_v == 2).astype(int)  # 0=S, 1=V
        y_pred_s_v_binary = (y_pred_s_v == 2).astype(int)

        print("\nPlotting Stage 2 confusion matrix (Supraventricular vs Ventricular)...")
        plot_confusion_matrix(
            y_true_s_v_binary,
            y_pred_s_v_binary,
            title='Stage 2: Sub-Classification (Supraventricular vs Ventricular)',
            filename='stage2_confusion_matrix.png',
            labels=['S (Supraventricular)', 'V (Ventricular)']
        )

    print("\nAll stage-wise confusion matrices generated successfully!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main testing script for hierarchical ECG classifier
    """
    print("\n" + "="*70)
    print(" HIERARCHICAL ECG CLASSIFICATION - TEST SCRIPT")
    print("="*70)
    print("\nTwo-Stage Classification:")
    print("  Stage 1: Normal vs Arrhythmia")
    print("  Stage 2: Supraventricular vs Ventricular (if arrhythmia)")
    print("="*70)

    config = Config()

    # Load test data
    X_test, y_test = load_test_data(config)

    # Initialize hierarchical classifier
    classifier = HierarchicalClassifier(
        stage1_path=config.STAGE1_MODEL,
        stage2_path=config.STAGE2_MODEL,
        stage1_threshold=config.STAGE1_THRESHOLD,
        stage2_threshold=config.STAGE2_THRESHOLD
    )

    # Predict
    y_pred, probabilities = classifier.predict_batch(X_test, verbose=True)

    # Evaluate
    results = evaluate_hierarchical(y_test, y_pred, probabilities, config)

    # Plot overall 3-class confusion matrix
    print("\nPlotting overall 3-class confusion matrix...")
    plot_confusion_matrix(y_test, y_pred,
                         title='Hierarchical Model (2-Stage) - Test Set Performance',
                         filename='hierarchical_confusion_matrix.png',
                         labels=['N', 'S', 'V'])

    # Plot stage-wise confusion matrices
    plot_stage_confusion_matrices(y_test, y_pred, probabilities, config)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTest Set Size: {len(y_test)} samples")
    print(f"  Normal (N): {np.sum(y_test == 0)} samples")
    print(f"  Supraventricular (S): {np.sum(y_test == 1)} samples")
    print(f"  Ventricular (V): {np.sum(y_test == 2)} samples")
    print(f"\nOverall 3-Class Accuracy: {results['accuracy']:.4f} ({100*results['accuracy']:.2f}%)")
    print(f"Stage 1 Accuracy (N vs Arr): {results['stage1_accuracy']:.4f}")
    if results['stage2_accuracy'] is not None:
        print(f"Stage 2 Accuracy (S vs V): {results['stage2_accuracy']:.4f}")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
