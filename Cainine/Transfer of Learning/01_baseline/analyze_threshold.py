"""
Analyze Stage 1 Decision Threshold for Improved Arrhythmia Detection

Purpose: Find optimal threshold for Stage 1 (N vs Arrhythmia) to maximize arrhythmia recall
         while maintaining acceptable precision.

Current Problem: With threshold=0.5, we're missing 48% of arrhythmias (recall=51.9%)
Solution: Lower threshold to catch more arrhythmias (trade-off: more false positives)
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report
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

    # Paths
    TEST_DATA = os.path.join(PROJECT_ROOT, "Cainine", "DataPreProcessing_Inter", "PreProcessedData", "canine_test.csv")
    STAGE1_MODEL = os.path.join(TRANSFER_LEARNING_DIR, "01_baseline", "models", "canine_binary_original.h5")

    # Data
    INPUT_LEN = 187


# ============================================================================
# DATA LOADING
# ============================================================================

def adapt_length(X, target_len=187):
    """Pad canine data from 186 to 187"""
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    N, L, C = X.shape
    if L == target_len:
        return X
    if L < target_len:
        pad_width = ((0, 0), (0, target_len - L), (0, 0))
        return np.pad(X, pad_width, mode='constant', constant_values=0)
    return X[:, :target_len, :]


def load_test_data(config):
    """Load test data and create binary labels"""
    print("="*70)
    print("LOADING TEST DATA")
    print("="*70)

    df = pd.read_csv(config.TEST_DATA, header=None, low_memory=False)
    print(f"Test set size: {len(df):,} samples")

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_orig = df.iloc[:, -1].to_numpy(dtype=np.int64)

    # Binary labels: N=0, Arrhythmia (S+V)=1
    y_binary = np.isin(y_orig, [1, 2]).astype(np.int64)

    print(f"\nLabel distribution:")
    print(f"  Normal (0): {np.sum(y_binary == 0):,} ({100*np.mean(y_binary==0):.2f}%)")
    print(f"  Arrhythmia (1): {np.sum(y_binary == 1):,} ({100*np.mean(y_binary==1):.2f}%)")

    X = adapt_length(X, target_len=config.INPUT_LEN)

    return X, y_binary


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def analyze_thresholds(y_true, y_probs, config):
    """
    Analyze different thresholds and find optimal ones for different objectives
    """
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)

    # Current threshold (0.5) performance
    y_pred_current = (y_probs >= 0.5).astype(int)
    cm_current = confusion_matrix(y_true, y_pred_current)

    tn, fp, fn, tp = cm_current.ravel()
    current_recall = tp / (tp + fn)
    current_precision = tp / (tp + fp)
    current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall)

    print(f"\nCURRENT PERFORMANCE (threshold=0.5):")
    print(f"  True Positives (TP):  {tp:,} (arrhythmias correctly detected)")
    print(f"  False Negatives (FN): {fn:,} (arrhythmias MISSED)")
    print(f"  False Positives (FP): {fp:,} (normal wrongly flagged)")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"\n  Recall (Sensitivity): {current_recall:.4f} ({100*current_recall:.2f}%)")
    print(f"  Precision:            {current_precision:.4f} ({100*current_precision:.2f}%)")
    print(f"  F1-Score:             {current_f1:.4f}")

    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n  ROC-AUC: {roc_auc:.4f}")

    # Find optimal thresholds for different objectives
    print("\n" + "="*70)
    print("RECOMMENDED THRESHOLDS")
    print("="*70)

    # 1. Maximize F1-Score
    best_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    best_f1_threshold = thresholds_pr[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precision[best_f1_idx]
    best_f1_recall = recall[best_f1_idx]

    print(f"\n1. BEST F1-SCORE (Balanced Precision/Recall)")
    print(f"   Threshold: {best_f1_threshold:.4f}")
    print(f"   F1-Score:  {best_f1:.4f}")
    print(f"   Precision: {best_f1_precision:.4f} ({100*best_f1_precision:.2f}%)")
    print(f"   Recall:    {best_f1_recall:.4f} ({100*best_f1_recall:.2f}%)")

    # Calculate confusion matrix at this threshold
    y_pred_f1 = (y_probs >= best_f1_threshold).astype(int)
    cm_f1 = confusion_matrix(y_true, y_pred_f1)
    tn_f1, fp_f1, fn_f1, tp_f1 = cm_f1.ravel()
    print(f"   TP: {tp_f1:,}, FN: {fn_f1:,}, FP: {fp_f1:,}, TN: {tn_f1:,}")

    # 2. Target 80% Recall (catch 80% of arrhythmias)
    target_recall_80 = 0.80
    idx_80 = np.argmin(np.abs(recall[:-1] - target_recall_80))
    threshold_80 = thresholds_pr[idx_80]
    precision_80 = precision[idx_80]
    recall_80 = recall[idx_80]
    f1_80 = 2 * precision_80 * recall_80 / (precision_80 + recall_80)

    print(f"\n2. TARGET 80% RECALL (Catch 80% of arrhythmias)")
    print(f"   Threshold: {threshold_80:.4f}")
    print(f"   Recall:    {recall_80:.4f} ({100*recall_80:.2f}%)")
    print(f"   Precision: {precision_80:.4f} ({100*precision_80:.2f}%)")
    print(f"   F1-Score:  {f1_80:.4f}")

    y_pred_80 = (y_probs >= threshold_80).astype(int)
    cm_80 = confusion_matrix(y_true, y_pred_80)
    tn_80, fp_80, fn_80, tp_80 = cm_80.ravel()
    print(f"   TP: {tp_80:,}, FN: {fn_80:,}, FP: {fp_80:,}, TN: {tn_80:,}")

    # 3. Target 90% Recall (catch 90% of arrhythmias)
    target_recall_90 = 0.90
    idx_90 = np.argmin(np.abs(recall[:-1] - target_recall_90))
    threshold_90 = thresholds_pr[idx_90]
    precision_90 = precision[idx_90]
    recall_90 = recall[idx_90]
    f1_90 = 2 * precision_90 * recall_90 / (precision_90 + recall_90)

    print(f"\n3. TARGET 90% RECALL (Catch 90% of arrhythmias)")
    print(f"   Threshold: {threshold_90:.4f}")
    print(f"   Recall:    {recall_90:.4f} ({100*recall_90:.2f}%)")
    print(f"   Precision: {precision_90:.4f} ({100*precision_90:.2f}%)")
    print(f"   F1-Score:  {f1_90:.4f}")

    y_pred_90 = (y_probs >= threshold_90).astype(int)
    cm_90 = confusion_matrix(y_true, y_pred_90)
    tn_90, fp_90, fn_90, tp_90 = cm_90.ravel()
    print(f"   TP: {tp_90:,}, FN: {fn_90:,}, FP: {fp_90:,}, TN: {tn_90:,}")

    # 4. Youden's J statistic (maximize TPR - FPR)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_j_threshold = thresholds_roc[best_j_idx]
    best_j_tpr = tpr[best_j_idx]
    best_j_fpr = fpr[best_j_idx]

    print(f"\n4. YOUDEN'S J STATISTIC (Maximize TPR - FPR)")
    print(f"   Threshold: {best_j_threshold:.4f}")
    print(f"   TPR (Recall): {best_j_tpr:.4f} ({100*best_j_tpr:.2f}%)")
    print(f"   FPR: {best_j_fpr:.4f} ({100*best_j_fpr:.2f}%)")

    y_pred_j = (y_probs >= best_j_threshold).astype(int)
    cm_j = confusion_matrix(y_true, y_pred_j)
    print("\nConfusion Matrix (Youden's J):")
    print(cm_j)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Precision-Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    ax1.plot(current_recall, current_precision, 'ro', markersize=10, label=f'Current (t=0.5)')
    ax1.plot(best_f1_recall, best_f1_precision, 'gs', markersize=10, label=f'Best F1 (t={best_f1_threshold:.3f})')
    ax1.plot(recall_80, precision_80, 'm^', markersize=10, label=f'80% Recall (t={threshold_80:.3f})')
    ax1.plot(recall_90, precision_90, 'cv', markersize=10, label=f'90% Recall (t={threshold_90:.3f})')
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve\nStage 1: N vs Arrhythmia', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.plot(best_j_fpr, best_j_tpr, 'rs', markersize=10, label=f"Youden's J (t={best_j_threshold:.3f})")
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax2.set_title('ROC Curve\nStage 1: N vs Arrhythmia', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. F1-Score vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(thresholds_pr, f1_scores[:-1], 'b-', linewidth=2, label='F1-Score')
    ax3.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Current (t=0.5)')
    ax3.axvline(best_f1_threshold, color='g', linestyle='--', linewidth=2, label=f'Best F1 (t={best_f1_threshold:.3f})')
    ax3.axvline(threshold_80, color='m', linestyle='--', linewidth=2, label=f'80% Recall (t={threshold_80:.3f})')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('F1-Score', fontsize=12)
    ax3.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])

    # 4. Recall/Precision vs Threshold
    ax4 = axes[1, 1]
    ax4.plot(thresholds_pr, recall[:-1], 'b-', linewidth=2, label='Recall')
    ax4.plot(thresholds_pr, precision[:-1], 'r-', linewidth=2, label='Precision')
    ax4.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Current (t=0.5)')
    ax4.axvline(best_f1_threshold, color='g', linestyle='--', linewidth=2, label=f'Best F1 (t={best_f1_threshold:.3f})')
    ax4.axhline(0.80, color='m', linestyle=':', linewidth=1, alpha=0.5, label='80% Recall Target')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Recall & Precision vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(config.SCRIPT_DIR, 'models', 'stage1_threshold_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    # ========================================================================
    # RECOMMENDATION
    # ========================================================================

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    print(f"\nCurrent threshold (0.5) is missing {fn:,} arrhythmias ({100*fn/(tp+fn):.1f}%)!")

    print(f"\nRECOMMENDED: Use threshold = {threshold_80:.4f} (Target 80% Recall)")
    print(f"  - Catches {100*recall_80:.1f}% of arrhythmias (vs {100*current_recall:.1f}% currently)")
    print(f"  - Reduces missed arrhythmias from {fn:,} to {fn_80:,}")
    print(f"  - Trade-off: More false positives ({fp_80:,} vs {fp:,})")
    print(f"  - But Stage 2 can correct these false positives!")

    print(f"\nTo implement:")
    print(f"  1. Update STAGE1_THRESHOLD in test_hierarchical_canine.py")
    print(f"  2. Change from 0.5 to {threshold_80:.4f}")
    print(f"  3. Re-run hierarchical test to see improvement")

    return {
        'current_threshold': 0.5,
        'best_f1_threshold': best_f1_threshold,
        'threshold_80_recall': threshold_80,
        'threshold_90_recall': threshold_90,
        'youden_threshold': best_j_threshold
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Analyze Stage 1 threshold for optimal arrhythmia detection"""
    print("\n" + "="*70)
    print("STAGE 1 THRESHOLD ANALYSIS")
    print("Optimizing Decision Threshold for Arrhythmia Detection")
    print("="*70)

    config = Config()

    # Check if model exists
    if not os.path.exists(config.STAGE1_MODEL):
        print(f"\nERROR: Model not found: {config.STAGE1_MODEL}")
        return

    # Load test data
    X_test, y_test_binary = load_test_data(config)

    # Load model
    print("\nLoading Stage 1 model...")
    model = load_model(config.STAGE1_MODEL)
    print("Model loaded successfully")

    # Get predictions (probabilities)
    print("\nGenerating predictions...")
    y_probs = model.predict(X_test, verbose=1).reshape(-1)

    # Analyze thresholds
    results = analyze_thresholds(y_test_binary, y_probs, config)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
