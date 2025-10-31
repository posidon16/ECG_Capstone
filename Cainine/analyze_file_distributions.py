"""
Analyze Class Distributions Across All Canine ECG Data Files

Purpose: Show N, S, V class counts for each numbered data file
Output: Summary table showing class distribution per file

This helps understand which files have what types of arrhythmias.
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to raw data (adjust if needed)
DATA_DIR = os.path.join(SCRIPT_DIR, "all_data")

# Label mapping
LABEL_NAMES = {0: 'N', 1: 'S', 2: 'V'}

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_file_distributions(data_dir):
    """
    Analyze class distributions across all data files

    Args:
        data_dir: Directory containing the numbered CSV files
    """
    print("="*80)
    print("CANINE ECG DATA - CLASS DISTRIBUTION BY FILE")
    print("="*80)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("\nPlease ensure the raw data is available.")
        print("The raw data should be in the 'all_data' folder.")
        return

    # Find all signal files (not label files)
    signal_files = []
    for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
        if not file_path.endswith("_labels.csv"):
            signal_files.append(file_path)

    if not signal_files:
        print(f"\nNo data files found in: {data_dir}")
        return

    # Sort files by number
    signal_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    print(f"\nFound {len(signal_files)} data files in: {data_dir}")
    print("\nAnalyzing class distributions...")
    print("="*80)

    # Store results
    results = []
    total_n = 0
    total_s = 0
    total_v = 0
    total_beats = 0

    # Analyze each file
    for signal_file in signal_files:
        # Get file number
        base_name = os.path.splitext(os.path.basename(signal_file))[0]
        labels_file = os.path.join(data_dir, f"{base_name}_labels.csv")

        if not os.path.exists(labels_file):
            print(f"Warning: No labels file found for {base_name}, skipping...")
            continue

        try:
            # Load labels
            labels_df = pd.read_csv(labels_file)

            # Get labels column (assuming it's named "Label" or similar)
            if 'Label' in labels_df.columns:
                labels = labels_df['Label'].astype(str).fillna("Q")
            elif 'label' in labels_df.columns:
                labels = labels_df['label'].astype(str).fillna("Q")
            else:
                # Try last column
                labels = labels_df.iloc[:, -1].astype(str).fillna("Q")

            # Count each class
            n_count = np.sum(labels.str.upper().str[0] == 'N')
            s_count = np.sum(labels.str.upper().str[0] == 'S')
            v_count = np.sum(labels.str.upper().str[0] == 'V')
            other_count = len(labels) - (n_count + s_count + v_count)

            total = len(labels)

            # Store results
            results.append({
                'file': base_name,
                'N': n_count,
                'S': s_count,
                'V': v_count,
                'Other': other_count,
                'Total': total,
                'N%': 100 * n_count / total if total > 0 else 0,
                'S%': 100 * s_count / total if total > 0 else 0,
                'V%': 100 * v_count / total if total > 0 else 0
            })

            # Update totals
            total_n += n_count
            total_s += s_count
            total_v += v_count
            total_beats += total

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    if not results:
        print("No files were successfully analyzed.")
        return

    print("\n" + "="*80)
    print("PER-FILE CLASS DISTRIBUTION")
    print("="*80)
    print(f"\n{'File':<8} {'N':>8} {'S':>8} {'V':>8} {'Other':>8} {'Total':>10} {'N%':>7} {'S%':>7} {'V%':>7}")
    print("-"*80)

    for r in results:
        print(f"{r['file']:<8} {r['N']:>8,} {r['S']:>8,} {r['V']:>8,} {r['Other']:>8,} "
              f"{r['Total']:>10,} {r['N%']:>6.1f}% {r['S%']:>6.1f}% {r['V%']:>6.1f}%")

    print("-"*80)
    print(f"{'TOTAL':<8} {total_n:>8,} {total_s:>8,} {total_v:>8,} "
          f"{total_beats - total_n - total_s - total_v:>8,} {total_beats:>10,} "
          f"{100*total_n/total_beats:>6.1f}% {100*total_s/total_beats:>6.1f}% "
          f"{100*total_v/total_beats:>6.1f}%")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal files analyzed: {len(results)}")
    print(f"Total beats: {total_beats:,}")
    print(f"\nOverall class distribution:")
    print(f"  N (Normal):           {total_n:>8,} ({100*total_n/total_beats:>5.2f}%)")
    print(f"  S (Supraventricular): {total_s:>8,} ({100*total_s/total_beats:>5.2f}%)")
    print(f"  V (Ventricular):      {total_v:>8,} ({100*total_v/total_beats:>5.2f}%)")

    # Imbalance ratios
    if total_s > 0:
        print(f"\n  N:S ratio = {total_n/total_s:.1f}:1")
    if total_v > 0:
        print(f"  N:V ratio = {total_n/total_v:.1f}:1")
    if total_s > 0 and total_v > 0:
        if total_s >= total_v:
            print(f"  S:V ratio = {total_s/total_v:.1f}:1")
        else:
            print(f"  S:V ratio = 1:{total_v/total_s:.1f}")

    # Files with most arrhythmias
    print("\n" + "="*80)
    print("FILES WITH MOST ARRHYTHMIAS")
    print("="*80)

    # Sort by total arrhythmias (S + V)
    results_sorted = sorted(results, key=lambda x: x['S'] + x['V'], reverse=True)

    print(f"\n{'File':<8} {'S':>8} {'V':>8} {'Total Arrhythmia':>18} {'% of File':>12}")
    print("-"*60)

    for i, r in enumerate(results_sorted[:10]):  # Top 10
        arrhythmia_total = r['S'] + r['V']
        arrhythmia_pct = 100 * arrhythmia_total / r['Total'] if r['Total'] > 0 else 0
        print(f"{r['file']:<8} {r['S']:>8,} {r['V']:>8,} {arrhythmia_total:>18,} {arrhythmia_pct:>11.2f}%")

    # Files with most S beats
    print("\n" + "="*80)
    print("FILES WITH MOST S (SUPRAVENTRICULAR) BEATS")
    print("="*80)

    results_sorted_s = sorted(results, key=lambda x: x['S'], reverse=True)

    print(f"\n{'File':<8} {'S Count':>10} {'% of File':>12} {'Total Beats':>15}")
    print("-"*50)

    for i, r in enumerate(results_sorted_s[:10]):  # Top 10
        s_pct = 100 * r['S'] / r['Total'] if r['Total'] > 0 else 0
        print(f"{r['file']:<8} {r['S']:>10,} {s_pct:>11.2f}% {r['Total']:>15,}")

    # Files with most V beats
    print("\n" + "="*80)
    print("FILES WITH MOST V (VENTRICULAR) BEATS")
    print("="*80)

    results_sorted_v = sorted(results, key=lambda x: x['V'], reverse=True)

    print(f"\n{'File':<8} {'V Count':>10} {'% of File':>12} {'Total Beats':>15}")
    print("-"*50)

    for i, r in enumerate(results_sorted_v[:10]):  # Top 10
        v_pct = 100 * r['V'] / r['Total'] if r['Total'] > 0 else 0
        print(f"{r['file']:<8} {r['V']:>10,} {v_pct:>11.2f}% {r['Total']:>15,}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # ========================================================================
    # SAVE TO CSV
    # ========================================================================

    output_file = os.path.join(SCRIPT_DIR, "file_class_distributions.csv")

    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results[['file', 'N', 'S', 'V', 'Other', 'Total', 'N%', 'S%', 'V%']]

    # Add total row
    total_row = pd.DataFrame([{
        'file': 'TOTAL',
        'N': total_n,
        'S': total_s,
        'V': total_v,
        'Other': total_beats - total_n - total_s - total_v,
        'Total': total_beats,
        'N%': 100 * total_n / total_beats,
        'S%': 100 * total_s / total_beats,
        'V%': 100 * total_v / total_beats
    }])

    df_results = pd.concat([df_results, total_row], ignore_index=True)

    # Save to CSV
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function to analyze file distributions
    """
    analyze_file_distributions(DATA_DIR)


if __name__ == "__main__":
    main()
