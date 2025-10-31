"""
Enhanced ECG Preprocessing for Canine Transfer Learning (Custom Test Files)
Based on "Transfer Learning in ECG Classification from Human to Horse" methodology

Features:
- Median filtering for baseline wander removal
- Discrete Wavelet Transform for noise reduction
- Proper normalization (global, not per-beat)
- Peak detection with species-specific considerations
- Batch processing and combining functionality
- CUSTOM: Specify which files to use for test set via command line

Usage:
    # Specify test files
    python enhanced_preprocessing_choose.py --test-files 413 102

    # Use single test file
    python enhanced_preprocessing_choose.py --test-files 301

    # Use default (413, 102)
    python enhanced_preprocessing_choose.py
"""

import csv
import glob
import os
import argparse
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.ndimage
import pywt
from scipy.signal import find_peaks, medfilt

# ==== CONFIGURATION ====
@dataclass
class PreprocessConfig:
    # Input files
    signal_file: str = "100.csv"
    labels_file: str = "100_labels.csv"
    output_file: str = "100_beats_enhanced.csv"

    # Signal processing
    sig_col: str = None  # Signal files have no header, just raw values
    idx_col: str = "Samples"
    lab_col: str = "Label"
    window_len: int = 500  # Use 500 like the paper (vs 187)

    # Filtering parameters (based on paper)
    median_filter_1: int = 200  # ms, remove P waves and QRS
    median_filter_2: int = 600  # ms, remove T waves
    sampling_rate: float = 500.0  # Hz

    # Wavelet denoising
    wavelet_name: str = 'coif2'  # Coiflet 2 as per paper
    wavelet_levels: int = 6
    threshold_mode: str = 'soft'

    # Normalization
    use_global_normalization: bool = True
    use_per_beat_normalization: bool = False

    # Peak detection
    detect_s_peaks: bool = True  # For canines (vs R peaks for humans)
    peak_distance_ms: float = 200.0  # Minimum distance between peaks

    # Labels
    fs: Optional[float] = None  # Set if labels are in seconds instead of samples

# Label mapping
LETTER_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def map_label(lbl: str) -> int:
    """Map letter labels to numeric IDs"""
    s = str(lbl).strip()
    if not s:
        return 4  # Unknown -> Q
    return LETTER_TO_ID.get(s[0].upper(), 4)

def median_baseline_filter(signal: np.ndarray, fs: float,
                          filter1_ms: int = 200, filter2_ms: int = 600) -> np.ndarray:
    """
    Remove baseline wander using dual median filtering as per paper

    Args:
        signal: Input ECG signal
        fs: Sampling rate in Hz
        filter1_ms: First median filter width (removes P waves, QRS)
        filter2_ms: Second median filter width (removes T waves)

    Returns:
        Baseline-corrected signal
    """
    # Convert ms to samples
    filter1_samp = int(filter1_ms * fs / 1000)
    filter2_samp = int(filter2_ms * fs / 1000)

    # Ensure odd filter lengths
    if filter1_samp % 2 == 0:
        filter1_samp += 1
    if filter2_samp % 2 == 0:
        filter2_samp += 1

    # First median filter - removes P waves and QRS complexes
    temp = medfilt(signal, kernel_size=filter1_samp)

    # Second median filter - removes T waves, leaves baseline
    baseline = medfilt(temp, kernel_size=filter2_samp)

    # Subtract baseline from original signal
    filtered_signal = signal - baseline

    return filtered_signal

def wavelet_denoise(signal: np.ndarray, wavelet: str = 'coif2',
                   levels: int = 6, mode: str = 'soft') -> np.ndarray:
    """
    Denoise signal using Discrete Wavelet Transform as per paper

    Args:
        signal: Input signal
        wavelet: Wavelet type (paper uses Coiflet 2)
        levels: Decomposition levels
        mode: Thresholding mode

    Returns:
        Denoised signal
    """
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=levels)

    # Calculate threshold using Daubechies method
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to detail coefficients
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode=mode)
                         for detail in coeffs[1:]]

    # Reconstruct signal
    denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Handle length mismatch due to wavelet transform
    if len(denoised) != len(signal):
        denoised = denoised[:len(signal)]

    return denoised

def detect_peaks_adaptive(signal: np.ndarray, fs: float,
                         detect_s_peaks: bool = True,
                         min_distance_ms: float = 200.0) -> np.ndarray:
    """
    Detect R-peaks (humans) or S-peaks (canines) in ECG signal

    Args:
        signal: Filtered ECG signal
        fs: Sampling rate
        detect_s_peaks: If True, detect S-peaks (negative); if False, R-peaks (positive)
        min_distance_ms: Minimum distance between peaks in ms

    Returns:
        Array of peak indices
    """
    min_distance_samp = int(min_distance_ms * fs / 1000)

    if detect_s_peaks:
        # For canines: detect S-peaks (negative peaks)
        # Invert signal to find negative peaks as positive
        search_signal = -signal
    else:
        # For humans: detect R-peaks (positive peaks)
        search_signal = signal

    # Adaptive threshold based on signal statistics
    threshold = np.mean(search_signal) + 2 * np.std(search_signal)

    # Find peaks
    peaks, _ = find_peaks(search_signal,
                         height=threshold,
                         distance=min_distance_samp)

    return peaks

def extract_window_robust(signal: np.ndarray, center: int, length: int) -> np.ndarray:
    """
    Extract window around center point with robust padding

    Args:
        signal: Input signal
        center: Center sample index
        length: Window length in samples

    Returns:
        Extracted window (zero-padded if necessary)
    """
    half = length // 2
    start = center - half
    end = start + length

    # Calculate padding needed
    left_pad = max(0, -start)
    right_pad = max(0, end - len(signal))

    # Extract valid portion
    start_clip = max(0, start)
    end_clip = min(len(signal), end)

    window = signal[start_clip:end_clip].astype(np.float32)

    # Apply padding if needed
    if left_pad > 0 or right_pad > 0:
        window = np.pad(window, (left_pad, right_pad), mode='constant', constant_values=0)

    return window

def global_normalize(signals: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float]:
    """
    Apply global normalization across all beats (preserves relative amplitudes)

    Args:
        signals: List of signal windows

    Returns:
        Normalized signals, global mean, global std
    """
    # Concatenate all signals to compute global statistics
    all_samples = np.concatenate(signals)
    global_mean = float(np.mean(all_samples))
    global_std = float(np.std(all_samples))

    # Normalize each signal using global statistics
    normalized_signals = [(sig - global_mean) / (global_std + 1e-8) for sig in signals]

    return normalized_signals, global_mean, global_std

def process_single_file(config: PreprocessConfig) -> Tuple[int, int]:
    """
    Process a single ECG file with enhanced preprocessing

    Returns:
        (written_count, skipped_count)
    """
    print(f"[processing] {config.signal_file} -> {config.output_file}")

    # Load signal data (raw values, no header)
    if config.sig_col is None:
        # Signal file contains only raw ECG values, one per line
        raw_signal = pd.read_csv(config.signal_file, header=None).iloc[:, 0].astype(np.float32).to_numpy()
    else:
        # Signal file has headers
        signal_df = pd.read_csv(config.signal_file)
        if config.sig_col not in signal_df.columns:
            raise ValueError(f"Signal column '{config.sig_col}' not found in {config.signal_file}")
        raw_signal = signal_df[config.sig_col].astype(np.float32).to_numpy()

    # Load labels
    labels_df = pd.read_csv(config.labels_file)
    if config.idx_col not in labels_df.columns or config.lab_col not in labels_df.columns:
        raise ValueError(f"Required columns not found in {config.labels_file}")

    r_indices = pd.to_numeric(labels_df[config.idx_col], errors="coerce")
    labels = labels_df[config.lab_col].astype(str).fillna("Q")

    # Convert time to samples if needed
    if config.fs is not None:
        r_indices = r_indices * config.fs

    # Filter valid indices
    valid_mask = np.isfinite(r_indices)
    r_indices = np.round(r_indices[valid_mask]).astype(np.int64)
    labels = labels[valid_mask].to_numpy()

    print(f"[signal] Length: {len(raw_signal)} samples")
    print(f"[labels] Found {len(r_indices)} valid peak annotations")

    # === SIGNAL PROCESSING ===
    print("[filtering] Applying median baseline filter...")
    filtered_signal = median_baseline_filter(raw_signal, config.sampling_rate,
                                           config.median_filter_1, config.median_filter_2)

    print("[filtering] Applying wavelet denoising...")
    denoised_signal = wavelet_denoise(filtered_signal, config.wavelet_name,
                                    config.wavelet_levels, config.threshold_mode)

    # === BEAT EXTRACTION ===
    print("[extraction] Extracting beat windows...")
    beat_windows = []
    valid_labels = []
    written = skipped = 0

    for idx, label in zip(r_indices, labels):
        if idx < 0 or idx >= len(denoised_signal):
            skipped += 1
            continue

        # Extract window around peak
        window = extract_window_robust(denoised_signal, int(idx), config.window_len)
        beat_windows.append(window)
        valid_labels.append(label)
        written += 1

    print(f"[extraction] Extracted {written} beats, skipped {skipped}")

    # === NORMALIZATION ===
    if config.use_global_normalization and beat_windows:
        print("[normalization] Applying global normalization...")
        beat_windows, global_mean, global_std = global_normalize(beat_windows)
        print(f"[normalization] Global stats - mean: {global_mean:.4f}, std: {global_std:.4f}")

    # === SAVE RESULTS ===
    print(f"[saving] Writing to {config.output_file}...")
    os.makedirs(os.path.dirname(config.output_file) or '.', exist_ok=True)

    with open(config.output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # NO HEADER - to be compatible with MIT-BIH test code
        # Write data only
        for window, label in zip(beat_windows, valid_labels):
            # Apply per-beat normalization if requested (not recommended)
            if config.use_per_beat_normalization and not config.use_global_normalization:
                mu, sigma = np.mean(window), np.std(window)
                window = (window - mu) / (sigma + 1e-8)

            label_id = map_label(label)
            row = window.tolist() + [int(label_id)]
            writer.writerow(row)

    print(f"[done] Wrote {written} beats to {config.output_file}")
    return written, skipped

def combine_batch_files(input_pattern: str, output_file: str,
                       config: PreprocessConfig) -> None:
    """
    Combine multiple processed files into a single dataset

    Args:
        input_pattern: Glob pattern for input files (e.g., "beats_*.csv")
        output_file: Combined output file
        config: Processing configuration
    """
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")

    print(f"[combining] Found {len(input_files)} files to combine")

    total_beats = 0
    header_written = False

    with open(output_file, 'w', newline='') as outf:
        writer = csv.writer(outf)

        for file_path in input_files:
            print(f"[combining] Processing {file_path}...")

            with open(file_path, 'r') as inf:
                reader = csv.reader(inf)

                # NO HEADER handling - files are headerless now
                # Copy all data rows directly
                file_beats = 0
                for row in reader:
                    writer.writerow(row)
                    file_beats += 1

                total_beats += file_beats
                print(f"[combining] Added {file_beats} beats from {file_path}")

    print(f"[done] Combined {total_beats} total beats into {output_file}")

def batch_process_directory(signal_pattern: str, labels_pattern: str,
                          output_dir: str, base_config: PreprocessConfig) -> None:
    """
    Process multiple ECG files in batch

    Args:
        signal_pattern: Glob pattern for signal files
        labels_pattern: Glob pattern for label files
        output_dir: Output directory
        base_config: Base configuration
    """
    signal_files = sorted(glob.glob(signal_pattern))
    labels_files = sorted(glob.glob(labels_pattern))

    if len(signal_files) != len(labels_files):
        print(f"Warning: Found {len(signal_files)} signal files but {len(labels_files)} label files")

    os.makedirs(output_dir, exist_ok=True)

    total_written = total_skipped = 0

    for i, (sig_file, lab_file) in enumerate(zip(signal_files, labels_files)):
        # Create file-specific config
        file_config = PreprocessConfig(
            signal_file=sig_file,
            labels_file=lab_file,
            output_file=os.path.join(output_dir, f"beats_{i:03d}.csv"),
            # Copy other settings from base_config
            window_len=base_config.window_len,
            sampling_rate=base_config.sampling_rate,
            use_global_normalization=base_config.use_global_normalization,
            detect_s_peaks=base_config.detect_s_peaks
        )

        written, skipped = process_single_file(file_config)
        total_written += written
        total_skipped += skipped

    print(f"[batch] Processed {len(signal_files)} files")
    print(f"[batch] Total: {total_written} beats written, {total_skipped} skipped")

    # Combine all files
    combined_file = os.path.join(output_dir, "combined_enhanced_beats.csv")
    combine_batch_files(os.path.join(output_dir, "beats_*.csv"), combined_file, base_config)

def main():
    """Process canine ECG data from Combined_data folder with custom test file selection"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced ECG preprocessing with custom test file selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use specific files for test set
  python enhanced_preprocessing_choose.py --test-files 413 102

  # Use single file for test set
  python enhanced_preprocessing_choose.py --test-files 301

  # Use default test files (413, 102)
  python enhanced_preprocessing_choose.py

  # List available files
  python enhanced_preprocessing_choose.py --list-files
        """
    )

    parser.add_argument(
        '--test-files',
        nargs='+',
        type=str,
        default=['413', '102'],
        help='File numbers to use for test set (default: 413 102). All other files will be train set.'
    )

    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List available files and exit'
    )

    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cainine_root = os.path.dirname(script_dir)

    # Base configuration for canine data
    base_config = PreprocessConfig(
        window_len=186,  # Match MIT-BIH model input length
        sampling_rate=500.0,
        use_global_normalization=True,
        use_per_beat_normalization=False,  # Disabled per-beat norm
        detect_s_peaks=True,  # For canines
        sig_col=None  # Raw signal files have no header
    )

    # Define input and output directories (relative paths)
    input_dir = os.path.join(cainine_root, "all_data")
    output_dir = os.path.join(script_dir, "PreProcessedData")

    # Check if raw data exists
    if not os.path.exists(input_dir):
        print("="*70)
        print("ERROR: RAW DATA NOT FOUND")
        print("="*70)
        print(f"\nThe raw canine ECG data folder does not exist:")
        print(f"  Expected location: {input_dir}")
        print(f"\nThe raw data is stored on Google Drive (too large for Git).")
        print(f"\nPlease download it first by running:")
        print(f"  cd {cainine_root}")
        print(f"  python download_data.py")
        print(f"\nThis will download and extract the data automatically.")
        print("="*70)
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all signal files (without _labels suffix)
    signal_files = []
    for file_path in glob.glob(os.path.join(input_dir, "*.csv")):
        if not file_path.endswith("_labels.csv"):
            signal_files.append(file_path)

    signal_files.sort()

    # Extract base names for listing
    available_files = [os.path.splitext(os.path.basename(f))[0] for f in signal_files]

    # List files mode
    if args.list_files:
        print("="*60)
        print("AVAILABLE FILES")
        print("="*60)
        print(f"\nFound {len(available_files)} signal files:")
        for file_num in available_files:
            print(f"  - {file_num}")
        print(f"\nDefault test files: {args.test_files}")
        print("\nTo use custom test files:")
        print("  python enhanced_preprocessing_choose.py --test-files 301 413")
        return

    # Use custom test files from command line
    TEST_FILES = args.test_files

    print("="*60)
    print("CANINE ECG ENHANCED PREPROCESSING (CUSTOM TEST FILES)")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nTest files: {TEST_FILES}")
    train_files = [f for f in available_files if f not in TEST_FILES]
    print(f"Train files: {train_files}")
    print(f"\nFound {len(signal_files)} signal files to process")

    total_written = total_skipped = 0
    train_files_written = 0
    train_files_skipped = 0
    test_files_written = 0
    test_files_skipped = 0

    for signal_file in signal_files:
        # Get base filename (e.g., "100" from "100.csv")
        base_name = os.path.splitext(os.path.basename(signal_file))[0]
        labels_file = os.path.join(input_dir, f"{base_name}_labels.csv")

        if not os.path.exists(labels_file):
            print(f"Warning: No labels file found for {signal_file}, skipping...")
            continue

        # Determine if this file goes to train or test
        is_test_file = base_name in TEST_FILES

        # Add suffix to distinguish train/test files
        if is_test_file:
            output_filename = f"{base_name}_beats_enhanced_TEST.csv"
            print(f"\n[TEST FILE] Processing {base_name}...")
        else:
            output_filename = f"{base_name}_beats_enhanced_TRAIN.csv"

        # Create file-specific config
        file_config = PreprocessConfig(
            signal_file=signal_file,
            labels_file=labels_file,
            output_file=os.path.join(output_dir, output_filename),
            window_len=base_config.window_len,
            sampling_rate=base_config.sampling_rate,
            use_global_normalization=base_config.use_global_normalization,
            use_per_beat_normalization=base_config.use_per_beat_normalization,
            detect_s_peaks=base_config.detect_s_peaks,
            sig_col=base_config.sig_col,
            idx_col=base_config.idx_col,
            lab_col=base_config.lab_col
        )

        try:
            written, skipped = process_single_file(file_config)

            if is_test_file:
                test_files_written += written
                test_files_skipped += skipped
            else:
                train_files_written += written
                train_files_skipped += skipped

            total_written += written
            total_skipped += skipped
        except Exception as e:
            print(f"Error processing {signal_file}: {e}")
            continue

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files processed: {len(signal_files)}")
    print(f"Total beats written: {total_written}")
    print(f"Total beats skipped: {total_skipped}")

    print(f"\nTRAIN SET:")
    print(f"  Beats written: {train_files_written:,}")
    print(f"  Beats skipped: {train_files_skipped:,}")

    print(f"\nTEST SET:")
    print(f"  Beats written: {test_files_written:,}")
    print(f"  Beats skipped: {test_files_skipped:,}")

    # Combine TRAIN files
    if train_files_written > 0:
        print("\n[combining] Creating TRAIN dataset...")
        train_file = os.path.join(output_dir, "canine_train.csv")
        try:
            combine_batch_files(
                os.path.join(output_dir, "*_beats_enhanced_TRAIN.csv"),
                train_file,
                base_config
            )
            print(f"[done] Train dataset saved to: {train_file}")
        except Exception as e:
            print(f"Error combining train files: {e}")

    # Combine TEST files
    if test_files_written > 0:
        print("\n[combining] Creating TEST dataset...")
        test_file = os.path.join(output_dir, "canine_test.csv")
        try:
            combine_batch_files(
                os.path.join(output_dir, "*_beats_enhanced_TEST.csv"),
                test_file,
                base_config
            )
            print(f"[done] Test dataset saved to: {test_file}")
        except Exception as e:
            print(f"Error combining test files: {e}")

    if total_written == 0:
        print("No data to combine.")

if __name__ == "__main__":
    main()