#!/usr/bin/env python3
"""
Batch convert multiple canine ECG signal+label files into MITBIH-style beat CSVs.
"""

import os
import glob
from PreProcessing import convert, parse_bool   # reuse the single-file script

def batch_convert(
    data_dir: str,
    out_dir: str,
    window_len: int = 187,
    normalize: bool = False,
    limit: int = None
):
    os.makedirs(out_dir, exist_ok=True)

    # Find all signal CSVs (skip *_labels.csv)
    signals = [f for f in glob.glob(os.path.join(data_dir, "*.csv")) if "_labels" not in f]

    for sig_path in signals:
        base = os.path.splitext(os.path.basename(sig_path))[0]  # e.g. "100"
        label_path = os.path.join(data_dir, f"{base}_labels.csv")
        if not os.path.exists(label_path):
            print(f"[warn] No labels file found for {base}, skipping")
            continue

        out_path = os.path.join(out_dir, f"{base}_mitbih.csv")
        print(f"[info] Converting {base}:")
        convert(
            signal_path=sig_path,
            labels_path=label_path,
            out_path=out_path,
            window_len=window_len,
            normalize=normalize,
            limit=limit,
            verbose_every=5000
        )

if __name__ == "__main__":
    # Example usage
    data_dir = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\Combined_data"   # folder with 100.csv, 100_labels.csv, 101.csv, 101_labels.csv, ...
    out_dir  = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\PreProcessedData"

    batch_convert(
        data_dir=data_dir,
        out_dir=out_dir,
        window_len=187,     
        normalize=False,    
        limit=None          
    )
