import csv
import numpy as np
import pandas as pd

# ==== SETTINGS (edit these directly) ====
signal_file = "101_signal.csv"
labels_file = "101_labels.csv"
out_file    = "101_beats.csv"

sig_col = "ecg"        # column in signal file
idx_col = "Samples"    # R-peak index column
lab_col = "Label"      # Label column
window_len = 187       # samples per beat
normalize = True       # z-score each beat
fs = None              # set to e.g. 250 if labels are in seconds
# ========================================

LETTER_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

def map_label(lbl: str) -> int:
    s = str(lbl).strip()
    if not s:
        return 4
    return LETTER_TO_ID.get(s[0].upper(), 4)

def extract_window(sig, center, L):
    half = L // 2
    start = center - half
    end   = start + L
    left_pad  = max(0, -start)
    right_pad = max(0, end - len(sig))
    start_c = max(0, start)
    end_c   = min(len(sig), end)
    w = sig[start_c:end_c].astype('float32')
    if left_pad or right_pad:
        w = np.pad(w, (left_pad, right_pad), mode='constant')
    return w

def per_beat_zscore(x):
    mu, sd = float(x.mean()), float(x.std())
    return (x - mu) / (sd + 1e-8)

# --- Load signal ---
S = pd.read_csv(signal_file)
sig = S[sig_col].astype('float32').to_numpy()

# --- Load labels ---
L = pd.read_csv(labels_file)
r_raw = pd.to_numeric(L[idx_col], errors="coerce")
labs  = L[lab_col].astype(str).fillna("Q")

if fs is not None:
    r_samp = np.round(r_raw.astype("float64") * fs)
else:
    r_samp = r_raw.astype("float64")

good = np.isfinite(r_samp)
r_samp = np.round(r_samp[good]).astype(np.int64)
labs   = labs[good].to_numpy()

# --- Convert ---
written = skipped = 0
with open(out_file, "w", newline="") as f:
    w = csv.writer(f)
    header = [f"s{i}" for i in range(window_len)] + ["label"]
    w.writerow(header)
    for r, lab in zip(r_samp, labs):
        if r < 0 or r >= len(sig):
            skipped += 1
            continue
        beat = extract_window(sig, int(r), window_len)
        if normalize:
            beat = per_beat_zscore(beat)
        cls = map_label(lab)
        w.writerow([*beat.tolist(), int(cls)])
        written += 1

print(f"[done] wrote {written} rows to {out_file}; skipped={skipped}")
