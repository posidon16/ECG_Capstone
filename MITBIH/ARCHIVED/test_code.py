import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

MODEL_FILE = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\MITBIH\Best_Model_arxiv-1805-00794\best_model.h5"   # <- adjust if different
model = load_model(MODEL_FILE, compile=False)

def _expected_len_and_channels(m):
    ish = m.input_shape
    if isinstance(ish, list):
        ish = ish[0]
    if len(ish) == 3:
        _, L, C = ish
        return int(L), int(C)
    elif len(ish) == 2:
        _, L = ish
        return int(L), 1
    raise ValueError(f"Unsupported input shape: {ish}")

_EXPECTED_LEN, _EXPECTED_CH = _expected_len_and_channels(model)

def _trim_or_pad(x1d: np.ndarray, L: int) -> np.ndarray:
    x = x1d.ravel().astype("float32")
    if x.shape[0] > L:
        x = x[:L]
    elif x.shape[0] < L:
        x = np.pad(x, (0, L - x.shape[0]), mode="constant")
    return x

def _per_beat_zscore(x1d: np.ndarray) -> np.ndarray:
    mu = x1d.mean()
    sd = x1d.std()
    return (x1d - mu) / (sd + 1e-8)

def _prepare_batch(X_2d: np.ndarray, zscore=True) -> np.ndarray:
    # X_2d: shape (N, features)
    X = X_2d.astype("float32")
    # adjust length
    X = np.stack([_trim_or_pad(r, _EXPECTED_LEN) for r in X], axis=0)
    # per-beat zscore (typical for this task)
    if zscore:
        X = np.stack([_per_beat_zscore(r) for r in X], axis=0)
    # add channel dim if needed
    if _EXPECTED_CH == 1:
        X = X[:, :, None]   # (N, L, 1)
    return X

def predict_single_beat(beat_1d, zscore=True):
    x = _trim_or_pad(np.asarray(beat_1d, dtype="float32"), _EXPECTED_LEN)
    if zscore:
        x = _per_beat_zscore(x)
    x = x[None, :, None] if _EXPECTED_CH == 1 else x[None, :]
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

def predict_from_kaggle_csv_row(csv_path: str, row_idx: int, zscore=True):
    df = pd.read_csv(csv_path, header=None).values
    beat = df[row_idx, :-1].astype("float32")
    true_idx = int(df[row_idx, -1])
    pred_idx, probs = predict_single_beat(beat, zscore=zscore)
    return pred_idx, probs, true_idx

def quick_eval(csv_path: str, n_rows: int = 2000, zscore=True):
    """Fast sanity check: evaluate accuracy on the first n_rows of the CSV."""
    df = pd.read_csv(csv_path, header=None).values
    X = df[:n_rows, :-1].astype("float32")
    y = df[:n_rows, -1].astype(int)
    Xp = _prepare_batch(X, zscore=zscore)
    probs = model.predict(Xp, verbose=0)
    pred = probs.argmax(axis=1)
    acc = (pred == y).mean()
    print(f"Sanity accuracy on first {len(y)} rows: {acc:.3f}")
    # Optional: class distribution to spot collapse
    unique, counts = np.unique(pred, return_counts=True)
    print("Pred class counts:", dict(zip(unique.tolist(), counts.tolist())))


#--------------------------------------Full Test--------------------------------------------------------------------


# OPTIONAL: map indices to label names if you want readable reports
INDEX_TO_LABEL = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

def full_file_eval(csv_path: str, zscore=True, limit=None):
    """
    Evaluate the whole CSV (or first `limit` rows) using the same preprocessing as predict_*.
    Prints class distribution, confusion matrix, and classification report.
    """
    df = pd.read_csv(csv_path, header=None).values
    if limit is not None:
        df = df[:limit]

    X = df[:, :-1].astype("float32")
    y_true = df[:, -1].astype(int)

    # Reuse the same prep as in quick_eval
    Xp = _prepare_batch(X, zscore=zscore)
    probs = model.predict(Xp, verbose=0)
    y_pred = probs.argmax(axis=1)

    # Distributions
    uniq_t, cnt_t = np.unique(y_true, return_counts=True)
    uniq_p, cnt_p = np.unique(y_pred, return_counts=True)
    print("True class counts:", dict(zip(uniq_t.tolist(), cnt_t.tolist())))
    print("Pred class counts:", dict(zip(uniq_p.tolist(), cnt_p.tolist())))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    print("\nConfusion matrix (rows=true, cols=pred) [0..4]:\n", cm)

    # Classification report
    target_names = [INDEX_TO_LABEL.get(i, str(i)) for i in range(5)]
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[0,1,2,3,4], target_names=target_names, digits=3))

def find_examples_by_true_label(csv_path: str, label: int, k: int = 5, zscore=True):
    """
    Grab k rows with a given TRUE label and print model predictions + probs.
    Useful to sanity-check minority classes.
    """
    df = pd.read_csv(csv_path, header=None).values
    indices = np.where(df[:, -1].astype(int) == label)[0]
    if len(indices) == 0:
        print(f"No rows with true label {label}")
        return
    for i in indices[:k]:
        beat = df[i, :-1].astype("float32")
        true_idx = int(df[i, -1])
        pred_idx, probs = predict_single_beat(beat, zscore=zscore)
        print(f"Row {i} | true={true_idx} ({INDEX_TO_LABEL.get(true_idx)}) "
              f"| pred={pred_idx} ({INDEX_TO_LABEL.get(pred_idx)}) "
              f"| top prob={probs.max():.3f}")
