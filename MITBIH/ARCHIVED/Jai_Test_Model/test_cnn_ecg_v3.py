import os
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import pandas as pd

MITBIH_PATH = "mit-bih"
MODEL_FILE = "cnn_ecg_model_best.h5"
LABEL_CLASSES_FILE = "label_classes.npy"
WINDOW_SIZE = 180

TEST_RECORDS = [
    '200','201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221','222',
    '223','228','230','231','232','233','234'
]

def extract_beats(record_name, window_size=180):
    record = wfdb.rdrecord(os.path.join(MITBIH_PATH, record_name))
    annotation = wfdb.rdann(os.path.join(MITBIH_PATH, record_name), 'atr')
    sig = record.p_signal[:, 0]
    beats, labels = [], []
    for i, r in enumerate(annotation.sample):
        sym = annotation.symbol[i]
        if sym in ['N', 'L', 'R', 'e', 'j']:
            label = 'N'
        elif sym in ['A', 'a', 'J', 'S']:
            label = 'S'
        elif sym in ['V', 'E']:
            label = 'V'
        elif sym in ['F']:
            label = 'F'
        else:
            continue
        start = r - window_size // 2
        end = r + window_size // 2
        if start >= 0 and end <= len(sig):
            beat = sig[start:end]
            beats.append(beat)
            labels.append(label)
    return beats, labels

def load_dataset(records):
    all_beats = []
    all_labels = []
    record_map = defaultdict(list)
    for rec in records:
        b, l = extract_beats(rec)
        all_beats.extend(b)
        all_labels.extend(l)
        record_map[rec] = l
    return np.array(all_beats), np.array(all_labels), record_map

# Load model and encoder
model = tf.keras.models.load_model(MODEL_FILE)
label_classes = np.load(LABEL_CLASSES_FILE)
le = LabelEncoder()
le.classes_ = label_classes

# Load data
X_raw, y_raw, record_map = load_dataset(TEST_RECORDS)
X = (X_raw - np.mean(X_raw, axis=1, keepdims=True)) / np.std(X_raw, axis=1, keepdims=True)
X = X[..., np.newaxis]
y = le.transform(y_raw)

# Predict and evaluate
y_pred = model.predict(X).argmax(axis=1)
print("Classification Report:")
print(classification_report(
    y, y_pred,
    labels=list(range(len(label_classes))),
    target_names=label_classes,
    zero_division=0
))


# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_classes, yticklabels=label_classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Per-record breakdown
record_results = {}
start_idx = 0
for rec in TEST_RECORDS:
    labels = record_map[rec]
    if not labels:
        continue
    end_idx = start_idx + len(labels)
    rec_X = X[start_idx:end_idx]
    rec_y_true = y[start_idx:end_idx]
    rec_y_pred = model.predict(rec_X).argmax(axis=1)
    rec_report = classification_report(
        rec_y_true,
        rec_y_pred,
        labels=list(range(len(label_classes))),
        target_names=label_classes,
        output_dict=True,
        zero_division=0  # prevents divide-by-zero warnings
    )
    record_results[rec] = rec_report
    start_idx = end_idx

# F1-score table
f1_per_record = {
    rec: {cls: round(r["f1-score"], 2) for cls, r in rep.items() if cls in label_classes}
    for rec, rep in record_results.items()
}
f1_df = pd.DataFrame(f1_per_record).T
f1_df.to_csv("per_record_f1_scores.csv")
print("\nSaved F1 scores per record to 'per_record_f1_scores.csv'")
