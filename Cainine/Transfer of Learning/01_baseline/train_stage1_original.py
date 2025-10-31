"""
Transfer learning for canine ECG beats using a pretrained MIT-BIH model.
Key features:
- Binary task: N (0) vs V (1)  <-- original V (2) is remapped to 1
- Oversampling of class (V) in the input pipeline
- Freezing of early layers for transfer learning
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)

# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # Get script directory for relative paths
    SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

    # --- Paths (relative) ---
    CANINE_TRAIN_GLOB: str = os.path.join(PROJECT_ROOT, "Cainine", "DataPreProcessing_TEST", "PreProcessedData", "canine_train.csv")
    BASE_MODEL_PATH:   str = os.path.join(PROJECT_ROOT, "MITBIH", "model_arxiv-1805-00794", "best_model.h5")
    OUTPUT_DIR:        str = os.path.join(SCRIPT_DIR, "models")

    # --- Data layout ---
    LABEL_COL_NAME:    str = "label"   # last column; integer-coded
    BASE_INPUT_LEN:    int = 187       # MIT-BIH model expects 187 samples
    BASE_INPUT_DEPTH:  int = 1

    # --- Task mode ---
    BINARY_MODE:       bool = True     # collapse to N vs Arrhythmia (S+V)
    POSITIVE_ORIG:     int  = 2        # original "V" label id
    INCLUDE_S_BEATS:   bool = True     # include S-beats as positive class (arrhythmia)
    # After remap: 0 (N) stays 0, V and S become 1 (Arrhythmia), others dropped

    # --- Oversampling ---
    USE_OVERSAMPLING:  bool = True
    TARGET_POS_RATIO:  float = 0.50    # aim ~50% V per epoch/batch

    # --- Freezing early layers ---
    FREEZE_EARLY_LAYERS: bool  = True
    FREEZE_FRACTION:      float = 0.50  # freeze first 50% of layers (excl. new head)

    # --- Training ---
    BATCH_SIZE:        int = 1024
    EPOCHS:            int = 10
    LR_FULL_FT:        float = 1e-3
    SHUFFLE_BUFFER:    int = 100_000
    VAL_SPLIT:         float = 0.15
    SEED:              int = 1337

    # --- Length adaptation ---
    CROP_INSTEAD_OF_PAD: bool = True

    # --- Eval ---
    PRINT_THRESHOLD_SWEEP: bool = True


# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_repro(seed=CFG.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def adapt_length(X: np.ndarray, target_len: int, crop_instead: bool = True) -> np.ndarray:
    """Center-crop or zero-pad to match target_len. X: (N, L) or (N, L, 1)"""
    if X.ndim == 2:
        X = X[:, :, None]
    N, L, C = X.shape
    if L == target_len:
        return X
    if L > target_len and crop_instead:
        start = (L - target_len) // 2
        end   = start + target_len
        return X[:, start:end, :]
    # pad
    pad_left  = max(0, (target_len - L) // 2)
    pad_right = max(0, target_len - L - pad_left)
    return np.pad(X, ((0,0),(pad_left,pad_right),(0,0)), mode="constant")

def filter_and_remap_binary(X: np.ndarray, y: np.ndarray, positive_orig: int, include_s_beats: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary classification: Normal vs Arrhythmia
    - Keep N (0) as 0 (Normal)
    - Map V (2) and optionally S (1) to 1 (Arrhythmia)
    - Drop other classes (F, Q)
    """
    if include_s_beats:
        # Include both S and V as arrhythmia (positive class)
        normal_classes = [0]  # N
        arrhythmia_classes = [1, positive_orig]  # S, V
        keep_classes = normal_classes + arrhythmia_classes
    else:
        # Original approach: only N vs V
        keep_classes = [0, positive_orig]  # N, V
        arrhythmia_classes = [positive_orig]

    # Filter to keep only relevant classes
    mask = np.isin(y, keep_classes)
    X2, y2 = X[mask], y[mask]

    # Remap: Normal=0, Arrhythmia=1
    if include_s_beats:
        y2_binary = np.isin(y2, arrhythmia_classes).astype(np.int64)
    else:
        y2_binary = (y2 == positive_orig).astype(np.int64)

    print(f"[binary] original classes: {dict(zip(*np.unique(y2, return_counts=True)))}")
    print(f"[binary] binary mapping: {dict(zip(*np.unique(y2_binary, return_counts=True)))}")

    return X2, y2_binary

def load_csv_files(path_glob: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one or more CSV files that each contain rows like:
    s0, s1, ..., sL-1, label
    Returns X (float32), y (int64).

    Enhanced preprocessing creates headerless files, so we handle that case.
    """
    files = glob.glob(path_glob)
    if not files:
        raise FileNotFoundError(f"No files matched: {path_glob}")
    print(f"[data] glob='{path_glob}' matched {len(files)} file(s):")
    for fp in files[:10]:
        print(f"   - {fp}")

    X_parts, y_parts = [], []
    for fp in files:
        # Enhanced preprocessing creates headerless files
        # Load as headerless and assign column names
        df = pd.read_csv(fp, header=None, low_memory=False)
        ncols = df.shape[1]
        if ncols < 2:
            raise ValueError(f"{fp}: expected at least 2 columns, got {ncols}")

        # Assign column names: s0, s1, ..., s{L-1}, label
        feat_names = [f"s{i}" for i in range(ncols - 1)]
        cols = feat_names + [label_col]
        df.columns = cols

        # Extract features/labels
        feat_cols = [c for c in df.columns if c != label_col]
        X = df[feat_cols].to_numpy(dtype=np.float32)
        y = df[label_col].to_numpy(dtype=np.int64)

        X_parts.append(X)
        y_parts.append(y)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    print(f"[data] loaded X shape={X_all.shape}, y shape={y_all.shape} (unique labels: {sorted(np.unique(y_all).tolist())})")
    return X_all, y_all



# -------------------------
# tf.data pipeline with oversampling
# -------------------------
def make_tfds(X: np.ndarray, y: np.ndarray, train: bool) -> tf.data.Dataset:
    """
    If training + binary + oversampling: duplicate minority class to reach target ratio.
    """
    X = X.astype(np.float32)
    if train and CFG.BINARY_MODE and CFG.USE_OVERSAMPLING:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        if len(idx_pos) == 0 or len(idx_neg) == 0:
            sel = np.arange(len(y))
        else:
            n_total = len(y)
            tgt_pos = int(CFG.TARGET_POS_RATIO * n_total)
            tgt_neg = n_total - tgt_pos
            rep_pos = int(math.ceil(tgt_pos / max(1, len(idx_pos))))
            rep_neg = int(math.ceil(tgt_neg / max(1, len(idx_neg))))
            sel_pos = np.tile(idx_pos, rep_pos)[:tgt_pos]
            sel_neg = np.tile(idx_neg, rep_neg)[:tgt_neg]
            sel = np.concatenate([sel_pos, sel_neg])
            np.random.shuffle(sel)
        X_use, y_use = X[sel], y[sel]
    else:
        X_use, y_use = X, y

    ds = tf.data.Dataset.from_tensor_slices((X_use, y_use))
    if train:
        ds = ds.shuffle(CFG.SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    ds = ds.batch(CFG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# -------------------------
# Model: load base, swap head, freeze early layers
# -------------------------
def load_base_model(path: str) -> tf.keras.Model:
    print(f"[model] loading base model from: {path}")
    return load_model(path, compile=False)

def swap_head_build(model_base: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    """
    Replace the last classification layer(s) with a new head.
    Assumes the penultimate layer is the one before final activation.
    """
    penultimate = model_base.layers[-2].output  # works for ... -> Dense -> Softmax/Sigmoid
    if num_classes == 2:
        logits = tf.keras.layers.Dense(1, name="canine_logit", dtype="float32")(penultimate)
        probs  = tf.keras.layers.Activation("sigmoid", name="canine_prob", dtype="float32")(logits)
        model  = tf.keras.Model(inputs=model_base.input, outputs=probs, name="canine_finetuned_bin")
    else:
        logits = tf.keras.layers.Dense(num_classes, name="canine_logits", dtype="float32")(penultimate)
        probs  = tf.keras.layers.Softmax(name="canine_probs", dtype="float32")(logits)
        model  = tf.keras.Model(inputs=model_base.input, outputs=probs, name="canine_finetuned_mc")
    # start with all trainable; we'll freeze early block next
    for l in model.layers:
        l.trainable = True
    return model

def freeze_early_layers(model: tf.keras.Model, fraction: float = 0.5):
    """
    Freeze the first `fraction` of layers (by index). Keep new head trainable.
    """
    n = len(model.layers)
    k = int(max(0.0, min(1.0, fraction)) * n)
    for lyr in model.layers[:k]:
        lyr.trainable = False
    for name in ("canine_logit", "canine_prob", "canine_logits", "canine_probs"):
        try:
            model.get_layer(name).trainable = True
        except Exception:
            pass
    # Quick summary of trainable counts
    trainable_params = int(np.sum([np.sum([tf.size(w).numpy() for w in lyr.trainable_weights]) for lyr in model.layers]))
    nontrain_params  = int(np.sum([np.sum([tf.size(w).numpy() for w in lyr.non_trainable_weights]) for lyr in model.layers]))
    print(f"[freeze] trainable params ≈ {trainable_params:,} | non-trainable params ≈ {nontrain_params:,}")


# -------------------------
# Callbacks (progress bar is verbose=1; this just logs LR/loss)
# -------------------------
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr = float(tf.keras.backend.get_value(opt.learning_rate))
        l  = logs or {}
        print(f"\n[epoch {epoch+1}] lr={lr:.3e} | loss={l.get('loss'):.4f} val_loss={l.get('val_loss'):.4f}")


# -------------------------
# Evaluation helpers
# -------------------------
def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, name="val"):
    y_true = y_true.astype(int)
    y_prob = y_prob.reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n[{name}] Confusion @0.5:\n{confusion_matrix(y_true, y_pred)}")
    print(classification_report(y_true, y_pred, target_names=["N","V"], digits=3))

    if CFG.PRINT_THRESHOLD_SWEEP:
        p, r, t = precision_recall_curve(y_true, y_prob)
        f1 = 2*p*r/(p+r+1e-12)
        best_idx = int(np.nanargmax(f1))
        best_thr = float(t[best_idx]) if best_idx < len(t) else 0.5
        pr_auc = float(auc(r, p))
        print(f"[{name}] PR-AUC={pr_auc:.3f} | Best-F1@thr≈{best_thr:.3f} "
              f"(P={p[best_idx]:.3f}, R={r[best_idx]:.3f}, F1={f1[best_idx]:.3f})")


# -------------------------
# Main
# -------------------------
def main():
    set_repro(CFG.SEED)
    ensure_dir(CFG.OUTPUT_DIR)

    print("[step] Loading canine data …")
    X, y = load_csv_files(CFG.CANINE_TRAIN_GLOB, CFG.LABEL_COL_NAME)

    # Binary remap: Normal vs Arrhythmia (including S-beats)
    if CFG.BINARY_MODE:
        X, y = filter_and_remap_binary(X, y, positive_orig=CFG.POSITIVE_ORIG, include_s_beats=CFG.INCLUDE_S_BEATS)
        print(f"[data] final binary distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Adapt length to base model input (N, 186, 1)
    X = adapt_length(X, CFG.BASE_INPUT_LEN, crop_instead=CFG.CROP_INSTEAD_OF_PAD).astype(np.float32)

    # Split train / val
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=CFG.VAL_SPLIT, random_state=CFG.SEED, stratify=y if len(np.unique(y))>1 else None
    )
    print(f"[data] train: {X_tr.shape}, val: {X_val.shape}")

    # Datasets
    ds_tr  = make_tfds(X_tr, y_tr, train=True)
    ds_val = make_tfds(X_val, y_val, train=False)

    # Load base model, swap head, freeze
    base = load_base_model(CFG.BASE_MODEL_PATH)
    model = swap_head_build(base, num_classes=(2 if CFG.BINARY_MODE else len(np.unique(y))))
    if CFG.FREEZE_EARLY_LAYERS:
        freeze_early_layers(model, fraction=CFG.FREEZE_FRACTION)

    # Compile
    if CFG.BINARY_MODE:
        loss_fn = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall_v")]
    else:
        loss_fn = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LR_FULL_FT),
        loss=loss_fn,
        metrics=metrics
    )

    model.summary(line_length=120)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CFG.OUTPUT_DIR, "checkpoint_best.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1
        ),
        LRLogger(),
    ]

    print("[step] Training … (oversampling={} | freeze_early={})".format(
        CFG.USE_OVERSAMPLING, CFG.FREEZE_EARLY_LAYERS
    ))
    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=CFG.EPOCHS,
        callbacks=callbacks,
        verbose=1,   # progress bar
    )

    # Evaluate (binary)
    if CFG.BINARY_MODE:
        print("[step] Evaluating on validation set …")
        y_val_prob = model.predict(ds_val, verbose=0).reshape(-1)
        eval_binary(y_val, y_val_prob, name="val")

        # Save probabilities for further analysis
        np.save(os.path.join(CFG.OUTPUT_DIR, "val_probs.npy"), y_val_prob)
        np.save(os.path.join(CFG.OUTPUT_DIR, "val_labels.npy"), y_val)
        model.save(os.path.join(CFG.OUTPUT_DIR, "canine_binary_original.h5"))
    else:
        print("[step] Evaluating (multi-class) …")
        val_metrics = model.evaluate(ds_val, verbose=0)
        print(dict(zip(model.metrics_names, val_metrics)))
        model.save(os.path.join(CFG.OUTPUT_DIR, "canine_multiclass_original.h5"))

if __name__ == "__main__":
    main()
