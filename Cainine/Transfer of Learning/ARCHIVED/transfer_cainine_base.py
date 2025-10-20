"""
Canine ECG Transfer Learning from MIT-BIH Pretrained Model
"""

import os, glob
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# =========================
# ========= CONFIG ========
# =========================
@dataclass
class Config:
    # --- Pretrained model ---
    PRETRAINED_MODEL_PATH: str = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\Transfer of Learning\best_model.h5"  # preferred: full model (arch+weights)
    PRETRAINED_WEIGHTS_PATH: str | None = None                       # optional: weights-only (if above not available)

    # --- Data (single file or glob pattern) ---
    CANINE_TRAIN_GLOB: str = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\Transfer of Learning\canine_all_mitbih.csv"
    CANINE_VAL_GLOB:   str = r""           # leave empty to auto-split from train
    CANINE_TEST_GLOB:  str = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\Cainine\PreProcessedData\100_mitbih.csv"           # optional

    AUTO_VAL_SPLIT: float = 0.15           # used only if CANINE_VAL_GLOB is empty
    LABEL_COL_NAME: str = "label"          # used when header is present

    # --- Data shape (will be aligned to model’s input automatically) ---
    WINDOW_LEN: int = 187                  # what your CSVs contain
    CHANNELS: int   = 1                    # single-lead

    # --- Labels ---
    NUM_CANINE_CLASSES: int = 5            # N/S/V/F/Q -> 0..4

    # --- Training ---
    BATCH_SIZE: int = 256
    EPOCHS: int = 30
    LR_FULL_FT: float = 1e-4
    EARLY_STOP_PATIENCE: int = 6
    REDUCE_LR_PATIENCE: int = 2
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-6
    SHUFFLE_BUFFER: int = 8192

    # --- Saving ---
    OUTPUT_DIR: str = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\canine_tl_out"
    OUTPUT_MODEL_NAME: str = "canine_tl_full_finetune.h5"

CFG = Config()

# =========================
# ===== UTILITIES =========
# =========================
def load_csv_files(file_glob: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSVs. Handles:
      - header with 'label' column, or
      - no header (last column is label).
    Returns:
      X: (N, WINDOW_LEN) float32
      y: (N,) int64
    """
    files: list[str] = []
    if file_glob:
        files = [file_glob] if os.path.isfile(file_glob) else sorted(glob.glob(file_glob))
    print(f"[data] glob='{file_glob}' matched {len(files)} file(s):")
    for fp in files: print("   -", fp)
    if not files:
        raise FileNotFoundError(f"No CSV files matched '{file_glob}'")

    X_list, y_list = [], []
    expected = CFG.WINDOW_LEN * CFG.CHANNELS

    for fp in files:
        # Try header mode first
        df = pd.read_csv(fp)
        used_header = CFG.LABEL_COL_NAME in df.columns
        if used_header:
            y = df[CFG.LABEL_COL_NAME].to_numpy()
            X = df.drop(columns=[CFG.LABEL_COL_NAME]).to_numpy(dtype=np.float32, copy=False)
        else:
            # No header → last column is label
            df = pd.read_csv(fp, header=None)
            if df.shape[1] < 2:
                df = pd.read_csv(fp, header=None, engine="python")
            X = df.iloc[:, :-1].to_numpy(dtype=np.float32, copy=False)
            y = df.iloc[:, -1].to_numpy()

        if X.shape[1] != expected:
            raise ValueError(
                f"{fp}: got {X.shape[1]} feature cols, expected {expected} "
                f"(WINDOW_LEN*CHANNELS = {CFG.WINDOW_LEN}*{CFG.CHANNELS})."
            )

        if y.dtype.kind in "UOS":
            raise ValueError(f"{fp}: label column is not numeric (dtype={y.dtype}). Map labels to ints 0..{CFG.NUM_CANINE_CLASSES-1}.")

        X_list.append(X)
        y_list.append(y.astype(np.int64, copy=False))

    X = np.vstack(X_list)   # (N, L)
    y = np.concatenate(y_list)  # (N,)
    return X, y

def ensure_3d(X: np.ndarray) -> np.ndarray:
    """Ensure input has shape (N, L, C)."""
    if X.ndim == 2:
        X = X[..., np.newaxis]
    return X

def adapt_length(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    Center-crop or zero-pad along time axis to match target_len.
    X shape: (N, L, C)
    """
    L = X.shape[1]
    if L == target_len:
        return X
    if L > target_len:
        # center-crop
        start = (L - target_len) // 2
        end   = start + target_len
        return X[:, start:end, :]
    else:
        # symmetric zero-pad
        pad_left  = (target_len - L) // 2
        pad_right = target_len - L - pad_left
        return np.pad(X, ((0,0),(pad_left,pad_right),(0,0)), mode="constant")

def compute_class_weights_robust(y: np.ndarray, num_classes: int) -> dict:
    """
    Balanced weights for classes present in y; fill absent classes with 0.0.
    """
    present = np.unique(y)
    w_present = compute_class_weight(class_weight="balanced", classes=present, y=y)
    weights = {int(c): float(w) for c, w in zip(present, w_present)}
    for c in range(num_classes):
        weights.setdefault(c, 0.0)
    return weights

def make_tfds(X: np.ndarray, y: np.ndarray, train: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(CFG.SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    return ds.batch(CFG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# === PRETRAINED MODEL ====
# =========================
def build_model_from_weights_only() -> tf.keras.Model:
    
    L, C = CFG.WINDOW_LEN, CFG.CHANNELS  # This will be adjusted later to match loaded weights if needed
    inp = tf.keras.Input(shape=(L, C), name="ecg")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(5, activation="softmax", name="mitbih_head")(x)
    return tf.keras.Model(inp, out, name="mitbih_backbone_stub")

def load_pretrained_backbone() -> tf.keras.Model:
    """
    Prefer loading a full model. If unavailable, rebuild arch and load weights.
    """
    if os.path.isfile(CFG.PRETRAINED_MODEL_PATH):
        try:
            print(f"[step] Loading full Keras model: {CFG.PRETRAINED_MODEL_PATH}")
            model = tf.keras.models.load_model(CFG.PRETRAINED_MODEL_PATH, compile=False)
            return model
        except Exception as e:
            print(f"[warn] Failed to load full model: {e}. Will try weights-only.")
    if not CFG.PRETRAINED_WEIGHTS_PATH:
        raise FileNotFoundError(
            "Could not load full Keras model and PRETRAINED_WEIGHTS_PATH is not set."
        )
    model = build_model_from_weights_only()
    print(f"[step] Loading weights: {CFG.PRETRAINED_WEIGHTS_PATH}")
    model.load_weights(CFG.PRETRAINED_WEIGHTS_PATH)
    return model

def swap_head_full_finetune(base: tf.keras.Model) -> tf.keras.Model:
    """
    Replace classifier head with a 5-class canine head; fine-tune ALL layers.
    """
    # Grab penultimate layer output
    penultimate = base.layers[-2].output
    logits = tf.keras.layers.Dense(CFG.NUM_CANINE_CLASSES, name="canine_logits")(penultimate)
    probs  = tf.keras.layers.Softmax(name="canine_probs")(logits)
    model  = tf.keras.Model(inputs=base.input, outputs=probs, name="canine_finetuned")

    for l in model.layers:
        l.trainable = True
    return model

# =========================
# ========= MAIN ==========
# =========================
def main():
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("[step] Loading canine data …")
    X_tr, y_tr = load_csv_files(CFG.CANINE_TRAIN_GLOB)

    if CFG.CANINE_VAL_GLOB.strip():
        X_val, y_val = load_csv_files(CFG.CANINE_VAL_GLOB)
    else:
        if X_tr.shape[0] < 2:
            raise ValueError("Need at least 2 samples to create a validation split.")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=CFG.AUTO_VAL_SPLIT, random_state=42, stratify=y_tr
        )

    X_te, y_te = (None, None)
    if CFG.CANINE_TEST_GLOB.strip():
        X_te, y_te = load_csv_files(CFG.CANINE_TEST_GLOB)

    print(f"[data] Train raw: {X_tr.shape}, Val raw: {X_val.shape}"
          + (f", Test raw: {X_te.shape}" if X_te is not None else ""))

    # --- Load pretrained model ---
    base = load_pretrained_backbone()
    print("[step] Pretrained model summary:")
    base.summary()

    # --- Harmonize data to model input (length & channels) ---
    exp_len = int(base.input_shape[1])      # e.g., 186
    exp_ch  = int(base.input_shape[2]) if len(base.input_shape) == 3 else 1

    X_tr  = ensure_3d(X_tr);  X_val = ensure_3d(X_val)
    if X_te is not None: X_te = ensure_3d(X_te)

    X_tr  = adapt_length(X_tr,  exp_len)
    X_val = adapt_length(X_val, exp_len)
    if X_te is not None: X_te = adapt_length(X_te, exp_len)

    if CFG.WINDOW_LEN != exp_len or CFG.CHANNELS != exp_ch:
        print(f"[info] adjusting CFG: WINDOW_LEN {CFG.WINDOW_LEN}→{exp_len}, CHANNELS {CFG.CHANNELS}→{exp_ch}")
        CFG.WINDOW_LEN = exp_len
        CFG.CHANNELS   = exp_ch

    print(f"[data] Train aligned: {X_tr.shape}, Val aligned: {X_val.shape}"
          + (f", Test aligned: {X_te.shape}" if X_te is not None else ""))

    # --- Label diagnostics ---
    uniq_tr, cnt_tr = np.unique(y_tr, return_counts=True)
    uniq_val, cnt_val = np.unique(y_val, return_counts=True)
    print("[info] train label distribution:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
    print("[info]   val label distribution:", dict(zip(uniq_val.tolist(), cnt_val.tolist())))

    # --- Swap head & compile ---
    model = swap_head_full_finetune(base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.LR_FULL_FT),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("[step] Fine-tune model summary:")
    model.summary()

    # --- Datasets & class weights ---
    ds_tr  = make_tfds(X_tr, y_tr, train=True)
    ds_val = make_tfds(X_val, y_val, train=False)

    class_wts = compute_class_weights_robust(y_tr, CFG.NUM_CANINE_CLASSES)
    print("[info] class weights:", class_wts)




    # --- Optional: log LR and losses at end of each epoch ---
    class LRLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            opt = self.model.optimizer
            lr  = float(tf.keras.backend.get_value(opt.learning_rate))
            print(f"\n[epoch {epoch+1}] lr={lr:.3e} | "
              f"loss={logs.get('loss'):.4f} val_loss={logs.get('val_loss'):.4f}")



    callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=CFG.REDUCE_LR_FACTOR,
        patience=CFG.REDUCE_LR_PATIENCE, min_lr=CFG.MIN_LR, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=CFG.EARLY_STOP_PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CFG.OUTPUT_DIR, "checkpoint_best.h5"),
        monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1
    ),
    LRLogger(),   # <--- added here
    ]

    # --- Train ---
    print("[step] Training … (full fine-tune)")
    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=CFG.EPOCHS,
        class_weight=class_wts,
        callbacks=callbacks,
        verbose=1
    )

    # --- Save ---
    out_path = os.path.join(CFG.OUTPUT_DIR, CFG.OUTPUT_MODEL_NAME)
    model.save(out_path)
    print(f"[done] Saved fine-tuned model to {out_path}")

    # --- Optional test ---
    if X_te is not None:
        ds_te = make_tfds(X_te, y_te, train=False)
        print("[step] Evaluating on test set …")
        test_metrics = model.evaluate(ds_te, verbose=1)
        print("[test] metrics:", dict(zip(model.metrics_names, test_metrics)))

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism = False
    main()
