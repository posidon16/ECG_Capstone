import os
import wfdb
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CONFIG
MITBIH_PATH = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\MITBIH\Jai_Test_Model\mit-bih"  # <-- set this to your actual path
MODEL_OUTPUT = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\MITBIH\Jai_Test_Model\cnn_ecg_model_best.h5"
LABEL_CLASSES_FILE = r"C:\Users\jaibi\Documents\GitHub\ECG_Capstone\MITBIH\Jai_Test_Model\label_classes.npy"
WINDOW_SIZE = 180
EPOCHS = 60
BATCH_SIZE = 128
SEED = 42

# For reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

TRAIN_RECORDS = [ '100','101','102','103','104','105','106','107','108','109',
                  '111','112','113','114','115','116','117','118','119','121',
                  '122','123','124' ]

TEST_RECORDS = [ '200','201','202','203','205','207','208','209','210',
                 '212','213','214','215','217','219','220','221','222',
                 '223','228','230','231','232','233','234' ]

def extract_beats(record_name, window_size=180):
    record = wfdb.rdrecord(os.path.join(MITBIH_PATH, record_name))
    annotation = wfdb.rdann(os.path.join(MITBIH_PATH, record_name), 'atr')
    sig = record.p_signal[:, 0]
    beats, labels = [], []
    for i, r in enumerate(annotation.sample):
        sym = annotation.symbol[i]
        if sym in ['N', 'L', 'R', 'e', 'j']: label = 'N'
        elif sym in ['A', 'a', 'J', 'S']:    label = 'S'
        elif sym in ['V', 'E']:              label = 'V'
        elif sym in ['F']:                  label = 'F'
        else:                                label = 'Q'
        start = r - window_size // 2
        end = r + window_size // 2
        if start >= 0 and end <= len(sig):
            beats.append(sig[start:end])
            labels.append(label)
    return beats, labels

def load_dataset(records):
    beats, labels = [], []
    for rec in records:
        b, l = extract_beats(rec)
        beats.extend(b)
        labels.extend(l)
    return np.array(beats), np.array(labels)

# Load
X_train_raw, y_train_raw = load_dataset(TRAIN_RECORDS)
X_test_raw, y_test_raw = load_dataset(TEST_RECORDS)

# Normalize
X_train = (X_train_raw - np.mean(X_train_raw, axis=1, keepdims=True)) / np.std(X_train_raw, axis=1, keepdims=True)
X_test = (X_test_raw - np.mean(X_test_raw, axis=1, keepdims=True)) / np.std(X_test_raw, axis=1, keepdims=True)

# Reshape for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
np.save(LABEL_CLASSES_FILE, le.classes_)

# Class balance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train-validation split
val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUTPUT, save_best_only=True, monitor='val_loss', mode='min')
]

# Train
model.fit(X_train, y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_val, y_val),
          class_weight=class_weights,
          callbacks=callbacks)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nPatient-wise split test accuracy: {test_acc:.4f}")

# Classification report
y_pred = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print(f"\nBest model saved to: {MODEL_OUTPUT}")
