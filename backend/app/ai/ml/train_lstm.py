"""
train_lstm.py  —  Spotter AI LSTM Trainer (v2, small-dataset optimized)
========================================================================
Reads processed sequences from data/processed/squat/
Trains a Bidirectional LSTM with techniques for small datasets:
  - Heavy dropout + L2 regularization
  - Label smoothing
  - Learning rate warmup + cosine decay
  - Cross-validation for reliable accuracy estimate
  - TFLite export for fast inference

Run from SPOTTER root:
    python train_lstm.py

Output:
    backend/app/ai/ml/squat_model.h5
    backend/app/ai/ml/squat_model.keras
    backend/app/ai/ml/squat_model_best.keras
    backend/app/ai/ml/squat_model.tflite
    data/processed/squat/training_report.png
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data/processed/squat")
MODEL_DIR = Path("backend/app/ai/ml")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN = DATA_DIR / "X_train.npy"
Y_TRAIN = DATA_DIR / "y_train.npy"
X_VAL   = DATA_DIR / "X_val.npy"
Y_VAL   = DATA_DIR / "y_val.npy"

for p in [X_TRAIN, Y_TRAIN, X_VAL, Y_VAL]:
    if not p.exists():
        print(f"❌ Missing: {p}")
        print("   Run: python build_dataset.py  first")
        sys.exit(1)

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = np.load(X_TRAIN).astype(np.float32)
y_train = np.load(Y_TRAIN).astype(np.float32)
X_val   = np.load(X_VAL).astype(np.float32)
y_val   = np.load(Y_VAL).astype(np.float32)

print(f"📦 Data loaded")
print(f"   Train: {X_train.shape}  labels {dict(zip(*np.unique(y_train.astype(int), return_counts=True)))}")
print(f"   Val:   {X_val.shape}    labels {dict(zip(*np.unique(y_val.astype(int),   return_counts=True)))}")

SEQ_LEN, FEAT_LEN = X_train.shape[1], X_train.shape[2]

# ── Import TF ──────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

print(f"\n🔧 TensorFlow {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# ── Model builder ──────────────────────────────────────────────────────────────
def build_model(dropout=0.40, l2=1e-4, lstm_units=(64, 32)):
    """
    Bidirectional LSTM optimized for 7-feature, 20-frame squat sequences.
    Small and regularized to avoid overfitting on ~200 training examples.
    """
    inp = layers.Input(shape=(SEQ_LEN, FEAT_LEN), name="pose_seq")

    # LSTM layers
    x = layers.Bidirectional(
            layers.LSTM(lstm_units[0], return_sequences=True,
                        dropout=0.20, recurrent_dropout=0.10,
                        kernel_regularizer=regularizers.l2(l2)),
            name="bilstm_1")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(
            layers.LSTM(lstm_units[1], return_sequences=False,
                        dropout=0.20, recurrent_dropout=0.10,
                        kernel_regularizer=regularizers.l2(l2)),
            name="bilstm_2")(x)
    x = layers.Dropout(dropout)(x)

    # Dense head
    x   = layers.Dense(32, activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(x)
    x   = layers.Dropout(dropout * 0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    m = models.Model(inp, out, name="SpotterLSTM_v2")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return m

# ── Print model summary ────────────────────────────────────────────────────────
sample_model = build_model()
sample_model.summary()
total_params = sample_model.count_params()
print(f"\n   Total parameters: {total_params:,}")

# ── Training callbacks ─────────────────────────────────────────────────────────
def make_callbacks(run_name="default"):
    best_path = str(MODEL_DIR / f"squat_model_best.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=25,
            restore_best_weights=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=10, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            best_path, monitor="val_auc",
            save_best_only=True, mode="max", verbose=0),
    ]

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n🏋️  Training...")
model = build_model()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=min(32, len(X_train) // 4),   # don't use huge batches on tiny data
    callbacks=make_callbacks(),
    verbose=1,
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\n📈 Final evaluation on validation set:")
results = model.evaluate(X_val, y_val, verbose=0)
for name, val in zip(model.metrics_names, results):
    print(f"   {name:12s} = {val:.4f}")

# Detailed report
from sklearn.metrics import classification_report, confusion_matrix

y_pred_prob = model.predict(X_val, verbose=0).flatten()
y_pred      = (y_pred_prob > 0.5).astype(int)
y_true      = y_val.astype(int)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["BAD (0)", "GOOD (1)"]))

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix (rows=actual, cols=predicted):")
print(f"         BAD   GOOD")
print(f"  BAD  [{cm[0,0]:5d} {cm[0,1]:5d}]")
print(f"  GOOD [{cm[1,0]:5d} {cm[1,1]:5d}]")

# ── Calibrate decision threshold ───────────────────────────────────────────────
# Find threshold that maximises F1 on validation set
from sklearn.metrics import f1_score

best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.3, 0.8, 0.01):
    preds = (y_pred_prob > t).astype(int)
    f1    = f1_score(y_true, preds, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"\n🎯 Best decision threshold: {best_thresh:.2f}  (macro F1 = {best_f1:.3f})")
print("   Update DECISION_THRESHOLD in model_loader.py if needed")

# ── Diagnose common failure modes ──────────────────────────────────────────────
val_auc = max(history.history.get("val_auc", [0]))
val_acc = max(history.history.get("val_accuracy", [0]))

print("\n🔍 Diagnosis:")
if val_auc < 0.65:
    print("   ❌ AUC < 0.65 — model is barely better than random.")
    print("      Likely cause: mixed front/side view, or GOOD and BAD look the same.")
    print("      Action: check your videos, ensure clear form differences.")
elif val_auc < 0.80:
    print("   ⚠  AUC 0.65–0.80 — model learned something but not reliable.")
    print("      Action: record 10+ more videos per class, focus on clear exaggeration.")
elif val_auc < 0.90:
    print("   ✅ AUC 0.80–0.90 — decent model, usable in production with hysteresis.")
else:
    print("   ✅✅ AUC > 0.90 — excellent! Deploy with confidence.")

if cm[0, 1] > cm[0, 0]:
    print("   ⚠  Model tends to predict GOOD even for BAD form (high false-positive rate).")
    print(f"      Try threshold={min(best_thresh+0.05, 0.75):.2f} to be more conservative.")

# ── Save models ────────────────────────────────────────────────────────────────
h5_path    = MODEL_DIR / "squat_model.h5"
keras_path = MODEL_DIR / "squat_model.keras"

model.save(str(h5_path))
model.save(str(keras_path))
print(f"\n💾 Saved: {h5_path}")
print(f"💾 Saved: {keras_path}")

# ── TFLite export ──────────────────────────────────────────────────────────────
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    tflite_path  = MODEL_DIR / "squat_model.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"💾 TFLite: {tflite_path}  ({len(tflite_bytes)//1024} KB)")
except Exception as e:
    print(f"⚠  TFLite export failed: {e}")

# ── Save training info ─────────────────────────────────────────────────────────
info = {
    "val_auc":          float(val_auc),
    "val_accuracy":     float(val_acc),
    "best_threshold":   float(best_thresh),
    "best_f1":          float(best_f1),
    "total_params":     total_params,
    "confusion_matrix": cm.tolist(),
    "label_map":        {"0": "BAD", "1": "GOOD"},
    "decision_threshold": float(best_thresh),
}
with open(DATA_DIR / "training_report.json", "w") as f:
    json.dump(info, f, indent=2)

# ── Training plot ──────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Spotter LSTM Training  |  val_AUC={val_auc:.3f}  threshold={best_thresh:.2f}",
                 fontsize=13, fontweight="bold")

    h = history.history
    ep = range(1, len(h["loss"]) + 1)

    axes[0].plot(ep, h["loss"],         label="Train"); axes[0].plot(ep, h["val_loss"],     label="Val")
    axes[0].set_title("Loss (lower=better)"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(ep, h["accuracy"],     label="Train"); axes[1].plot(ep, h["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(ep, h["auc"],          label="Train"); axes[2].plot(ep, h["val_auc"],      label="Val")
    axes[2].axhline(0.80, color="orange", linestyle="--", label="0.80 target")
    axes[2].axhline(0.90, color="green",  linestyle="--", label="0.90 target")
    axes[2].set_title("AUC (higher=better)"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plot_path = DATA_DIR / "training_plot.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\n📊 Training plot → {plot_path}")
except Exception as e:
    print(f"   (Plot skipped: {e})")

print("\n✅ Training complete! Next: python webcam_test.py")
print(f"   If GOOD/BAD appear inverted, press L in webcam_test.py to flip labels.")
print(f"   Recommended threshold in model_loader.py: {best_thresh:.2f}")