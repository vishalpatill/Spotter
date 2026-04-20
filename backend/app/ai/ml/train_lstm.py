"""
train_lstm_v2.py
================
Place at: SPOTTER/backend/app/ai/ml/train_lstm_v2.py

Run from SPOTTER root:
    python backend/app/ai/ml/train_lstm_v2.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ── Paths — this file lives at SPOTTER/backend/app/ai/ml/train_lstm_v2.py ────
THIS_FILE    = Path(__file__).resolve()
ML_DIR       = THIS_FILE.parent                          # .../backend/app/ai/ml/
SPOTTER_ROOT = ML_DIR.parents[3]                         # SPOTTER/

DATA_DIR  = SPOTTER_ROOT / "data" / "processed" / "squat"
MODEL_OUT = ML_DIR / "squat_model.keras"
HIST_OUT  = ML_DIR / "training_history.json"

print(f"📁 Data  : {DATA_DIR}")
print(f"💾 Model : {MODEL_OUT}")

# Verify data exists before importing keras (saves time if path wrong)
if not (DATA_DIR / "X.npy").exists():
    raise FileNotFoundError(
        f"\n❌ Dataset not found at: {DATA_DIR}\n"
        f"   Run dataset_builder.py first."
    )

EPOCHS     = 100
BATCH_SIZE = 32
LR         = 1e-3


def build_model(seq_len: int, feat_count: int) -> Sequential:
    model = Sequential(name="SpotterSquatLSTM_v2")
    model.add(Input(shape=(seq_len, feat_count)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    print(f"\n✅ Dataset: X={X.shape}  y={y.shape}")
    print(f"   Good: {int(np.sum(y==1))}  |  Bad: {int(np.sum(y==0))}")

    seq_len, feat_count = X.shape[1], X.shape[2]

    # ── Train/test split ──────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Class weights — handles good/bad imbalance ────────────────────────────
    weights       = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(weights))
    print(f"   Class weights: {class_weights}")

    y_tr_cat = to_categorical(y_tr, num_classes=2)
    y_te_cat = to_categorical(y_te, num_classes=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(seq_len, feat_count)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_path = str(MODEL_OUT).replace(".keras", "_best.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            best_path, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n🚀 Training on {len(X_tr)} sequences, validating on {len(X_te)}...\n")
    history = model.fit(
        X_tr, y_tr_cat,
        validation_data=(X_te, y_te_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_te, y_te_cat, verbose=0)
    print(f"\n{'='*50}")
    print(f"  🏆 Test accuracy : {acc*100:.1f}%")
    print(f"  📉 Test loss     : {loss:.4f}")
    print(f"{'='*50}")

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(str(MODEL_OUT))
    print(f"✅ Model saved → {MODEL_OUT}")

    # Save training history for later plotting
    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(HIST_OUT, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"📈 History  → {HIST_OUT}")

    # ── Advice ────────────────────────────────────────────────────────────────
    print()
    if acc < 0.75:
        print("⚠️  Accuracy below 75% — consider adding more varied videos")
    elif acc < 0.85:
        print("✅ Decent accuracy. Model is usable. More data will improve further.")
    else:
        print("🔥 Excellent accuracy! Model is production-ready.")

    print(f"\nNext: start the API with:")
    print(f"  uvicorn backend.app.ai.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()