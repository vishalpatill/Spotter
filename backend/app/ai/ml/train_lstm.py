import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# LOAD DATA
# --------------------------
X = np.load("data/processed/squat/X.npy")
y = np.load("data/processed/squat/y.npy")

print("Loaded dataset:", X.shape, y.shape)

# --------------------------
# TRAIN TEST SPLIT
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# one-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# --------------------------
# BUILD MODEL
# --------------------------
model = Sequential()

# ✅ FIXED INPUT SHAPE (7 features now)
model.add(Input(shape=(20, 7)))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------
# EARLY STOPPING (IMPORTANT)
# --------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --------------------------
# TRAIN
# --------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=50,              # 🔥 increased
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# --------------------------
# SAVE MODEL
# --------------------------
model.save("backend/app/ai/ml/squat_model.keras")  # ✅ modern format

print("\n🔥 Model training complete and saved!")