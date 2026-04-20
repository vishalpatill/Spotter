import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model

SEQUENCE_LENGTH = 20
FEATURE_COUNT = 7

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "squat_model.keras"
)

_MODEL = None


def _build_squat_model():
    return Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(2, activation="softmax"),
    ])


def get_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    try:
        _MODEL = load_model(MODEL_PATH, compile=False)
    except (TypeError, ValueError):
        _MODEL = _build_squat_model()
        _MODEL.load_weights(MODEL_PATH)

    return _MODEL


def predict_sequence(sequence):
    """
    sequence shape: (20, 7)
    """

    sequence = np.asarray(sequence, dtype=np.float32)
    expected_shape = (SEQUENCE_LENGTH, FEATURE_COUNT)
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Expected sequence shape {expected_shape}, got {sequence.shape}"
        )

    sequence = np.expand_dims(sequence, axis=0)

    prediction = get_model().predict(sequence, verbose=0)[0]

    label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return label, confidence
