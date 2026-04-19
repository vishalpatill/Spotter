import numpy as np
import os
from tensorflow.keras.models import load_model

# ✅ FIX: dynamic path (works on any machine)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "squat_model.h5"
)

model = load_model(MODEL_PATH)


def predict_sequence(sequence):
    """
    sequence shape: (20, 4)
    """

    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence, verbose=0)[0]

    label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return label, confidence