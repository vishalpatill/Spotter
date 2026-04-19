import os
import cv2
import numpy as np

from backend.app.ai.pose.mediapipe_engine import detect_pose
from backend.app.ai.biomechanics.angle_calculator import compute_relevant_angles

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = "data/raw/squat"
OUTPUT_DIR = "data/processed/squat"

SEQUENCE_LENGTH = 20  # matches webcam + model

LABEL_MAP = {
    "good": 1,
    "bad": 0
}

# --------------------------
# dict → indexed list
# --------------------------
def convert_landmarks_dict_to_list(landmarks_dict):
    mapping = {
        11: "LEFT_SHOULDER",
        12: "RIGHT_SHOULDER",
        13: "LEFT_ELBOW",
        14: "RIGHT_ELBOW",
        15: "LEFT_WRIST",
        16: "RIGHT_WRIST",
        23: "LEFT_HIP",
        24: "RIGHT_HIP",
        25: "LEFT_KNEE",
        26: "RIGHT_KNEE",
        27: "LEFT_ANKLE",
        28: "RIGHT_ANKLE",
    }

    landmarks_list = [{"x": 0, "y": 0} for _ in range(33)]

    for idx, name in mapping.items():
        if name in landmarks_dict:
            x, y = landmarks_dict[name]
            landmarks_list[idx] = {"x": x, "y": y}

    return landmarks_list


# --------------------------
# FEATURE EXTRACTION (UPDATED)
# --------------------------
def extract_features(landmarks_dict):
    landmarks_list = convert_landmarks_dict_to_list(landmarks_dict)

    angles = compute_relevant_angles(landmarks_list)

    left_knee = angles.get("left_knee", 0)
    right_knee = angles.get("right_knee", 0)
    left_hip = angles.get("left_hip", 0)
    right_hip = angles.get("right_hip", 0)

    # 🔥 NEW FEATURES
    knee_diff = abs(left_knee - right_knee)
    hip_diff = abs(left_hip - right_hip)
    avg_knee = (left_knee + right_knee) / 2

    # ✅ NORMALIZED FEATURES (0–1)
    feature_vector = [
        left_knee / 180.0,
        right_knee / 180.0,
        left_hip / 180.0,
        right_hip / 180.0,
        knee_diff / 180.0,
        hip_diff / 180.0,
        avg_knee / 180.0,
    ]

    return feature_vector


# --------------------------
# PROCESS VIDEO
# --------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_pose(frame)

        if landmarks is None:
            continue

        features = extract_features(landmarks)
        sequence.append(features)

    cap.release()

    return sequence


# --------------------------
# CREATE SEQUENCES
# --------------------------
def create_sequences(sequence):
    sequences = []

    for i in range(len(sequence) - SEQUENCE_LENGTH):
        window = sequence[i:i + SEQUENCE_LENGTH]

        if len(window) == SEQUENCE_LENGTH:
            sequences.append(window)

    return sequences


# --------------------------
# PROCESS FOLDER
# --------------------------
def process_folder(folder_path, label):
    X = []
    y = []

    for file in os.listdir(folder_path):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(folder_path, file)
        print(f"Processing: {video_path}")

        sequence = process_video(video_path)

        if len(sequence) < SEQUENCE_LENGTH:
            print("⚠️ Skipping short video")
            continue

        windows = create_sequences(sequence)

        for w in windows:
            X.append(w)
            y.append(label)

    return X, y


# --------------------------
# MAIN
# --------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔥 Building dataset...")

    good_X, good_y = process_folder(
        os.path.join(DATA_DIR, "good"), LABEL_MAP["good"]
    )

    bad_X, bad_y = process_folder(
        os.path.join(DATA_DIR, "bad"), LABEL_MAP["bad"]
    )

    X = np.array(good_X + bad_X)
    y = np.array(good_y + bad_y)

    print("\n✅ Dataset created!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\n💾 Saved to data/processed/squat/")


if __name__ == "__main__":
    main()