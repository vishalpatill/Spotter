"""
pose_pipeline.py
================
Used by the FastAPI server to process uploaded image frames.
Uses IMAGE mode (not VIDEO mode) — correct for isolated frames.
"""

import cv2
import numpy as np
from backend.app.ai.pose.mediapipe_engine import detect_pose_image


LANDMARK_INDEX_BY_NAME = {
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
}


def _landmark_dict_to_list(landmark_dict: dict) -> list:
    """Convert named landmark dict → 33-item list indexed by MediaPipe index."""
    landmarks = [{"x": 0.0, "y": 0.0} for _ in range(33)]
    for name, index in LANDMARK_INDEX_BY_NAME.items():
        if name in landmark_dict:
            x, y = landmark_dict[name]
            landmarks[index] = {"x": float(x), "y": float(y)}
    return landmarks


def run_blazepose_on_image(image_bytes: bytes) -> dict | None:
    """
    Decode image bytes and run pose detection.

    Returns:
        {
            "landmarks":     33-item list [{x,y},...],  ← for angle_calculator
            "landmark_dict": named dict {name: (x,y)},  ← for angle_calculator
        }
        or None if no person detected / bad image
    """
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    landmark_dict = detect_pose_image(frame)   # ← IMAGE mode

    if landmark_dict is None:
        return None

    return {
        "landmarks":     _landmark_dict_to_list(landmark_dict),
        "landmark_dict": landmark_dict,
    }