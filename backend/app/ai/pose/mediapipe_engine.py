"""
mediapipe_engine.py
===================
Uses mp.solutions.pose (classic API) — stable with your current protobuf/TF setup.
The new Tasks API (PoseLandmarker) conflicts with TensorFlow's protobuf version.
"""

import os
import tempfile
import time
import cv2
import mediapipe as mp

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "spotter_matplotlib"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# ── One model instance, lazy loaded ───────────────────────────────────────────
_POSE = None

def _get_pose():
    global _POSE
    if _POSE is None:
        _POSE = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _POSE


# ── Landmark index → name mapping ─────────────────────────────────────────────
LANDMARK_NAMES = {
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
}


def _extract(results, frame_shape):
    """Convert MediaPipe results → named pixel-coord dict."""
    if not results.pose_landmarks:
        return None
    h, w    = frame_shape[0], frame_shape[1]
    lm      = results.pose_landmarks.landmark
    return {
        name: (lm[idx].x * w, lm[idx].y * h)
        for idx, name in LANDMARK_NAMES.items()
    }


def detect_pose(frame):
    """
    For webcam_test.py — BGR frame from cv2.VideoCapture.
    Returns named landmark dict or None.
    """
    pose    = _get_pose()
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    return _extract(results, frame.shape)


def detect_pose_image(frame):
    """
    For FastAPI — BGR frame decoded from uploaded image bytes.
    Same model, same function — static_image_mode=False handles both fine.
    Returns named landmark dict or None.
    """
    return detect_pose(frame)