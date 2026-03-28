import cv2
import numpy as np
from typing import List, Dict, Optional
from math import atan2, degrees
import os
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from backend.app.ai.exercise_engine.engine import ExerciseEngine
from backend.app.ai.exercise_engine.loader import load_exercises
from backend.app.ai.pose.mediapipe_engine import detect_pose

EXERCISE_ENGINE = None
# Load MediaPipe Pose model once (production-safe)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "pose_landmarker_lite.task"
)


BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

def load_exercise_engine():
    exercises = load_exercises("app/ai/exercise_engine/definitions")
    return ExerciseEngine(exercises)

def load_pose_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


# Create model singleton
POSE_LANDMARKER = None

def image_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_blazepose_on_image(image_bytes: bytes) -> Optional[Dict]:
    global POSE_LANDMARKER

    if POSE_LANDMARKER is None:
        POSE_LANDMARKER = load_pose_landmarker()

    img_rgb = image_from_bytes(image_bytes)
    h, w, _ = img_rgb.shape

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=img_rgb
    )

    result = POSE_LANDMARKER.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = []
    for lm in result.pose_landmarks[0]:
        landmarks.append({
            "x": float(lm.x * w),
            "y": float(lm.y * h),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })

    global EXERCISE_ENGINE

    if EXERCISE_ENGINE is None:
        EXERCISE_ENGINE = load_exercise_engine()
        EXERCISE_ENGINE.set_exercise("squat") 

    angles = compute_relevant_angles(landmarks)

    result = EXERCISE_ENGINE.process_frame(angles)

    return {
        "landmarks": landmarks,
        "angles": angles,
        "exercise_result": result,
        "image_shape": (h, w)
    }

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)
    ang = degrees(atan2(v2[1], v2[0]) - atan2(v1[1], v1[0]))
    return abs((ang + 180) % 360 - 180)


def compute_relevant_angles(landmarks: List[Dict]) -> Dict[str, float]:
    def pt(i):
        return (landmarks[i]["x"], landmarks[i]["y"])

    # Landmark indices (MediaPipe Pose)
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_SH, R_SH = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WR, R_WR = 15, 16

    angles = {}
    try:
        angles["left_knee"] = angle_between(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE))
        angles["right_knee"] = angle_between(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE))
        angles["left_hip"] = angle_between(pt(L_SH), pt(L_HIP), pt(L_KNEE))
        angles["right_hip"] = angle_between(pt(R_SH), pt(R_HIP), pt(R_KNEE))
        angles["left_elbow"] = angle_between(pt(L_SH), pt(L_ELB), pt(L_WR))
        angles["right_elbow"] = angle_between(pt(R_SH), pt(R_ELB), pt(R_WR))
    except Exception:
        pass

    return angles


def estimate_forward_lean(landmarks: List[Dict]) -> float:
    def midpoint(a, b):
        return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0)

    left_sh, right_sh = landmarks[11], landmarks[12]
    left_hip, right_hip = landmarks[23], landmarks[24]

    ms = midpoint(left_sh, right_sh)
    mh = midpoint(left_hip, right_hip)
    return ms[0] - mh[0]
