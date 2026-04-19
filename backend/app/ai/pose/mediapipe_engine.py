import cv2
import os
import time
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "pose_landmarker_lite.task"
)

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

POSE_LANDMARKER = None


# --------------------------
# LOAD MODEL (VIDEO MODE ✅)
# --------------------------
def load_model():
    global POSE_LANDMARKER

    if POSE_LANDMARKER is None:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,   # ✅ FIXED BACK
            num_poses=1,
        )
        POSE_LANDMARKER = PoseLandmarker.create_from_options(options)

    return POSE_LANDMARKER


# --------------------------
# DETECT POSE (TRACKING ENABLED)
# --------------------------
def detect_pose(frame):
    model = load_model()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=img_rgb
    )

    # ✅ REQUIRED for VIDEO mode
    timestamp_ms = int(time.time() * 1000)

    result = model.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks[0]

    h, w, _ = frame.shape

    landmark_dict = {
        "LEFT_HIP": (lm[23].x * w, lm[23].y * h),
        "LEFT_KNEE": (lm[25].x * w, lm[25].y * h),
        "LEFT_ANKLE": (lm[27].x * w, lm[27].y * h),

        "RIGHT_HIP": (lm[24].x * w, lm[24].y * h),
        "RIGHT_KNEE": (lm[26].x * w, lm[26].y * h),
        "RIGHT_ANKLE": (lm[28].x * w, lm[28].y * h),

        "LEFT_SHOULDER": (lm[11].x * w, lm[11].y * h),
        "RIGHT_SHOULDER": (lm[12].x * w, lm[12].y * h),
    }

    return landmark_dict