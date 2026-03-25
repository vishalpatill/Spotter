from typing import List, Dict

def detect_exercise_from_angles(angles: Dict[str, float]) -> str:
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")
    le = angles.get("left_elbow")
    re = angles.get("right_elbow")

    if lk is not None and rk is not None and lk < 100 and rk < 100:
        return "squat"

    if le is not None and re is not None and le < 100 and re < 100:
        return "pushup"

    if lk is not None and rk is not None:
        if lk < 80 and rk > 120:
            return "lunge_left"
        if rk < 80 and lk > 120:
            return "lunge_right"

    return "unknown"

def posture_state_from_angles(angles: Dict[str, float]) -> str:
    lk = angles.get("left_knee", 180)
    rk = angles.get("right_knee", 180)
    lh = angles.get("left_hip", 180)
    rh = angles.get("right_hip", 180)

    avg_knee = (lk + rk) / 2.0
    avg_hip = (lh + rh) / 2.0

    if avg_knee > 120 and avg_hip > 110:
        return "good"
    if avg_knee > 90 and avg_hip > 80:
        return "warning"
    return "bad"

def mp_index(name: str) -> int:
    mapping = {
        "LEFT_KNEE": 25,
        "LEFT_ANKLE": 27,
        "RIGHT_KNEE": 26,
        "RIGHT_ANKLE": 28,
    }
    return mapping[name]

def estimate_forward_lean(landmarks):
    from .pose_pipeline import estimate_forward_lean as _efl
    return _efl(landmarks)

def detect_danger(landmarks: List[Dict], angles: Dict[str, float]) -> List[str]:
    alerts = []

    lh = angles.get("left_hip")
    rh = angles.get("right_hip")
    if (lh is not None and lh < 60) or (rh is not None and rh < 60):
        alerts.append("back_rounding")

    try:
        L_KNEE = mp_index("LEFT_KNEE")
        L_ANKLE = mp_index("LEFT_ANKLE")
        R_KNEE = mp_index("RIGHT_KNEE")
        R_ANKLE = mp_index("RIGHT_ANKLE")

        left_knee = landmarks[L_KNEE]
        left_ankle = landmarks[L_ANKLE]
        right_knee = landmarks[R_KNEE]
        right_ankle = landmarks[R_ANKLE]

        if abs(left_knee["x"] - left_ankle["x"]) < 20:
            alerts.append("knee_valgus_left")
        if abs(right_knee["x"] - right_ankle["x"]) < 20:
            alerts.append("knee_valgus_right")
    except Exception:
        pass

    try:
        lean = estimate_forward_lean(landmarks)
        if lean > 40:
            alerts.append("excessive_forward_lean")
    except Exception:
        pass

    return alerts
