from typing import Dict


# --------------------------
# DETECT EXERCISE
# --------------------------
def detect_exercise_from_angles(angles: Dict[str, float]) -> str:
    lk = angles.get("left_knee")
    rk = angles.get("right_knee")

    if lk is not None and rk is not None:
        if lk < 110 and rk < 110:
            return "squat"

    return "unknown"


# --------------------------
# POSTURE POSITION (FIXED)
# --------------------------
def posture_state_from_angles(angles: Dict[str, float]) -> str:
    lk = angles.get("left_knee", 180)
    rk = angles.get("right_knee", 180)

    avg_knee = (lk + rk) / 2

    if avg_knee > 150:
        return "up"

    elif avg_knee > 100:
        return "mid"

    else:
        return "down"


# --------------------------
# FORM QUALITY (NEW - IMPORTANT)
# --------------------------
def posture_quality_from_angles(angles: Dict[str, float]) -> str:
    lh = angles.get("left_hip", 180)
    rh = angles.get("right_hip", 180)
    lk = angles.get("left_knee", 180)
    rk = angles.get("right_knee", 180)

    # back rounding
    if lh < 70 or rh < 70:
        return "bad"

    # imbalance
    if abs(lk - rk) > 25:
        return "bad"

    return "good"


# --------------------------
# FORM / DANGER DETECTION
# --------------------------
def detect_danger(angles: Dict[str, float]) -> list:
    alerts = []

    lh = angles.get("left_hip", 180)
    rh = angles.get("right_hip", 180)
    lk = angles.get("left_knee", 180)
    rk = angles.get("right_knee", 180)

    # ❌ back rounding
    if lh < 60 or rh < 60:
        alerts.append("back_rounding")

    # ❌ too shallow squat (only when trying to squat)
    if 100 < lk < 150 and 100 < rk < 150:
        alerts.append("not_deep_enough")

    # ❌ imbalance
    if abs(lk - rk) > 25:
        alerts.append("imbalance")

    return alerts