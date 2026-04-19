from math import atan2, degrees


# --------------------------
# CORE ANGLE FUNCTION
# --------------------------
def angle_between(p1, p2, p3):
    """
    Calculates angle at point p2 formed by p1 -> p2 -> p3
    Returns angle in range [0, 180]
    """

    try:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        v1 = (x1 - x2, y1 - y2)
        v2 = (x3 - x2, y3 - y2)

        angle = degrees(atan2(v2[1], v2[0]) - atan2(v1[1], v1[0]))
        angle = abs((angle + 180) % 360 - 180)

        return angle

    except Exception:
        return 0  # ✅ safe fallback


# --------------------------
# MAIN ANGLE COMPUTATION
# --------------------------
def compute_relevant_angles(landmarks):
    """
    landmarks = list of 33 dicts:
    [{"x": ..., "y": ...}, ...]
    """

    def pt(i):
        return (landmarks[i]["x"], landmarks[i]["y"])

    angles = {}

    try:
        # LOWER BODY (MOST IMPORTANT FOR SQUATS)
        angles["left_knee"] = angle_between(pt(23), pt(25), pt(27))
        angles["right_knee"] = angle_between(pt(24), pt(26), pt(28))

        angles["left_hip"] = angle_between(pt(11), pt(23), pt(25))
        angles["right_hip"] = angle_between(pt(12), pt(24), pt(26))

        # OPTIONAL UPPER BODY
        angles["left_elbow"] = angle_between(pt(11), pt(13), pt(15))
        angles["right_elbow"] = angle_between(pt(12), pt(14), pt(16))

        # ✅ NEW: average values (useful everywhere)
        angles["avg_knee"] = (
            angles["left_knee"] + angles["right_knee"]
        ) / 2

        angles["avg_hip"] = (
            angles["left_hip"] + angles["right_hip"]
        ) / 2

    except Exception as e:
        print("⚠️ Angle calculation error:", e)

    return angles