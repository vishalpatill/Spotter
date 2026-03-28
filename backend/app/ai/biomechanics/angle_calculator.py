from math import atan2, degrees

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    ang = degrees(atan2(v2[1], v2[0]) - atan2(v1[1], v1[0]))
    return abs((ang + 180) % 360 - 180)


def compute_relevant_angles(landmarks):
    def pt(i):
        return (landmarks[i]["x"], landmarks[i]["y"])

    angles = {}

    try:
        # LOWER BODY
        angles["left_knee"] = angle_between(pt(23), pt(25), pt(27))
        angles["right_knee"] = angle_between(pt(24), pt(26), pt(28))

        angles["left_hip"] = angle_between(pt(11), pt(23), pt(25))
        angles["right_hip"] = angle_between(pt(12), pt(24), pt(26))

        # UPPER BODY
        angles["left_elbow"] = angle_between(pt(11), pt(13), pt(15))
        angles["right_elbow"] = angle_between(pt(12), pt(14), pt(16))

    except Exception as e:
        print("Angle calculation error:", e)

    return angles