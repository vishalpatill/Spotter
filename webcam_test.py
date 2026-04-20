"""
webcam_test.py  —  Spotter AI Live Demo
========================================
Run from SPOTTER root:
    python webcam_test.py

Keys:
    Q  — quit
    R  — reset rep counter
    S  — save screenshot
"""

import sys
import os
import time
import cv2
import numpy as np
from collections import deque
from math import atan2, degrees

# ── Path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.ai.pose.mediapipe_engine import detect_pose
from backend.app.ai.ml.model_loader import predict_sequence
from backend.app.ai.pose.rep_counter import RepCounter
from backend.app.ai.pose.movement_logic import detect_danger, posture_quality_from_angles

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN      = (0,   220,  80)
RED        = (0,    60, 220)
YELLOW     = (0,   200, 255)
WHITE      = (255, 255, 255)
DARK       = (20,   20,  20)
LIGHT_GREY = (180, 180, 180)
ORANGE     = (0,   160, 255)

# ── Skeleton connections ───────────────────────────────────────────────────────
SKELETON = [
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
]

# ── Angle helper ──────────────────────────────────────────────────────────────
def angle_at(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def build_angles(lm):
    lk = angle_at(lm["LEFT_HIP"],   lm["LEFT_KNEE"],   lm["LEFT_ANKLE"])
    rk = angle_at(lm["RIGHT_HIP"],  lm["RIGHT_KNEE"],  lm["RIGHT_ANKLE"])
    lh = angle_at(lm["LEFT_SHOULDER"],  lm["LEFT_HIP"],  lm["LEFT_KNEE"])
    rh = angle_at(lm["RIGHT_SHOULDER"], lm["RIGHT_HIP"], lm["RIGHT_KNEE"])
    return {
        "left_knee":  lk, "right_knee": rk,
        "left_hip":   lh, "right_hip":  rh,
        "avg_knee":   (lk + rk) / 2,
        "avg_hip":    (lh + rh) / 2,
    }


def build_features(angles):
    lk = angles["left_knee"];  rk = angles["right_knee"]
    lh = angles["left_hip"];   rh = angles["right_hip"]
    return [
        lk/180, rk/180, lh/180, rh/180,
        abs(lk-rk)/180, abs(lh-rh)/180,
        angles["avg_knee"]/180,
    ]

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, form):
    colour = GREEN if form == "GOOD" else RED if form == "BAD" else LIGHT_GREY

    for a, b in SKELETON:
        if a in lm and b in lm:
            p1 = (int(lm[a][0]), int(lm[a][1]))
            p2 = (int(lm[b][0]), int(lm[b][1]))
            cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)

    for name, (x, y) in lm.items():
        cv2.circle(frame, (int(x), int(y)), 7, WHITE,  -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 7, colour,  2, cv2.LINE_AA)


def draw_angle_arc(frame, lm, joint, p1_name, p2_name, angle, form):
    if joint not in lm or p1_name not in lm or p2_name not in lm:
        return
    cx, cy = int(lm[joint][0]), int(lm[joint][1])
    colour  = GREEN if angle > 150 else (YELLOW if angle > 100 else RED)
    cv2.putText(frame, f"{int(angle)}°", (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def draw_panel(frame, rep_count, stage, form, confidence, angles, danger_alerts, adaptive_feedback):
    h, w = frame.shape[:2]

    # ── Left panel — reps + stage ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (200, 140), DARK, -1)
    cv2.rectangle(frame, (0, 0), (200, 140), LIGHT_GREY, 1)

    cv2.putText(frame, "REPS", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, LIGHT_GREY, 1)
    cv2.putText(frame, str(rep_count), (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN, 4, cv2.LINE_AA)
    cv2.putText(frame, stage.upper(), (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)

    # ── Top centre — FORM badge ────────────────────────────────────────────
    if form != "UNKNOWN":
        colour    = GREEN if form == "GOOD" else RED
        label     = f"{'✓' if form == 'GOOD' else '✗'}  {form}  {confidence*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x0 = w // 2 - tw // 2 - 12
        cv2.rectangle(frame, (x0, 8), (x0 + tw + 24, 48), colour, -1)
        cv2.putText(frame, label, (x0 + 12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

    # ── Right panel — angles ───────────────────────────────────────────────
    cv2.rectangle(frame, (w - 185, 0), (w, 130), DARK, -1)
    cv2.rectangle(frame, (w - 185, 0), (w, 130), LIGHT_GREY, 1)

    lines = [
        ("L Knee", angles.get("left_knee",  0)),
        ("R Knee", angles.get("right_knee", 0)),
        ("L Hip",  angles.get("left_hip",   0)),
        ("R Hip",  angles.get("right_hip",  0)),
    ]
    for i, (name, val) in enumerate(lines):
        col = GREEN if val > 150 else (YELLOW if val > 100 else RED)
        cv2.putText(frame, f"{name}: {int(val)}°",
                    (w - 180, 25 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)

    # ── Danger alerts — bottom centre ─────────────────────────────────────
    DANGER_LABELS = {
        "back_rounding":   "⚠  Back rounding — chest up!",
        "not_deep_enough": "⚠  Go deeper!",
        "imbalance":       "⚠  Uneven — balance both legs",
    }
    for i, code in enumerate(danger_alerts):
        msg = DANGER_LABELS.get(code, code)
        cv2.rectangle(frame, (0, h - 45 - i*38), (w, h - 10 - i*38), (0, 0, 150), -1)
        cv2.putText(frame, msg, (15, h - 18 - i*38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)

    # ── Adaptive coaching tip ──────────────────────────────────────────────
    if adaptive_feedback:
        cv2.rectangle(frame, (0, h//2 - 30), (w, h//2 + 10), (20, 80, 20), -1)
        cv2.putText(frame, adaptive_feedback, (15, h//2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    # ── Controls hint ──────────────────────────────────────────────────────
    cv2.putText(frame, "Q: quit   R: reset   S: screenshot",
                (10, h - 10 - len(danger_alerts) * 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, LIGHT_GREY, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 Spotter AI — Live Demo")
    print("   Opening webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    rep_counter   = RepCounter()
    seq_buffer    = deque(maxlen=20)
    pred_buffer   = deque(maxlen=10)

    last_pred_time   = 0.0
    last_label       = 1       # 1=good, 0=bad
    last_label_time  = 0.0
    bad_streak       = 0
    adaptive_msg     = None
    adaptive_msg_until = 0.0
    screenshot_n     = 0

    print("   ✅ Ready! Perform squats in front of the camera.")
    print("   Press Q to quit, R to reset reps, S to screenshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        lm    = detect_pose(frame)

        form            = "UNKNOWN"
        form_confidence = 0.0
        danger_codes    = []

        if lm:
            # Skeleton + angles
            angles = build_angles(lm)

            # Danger / quality
            danger_codes = detect_danger(angles)

            # Rep counter
            rep_counter.update(angles["avg_knee"])

            # LSTM sequence
            seq_buffer.append(build_features(angles))

            if len(seq_buffer) == 20 and time.time() - last_pred_time > 0.4:
                try:
                    label, conf = predict_sequence(list(seq_buffer))
                    if conf > 0.65:
                        pred_buffer.append(label)

                    if len(pred_buffer) >= 6:
                        final = max(set(pred_buffer), key=pred_buffer.count)
                    else:
                        final = last_label

                    if time.time() - last_label_time > 0.8:
                        last_label      = final
                        last_label_time = time.time()

                    form            = "GOOD" if last_label == 1 else "BAD"
                    form_confidence = conf
                    last_pred_time  = time.time()
                except Exception:
                    pass

            # Adaptive coaching
            if form == "BAD" or danger_codes:
                bad_streak += 1
            else:
                bad_streak = 0

            if bad_streak >= 3:
                if "back_rounding" in danger_codes:
                    adaptive_msg = "Your back keeps rounding — chest up!"
                elif "imbalance" in danger_codes:
                    adaptive_msg = "Consistently uneven — reset your stance."
                elif "not_deep_enough" in danger_codes:
                    adaptive_msg = "Keep stopping short — push deeper!"
                else:
                    adaptive_msg = "Repeated form issues — slow down and reset."
                adaptive_msg_until = time.time() + 3.0
                bad_streak = 0

            if time.time() > adaptive_msg_until:
                adaptive_msg = None

            # Draw skeleton BEFORE panel (panel overlays on top)
            draw_skeleton(frame, lm, form)

            # Angle labels at joints
            draw_angle_arc(frame, lm, "LEFT_KNEE",
                           "LEFT_HIP", "LEFT_ANKLE",
                           angles["left_knee"], form)
            draw_angle_arc(frame, lm, "RIGHT_KNEE",
                           "RIGHT_HIP", "RIGHT_ANKLE",
                           angles["right_knee"], form)
        else:
            # No person detected
            h, w = frame.shape[:2]
            cv2.putText(frame, "No person detected — step into frame",
                        (w//2 - 280, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)
            angles = {k: 0 for k in ["left_knee","right_knee","left_hip","right_hip","avg_knee"]}

        # Draw UI panel
        draw_panel(
            frame,
            rep_count        = rep_counter.get_count(),
            stage            = rep_counter.get_stage(),
            form             = form,
            confidence       = form_confidence,
            angles           = angles,
            danger_alerts    = danger_codes,
            adaptive_feedback= adaptive_msg,
        )

        cv2.imshow("Spotter AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            rep_counter = RepCounter()
            seq_buffer.clear()
            pred_buffer.clear()
            print("🔄 Reps reset")
        elif key == ord("s"):
            name = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(name, frame)
            print(f"📸 Saved {name}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Session ended — {rep_counter.get_count()} reps completed")


if __name__ == "__main__":
    main()