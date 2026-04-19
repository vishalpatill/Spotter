import cv2
from backend.app.ai.pose.mediapipe_engine import detect_pose
from collections import deque
from backend.app.ai.ml.model_loader import predict_sequence
from backend.app.ai.rep_counter import RepCounter
import time
import numpy as np

print("🚀 Starting Spotter AI...")

VIDEO_PATH = None


def draw_skeleton(frame, landmarks):
    connections = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_HIP", "LEFT_KNEE"),
        ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"),
        ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]

    for key, (x, y) in landmarks.items():
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    for a, b in connections:
        if a in landmarks and b in landmarks:
            x1, y1 = landmarks[a]
            x2, y2 = landmarks[b]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


def main():
    if VIDEO_PATH:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"🎥 Testing video: {VIDEO_PATH}")
    else:
        cap = cv2.VideoCapture(0)
        print("📷 Webcam mode")

    if not cap.isOpened():
        print("❌ Cannot open source")
        return

    rep_counter = RepCounter()

    sequence_buffer = deque(maxlen=20)
    prediction_buffer = deque(maxlen=10)

    last_prediction_time = 0

    # ✅ NEW: stability variables
    last_label = 1
    last_label_update_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_pose(frame)

        if landmarks:
            draw_skeleton(frame, landmarks)

            # KEY POINTS
            hip_l = landmarks["LEFT_HIP"]
            knee_l = landmarks["LEFT_KNEE"]
            ankle_l = landmarks["LEFT_ANKLE"]

            hip_r = landmarks["RIGHT_HIP"]
            knee_r = landmarks["RIGHT_KNEE"]
            ankle_r = landmarks["RIGHT_ANKLE"]

            shoulder_l = landmarks["LEFT_SHOULDER"]
            shoulder_r = landmarks["RIGHT_SHOULDER"]

            # ANGLES
            angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            angle_r = calculate_angle(hip_r, knee_r, ankle_r)

            knee_angle = (angle_l + angle_r) / 2

            left_hip_angle = calculate_angle(shoulder_l, hip_l, knee_l)
            right_hip_angle = calculate_angle(shoulder_r, hip_r, knee_r)

            print(f"L:{int(angle_l)} R:{int(angle_r)} AVG:{int(knee_angle)}")

            # REP COUNTER
            rep_counter.update(knee_angle)
            counter = rep_counter.get_count()
            stage = rep_counter.get_stage()

            # ML FEATURES
            features = [
                angle_l / 180.0,
                angle_r / 180.0,
                left_hip_angle / 180.0,
                right_hip_angle / 180.0
            ]

            sequence_buffer.append(features)

            # 🔥 FINAL STABLE ML PREDICTION
            if len(sequence_buffer) == 20 and (time.time() - last_prediction_time > 0.5):

                label, confidence = predict_sequence(list(sequence_buffer))

                # accept only confident predictions
                if confidence > 0.7:
                    prediction_buffer.append(label)

                # strong majority voting
                if len(prediction_buffer) >= 8:
                    final_label = max(set(prediction_buffer), key=prediction_buffer.count)
                else:
                    final_label = last_label

                # lock label for stability (1 sec)
                if time.time() - last_label_update_time > 1.0:
                    last_label = final_label
                    last_label_update_time = time.time()

                form = "GOOD" if last_label == 1 else "BAD"

                last_prediction_time = time.time()

                cv2.putText(frame, f"Form: {form}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if last_label == 1 else (0, 0, 255), 2)

                cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

            # DISPLAY
            cv2.putText(frame, f"Reps: {counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Angle: {int(knee_angle)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, f"Stage: {stage}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Spotter AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()