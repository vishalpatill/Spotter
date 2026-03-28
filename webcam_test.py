import cv2
from backend.app.ai.pose.mediapipe_engine import detect_pose
import math
import time 
import mediapipe as mp

print("🚀 Starting Spotter AI...")

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

    # draw joints
    for key, (x, y) in landmarks.items():
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # draw connections
    for a, b in connections:
        if a in landmarks and b in landmarks:
            x1, y1 = landmarks[a]
            x2, y2 = landmarks[b]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Webcam running. Press 'q' to quit.")

    counter = 0
    stage = None
    last_count_time = 0
    down_start_time = None
    angle_buffer = []   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_pose(frame)

        if landmarks:
            draw_skeleton(frame, landmarks)

        if landmarks:
            # angle function FIRST (fix order)
            def calculate_angle(a, b, c):
                angle = math.degrees(
                    math.atan2(c[1] - b[1], c[0] - b[0]) -
                    math.atan2(a[1] - b[1], a[0] - b[0])
                )
                return abs(angle)

            # get key points
            hip_l = landmarks["LEFT_HIP"]
            knee_l = landmarks["LEFT_KNEE"]
            ankle_l = landmarks["LEFT_ANKLE"]

            hip_r = landmarks["RIGHT_HIP"]
            knee_r = landmarks["RIGHT_KNEE"]
            ankle_r = landmarks["RIGHT_ANKLE"]

            # calculate both legs
            angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            angle_r = calculate_angle(hip_r, knee_r, ankle_r)

            # average
            raw_angle = (angle_l + angle_r) / 2

            # smoothing
            angle_buffer.append(raw_angle)
            if len(angle_buffer) > 5:
                angle_buffer.pop(0)

            knee_angle = sum(angle_buffer) / len(angle_buffer)

            print("KNEE:", knee_angle)

            # squat logic
            if knee_angle < 95:
                if down_start_time is None:
                    down_start_time = time.time()

                if time.time() - down_start_time > 0.5:
                    stage = "down"

            elif knee_angle > 160:
                down_start_time = None

                if stage == "down":
                    if time.time() - last_count_time > 1:
                        stage = "up"
                        counter += 1
                        last_count_time = time.time()

            # display
            cv2.putText(frame, f"Reps: {counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Angle: {int(knee_angle)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Spotter AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()