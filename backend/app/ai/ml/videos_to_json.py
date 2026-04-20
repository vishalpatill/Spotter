"""
videos_to_json.py  (self-contained version)
============================================
Place at: SPOTTER/backend/app/ai/ml/videos_to_json.py

Fix protobuf FIRST:
    pip install "protobuf==4.25.3"

Then run from SPOTTER root:
    python backend/app/ai/ml/videos_to_json.py
"""

import json
import sys
from pathlib import Path
import cv2

THIS_FILE    = Path(__file__).resolve()
SPOTTER_ROOT = THIS_FILE.parents[4]
DATA_DIR     = SPOTTER_ROOT / "data" / "raw" / "squat"

print(f"📁 Videos folder: {DATA_DIR}")

_pose_instance = None

def get_pose():
    global _pose_instance
    if _pose_instance is None:
        import mediapipe as mp
        _pose_instance = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("✅ MediaPipe Pose loaded")
    return _pose_instance

def detect_landmarks(frame):
    pose   = get_pose()
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    return [{"x": float(lm.x), "y": float(lm.y)} for lm in result.pose_landmarks.landmark]

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ❌ Cannot open: {video_path.name}")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps / 10))

    print(f"    {fps:.0f}fps | {total} frames | ~{total/fps:.1f}s → every {frame_step} frames")

    frames_out, frame_idx, sampled, detected = [], 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            sampled += 1
            lm = detect_landmarks(frame)
            if lm:
                frames_out.append({"landmarks": lm})
                detected += 1
        frame_idx += 1

    cap.release()
    rate = (detected / sampled * 100) if sampled > 0 else 0
    print(f"    Sampled: {sampled} | Found: {detected} | Rate: {rate:.0f}%")
    if rate < 50:
        print("    ⚠️  Low detection — ensure full body visible in video")
    return frames_out

def main():
    if not DATA_DIR.exists():
        print(f"\n❌ Not found: {DATA_DIR}"); sys.exit(1)

    try:
        get_pose()
    except Exception as e:
        print(f"\n❌ MediaPipe failed: {e}")
        print('👉 Run: pip install "protobuf==4.25.3"  then retry')
        sys.exit(1)

    saved = failed = skipped = 0

    for label in ["good", "bad"]:
        folder = DATA_DIR / label
        if not folder.exists():
            print(f"\n⚠️  Missing: {folder}"); continue

        videos = (sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.MP4")) +
                  sorted(folder.glob("*.mov")) + sorted(folder.glob("*.MOV")))

        print(f"\n{'='*60}\n  {label.upper()}  —  {len(videos)} video(s)\n{'='*60}")

        for vp in videos:
            jp = vp.with_suffix(".json")
            if jp.exists():
                print(f"\n  ⏭️  {vp.name}  →  skipping (JSON exists)")
                skipped += 1; continue

            print(f"\n  🎬 {vp.name}")
            frames = process_video(vp)

            if len(frames) < 15:
                print(f"    ❌ Only {len(frames)} frames — skipping")
                failed += 1; continue

            with open(jp, "w") as f:
                json.dump(frames, f)
            print(f"    ✅ {jp.name}  ({len(frames)} frames, {jp.stat().st_size//1024} KB)")
            saved += 1

    print(f"\n{'='*60}")
    print(f"  ✅ Converted: {saved}  ⏭️  Skipped: {skipped}  ❌ Failed: {failed}")
    print(f"{'='*60}")
    if saved + skipped > 0:
        print("\n✅ Next:  python backend/app/ai/ml/dataset_builder_v2.py")
    else:
        print("\n❌ Nothing converted."); sys.exit(1)

if __name__ == "__main__":
    main()