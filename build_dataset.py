"""
build_dataset.py  —  Spotter AI Dataset Builder (v2)
=====================================================
Reads side-view squat mp4s from data/sorted/side/{good,bad}/
Extracts MediaPipe pose landmarks frame-by-frame.
Slides a 20-frame window across each video to create sequences.
Applies heavy augmentation for small datasets.
Saves training-ready numpy arrays.

Run from SPOTTER root:
    python build_dataset.py

Output:
    data/processed/squat/X_train.npy   (N, 20, 7)
    data/processed/squat/y_train.npy   (N,)
    data/processed/squat/X_val.npy
    data/processed/squat/y_val.npy
    data/processed/squat/dataset_info.json

Label convention:  0 = BAD,  1 = GOOD  (matches webcam_test.py)
"""

import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Try to use existing mediapipe engine; fall back to direct mediapipe ───────
try:
    from backend.app.ai.pose.mediapipe_engine import detect_pose
    print("✅ Using project mediapipe_engine")
    USE_PROJECT_ENGINE = True
except ImportError:
    print("⚠  Using direct mediapipe (project engine not found)")
    USE_PROJECT_ENGINE = False
    import mediapipe as mp

# ── Paths ──────────────────────────────────────────────────────────────────────
GOOD_DIR  = Path("data/sorted/side/good")
BAD_DIR   = Path("data/sorted/side/bad")
OUT_DIR   = Path("data/processed/squat")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN   = 20     # frames per sequence
FEAT_LEN  = 7      # features per frame
STRIDE    = 5      # sliding window stride (lower = more sequences per video)

# ── Mediapipe fallback ────────────────────────────────────────────────────────
if not USE_PROJECT_ENGINE:
    mp_pose    = mp.solutions.pose
    _pose_inst = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    _LM_NAMES = {
        "LEFT_SHOULDER":  11, "RIGHT_SHOULDER": 12,
        "LEFT_HIP":       23, "RIGHT_HIP":       24,
        "LEFT_KNEE":      25, "RIGHT_KNEE":      26,
        "LEFT_ANKLE":     27, "RIGHT_ANKLE":     28,
    }

    def detect_pose(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _pose_inst.process(rgb)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        h, w = frame.shape[:2]
        return {
            name: (lms[idx].x * w, lms[idx].y * h)
            for name, idx in _LM_NAMES.items()
        }

# ── Angle + feature helpers ───────────────────────────────────────────────────
def angle_at(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_angles(lm):
    lk = angle_at(lm["LEFT_HIP"],       lm["LEFT_KNEE"],   lm["LEFT_ANKLE"])
    rk = angle_at(lm["RIGHT_HIP"],      lm["RIGHT_KNEE"],  lm["RIGHT_ANKLE"])
    lh = angle_at(lm["LEFT_SHOULDER"],  lm["LEFT_HIP"],    lm["LEFT_KNEE"])
    rh = angle_at(lm["RIGHT_SHOULDER"], lm["RIGHT_HIP"],   lm["RIGHT_KNEE"])
    avg_k = (lk + rk) / 2
    return lk, rk, lh, rh, avg_k


def build_features(lk, rk, lh, rh):
    avg_k = (lk + rk) / 2
    return [
        lk / 180,
        rk / 180,
        lh / 180,
        rh / 180,
        abs(lk - rk) / 180,
        abs(lh - rh) / 180,
        avg_k / 180,
    ]

# ── Video → sequence extractor ────────────────────────────────────────────────
def extract_sequences_from_video(video_path, label_name):
    """
    Extract all valid 20-frame sliding window sequences from a video.
    Returns list of (seq, quality_score) tuples.
    quality_score = how much knee angle varied (higher = more dynamic squat).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   ⚠  Cannot open: {video_path.name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"   📹 {video_path.name}  ({total_frames} frames @ {fps:.0f}fps)")

    all_features = []   # list of 7-dim feature vectors, one per frame
    bad_frames   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm = detect_pose(frame)
        if lm is None:
            bad_frames += 1
            # If we get too many consecutive bad frames, insert a neutral one
            if bad_frames <= 3 and all_features:
                all_features.append(all_features[-1])   # repeat last
            continue

        bad_frames = 0
        try:
            lk, rk, lh, rh, avg_k = compute_angles(lm)
            feat = build_features(lk, rk, lh, rh)
            all_features.append(feat)
        except Exception:
            if all_features:
                all_features.append(all_features[-1])

    cap.release()

    if len(all_features) < SEQ_LEN:
        print(f"   ⚠  Too short ({len(all_features)} usable frames) — skipping")
        return []

    # ── Sliding window ─────────────────────────────────────────────────────────
    sequences = []
    feat_arr  = np.array(all_features)
    avg_knees = feat_arr[:, 6] * 180   # feature index 6 = avg_knee/180

    for start in range(0, len(all_features) - SEQ_LEN + 1, STRIDE):
        seq    = all_features[start : start + SEQ_LEN]
        # Quality score: range of average knee angle in this window
        window_knees = avg_knees[start : start + SEQ_LEN]
        quality      = float(np.max(window_knees) - np.min(window_knees))
        sequences.append((seq, quality))

    # ── Filter low-quality static windows (no squat movement) ─────────────────
    # For BAD sequences we're more lenient (bad form at any depth counts)
    min_range = 8.0 if label_name == "good" else 4.0
    filtered  = [(s, q) for s, q in sequences if q >= min_range]

    print(f"      {len(all_features)} frames → {len(sequences)} windows → "
          f"{len(filtered)} kept (min_range={min_range}°)")
    return filtered

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_sequences(seqs, rng, factor=4):
    """
    Augment sequences using:
    - Gaussian noise
    - Temporal jitter (roll ±2 frames)
    - Feature scaling (simulate different body proportions)
    - Mirroring (swap left/right)
    """
    original   = np.array(seqs, dtype=np.float32)
    augmented  = [original]

    for _ in range(factor):
        # Noise
        noisy = original + rng.normal(0, 0.007, original.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 1.0)
        augmented.append(noisy)

        # Temporal roll ±2
        shift   = rng.integers(-2, 3)
        shifted = np.roll(original, shift, axis=1)
        augmented.append(shifted)

        # Scale angles ±5%
        scale  = rng.uniform(0.95, 1.05, (len(original), 1, 7)).astype(np.float32)
        scaled = np.clip(original * scale, 0.0, 1.0)
        augmented.append(scaled)

    # Mirror: swap left/right features
    # Features: [lk, rk, lh, rh, knee_diff, hip_diff, avg_knee]
    # Swap lk↔rk and lh↔rh; diff features stay the same; avg stays same
    mirrored = original.copy()
    mirrored[:, :, 0], mirrored[:, :, 1] = original[:, :, 1].copy(), original[:, :, 0].copy()
    mirrored[:, :, 2], mirrored[:, :, 3] = original[:, :, 3].copy(), original[:, :, 2].copy()
    augmented.append(mirrored)

    return np.concatenate(augmented, axis=0)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🔧 Spotter — Dataset Builder v2")

    if not GOOD_DIR.exists() or not BAD_DIR.exists():
        print(f"❌ Sorted video folders not found.")
        print(f"   Expected: {GOOD_DIR}")
        print(f"   Expected: {BAD_DIR}")
        print("   Run: python sort_videos.py  first")
        return

    good_vids = sorted(GOOD_DIR.glob("*.mp4"))
    bad_vids  = sorted(BAD_DIR.glob("*.mp4"))

    print(f"\n📁 Found {len(good_vids)} GOOD, {len(bad_vids)} BAD side-view videos")

    if len(good_vids) == 0 or len(bad_vids) == 0:
        print("❌ No videos found in sorted folders. Run sort_videos.py first.")
        return

    # ── Extract raw sequences ─────────────────────────────────────────────────
    print("\n── Extracting GOOD sequences ──────────────────────────────")
    good_raw = []
    for vid in good_vids:
        seqs = extract_sequences_from_video(vid, "good")
        good_raw.extend([s for s, q in seqs])

    print(f"\n── Extracting BAD sequences ───────────────────────────────")
    bad_raw = []
    for vid in bad_vids:
        seqs = extract_sequences_from_video(vid, "bad")
        bad_raw.extend([s for s, q in seqs])

    print(f"\n📊 Raw sequences  GOOD={len(good_raw)}  BAD={len(bad_raw)}")

    if len(good_raw) < 5 or len(bad_raw) < 5:
        print("❌ Too few sequences extracted. Check your videos and try again.")
        return

    # ── Augment ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    print("\n🔧 Augmenting...")

    aug_factor = max(2, 120 // max(len(good_raw), 1))   # aim for ~200+ per class
    good_aug   = augment_sequences(good_raw, rng, factor=aug_factor)
    bad_aug    = augment_sequences(bad_raw,  rng, factor=aug_factor)

    print(f"   After augmentation  GOOD={len(good_aug)}  BAD={len(bad_aug)}")

    # ── Balance ───────────────────────────────────────────────────────────────
    min_n    = min(len(good_aug), len(bad_aug))
    good_bal = good_aug[rng.choice(len(good_aug), min_n, replace=False)]
    bad_bal  = bad_aug[rng.choice(len(bad_aug),  min_n, replace=False)]
    print(f"   After balancing     GOOD={len(good_bal)}  BAD={len(bad_bal)}")

    # Labels: 0=BAD, 1=GOOD
    X = np.concatenate([bad_bal, good_bal], axis=0).astype(np.float32)
    y = np.array([0]*len(bad_bal) + [1]*len(good_bal), dtype=np.int32)

    # Shuffle
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # ── Train / val split ─────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy",   X_val)
    np.save(OUT_DIR / "y_val.npy",   y_val)

    info = {
        "good_videos": len(good_vids),
        "bad_videos":  len(bad_vids),
        "good_raw_seqs": len(good_raw),
        "bad_raw_seqs":  len(bad_raw),
        "train_samples": len(X_train),
        "val_samples":   len(X_val),
        "seq_len":  SEQ_LEN,
        "feat_len": FEAT_LEN,
        "features": ["left_knee/180","right_knee/180","left_hip/180",
                     "right_hip/180","knee_diff/180","hip_diff/180","avg_knee/180"],
        "label_map": {"0": "BAD", "1": "GOOD"},
        "camera_view": "side-only",
    }
    with open(OUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Dataset saved to {OUT_DIR}/")
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"\n   Now run: python train_lstm.py")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n🔍 Sanity check (feature index 6 = avg_knee/180):")
    good_mask = y_train == 1
    bad_mask  = y_train == 0
    print(f"   GOOD seqs avg-knee range: "
          f"{X_train[good_mask, :, 6].min()*180:.1f}° – "
          f"{X_train[good_mask, :, 6].max()*180:.1f}°")
    print(f"   BAD  seqs avg-knee range: "
          f"{X_train[bad_mask,  :, 6].min()*180:.1f}° – "
          f"{X_train[bad_mask,  :, 6].max()*180:.1f}°")

    good_min_knee = X_train[good_mask, :, 6].min(axis=1).mean() * 180
    bad_min_knee  = X_train[bad_mask,  :, 6].min(axis=1).mean() * 180
    print(f"\n   GOOD avg minimum knee: {good_min_knee:.1f}°  (should be < 130° for deep squats)")
    print(f"   BAD  avg minimum knee: {bad_min_knee:.1f}°")

    if abs(good_min_knee - bad_min_knee) < 5:
        print("\n⚠  WARNING: GOOD and BAD knee angles look very similar.")
        print("   This likely means your side-view videos don't show clear form differences.")
        print("   Check that 'bad' videos actually show bad form (rounded back, shallow depth, etc.)")
    else:
        print("\n✅ GOOD/BAD separation looks reasonable — proceed to training.")


if __name__ == "__main__":
    main()