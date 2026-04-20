"""
dataset_builder_v3.py  — Sliding Window Edition
================================================
Place at: SPOTTER/backend/app/ai/ml/dataset_builder_v3.py

The CRITICAL fix: instead of 1 sequence per video (what v2 did),
this slides a window of 20 frames across every video to extract
as many sequences as possible.

Example:
    Video with 198 frames, window=20, stride=4
    → (198 - 20) / 4 = ~44 sequences from ONE video

This turns your ~70 videos into 500-1000+ sequences → proper training data.

Run from SPOTTER root:
    python backend/app/ai/ml/dataset_builder_v3.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from math import atan2, degrees

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_FILE    = Path(__file__).resolve()
SPOTTER_ROOT = THIS_FILE.parents[4]
RAW_DIR      = SPOTTER_ROOT / "data" / "raw" / "squat"
OUT_DIR      = SPOTTER_ROOT / "data" / "processed" / "squat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 20    # frames per sequence (must match model input)
STRIDE      = 4     # step between windows — smaller = more sequences, more overlap
                    # 4 means ~80% overlap between consecutive windows — aggressive augmentation

# ── Angle maths (inline, no imports needed) ───────────────────────────────────
def angle_between(p1, p2, p3) -> float:
    x1, y1 = p1;  x2, y2 = p2;  x3, y3 = p3
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)
    norm1 = (v1[0]**2 + v1[1]**2) ** 0.5
    norm2 = (v2[0]**2 + v2[1]**2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_a = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return degrees(atan2(
        (v2[1]*v1[0] - v2[0]*v1[1]),   # cross product for direction
        dot
    ))


def angle_at(lm, i, j, k) -> float:
    """Angle at landmark j, formed by i→j→k."""
    p1 = (lm[i]["x"], lm[i]["y"])
    p2 = (lm[j]["x"], lm[j]["y"])
    p3 = (lm[k]["x"], lm[k]["y"])
    # Use atan2 approach for stable 0-180 range
    a = (p1[0]-p2[0], p1[1]-p2[1])
    b = (p3[0]-p2[0], p3[1]-p2[1])
    dot   = a[0]*b[0] + a[1]*b[1]
    cross = a[0]*b[1] - a[1]*b[0]
    return abs(degrees(atan2(abs(cross), dot)))


def build_features(lm: list) -> list[float] | None:
    """
    7-feature vector from one frame.
    Returns None if landmarks are missing/zero (bad frame).
    """
    try:
        lk = angle_at(lm, 23, 25, 27)   # left knee
        rk = angle_at(lm, 24, 26, 28)   # right knee
        lh = angle_at(lm, 11, 23, 25)   # left hip
        rh = angle_at(lm, 12, 24, 26)   # right hip

        # Reject frames where key landmarks are at 0,0 (not detected)
        key_pts = [lm[i] for i in [23, 24, 25, 26, 27, 28]]
        if any(p["x"] == 0.0 and p["y"] == 0.0 for p in key_pts):
            return None

        avg_knee = (lk + rk) / 2

        return [
            lk / 180.0,
            rk / 180.0,
            lh / 180.0,
            rh / 180.0,
            abs(lk - rk) / 180.0,
            abs(lh - rh) / 180.0,
            avg_knee / 180.0,
        ]
    except Exception:
        return None


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Swap left/right — free augmentation."""
    m = seq.copy()
    m[:, 0], m[:, 1] = seq[:, 1].copy(), seq[:, 0].copy()  # knee L↔R
    m[:, 2], m[:, 3] = seq[:, 3].copy(), seq[:, 2].copy()  # hip  L↔R
    return m


def extract_sequences_from_json(json_path: Path) -> list[np.ndarray]:
    """
    Sliding window over all frames in a JSON file.
    Returns list of (WINDOW_SIZE, 7) arrays.
    """
    with open(json_path) as f:
        recording = json.load(f)

    # Build feature for each frame
    feature_frames = []
    for frame in recording:
        lm = frame.get("landmarks", [])
        if len(lm) < 29:
            continue
        feat = build_features(lm)
        if feat is not None:
            feature_frames.append(feat)

    if len(feature_frames) < WINDOW_SIZE:
        return []   # video too short

    # Slide window
    sequences = []
    for start in range(0, len(feature_frames) - WINDOW_SIZE + 1, STRIDE):
        window = feature_frames[start : start + WINDOW_SIZE]
        seq    = np.array(window, dtype=np.float32)   # (20, 7)
        sequences.append(seq)

    return sequences


def main():
    X_list, y_list = [], []
    stats = {}

    for label_name, label_val in [("good", 1), ("bad", 0)]:
        folder = RAW_DIR / label_name
        if not folder.exists():
            print(f"⚠️  Missing folder: {folder}")
            continue

        jsons = sorted(folder.glob("*.json"))
        print(f"\n{'='*60}")
        print(f"  {label_name.upper()}  —  {len(jsons)} JSON file(s)")
        print(f"{'='*60}")

        label_seq_count = 0

        for jp in jsons:
            seqs = extract_sequences_from_json(jp)

            if not seqs:
                print(f"  ⚠️  {jp.name}  — too short, skipping")
                continue

            for seq in seqs:
                X_list.append(seq)
                y_list.append(label_val)
                # Mirror augmentation
                X_list.append(mirror_sequence(seq))
                y_list.append(label_val)

            label_seq_count += len(seqs)
            print(f"  ✅ {jp.name:<45}  {len(seqs):>4} sequences → {len(seqs)*2} with mirror")

        stats[label_name] = label_seq_count * 2   # *2 for mirror
        print(f"\n  {label_name.upper()} total (with mirror): {stats[label_name]} sequences")

    if not X_list:
        print("\n❌ No data built.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)

    # Shuffle before saving
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)

    import json as _json
    meta = {
        "total_sequences":  int(X.shape[0]),
        "sequence_length":  int(X.shape[1]),
        "feature_count":    int(X.shape[2]),
        "good_count":       int(np.sum(y == 1)),
        "bad_count":        int(np.sum(y == 0)),
        "window_size":      WINDOW_SIZE,
        "stride":           STRIDE,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        _json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  🎉 Dataset built!")
    print(f"  Total sequences : {X.shape[0]}")
    print(f"  Shape           : X={X.shape}  y={y.shape}")
    print(f"  Good            : {meta['good_count']}")
    print(f"  Bad             : {meta['bad_count']}")
    print(f"  Saved →  {OUT_DIR}")
    print(f"{'='*60}")
    print(f"\n✅ Next:")
    print(f"  python backend/app/ai/ml/train_lstm_v2.py")

    if X.shape[0] < 200:
        print(f"\n⚠️  Still under 200 sequences. Consider reducing STRIDE to 2.")
    elif X.shape[0] > 500:
        print(f"\n🔥 Great dataset size! Model should train well.")


if __name__ == "__main__":
    main()