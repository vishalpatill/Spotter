"""
sort_videos.py  —  Spotter AI Video Sorter
==========================================
Quickly review your existing mp4 videos and sort them by camera angle.
Opens each video, you press a key to classify it.

Run from SPOTTER root:
    python sort_videos.py

Keys during review:
    S  — Side view  (KEEP — good for squat analysis)
    F  — Front view (skip — bad for knee angle measurement)
    X  — Discard    (bad quality, wrong exercise, partial body)
    SPACE — Skip / decide later
    Q  — Quit and save progress

Output:
    data/sorted/side/good/   ← symlinks to usable good-form side-view videos
    data/sorted/side/bad/    ← symlinks to usable bad-form side-view videos
    data/sort_log.json       ← full classification log (resume if interrupted)
"""

import sys
import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_GOOD   = Path("data/raw/squat/good")
RAW_BAD    = Path("data/raw/squat/bad")
SORTED_DIR = Path("data/sorted/side")
LOG_PATH   = Path("data/sort_log.json")

(SORTED_DIR / "good").mkdir(parents=True, exist_ok=True)
(SORTED_DIR / "bad").mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_GREEN  = (50,  205,  50)
C_RED    = (50,   50, 220)
C_YELLOW = (0,   200, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (18,   18,  18)
C_GREY   = (120, 120, 120)
C_ORANGE = (0,   165, 255)


def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}


def save_log(log):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def get_all_videos():
    """Return list of (path, label) tuples for all mp4 files."""
    vids = []
    for p in sorted(RAW_GOOD.glob("*.mp4")):
        vids.append((p, "good"))
    for p in sorted(RAW_BAD.glob("*.mp4")):
        vids.append((p, "bad"))
    return vids


def get_sample_frames(cap, n=6):
    """Get n evenly-spaced frames from a video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = [int(total * i / n) for i in range(n)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def make_contact_sheet(frames, label, filename):
    """
    Combine frames into a 3×2 contact sheet with UI overlay.
    """
    if not frames:
        return None

    target_h, target_w = 240, 320
    resized = [cv2.resize(f, (target_w, target_h)) for f in frames[:6]]

    # Pad to 6
    while len(resized) < 6:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    row1 = np.hstack(resized[:3])
    row2 = np.hstack(resized[3:])
    sheet = np.vstack([row1, row2])

    # UI overlay
    h, w = sheet.shape[:2]
    ui_h = 120
    ui   = np.zeros((ui_h, w, 3), dtype=np.uint8)
    ui[:] = (18, 18, 18)

    label_col = C_GREEN if label == "good" else C_RED
    cv2.putText(ui, f"Label: {label.upper()} FORM", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_col, 2)
    cv2.putText(ui, filename, (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREY, 1)

    # Key hints
    hints = [
        ("[S] SIDE VIEW — keep",   C_GREEN),
        ("[F] FRONT VIEW — skip",  C_YELLOW),
        ("[X] DISCARD — bad quality", C_RED),
        ("[SPACE] skip for now",   C_GREY),
        ("[Q] quit & save",        C_WHITE),
    ]
    for i, (text, col) in enumerate(hints):
        x = 20 + (i % 3) * (w // 3)
        y = 82 + (i // 3) * 22
        cv2.putText(ui, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    return np.vstack([sheet, ui])


def copy_to_sorted(src_path, label, decision):
    """Copy (not symlink) accepted videos to sorted folder."""
    dest = SORTED_DIR / label / src_path.name
    if not dest.exists():
        shutil.copy2(str(src_path), str(dest))


def main():
    print("🎬 Spotter — Video Sorter")
    print(f"   Good videos: {RAW_GOOD}")
    print(f"   Bad  videos: {RAW_BAD}")
    print(f"   Output:      {SORTED_DIR}\n")

    videos = get_all_videos()
    if not videos:
        print(f"❌ No mp4 files found in {RAW_GOOD} or {RAW_BAD}")
        return

    log = load_log()

    # Stats
    side_good = 0; side_bad = 0; front = 0; discarded = 0; skipped = 0

    # Recount from log
    for key, entry in log.items():
        d = entry.get("decision", "skip")
        l = entry.get("label",    "good")
        if d == "side":
            if l == "good": side_good += 1
            else:           side_bad  += 1
        elif d == "front":   front     += 1
        elif d == "discard": discarded += 1

    print(f"   Resuming — {len(log)} / {len(videos)} already classified")

    cv2.namedWindow("Video Sorter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Sorter", 1000, 620)

    for idx, (vid_path, label) in enumerate(videos):
        key_str = str(vid_path)

        if key_str in log:
            print(f"   ⏭  Skip (already done): {vid_path.name}")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"   ⚠  Cannot open: {vid_path.name}")
            log[key_str] = {"label": label, "decision": "error"}
            save_log(log)
            continue

        frames = get_sample_frames(cap, n=6)
        cap.release()

        # Also play a short preview loop
        sheet = make_contact_sheet(frames, label, vid_path.name)
        if sheet is None:
            print(f"   ⚠  No frames: {vid_path.name}")
            log[key_str] = {"label": label, "decision": "error"}
            save_log(log)
            continue

        print(f"\n[{idx+1}/{len(videos)}] {vid_path.name}  ({label})")
        print(f"   GOOD(side)={side_good}  BAD(side)={side_bad}  front={front}  discarded={discarded}")

        decision = None
        while True:
            progress = f"{idx+1}/{len(videos)}"
            info_sheet = sheet.copy()

            # Progress bar at top
            bar_w = int(sheet.shape[1] * (idx / len(videos)))
            cv2.rectangle(info_sheet, (0, 0), (bar_w, 4), C_GREEN, -1)
            cv2.putText(info_sheet, progress, (8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1)

            cv2.imshow("Video Sorter", info_sheet)
            k = cv2.waitKey(0) & 0xFF

            if k == ord("s"):
                decision = "side"
                copy_to_sorted(vid_path, label, decision)
                if label == "good": side_good += 1
                else:               side_bad  += 1
                print(f"   ✅ SIDE ({label})")
                break
            elif k == ord("f"):
                decision = "front"
                front += 1
                print(f"   ↩  FRONT (skipped)")
                break
            elif k == ord("x"):
                decision = "discard"
                discarded += 1
                print(f"   🗑  DISCARDED")
                break
            elif k == ord(" "):
                decision = "skip"
                skipped += 1
                print(f"   ⏭  SKIPPED (undecided)")
                break
            elif k == ord("q"):
                print("\n   Saving and quitting...")
                save_log(log)
                cv2.destroyAllWindows()
                print_summary(side_good, side_bad, front, discarded, skipped)
                return

        log[key_str] = {"label": label, "decision": decision, "file": vid_path.name}
        save_log(log)

    cv2.destroyAllWindows()
    save_log(log)
    print_summary(side_good, side_bad, front, discarded, skipped)


def print_summary(side_good, side_bad, front, discarded, skipped):
    print("\n" + "="*50)
    print("📊 Sort Summary")
    print(f"   ✅ Side GOOD : {side_good}  → data/sorted/side/good/")
    print(f"   ✅ Side BAD  : {side_bad}  → data/sorted/side/bad/")
    print(f"   ↩  Front view: {front}  (skipped)")
    print(f"   🗑  Discarded : {discarded}")
    print(f"   ⏭  Undecided : {skipped}")
    print()

    min_needed = 12
    if side_good < min_needed or side_bad < min_needed:
        print(f"⚠  You have fewer than {min_needed} usable videos per class.")
        print("   Recommendation: record 10–15 more side-view videos of each class.")
        print("   Use data_recorder.py for real-time collection, or film new videos")
        print("   and drop them in data/raw/squat/good or bad.")
    else:
        print("✅ Enough videos! Now run:")
        print("   python build_dataset.py")
        print("   python train_lstm.py")


if __name__ == "__main__":
    main()