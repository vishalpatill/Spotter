# 🏋️ Spotter AI
### Real-Time AI Fitness Coach — Pose Detection, Form Analysis & Rep Counting

## Overview

Spotter AI is a real-time exercise coaching system that uses computer vision and deep learning to:

- Detect human pose from a webcam or uploaded image frames
- Classify squat form as **GOOD** or **BAD** in real time
- Count repetitions automatically
- Detect specific form errors (back rounding, knee imbalance, insufficient depth)
- Provide adaptive coaching feedback when mistakes are repeated

The system works in two modes:
- **Local demo** — OpenCV window with live skeleton overlay and colour-coded feedback
- **Web API** — FastAPI backend that a frontend can connect to via HTTP or WebSocket

---

## System Architecture

```
Webcam / Camera Frame
        │
        ▼
┌─────────────────────┐
│   MediaPipe Pose    │  ← 33 body landmarks detected
│  (mp.solutions.pose)│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Angle Calculator   │  ← knee, hip angles computed per frame
│ biomechanics/       │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐     ┌──────────────────────┐
│  Sliding Window     │     │   Rule-Based Checks   │
│  LSTM Classifier    │     │   movement_logic.py   │
│  (20 frames, 7 ft.) │     │   back rounding etc.  │
└─────────────────────┘     └──────────────────────┘
        │                            │
        ▼                            ▼
┌──────────────────────────────────────────┐
│           Output Layer                    │
│  form: GOOD/BAD | reps | danger alerts   │
│  adaptive coaching | stage: up/down      │
└──────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Pose Detection** | MediaPipe detects 33 body landmarks per frame at real-time speed |
| **Form Classification** | Bidirectional LSTM trained on 2,800+ sequences, 93.7% test accuracy |
| **Rep Counting** | Angle-threshold + smoothing, counts full range-of-motion reps only |
| **Danger Detection** | Rule-based checks for back rounding, knee imbalance, shallow depth |
| **Adaptive Coaching** | Triggers specific tips when the same mistake is repeated 3× in a row |
| **Session Tracking** | Per-session rep history, form score, and end-of-session grade (A–D) |
| **REST API** | Full FastAPI backend with auto-generated Swagger docs at `/docs` |
| **Live Demo Mode** | OpenCV window with skeleton overlay, angle labels, colour-coded form |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pose Detection | MediaPipe `mp.solutions.pose` |
| ML Model | TensorFlow / Keras — Bidirectional LSTM |
| API Framework | FastAPI + Uvicorn |
| Computer Vision | OpenCV |
| Data Processing | NumPy, scikit-learn |
| Language | Python 3.10 |

---

## Project Structure

```
SPOTTER/
│
├── backend/
│   └── app/
│       └── ai/
│           ├── main.py                    ← FastAPI app (entry point)
│           │
│           ├── biomechanics/
│           │   └── angle_calculator.py    ← joint angle computation
│           │
│           ├── pose/
│           │   ├── mediapipe_engine.py    ← pose detection (webcam + API)
│           │   ├── pose_pipeline.py       ← image bytes → landmark dict
│           │   ├── movement_logic.py      ← danger detection, form rules
│           │   └── rep_counter.py         ← rep counting with smoothing
│           │
│           └── ml/
│               ├── model_loader.py        ← LSTM model loader + predict
│               ├── squat_model.keras      ← trained model (93.7% accuracy)
│               ├── dataset_builder_v3.py  ← sliding window dataset builder
│               ├── videos_to_json.py      ← MP4 → MediaPipe JSON converter
│               └── train_lstm_v2.py       ← training script
│
├── data/
│   ├── raw/
│   │   └── squat/
│   │       ├── good/    ← good form MP4s + generated JSONs
│   │       └── bad/     ← bad form MP4s + generated JSONs
│   └── processed/
│       └── squat/
│           ├── X.npy    ← training sequences (N, 20, 7)
│           └── y.npy    ← labels (N,)
│
├── webcam_test.py        ← live demo with overlays (no API needed)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10
- pip
- A webcam (for live demo mode)

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Spotter
python3.10 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Training the Model

If you want to retrain from scratch with your own videos:

### Step 1 — Organise your videos

```
data/raw/squat/good/   ← place good form .mp4 files here
data/raw/squat/bad/    ← place bad form .mp4 files here
```

Aim for at least 20 videos per class. More is better.

### Step 2 — Extract pose landmarks from videos

```bash
python backend/app/ai/ml/videos_to_json.py
```

This runs MediaPipe on every video and saves landmark JSON files next to each MP4.

### Step 3 — Build the training dataset

```bash
python backend/app/ai/ml/dataset_builder_v3.py
```

Uses a sliding window (size=20, stride=4) across all JSON files.
Mirror augmentation is applied automatically (doubles dataset size).
Saves `data/processed/squat/X.npy` and `y.npy`.

### Step 4 — Train the model

```bash
python backend/app/ai/ml/train_lstm_v2.py
```

Trains a Bidirectional LSTM with:
- Class-weight balancing for unequal good/bad split
- Early stopping on validation accuracy
- ReduceLROnPlateau scheduler
- Saves best checkpoint automatically

Output: `backend/app/ai/ml/squat_model.keras`

---

## Running the App

### Live Webcam Demo (no frontend needed)

```bash
python webcam_test.py
```

| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Reset rep counter |
| `S` | Save screenshot |

What you'll see:
- **Green skeleton** = good form, **Red skeleton** = bad form
- Large rep counter top-left
- GOOD/BAD badge with confidence % top-centre
- Live joint angles top-right
- Danger alert banners at the bottom
- Adaptive coaching tips when mistakes are repeated

### FastAPI Backend (for web frontend)

```bash
python -m uvicorn backend.app.ai.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**


## Model Performance

| Metric | Value |
|---|---|
| Architecture | Bidirectional LSTM |
| Input shape | (20 frames × 14 features) |
| Test accuracy | **93.7%** |
| Test loss | 0.1882 |
| Training sequences | ~2,800 (with mirror augmentation) |
| Source videos | 68 MP4s (28 good + 40 bad form) |

**14 features per frame:**

### Angles (7)
- Left knee angle
- Right knee angle
- Left hip angle
- Right hip angle
- Knee difference
- Hip difference
- Average knee angle

### Position-based (7)
- Left hip (x, y)
- Right hip (x, y)
- Left knee (x, y)
- Torso alignment

---

## How It Works

### Pose Detection
MediaPipe `mp.solutions.pose` detects 33 body landmarks (x, y normalised coordinates) from each frame. We extract 8 key landmarks: shoulders, hips, knees, ankles.

### Angle Computation
Joint angles are computed using vector dot product:

```
angle at B = arccos( (BA · BC) / (|BA| × |BC|) )
```

Applied to knee (hip→knee→ankle) and hip (shoulder→hip→knee) joints.

### LSTM Classification
A sliding window of 20 consecutive frames is fed into a Bidirectional LSTM. The model outputs a probability for [BAD, GOOD]. Predictions are smoothed using majority voting over the last 10 predictions to prevent flickering.

### Rep Counting
Reps are counted when:
1. Knee angle drops below 120° (enters "down" phase)
2. Time in down position > 0.4s (filters out noise)
3. Knee angle rises above 150° (full return to "up")
4. Time since last count > 0.3s (prevents double counting)

A 5-frame smoothing window is applied to the angle signal.

### Danger Detection
Rule-based checks run on every frame independently of the LSTM:
- **Back rounding** — hip angle < 70°
- **Imbalance** — left/right knee difference > 25°
- **Not deep enough** — both knees between 100–150° (stuck in mid position)

---

## Known Limitations

- Currently trained on **squat only** — other exercises return `exercise_detected: "unknown"`
- Model accuracy may drop for camera angles significantly different from training data (front/side views are best)
- LSTM requires 20 frames to warm up before first prediction — expect `form: "UNKNOWN"` for the first ~2 seconds

---

## Future Work

- [ ] Add more exercises (lunges, push-ups, deadlifts)
- [ ] Expand to 14-feature vectors (normalised keypoint positions + angles)
- [ ] Collect data from multiple body types and camera angles
- [ ] Add user authentication and persistent workout history
- [ ] Mobile app with on-device inference

---
