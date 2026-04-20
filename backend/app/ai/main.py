"""
main.py  —  Spotter AI Backend
===============================
Place at: SPOTTER/backend/app/ai/main.py

Run from SPOTTER root:
    uvicorn backend.app.ai.main:app --reload --host 0.0.0.0 --port 8000

Docs UI:
    http://localhost:8000/docs
"""

# ── PATH FIX — must be before all other imports ───────────────────────────────
import sys
import os
_SPOTTER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
if _SPOTTER_ROOT not in sys.path:
    sys.path.insert(0, _SPOTTER_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import time
import uuid
from collections import deque
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.ai.pose.pose_pipeline import run_blazepose_on_image
from backend.app.ai.biomechanics.angle_calculator import compute_relevant_angles
from backend.app.ai.pose.movement_logic import (
    detect_danger,
    detect_exercise_from_angles,
    posture_quality_from_angles,
    posture_state_from_angles,
)
from backend.app.ai.pose.rep_counter import RepCounter
from backend.app.ai.ml.model_loader import predict_sequence

app = FastAPI(
    title="Spotter AI",
    description="""
Real-time AI fitness posture correction API.

## Quick start for frontend dev
1. `POST /session/start` → get a `session_id`
2. `POST /session/frame` (multipart: file + session_id) → real-time feedback
3. `GET /session/{id}/summary` → end-of-workout results
4. `DELETE /session/{id}` → cleanup
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict = {}
SESSION_TTL    = 3600


def make_session(exercise: str = "squat") -> dict:
    return {
        "exercise":        exercise,
        "rep_counter":     RepCounter(),
        "seq_buffer":      deque(maxlen=20),
        "pred_buffer":     deque(maxlen=10),
        "last_pred_time":  0.0,
        "last_label":      1,
        "last_label_time": 0.0,
        "bad_streak":      0,
        "form_scores":     [],
        "danger_log":      [],
        "created_at":      time.time(),
        "frame_count":     0,
    }


def cleanup_old_sessions():
    now     = time.time()
    expired = [sid for sid, s in SESSIONS.items()
               if now - s["created_at"] > SESSION_TTL]
    for sid in expired:
        del SESSIONS[sid]


def build_features(angles: dict) -> list:
    lk = angles.get("left_knee",  180)
    rk = angles.get("right_knee", 180)
    lh = angles.get("left_hip",   180)
    rh = angles.get("right_hip",  180)
    return [
        lk / 180.0, rk / 180.0, lh / 180.0, rh / 180.0,
        abs(lk - rk) / 180.0, abs(lh - rh) / 180.0,
        ((lk + rk) / 2) / 180.0,
    ]


DANGER_MESSAGES = {
    "back_rounding":   "Keep your back straight — chest up!",
    "not_deep_enough": "Go deeper — aim for thighs parallel to floor.",
    "imbalance":       "You're favouring one side — keep both legs even.",
}


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "Spotter AI", "version": "1.0.0"}


@app.get("/exercises", tags=["Info"])
def list_exercises():
    return {"exercises": ["squat"]}


@app.post("/session/start", tags=["Session"])
def start_session(exercise: str = "squat"):
    """Create a new workout session. Returns session_id."""
    sid           = str(uuid.uuid4())
    SESSIONS[sid] = make_session(exercise)
    return {"session_id": sid, "exercise": exercise, "message": "Session started"}


@app.delete("/session/{session_id}", tags=["Session"])
def end_session(session_id: str):
    """End and delete a session."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    del SESSIONS[session_id]
    return {"message": "Session ended"}


@app.get("/session/{session_id}/summary", tags=["Session"])
def get_summary(session_id: str):
    """End-of-workout summary — grade, reps, duration, common mistakes."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
    s        = SESSIONS[session_id]
    duration = time.time() - s["created_at"]
    scores   = s["form_scores"]
    avg      = float(sum(scores) / len(scores)) if scores else 0.0
    from collections import Counter
    top_dangers = [d for d, _ in Counter(s["danger_log"]).most_common(3)]
    grade = "A" if avg >= 0.85 else "B" if avg >= 0.70 else "C" if avg >= 0.50 else "D"
    return {
        "session_id": session_id, "exercise": s["exercise"],
        "total_reps": s["rep_counter"].get_count(),
        "duration_seconds": round(duration, 1),
        "avg_form_score": round(avg, 3),
        "common_dangers": top_dangers, "final_grade": grade,
    }


@app.post("/session/frame", tags=["Session"])
async def process_frame(
    file:       UploadFile     = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Send one camera frame, get real-time feedback.

    POST multipart/form-data:
    - file: JPEG image from camera
    - session_id: from /session/start

    Call at 5-10fps.
    """
    cleanup_old_sessions()

    if not session_id or session_id not in SESSIONS:
        session_id        = str(uuid.uuid4())
        SESSIONS[session_id] = make_session("squat")

    s = SESSIONS[session_id]
    s["frame_count"] += 1

    image_bytes     = await file.read()
    pipeline_result = run_blazepose_on_image(image_bytes)

    if pipeline_result is None:
        return {
            "ok": False, "session_id": session_id,
            "msg": "no-person-detected",
            "reps": s["rep_counter"].get_count(),
            "stage": s["rep_counter"].get_stage(),
            "form": "UNKNOWN", "form_confidence": 0.0,
            "danger_alerts": [], "posture_quality": "unknown",
            "exercise_detected": "unknown", "angles": {},
            "adaptive_feedback": None, "landmarks": None,
        }

    landmark_list = pipeline_result["landmarks"]
    angles        = compute_relevant_angles(landmark_list)
    exercise      = detect_exercise_from_angles(angles)
    posture_quality = posture_quality_from_angles(angles)
    danger_codes  = detect_danger(angles)
    s["danger_log"].extend(danger_codes)

    danger_alerts = [
        {"code": c, "message": DANGER_MESSAGES.get(c, c)}
        for c in danger_codes
    ]

    s["rep_counter"].update(angles.get("avg_knee", 180))
    s["seq_buffer"].append(build_features(angles))

    form = "UNKNOWN"
    form_confidence = 0.0

    if len(s["seq_buffer"]) == 20 and time.time() - s["last_pred_time"] > 0.4:
        try:
            label, confidence = predict_sequence(list(s["seq_buffer"]))
            if confidence > 0.65:
                s["pred_buffer"].append(label)
            final_label = (
                max(set(s["pred_buffer"]), key=s["pred_buffer"].count)
                if len(s["pred_buffer"]) >= 6 else s["last_label"]
            )
            if time.time() - s["last_label_time"] > 0.8:
                s["last_label"]      = final_label
                s["last_label_time"] = time.time()
            form            = "GOOD" if s["last_label"] == 1 else "BAD"
            form_confidence = round(confidence, 3)
            s["last_pred_time"] = time.time()
            s["form_scores"].append(1.0 if s["last_label"] == 1 else 0.0)
        except Exception:
            form = "UNKNOWN"

    adaptive_feedback = None
    s["bad_streak"] = s["bad_streak"] + 1 if (form == "BAD" or danger_codes) else 0
    if s["bad_streak"] >= 3:
        if "back_rounding" in danger_codes:
            adaptive_feedback = "Your back keeps rounding — chest up, slow down."
        elif "imbalance" in danger_codes:
            adaptive_feedback = "You're consistently favouring one side — reset and focus."
        elif "not_deep_enough" in danger_codes:
            adaptive_feedback = "You keep stopping short — try to go deeper each rep."
        else:
            adaptive_feedback = "Focus on your form — slow down and reset."
        s["bad_streak"] = 0

    return {
        "ok": True, "session_id": session_id,
        "frame_count": s["frame_count"],
        "reps": s["rep_counter"].get_count(),
        "stage": s["rep_counter"].get_stage(),
        "form": form, "form_confidence": form_confidence,
        "exercise_detected": exercise,
        "posture_quality": posture_quality,
        "angles": {k: round(v, 1) for k, v in angles.items()},
        "danger_alerts": danger_alerts,
        "adaptive_feedback": adaptive_feedback,
        "landmarks": landmark_list,
    }


@app.post("/predict/", tags=["Legacy"])
async def predict_legacy(
    file:       UploadFile     = File(...),
    session_id: Optional[str] = Form(None),
    user_id:    Optional[str] = Form(None),
):
    """Legacy endpoint. Use /session/frame instead."""
    return await process_frame(file=file, session_id=session_id)