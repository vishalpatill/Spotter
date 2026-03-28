"""
Spotter Backend - Hybrid Version

This version keeps your old /predict/ endpoint while adding new Learn Mode endpoints.
Use this to test the new system without breaking your existing setup.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import time

# Your existing imports
from app.ai.pose_pipeline import run_blazepose_on_image, compute_relevant_angles
from app.ai.movement_logic import detect_exercise_from_angles, posture_state_from_angles, detect_danger
from app.services.session_service import get_session, remove_old_sessions

# New imports
from app.ai.exercise_library import get_exercise, detect_exercise_from_pose, get_all_exercises
from app.ai.biomechanics_engine import BiomechanicsEngine
from app.ai.scoring_engine import ScoringEngine
from app.ai.rep_counter import RepCounter


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Spotter Backend - Hybrid",
    description="Old /predict/ endpoint + New Learn Mode endpoints",
    version="2.0.0-hybrid"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Session Manager for NEW endpoints
# ============================================================================

class SessionManager:
    """Manages Learn Mode sessions"""
    
    def __init__(self):
        self.sessions = {}
        self.biomech_engine = BiomechanicsEngine()
    
    def create_session(self, session_id: str, exercise_type: str, mode: str = "learn"):
        """Create new session"""
        self.sessions[session_id] = {
            "session_id": session_id,
            "exercise_type": exercise_type,
            "mode": mode,
            "rep_counter": RepCounter(exercise_type),
            "scoring_engine": ScoringEngine(),
            "started_at": time.time(),
            "frame_count": 0,
            "rep_scores": [],
            "current_rep_frames": [],
            "status": "active"
        }
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str):
        """End a session"""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "completed"
            self.sessions[session_id]["ended_at"] = time.time()


session_manager = SessionManager()


# ============================================================================
# OLD ENDPOINT (Your Original Code - UNCHANGED)
# ============================================================================

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    """
    OLD ENDPOINT - Your original code
    
    This is kept for backward compatibility.
    For new features, use /api/learn-mode/* endpoints instead.
    """
    
    contents = await file.read()
    result = run_blazepose_on_image(contents)
    
    if result is None:
        return {"ok": False, "msg": "no-person-detected"}
    
    landmarks = result["landmarks"]
    angles = compute_relevant_angles(landmarks)
    exercise = detect_exercise_from_angles(angles)
    posture_state = posture_state_from_angles(angles)
    danger_alerts = detect_danger(landmarks, angles)
    
    if not session_id:
        session_id = f"anon-{int(time.time() * 1000)}"
    
    session = get_session(session_id)
    adaptive_feedback = None
    rep_completed = False
    
    if session.get("last_exercise") != exercise:
        session["state"] = "up"
        session["last_exercise"] = exercise
    
    try:
        if exercise in ("squat", "pushup"):
            if exercise == "squat":
                lk = angles.get("left_knee")
                rk = angles.get("right_knee")
                angle_val = (lk + rk) / 2.0 if (lk is not None and rk is not None) else None
                down_thresh, up_thresh = 90.0, 150.0
            else:
                le = angles.get("left_elbow")
                re = angles.get("right_elbow")
                angle_val = (le + re) / 2.0 if (le is not None and re is not None) else None
                down_thresh, up_thresh = 70.0, 150.0
            
            if angle_val is not None:
                if session["state"] == "up" and angle_val < down_thresh:
                    session["state"] = "down"
                elif session["state"] == "down" and angle_val > up_thresh:
                    session["reps"] += 1
                    rep_completed = True
                    session["state"] = "up"
                    session["bad_count"] = 0
    except Exception:
        pass
    
    if posture_state == "bad" or len(danger_alerts) > 0:
        session["bad_count"] += 1
    else:
        session["bad_count"] = 0
    
    if session["bad_count"] >= 3:
        adaptive_feedback = "You keep repeating a posture mistake. Slow down and focus on alignment."
        session["bad_count"] = 0
    
    session["history"].append({
        "ts": time.time(),
        "angles": angles,
        "exercise": exercise,
        "posture_state": posture_state,
        "danger_alerts": danger_alerts,
        "rep_completed": rep_completed
    })
    
    remove_old_sessions()
    
    return {
        "ok": True,
        "session_id": session_id,
        "angles": angles,
        "exercise": exercise,
        "posture_state": posture_state,
        "danger_alerts": danger_alerts,
        "adaptive_feedback": adaptive_feedback,
        "rep_completed": rep_completed,
        "reps_total": session.get("reps", 0),
        "landmark_count": len(landmarks),
    }


# ============================================================================
# NEW ENDPOINTS (Learn Mode with Advanced Features)
# ============================================================================

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "2.0.0-hybrid",
        "service": "Spotter Backend",
        "endpoints": {
            "old": "/predict/",
            "new": ["/api/learn-mode/start", "/api/learn-mode/frame", "/api/learn-mode/finish"]
        }
    }


@app.get("/api/exercises/")
def list_exercises():
    """Get all supported exercises"""
    exercises = get_all_exercises()
    return {
        "total": len(exercises),
        "exercises": [
            {
                "id": ex.id,
                "name": ex.display_name,
                "category": ex.category.value,
                "difficulty": ex.difficulty.value
            }
            for ex in exercises
        ]
    }


@app.post("/api/learn-mode/start")
async def start_learn_mode(
    user_id: str = Form(...),
    exercise_type: str = Form(...),
):
    """
    NEW ENDPOINT - Start a Learn Mode session
    
    Learn Mode provides:
    - Real-time form scoring (0-100)
    - Component scores (depth, symmetry, stability, tempo, alignment)
    - Better rep counting
    - Coaching tips
    """
    
    # Validate exercise
    exercise_def = get_exercise(exercise_type)
    if not exercise_def:
        raise HTTPException(
            status_code=400,
            detail=f"Exercise '{exercise_type}' not supported. Use /api/exercises/ to see available exercises."
        )
    
    # Create session
    session_id = f"{user_id}_{int(time.time() * 1000)}"
    session = session_manager.create_session(session_id, exercise_type, mode="learn")
    
    return {
        "ok": True,
        "session_id": session_id,
        "exercise_type": exercise_type,
        "mode": "learn",
        "message": f"Learn Mode started for {exercise_def.display_name}",
        "tips": exercise_def.tips
    }


@app.post("/api/learn-mode/frame")
async def process_learn_mode_frame(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    NEW ENDPOINT - Process a single frame in Learn Mode
    
    Returns comprehensive analysis:
    - Form score (0-100)
    - Component scores
    - Rep count
    - Danger alerts
    - Coaching tips
    """
    
    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Read image
    contents = await file.read()
    
    # 1. Pose Detection
    pose_result = run_blazepose_on_image(contents)
    if pose_result is None:
        return {
            "ok": False,
            "error": "no_person_detected",
            "message": "Could not detect person in frame"
        }
    
    landmarks = pose_result["landmarks"]
    
    # 2. Calculate angles
    angles = compute_relevant_angles(landmarks)
    
    # 3. Auto-detect exercise
    detected_exercise = detect_exercise_from_pose(angles, landmarks)
    
    # 4. Biomechanics Analysis (NEW)
    biomech_engine = session_manager.biomech_engine
    biomechanics = biomech_engine.analyze(landmarks, session["exercise_type"])
    
    # 5. Scoring (NEW)
    scoring_engine = session["scoring_engine"]
    frame_score = scoring_engine.score_frame(
        {
            "angles": angles,
            "symmetry_score": biomechanics.symmetry_score,
            "balance_score": biomechanics.balance_score,
            "spine_alignment": biomechanics.spine_alignment,
        },
        session["exercise_type"]
    )
    
    # 6. Rep Counting (NEW)
    rep_counter = session["rep_counter"]
    rep_event = rep_counter.update(angles)
    
    # Store frame data
    session["current_rep_frames"].append({
        "timestamp": time.time(),
        "frame_score": frame_score,
        "biomechanics": biomechanics,
        "angles": angles
    })
    
    # If rep completed
    rep_completed = False
    rep_score = None
    coaching_tip = None
    
    if rep_event:
        rep_completed = True
        
        # Score the completed rep
        frame_scores = [f["frame_score"] for f in session["current_rep_frames"]]
        rep_score_obj = scoring_engine.score_rep(
            frame_scores=frame_scores,
            rep_number=rep_event.rep_number,
            rep_duration_sec=rep_event.duration_sec,
            danger_alerts=biomechanics.danger_alerts,
            exercise_type=session["exercise_type"]
        )
        
        session["rep_scores"].append(rep_score_obj)
        session["current_rep_frames"] = []
        
        rep_score = {
            "rep_number": rep_score_obj.rep_number,
            "form_score": round(rep_score_obj.form_score, 1),
            "depth_score": round(rep_score_obj.depth_score, 1),
            "symmetry_score": round(rep_score_obj.symmetry_score, 1),
            "stability_score": round(rep_score_obj.stability_score, 1),
            "tempo_score": round(rep_score_obj.tempo_score, 1),
            "alignment_score": round(rep_score_obj.alignment_score, 1),
            "grade": rep_score_obj.grade,
            "duration_sec": round(rep_score_obj.rep_duration_sec, 2)
        }
        
        coaching_tip = rep_score_obj.coaching_tip
    
    session["frame_count"] += 1
    
    # Response
    return {
        "ok": True,
        "session_id": session_id,
        
        # Exercise info
        "exercise_detected": detected_exercise,
        "exercise_expected": session["exercise_type"],
        
        # Real-time scores (NEW)
        "form_score": round(frame_score.form_score, 1),
        "depth_score": round(frame_score.depth_score, 1),
        "symmetry_score": round(frame_score.symmetry_score, 1),
        "stability_score": round(frame_score.stability_score, 1),
        
        # Rep tracking
        "current_phase": rep_counter.get_current_phase(),
        "total_reps": rep_counter.get_rep_count(),
        "rep_completed": rep_completed,
        "rep_score": rep_score,
        
        # Safety
        "danger_alerts": biomechanics.danger_alerts,
        "danger_severity": biomechanics.danger_severity,
        
        # Coaching (NEW)
        "coaching_tip": coaching_tip,
        
        # Debug info
        "angles": {k: round(v, 1) for k, v in angles.items()},
    }


@app.post("/api/learn-mode/finish")
async def finish_learn_mode(
    session_id: str = Form(...),
):
    """
    NEW ENDPOINT - Finish a Learn Mode session
    
    Returns session summary with grades and analytics
    """
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # End session
    session_manager.end_session(session_id)
    
    # Calculate session stats
    rep_scores = session["rep_scores"]
    
    if not rep_scores:
        return {
            "ok": True,
            "session_id": session_id,
            "total_reps": 0,
            "message": "No reps completed"
        }
    
    form_scores = [r.form_score for r in rep_scores]
    
    avg_form = sum(form_scores) / len(form_scores)
    best_rep = max(form_scores)
    worst_rep = min(form_scores)
    
    # Calculate grade
    if avg_form >= 90:
        grade = "A"
        summary = "Excellent session! Your form was outstanding."
    elif avg_form >= 80:
        grade = "B"
        summary = "Great session! Strong form overall."
    elif avg_form >= 70:
        grade = "C"
        summary = "Good session. Work on consistency."
    else:
        grade = "D"
        summary = "Session complete. Focus on form fundamentals."
    
    return {
        "ok": True,
        "session_id": session_id,
        "exercise_type": session["exercise_type"],
        "mode": session["mode"],
        
        # Session stats
        "total_reps": len(rep_scores),
        "avg_form_score": round(avg_form, 1),
        "best_rep_score": round(best_rep, 1),
        "worst_rep_score": round(worst_rep, 1),
        "grade": grade,
        "summary": summary,
        
        # Duration
        "duration_sec": round(time.time() - session["started_at"], 1),
        
        # Per-rep breakdown
        "reps": [
            {
                "rep": r.rep_number,
                "score": round(r.form_score, 1),
                "grade": r.grade,
                "tip": r.coaching_tip
            }
            for r in rep_scores
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)