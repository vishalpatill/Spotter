"""
Spotter Backend - Unified Main API

This integrates:
- yakup's YAML exercise definitions (from exercise_engine)
- Your scoring system (from pose/)
- Your FastAPI endpoints
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
import time
import yaml
import os

# Your pose detection (original)
from app.ai.pose_pipeline import run_blazepose_on_image, compute_relevant_angles

# Your scoring & rep counting (what we built)
from app.ai.scoring_engine import ScoringEngine, FrameScore
from app.ai.rep_counter import RepCounter

# yakup's exercise loader
from app.ai.exercise_engine.loader import ExerciseLoader

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Spotter - Ultimate Hybrid",
    description="YAML exercises + Advanced scoring + FastAPI",
    version="3.0.0-hybrid"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Load Exercises from YAML
# ============================================================================

class YAMLExerciseManager:
    """Load and manage exercises from YAML files"""
    
    def __init__(self, definitions_dir: str = "app/ai/exercise_engine/definitions"):
        self.definitions_dir = definitions_dir
        self.exercises = {}
        self.load_all()
    
    def load_all(self):
        """Load all YAML exercise files"""
        
        if not os.path.exists(self.definitions_dir):
            print(f"⚠️ Warning: {self.definitions_dir} not found")
            return
        
        for filename in os.listdir(self.definitions_dir):
            if filename.endswith('.yaml'):
                exercise_id = filename.replace('.yaml', '')
                filepath = os.path.join(self.definitions_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        exercise_data = yaml.safe_load(f)
                        self.exercises[exercise_id] = exercise_data
                        print(f"✅ Loaded exercise: {exercise_id}")
                except Exception as e:
                    print(f"❌ Failed to load {filename}: {e}")
    
    def get_exercise(self, exercise_id: str) -> Optional[Dict]:
        """Get exercise definition"""
        return self.exercises.get(exercise_id)
    
    def list_all(self) -> List[Dict]:
        """List all available exercises"""
        return [
            {
                "id": ex_id,
                "name": ex_data.get('name', ex_id),
                "display_name": ex_data.get('display_name', ex_id.title()),
            }
            for ex_id, ex_data in self.exercises.items()
        ]
    
    def get_thresholds(self, exercise_id: str) -> tuple:
        """Get rep detection thresholds for exercise"""
        exercise = self.get_exercise(exercise_id)
        if not exercise:
            return (90.0, 150.0)  # Default
        
        # Parse from YAML
        # yakup's format: angles → knee → down_threshold, up_threshold
        angles = exercise.get('angles', {})
        primary_joint = exercise.get('primary_joint', 'knee')
        
        joint_config = angles.get(primary_joint, {})
        down_thresh = joint_config.get('down_threshold', 90.0)
        up_thresh = joint_config.get('up_threshold', 150.0)
        
        return (down_thresh, up_thresh)


# Global exercise manager
exercise_manager = YAMLExerciseManager()

# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """Manages workout sessions"""
    
    def __init__(self):
        self.sessions = {}
        self.scoring_engine = ScoringEngine()
    
    def create_session(self, session_id: str, exercise_type: str):
        """Create new session"""
        
        # Get thresholds from YAML
        exercise_def = exercise_manager.get_exercise(exercise_type)
        if not exercise_def:
            raise ValueError(f"Exercise '{exercise_type}' not found")
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "exercise_type": exercise_type,
            "exercise_def": exercise_def,
            "rep_counter": RepCounter(exercise_type),
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
# API Endpoints
# ============================================================================

@app.get("/")
def health_check():
    """Health check"""
    return {
        "status": "online",
        "version": "3.0.0-hybrid",
        "service": "Spotter Ultimate",
        "exercises_loaded": len(exercise_manager.exercises),
        "features": [
            "YAML exercise definitions",
            "Advanced AI scoring",
            "Real-time form feedback",
            "Rep counting with FSM"
        ]
    }


@app.get("/api/exercises/")
def list_exercises():
    """Get all exercises from YAML files"""
    exercises = exercise_manager.list_all()
    return {
        "total": len(exercises),
        "exercises": exercises,
        "source": "YAML files from yakup's repo"
    }


@app.get("/api/exercises/{exercise_id}")
def get_exercise_details(exercise_id: str):
    """Get detailed exercise definition"""
    exercise = exercise_manager.get_exercise(exercise_id)
    
    if not exercise:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    
    return {
        "id": exercise_id,
        "definition": exercise
    }


@app.post("/api/learn-mode/start")
async def start_learn_mode(
    user_id: str = Form(...),
    exercise_type: str = Form(...),
):
    """Start Learn Mode session with YAML exercise"""
    
    # Validate exercise exists
    exercise_def = exercise_manager.get_exercise(exercise_type)
    if not exercise_def:
        available = [ex['id'] for ex in exercise_manager.list_all()]
        raise HTTPException(
            status_code=400,
            detail=f"Exercise '{exercise_type}' not found. Available: {available}"
        )
    
    # Create session
    session_id = f"{user_id}_{int(time.time() * 1000)}"
    session = session_manager.create_session(session_id, exercise_type)
    
    return {
        "ok": True,
        "session_id": session_id,
        "exercise_type": exercise_type,
        "exercise_name": exercise_def.get('display_name', exercise_type),
        "mode": "learn",
        "message": f"Learn Mode started for {exercise_def.get('display_name', exercise_type)}",
        "tips": exercise_def.get('tips', [])
    }


@app.post("/api/learn-mode/frame")
async def process_frame(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Process frame with YAML + Scoring hybrid"""
    
    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Read image
    contents = await file.read()
    
    # 1. Pose Detection (your code)
    pose_result = run_blazepose_on_image(contents)
    if pose_result is None:
        return {
            "ok": False,
            "error": "no_person_detected",
            "message": "Could not detect person in frame"
        }
    
    landmarks = pose_result["landmarks"]
    
    # 2. Calculate angles (your code)
    angles = compute_relevant_angles(landmarks)
    
    # 3. Simple biomechanics (for now)
    # In future, integrate yakup's angle_calculator if needed
    danger_alerts = []
    
    # Basic danger check (simplified)
    left_knee = angles.get("left_knee", 180)
    right_knee = angles.get("right_knee", 180)
    
    if abs(left_knee - right_knee) > 20:
        danger_alerts.append("asymmetric_movement")
    
    # 4. Scoring (YOUR advanced system - yakup doesn't have this!)
    frame_score = session_manager.scoring_engine.score_frame(
        {
            "angles": angles,
            "symmetry_score": 100 - abs(left_knee - right_knee),
            "balance_score": 85.0,  # Simplified
            "spine_alignment": 25.0,
        },
        session["exercise_type"]
    )
    
    # 5. Rep Counting (your code)
    rep_counter = session["rep_counter"]
    rep_event = rep_counter.update(angles)
    
    # Track frame
    session["current_rep_frames"].append({
        "timestamp": time.time(),
        "frame_score": frame_score,
        "angles": angles
    })
    
    # If rep completed
    rep_completed = False
    rep_score = None
    coaching_tip = None
    
    if rep_event:
        rep_completed = True
        
        # Score the rep
        frame_scores = [f["frame_score"] for f in session["current_rep_frames"]]
        rep_score_obj = session_manager.scoring_engine.score_rep(
            frame_scores=frame_scores,
            rep_number=rep_event.rep_number,
            rep_duration_sec=rep_event.duration_sec,
            danger_alerts=danger_alerts,
            exercise_type=session["exercise_type"]
        )
        
        session["rep_scores"].append(rep_score_obj)
        session["current_rep_frames"] = []
        
        rep_score = {
            "rep_number": rep_score_obj.rep_number,
            "form_score": round(rep_score_obj.form_score, 1),
            "grade": rep_score_obj.grade,
            "duration_sec": round(rep_score_obj.rep_duration_sec, 2)
        }
        
        coaching_tip = rep_score_obj.coaching_tip
    
    session["frame_count"] += 1
    
    # Get coaching cues from YAML
    exercise_def = session["exercise_def"]
    available_tips = exercise_def.get('tips', [])
    
    return {
        "ok": True,
        "session_id": session_id,
        
        # Exercise info
        "exercise_type": session["exercise_type"],
        
        # Real-time scores (YOUR ADVANCED SYSTEM!)
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
        "danger_alerts": danger_alerts,
        
        # Coaching
        "coaching_tip": coaching_tip if coaching_tip else (available_tips[0] if available_tips else None),
        
        # Debug
        "angles": {k: round(v, 1) for k, v in angles.items()},
        "source": "YAML exercise + YOUR scoring system"
    }


@app.post("/api/learn-mode/finish")
async def finish_session(
    session_id: str = Form(...),
):
    """Finish session and get summary"""
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.end_session(session_id)
    
    rep_scores = session["rep_scores"]
    
    if not rep_scores:
        return {
            "ok": True,
            "session_id": session_id,
            "total_reps": 0,
            "message": "No reps completed"
        }
    
    # Calculate stats
    form_scores = [r.form_score for r in rep_scores]
    avg_form = sum(form_scores) / len(form_scores)
    
    # Grade
    if avg_form >= 90:
        grade = "A"
        summary = "Excellent session!"
    elif avg_form >= 80:
        grade = "B"
        summary = "Great session!"
    else:
        grade = "C"
        summary = "Good effort!"
    
    return {
        "ok": True,
        "session_id": session_id,
        "exercise_type": session["exercise_type"],
        "total_reps": len(rep_scores),
        "avg_form_score": round(avg_form, 1),
        "grade": grade,
        "summary": summary,
        "duration_sec": round(time.time() - session["started_at"], 1),
        "reps": [
            {
                "rep": r.rep_number,
                "score": round(r.form_score, 1),
                "grade": r.grade
            }
            for r in rep_scores
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)