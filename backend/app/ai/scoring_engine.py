"""
Scoring Engine - Calculates form scores (0-100) from biomechanics data

This is the bridge between biomechanics analysis and user feedback.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FrameScore:
    """Score for a single frame"""
    form_score: float  # 0-100
    depth_score: float
    symmetry_score: float
    stability_score: float
    tempo_score: float
    alignment_score: float


@dataclass
class RepScore:
    """Score for a complete rep"""
    rep_number: int
    form_score: float
    depth_score: float
    symmetry_score: float
    stability_score: float
    tempo_score: float
    alignment_score: float
    rep_duration_sec: float
    danger_alerts: List[str]
    coaching_tip: str
    grade: str  # A+, A, B, C, D, F


class ScoringEngine:
    """
    Calculates form scores based on biomechanics data
    """
    
    def __init__(self):
        self.angle_history = []
        
    def score_frame(
        self,
        biomechanics_data: Dict,
        exercise_type: str
    ) -> FrameScore:
        """
        Score a single frame
        
        Args:
            biomechanics_data: Output from BiomechanicsEngine.analyze()
            exercise_type: "squat", "bench", "deadlift", etc.
        
        Returns:
            FrameScore with all component scores
        """
        
        # Extract data
        angles = biomechanics_data.get("angles", {})
        symmetry_score = biomechanics_data.get("symmetry_score", 85.0)
        balance_score = biomechanics_data.get("balance_score", 85.0)
        spine_alignment = biomechanics_data.get("spine_alignment", 25.0)
        
        # Calculate component scores
        depth = self._score_depth(angles, exercise_type)
        symmetry = symmetry_score
        stability = balance_score
        tempo = 85.0  # Placeholder (requires temporal analysis)
        alignment = self._score_alignment(spine_alignment)
        
        # Weighted composite score
        form_score = self._composite_score(
            depth, symmetry, stability, tempo, alignment, exercise_type
        )
        
        return FrameScore(
            form_score=form_score,
            depth_score=depth,
            symmetry_score=symmetry,
            stability_score=stability,
            tempo_score=tempo,
            alignment_score=alignment
        )
    
    def score_rep(
        self,
        frame_scores: List[FrameScore],
        rep_number: int,
        rep_duration_sec: float,
        danger_alerts: List[str],
        exercise_type: str
    ) -> RepScore:
        """
        Score a completed rep
        
        Args:
            frame_scores: List of FrameScore for each frame in the rep
            rep_number: Which rep this is (1, 2, 3, ...)
            rep_duration_sec: How long the rep took
            danger_alerts: List of danger flags from biomechanics
            exercise_type: Exercise being performed
        
        Returns:
            RepScore with aggregated analysis
        """
        
        # Aggregate frame scores
        avg_form = np.mean([f.form_score for f in frame_scores])
        avg_depth = np.mean([f.depth_score for f in frame_scores])
        avg_symmetry = np.mean([f.symmetry_score for f in frame_scores])
        avg_stability = np.mean([f.stability_score for f in frame_scores])
        avg_tempo = self._score_rep_tempo(rep_duration_sec, exercise_type)
        avg_alignment = np.mean([f.alignment_score for f in frame_scores])
        
        # Generate coaching tip
        coaching_tip = self._generate_coaching_tip(
            avg_depth, avg_symmetry, avg_stability, avg_tempo, 
            avg_alignment, danger_alerts
        )
        
        # Calculate grade
        grade = self._calculate_grade(avg_form)
        
        return RepScore(
            rep_number=rep_number,
            form_score=avg_form,
            depth_score=avg_depth,
            symmetry_score=avg_symmetry,
            stability_score=avg_stability,
            tempo_score=avg_tempo,
            alignment_score=avg_alignment,
            rep_duration_sec=rep_duration_sec,
            danger_alerts=danger_alerts,
            coaching_tip=coaching_tip,
            grade=grade
        )
    
    def _score_depth(self, angles: Dict[str, float], exercise_type: str) -> float:
        """Score movement depth"""
        
        if exercise_type == "squat":
            left_knee = angles.get("left_knee", 180)
            right_knee = angles.get("right_knee", 180)
            avg_knee = (left_knee + right_knee) / 2
            
            # Scoring: < 90° = parallel or below (good)
            if avg_knee < 70:
                return 100.0  # Deep squat
            elif avg_knee < 90:
                return 90 + ((90 - avg_knee) / 20) * 10
            elif avg_knee < 110:
                return 70 + ((110 - avg_knee) / 20) * 20
            else:
                return max(50, 70 - ((avg_knee - 110) / 2))
        
        elif exercise_type == "bench":
            left_elbow = angles.get("left_elbow", 180)
            right_elbow = angles.get("right_elbow", 180)
            avg_elbow = (left_elbow + right_elbow) / 2
            
            if avg_elbow < 60:
                return 100.0
            elif avg_elbow < 90:
                return 80 + ((90 - avg_elbow) / 30) * 20
            else:
                return max(50, 80 - ((avg_elbow - 90) / 3))
        
        elif exercise_type == "deadlift":
            left_hip = angles.get("left_hip", 180)
            right_hip = angles.get("right_hip", 180)
            avg_hip = (left_hip + right_hip) / 2
            
            if avg_hip > 170:
                return 100.0
            elif avg_hip > 160:
                return 90 + ((avg_hip - 160) / 10) * 10
            else:
                return max(50, 90 - ((160 - avg_hip) / 2))
        
        return 85.0
    
    def _score_alignment(self, spine_deviation: float) -> float:
        """Score spinal alignment"""
        
        if spine_deviation < 20:
            return 100.0
        elif spine_deviation < 40:
            return 80 + ((40 - spine_deviation) / 20) * 20
        else:
            return max(50, 80 - ((spine_deviation - 40) / 2))
    
    def _score_rep_tempo(self, rep_duration_sec: float, exercise_type: str) -> float:
        """Score rep tempo"""
        
        ideal_ranges = {
            "squat": (3.0, 5.0),
            "bench": (2.5, 4.0),
            "deadlift": (2.0, 4.0),
            "overhead_press": (3.0, 5.0),
            "pullup": (3.0, 6.0),
        }
        
        ideal_min, ideal_max = ideal_ranges.get(exercise_type, (2.0, 5.0))
        
        if ideal_min <= rep_duration_sec <= ideal_max:
            return 100.0
        elif rep_duration_sec < ideal_min:
            return max(70, 100 - ((ideal_min - rep_duration_sec) * 10))
        else:
            return max(70, 100 - ((rep_duration_sec - ideal_max) * 5))
    
    def _composite_score(
        self,
        depth: float,
        symmetry: float,
        stability: float,
        tempo: float,
        alignment: float,
        exercise_type: str
    ) -> float:
        """Calculate weighted composite score"""
        
        # Exercise-specific weights
        weights = {
            "squat": {
                "depth": 0.35,
                "symmetry": 0.25,
                "stability": 0.20,
                "tempo": 0.10,
                "alignment": 0.10
            },
            "bench": {
                "depth": 0.30,
                "symmetry": 0.30,
                "stability": 0.15,
                "tempo": 0.15,
                "alignment": 0.10
            },
            "deadlift": {
                "depth": 0.20,
                "symmetry": 0.20,
                "stability": 0.20,
                "tempo": 0.15,
                "alignment": 0.25
            },
        }
        
        w = weights.get(exercise_type, {
            "depth": 0.30,
            "symmetry": 0.25,
            "stability": 0.20,
            "tempo": 0.15,
            "alignment": 0.10
        })
        
        composite = (
            depth * w["depth"] +
            symmetry * w["symmetry"] +
            stability * w["stability"] +
            tempo * w["tempo"] +
            alignment * w["alignment"]
        )
        
        return min(100.0, max(0.0, composite))
    
    def _generate_coaching_tip(
        self,
        depth: float,
        symmetry: float,
        stability: float,
        tempo: float,
        alignment: float,
        dangers: List[str]
    ) -> str:
        """Generate coaching feedback"""
        
        # Priority 1: Dangers
        if dangers:
            danger_tips = {
                "knee_valgus_left": "Push your left knee outward",
                "knee_valgus_right": "Push your right knee outward",
                "rounded_back": "Keep your chest up and core tight",
                "excessive_forward_lean": "Shift your weight back to your heels",
            }
            return danger_tips.get(dangers[0], "Focus on form over weight")
        
        # Priority 2: Worst component
        scores = {
            "depth": (depth, "Go deeper - aim for parallel or below"),
            "symmetry": (symmetry, "Keep both sides moving evenly"),
            "stability": (stability, "Control the movement - don't rush"),
            "tempo": (tempo, "Adjust your tempo - 2-3 sec down, 1-2 sec up"),
            "alignment": (alignment, "Keep your spine neutral")
        }
        
        worst_component = min(scores.items(), key=lambda x: x[1][0])
        
        if worst_component[1][0] < 75:
            return worst_component[1][1]
        
        return "Great rep! Keep it up."
    
    def _calculate_grade(self, form_score: float) -> str:
        """Convert score to letter grade"""
        
        if form_score >= 97:
            return "A+"
        elif form_score >= 93:
            return "A"
        elif form_score >= 90:
            return "A-"
        elif form_score >= 87:
            return "B+"
        elif form_score >= 83:
            return "B"
        elif form_score >= 80:
            return "B-"
        elif form_score >= 77:
            return "C+"
        elif form_score >= 73:
            return "C"
        elif form_score >= 70:
            return "C-"
        elif form_score >= 60:
            return "D"
        else:
            return "F"