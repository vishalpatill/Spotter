"""
Rep Counter - Detects and counts repetitions using state machine logic

This module tracks movement phases (up/down) and counts completed reps.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class RepPhase(Enum):
    """Movement phases"""
    UP = "up"
    DOWN = "down"
    TRANSITION = "transition"


@dataclass
class RepEvent:
    """Event when a rep is completed"""
    rep_number: int
    timestamp: float
    duration_sec: float
    peak_angle: float
    min_angle: float


class RepCounter:
    """
    State machine for counting reps
    
    Uses joint angles to detect when movement crosses thresholds
    """
    
    def __init__(self, exercise_type: str):
        self.exercise_type = exercise_type
        self.current_phase = RepPhase.UP
        self.rep_count = 0
        
        # Timing
        self.phase_start_time = time.time()
        self.rep_start_time = time.time()
        
        # Angle tracking
        self.current_rep_angles = []
        
        # Thresholds (exercise-specific)
        self.down_threshold, self.up_threshold = self._get_thresholds(exercise_type)
        
    def _get_thresholds(self, exercise_type: str) -> Tuple[float, float]:
        """Get angle thresholds for detecting rep phases"""
        
        thresholds = {
            "squat": (90.0, 150.0),      # knee angle
            "bench": (70.0, 150.0),      # elbow angle
            "deadlift": (90.0, 160.0),   # hip angle
            "overhead_press": (90.0, 165.0),  # elbow angle
            "pullup": (60.0, 165.0),     # elbow angle
            "pushup": (80.0, 160.0),     # elbow angle
        }
        
        return thresholds.get(exercise_type, (90.0, 150.0))
    
    def update(self, angles: Dict[str, float]) -> Optional[RepEvent]:
        """
        Update state with new frame
        
        Args:
            angles: Joint angles from biomechanics engine
        
        Returns:
            RepEvent if a rep was completed, None otherwise
        """
        
        # Get primary angle for this exercise
        angle = self._get_primary_angle(angles)
        
        if angle is None:
            return None
        
        # Track angle
        self.current_rep_angles.append(angle)
        
        # State machine
        current_time = time.time()
        
        if self.current_phase == RepPhase.UP:
            # Check if moving down
            if angle < self.down_threshold:
                self.current_phase = RepPhase.DOWN
                self.phase_start_time = current_time
                
        elif self.current_phase == RepPhase.DOWN:
            # Check if moving back up
            if angle > self.up_threshold:
                # Rep completed!
                self.current_phase = RepPhase.UP
                self.rep_count += 1
                
                # Calculate rep stats
                rep_duration = current_time - self.rep_start_time
                peak_angle = max(self.current_rep_angles)
                min_angle = min(self.current_rep_angles)
                
                # Create event
                event = RepEvent(
                    rep_number=self.rep_count,
                    timestamp=current_time,
                    duration_sec=rep_duration,
                    peak_angle=peak_angle,
                    min_angle=min_angle
                )
                
                # Reset for next rep
                self.current_rep_angles = []
                self.rep_start_time = current_time
                self.phase_start_time = current_time
                
                return event
        
        return None
    
    def _get_primary_angle(self, angles: Dict[str, float]) -> Optional[float]:
        """Get the primary angle for this exercise"""
        
        if self.exercise_type == "squat":
            # Use average knee angle
            left_knee = angles.get("left_knee")
            right_knee = angles.get("right_knee")
            if left_knee is not None and right_knee is not None:
                return (left_knee + right_knee) / 2
        
        elif self.exercise_type in ["bench", "overhead_press", "pushup"]:
            # Use average elbow angle
            left_elbow = angles.get("left_elbow")
            right_elbow = angles.get("right_elbow")
            if left_elbow is not None and right_elbow is not None:
                return (left_elbow + right_elbow) / 2
        
        elif self.exercise_type == "deadlift":
            # Use average hip angle
            left_hip = angles.get("left_hip")
            right_hip = angles.get("right_hip")
            if left_hip is not None and right_hip is not None:
                return (left_hip + right_hip) / 2
        
        elif self.exercise_type == "pullup":
            # Use average elbow angle
            left_elbow = angles.get("left_elbow")
            right_elbow = angles.get("right_elbow")
            if left_elbow is not None and right_elbow is not None:
                return (left_elbow + right_elbow) / 2
        
        return None
    
    def reset(self):
        """Reset counter for new session"""
        self.current_phase = RepPhase.UP
        self.rep_count = 0
        self.current_rep_angles = []
        self.rep_start_time = time.time()
        self.phase_start_time = time.time()
    
    def get_current_phase(self) -> str:
        """Get current phase as string"""
        return self.current_phase.value
    
    def get_rep_count(self) -> int:
        """Get total reps counted"""
        return self.rep_count