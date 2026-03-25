"""
Biomechanics Engine - Advanced joint angle calculations and movement analysis

This module handles:
- Joint angle calculations from MediaPipe landmarks
- Symmetry analysis (left vs right)
- Range of motion tracking
- Danger pattern detection
- Alignment analysis
- Stability metrics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from math import atan2, degrees, sqrt
from dataclasses import dataclass


# MediaPipe Pose Landmark Indices
# Full reference: https://google.github.io/mediapipe/solutions/pose.html
class LandmarkIndex:
    """MediaPipe Pose landmark indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class BiomechanicsData:
    """Complete biomechanics analysis for a frame"""
    # Joint angles (degrees)
    angles: Dict[str, float]
    
    # Symmetry metrics
    symmetry_score: float  # 0-100
    left_right_differences: Dict[str, float]
    
    # Alignment
    spine_alignment: float  # Deviation from neutral
    shoulder_level: float   # Difference in shoulder height
    hip_level: float        # Difference in hip height
    
    # Stability
    center_of_mass: Tuple[float, float]
    balance_score: float  # 0-100
    
    # Range of motion
    rom_percentages: Dict[str, float]  # % of expected ROM
    
    # Danger flags
    danger_alerts: List[str]
    danger_severity: str  # "none", "low", "medium", "high", "critical"


class BiomechanicsEngine:
    """
    Advanced biomechanics analysis engine
    
    Responsibilities:
    - Calculate all joint angles
    - Detect asymmetries
    - Identify dangerous patterns
    - Track stability
    """
    
    def __init__(self):
        self.landmark_idx = LandmarkIndex()
        
    def analyze(self, landmarks: List[Dict], exercise_type: str = None) -> BiomechanicsData:
        """
        Complete biomechanics analysis of a pose
        
        Args:
            landmarks: List of MediaPipe landmarks with x, y, z, visibility
            exercise_type: Optional exercise type for context-specific analysis
            
        Returns:
            BiomechanicsData with comprehensive analysis
        """
        
        # Calculate all joint angles
        angles = self.calculate_all_angles(landmarks)
        
        # Symmetry analysis
        symmetry_score, lr_diffs = self.analyze_symmetry(angles, landmarks)
        
        # Alignment analysis
        spine_align = self.calculate_spine_alignment(landmarks)
        shoulder_level = self.calculate_shoulder_level(landmarks)
        hip_level = self.calculate_hip_level(landmarks)
        
        # Stability
        com = self.calculate_center_of_mass(landmarks)
        balance = self.calculate_balance_score(landmarks, com)
        
        # Range of motion
        rom_pcts = self.calculate_rom_percentages(angles, exercise_type)
        
        # Danger detection
        dangers, severity = self.detect_all_dangers(
            landmarks, angles, exercise_type
        )
        
        return BiomechanicsData(
            angles=angles,
            symmetry_score=symmetry_score,
            left_right_differences=lr_diffs,
            spine_alignment=spine_align,
            shoulder_level=shoulder_level,
            hip_level=hip_level,
            center_of_mass=com,
            balance_score=balance,
            rom_percentages=rom_pcts,
            danger_alerts=dangers,
            danger_severity=severity
        )
    
    # ========================================================================
    # ANGLE CALCULATIONS
    # ========================================================================
    
    def calculate_all_angles(self, landmarks: List[Dict]) -> Dict[str, float]:
        """Calculate all relevant joint angles"""
        
        angles = {}
        
        try:
            # Lower body angles
            angles["left_knee"] = self._angle_between_points(
                landmarks[self.landmark_idx.LEFT_HIP],
                landmarks[self.landmark_idx.LEFT_KNEE],
                landmarks[self.landmark_idx.LEFT_ANKLE]
            )
            
            angles["right_knee"] = self._angle_between_points(
                landmarks[self.landmark_idx.RIGHT_HIP],
                landmarks[self.landmark_idx.RIGHT_KNEE],
                landmarks[self.landmark_idx.RIGHT_ANKLE]
            )
            
            angles["left_hip"] = self._angle_between_points(
                landmarks[self.landmark_idx.LEFT_SHOULDER],
                landmarks[self.landmark_idx.LEFT_HIP],
                landmarks[self.landmark_idx.LEFT_KNEE]
            )
            
            angles["right_hip"] = self._angle_between_points(
                landmarks[self.landmark_idx.RIGHT_SHOULDER],
                landmarks[self.landmark_idx.RIGHT_HIP],
                landmarks[self.landmark_idx.RIGHT_KNEE]
            )
            
            angles["left_ankle"] = self._angle_between_points(
                landmarks[self.landmark_idx.LEFT_KNEE],
                landmarks[self.landmark_idx.LEFT_ANKLE],
                landmarks[self.landmark_idx.LEFT_FOOT_INDEX]
            )
            
            angles["right_ankle"] = self._angle_between_points(
                landmarks[self.landmark_idx.RIGHT_KNEE],
                landmarks[self.landmark_idx.RIGHT_ANKLE],
                landmarks[self.landmark_idx.RIGHT_FOOT_INDEX]
            )
            
            # Upper body angles
            angles["left_shoulder"] = self._angle_between_points(
                landmarks[self.landmark_idx.LEFT_HIP],
                landmarks[self.landmark_idx.LEFT_SHOULDER],
                landmarks[self.landmark_idx.LEFT_ELBOW]
            )
            
            angles["right_shoulder"] = self._angle_between_points(
                landmarks[self.landmark_idx.RIGHT_HIP],
                landmarks[self.landmark_idx.RIGHT_SHOULDER],
                landmarks[self.landmark_idx.RIGHT_ELBOW]
            )
            
            angles["left_elbow"] = self._angle_between_points(
                landmarks[self.landmark_idx.LEFT_SHOULDER],
                landmarks[self.landmark_idx.LEFT_ELBOW],
                landmarks[self.landmark_idx.LEFT_WRIST]
            )
            
            angles["right_elbow"] = self._angle_between_points(
                landmarks[self.landmark_idx.RIGHT_SHOULDER],
                landmarks[self.landmark_idx.RIGHT_ELBOW],
                landmarks[self.landmark_idx.RIGHT_WRIST]
            )
            
            # Torso angles
            angles["torso_lean"] = self._calculate_torso_lean(landmarks)
            angles["spine_angle"] = self._calculate_spine_curvature(landmarks)
            
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return angles
    
    def _angle_between_points(
        self, 
        point1: Dict, 
        point2: Dict, 
        point3: Dict
    ) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3
        
        Returns angle in degrees (0-180)
        """
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]
        x3, y3 = point3["x"], point3["y"]
        
        # Vectors
        v1 = np.array([x1 - x2, y1 - y2])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Angle using atan2
        angle_rad = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
        angle_deg = abs(degrees(angle_rad))
        
        # Normalize to 0-180
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
            
        return angle_deg
    
    def _calculate_torso_lean(self, landmarks: List[Dict]) -> float:
        """
        Calculate forward/backward lean of torso
        
        Returns:
            Positive = forward lean (degrees from vertical)
            Negative = backward lean
        """
        shoulder_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_SHOULDER],
            landmarks[self.landmark_idx.RIGHT_SHOULDER]
        )
        
        hip_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_HIP],
            landmarks[self.landmark_idx.RIGHT_HIP]
        )
        
        # Calculate angle from vertical
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        
        # atan2 returns angle from horizontal, convert to from vertical
        angle = degrees(atan2(dx, dy))
        
        return angle
    
    def _calculate_spine_curvature(self, landmarks: List[Dict]) -> float:
        """
        Estimate spine curvature (rounding)
        
        Returns:
            Deviation from neutral spine (0 = neutral, higher = more rounded)
        """
        nose = landmarks[self.landmark_idx.NOSE]
        shoulder_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_SHOULDER],
            landmarks[self.landmark_idx.RIGHT_SHOULDER]
        )
        hip_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_HIP],
            landmarks[self.landmark_idx.RIGHT_HIP]
        )
        
        # Check if upper back is curved forward
        # In neutral spine, nose should be roughly above shoulders
        horizontal_deviation = abs(nose["x"] - shoulder_mid[0])
        
        return horizontal_deviation
    
    # ========================================================================
    # SYMMETRY ANALYSIS
    # ========================================================================
    
    def analyze_symmetry(
        self, 
        angles: Dict[str, float],
        landmarks: List[Dict]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze left-right symmetry
        
        Returns:
            (symmetry_score, differences_dict)
            symmetry_score: 0-100 (100 = perfect symmetry)
        """
        
        differences = {}
        total_asymmetry = 0
        count = 0
        
        # Compare bilateral joints
        pairs = [
            ("left_knee", "right_knee"),
            ("left_hip", "right_hip"),
            ("left_ankle", "right_ankle"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
        ]
        
        for left_joint, right_joint in pairs:
            if left_joint in angles and right_joint in angles:
                left_angle = angles[left_joint]
                right_angle = angles[right_joint]
                diff = abs(left_angle - right_angle)
                
                differences[f"{left_joint}_vs_{right_joint}"] = diff
                total_asymmetry += diff
                count += 1
        
        # Calculate symmetry score
        if count > 0:
            avg_asymmetry = total_asymmetry / count
            # Convert to 0-100 score (10° diff = 90 score, 30° diff = 70 score)
            symmetry_score = max(0, 100 - (avg_asymmetry * 2))
        else:
            symmetry_score = 100.0
        
        return symmetry_score, differences
    
    # ========================================================================
    # ALIGNMENT ANALYSIS
    # ========================================================================
    
    def calculate_spine_alignment(self, landmarks: List[Dict]) -> float:
        """
        Calculate spine alignment score
        
        Returns deviation from neutral (0 = perfect, higher = worse)
        """
        
        # Get key spinal reference points
        nose = landmarks[self.landmark_idx.NOSE]
        shoulder_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_SHOULDER],
            landmarks[self.landmark_idx.RIGHT_SHOULDER]
        )
        hip_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_HIP],
            landmarks[self.landmark_idx.RIGHT_HIP]
        )
        
        # Calculate alignment score
        # Ideal: nose, shoulders, hips in vertical line
        shoulder_hip_dx = abs(shoulder_mid[0] - hip_mid[0])
        nose_shoulder_dx = abs(nose["x"] - shoulder_mid[0])
        
        total_deviation = shoulder_hip_dx + nose_shoulder_dx
        
        return total_deviation
    
    def calculate_shoulder_level(self, landmarks: List[Dict]) -> float:
        """Calculate difference in shoulder height"""
        left_shoulder = landmarks[self.landmark_idx.LEFT_SHOULDER]
        right_shoulder = landmarks[self.landmark_idx.RIGHT_SHOULDER]
        
        return abs(left_shoulder["y"] - right_shoulder["y"])
    
    def calculate_hip_level(self, landmarks: List[Dict]) -> float:
        """Calculate difference in hip height"""
        left_hip = landmarks[self.landmark_idx.LEFT_HIP]
        right_hip = landmarks[self.landmark_idx.RIGHT_HIP]
        
        return abs(left_hip["y"] - right_hip["y"])
    
    # ========================================================================
    # STABILITY & BALANCE
    # ========================================================================
    
    def calculate_center_of_mass(
        self, 
        landmarks: List[Dict]
    ) -> Tuple[float, float]:
        """
        Estimate center of mass
        
        Simplified: weighted average of key body parts
        """
        
        # Weight distribution (approximate)
        weights = {
            "head": 0.08,
            "torso": 0.50,
            "arms": 0.10,
            "legs": 0.32,
        }
        
        # Head
        nose = landmarks[self.landmark_idx.NOSE]
        head_com = (nose["x"], nose["y"])
        
        # Torso (shoulder to hip midpoint)
        shoulder_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_SHOULDER],
            landmarks[self.landmark_idx.RIGHT_SHOULDER]
        )
        hip_mid = self._midpoint(
            landmarks[self.landmark_idx.LEFT_HIP],
            landmarks[self.landmark_idx.RIGHT_HIP]
        )
        torso_com = self._midpoint_tuples(shoulder_mid, hip_mid)
        
        # Arms
        left_elbow = landmarks[self.landmark_idx.LEFT_ELBOW]
        right_elbow = landmarks[self.landmark_idx.RIGHT_ELBOW]
        arms_com = ((left_elbow["x"] + right_elbow["x"]) / 2,
                    (left_elbow["y"] + right_elbow["y"]) / 2)
        
        # Legs
        left_knee = landmarks[self.landmark_idx.LEFT_KNEE]
        right_knee = landmarks[self.landmark_idx.RIGHT_KNEE]
        legs_com = ((left_knee["x"] + right_knee["x"]) / 2,
                    (left_knee["y"] + right_knee["y"]) / 2)
        
        # Weighted average
        com_x = (
            head_com[0] * weights["head"] +
            torso_com[0] * weights["torso"] +
            arms_com[0] * weights["arms"] +
            legs_com[0] * weights["legs"]
        )
        
        com_y = (
            head_com[1] * weights["head"] +
            torso_com[1] * weights["torso"] +
            arms_com[1] * weights["arms"] +
            legs_com[1] * weights["legs"]
        )
        
        return (com_x, com_y)
    
    def calculate_balance_score(
        self, 
        landmarks: List[Dict],
        com: Tuple[float, float]
    ) -> float:
        """
        Calculate balance/stability score
        
        Returns: 0-100 (100 = perfect balance)
        """
        
        # Base of support (between feet)
        left_foot = landmarks[self.landmark_idx.LEFT_ANKLE]
        right_foot = landmarks[self.landmark_idx.RIGHT_ANKLE]
        
        foot_mid_x = (left_foot["x"] + right_foot["x"]) / 2
        foot_mid_y = (left_foot["y"] + right_foot["y"]) / 2
        
        # COM should be over base of support
        com_offset = sqrt(
            (com[0] - foot_mid_x) ** 2 + 
            (com[1] - foot_mid_y) ** 2
        )
        
        # Convert to score (lower offset = better balance)
        balance_score = max(0, 100 - (com_offset * 2))
        
        return balance_score
    
    # ========================================================================
    # RANGE OF MOTION
    # ========================================================================
    
    def calculate_rom_percentages(
        self,
        angles: Dict[str, float],
        exercise_type: Optional[str]
    ) -> Dict[str, float]:
        """
        Calculate what % of expected ROM is being used
        
        Useful for depth scoring in exercises
        """
        
        rom_pcts = {}
        
        # Expected ROM ranges (degrees) for major joints
        expected_rom = {
            "knee": (0, 140),      # Full extension to full flexion
            "hip": (0, 120),
            "elbow": (0, 145),
            "shoulder": (0, 180),
            "ankle": (70, 110),    # Dorsiflexion to plantarflexion
        }
        
        for joint in ["knee", "hip", "elbow", "shoulder", "ankle"]:
            left_key = f"left_{joint}"
            right_key = f"right_{joint}"
            
            if left_key in angles:
                min_rom, max_rom = expected_rom[joint]
                current = angles[left_key]
                
                # % through ROM
                pct = ((current - min_rom) / (max_rom - min_rom)) * 100
                pct = max(0, min(100, pct))  # Clamp 0-100
                
                rom_pcts[left_key] = pct
            
            if right_key in angles:
                min_rom, max_rom = expected_rom[joint]
                current = angles[right_key]
                
                pct = ((current - min_rom) / (max_rom - min_rom)) * 100
                pct = max(0, min(100, pct))
                
                rom_pcts[right_key] = pct
        
        return rom_pcts
    
    # ========================================================================
    # DANGER DETECTION
    # ========================================================================
    
    def detect_all_dangers(
        self,
        landmarks: List[Dict],
        angles: Dict[str, float],
        exercise_type: Optional[str]
    ) -> Tuple[List[str], str]:
        """
        Detect all dangerous movement patterns
        
        Returns:
            (list of danger alerts, overall severity)
        """
        
        dangers = []
        
        # Generic dangers (apply to all exercises)
        dangers.extend(self.check_knee_valgus(landmarks, angles))
        dangers.extend(self.check_spine_rounding(landmarks, angles))
        dangers.extend(self.check_excessive_lean(landmarks))
        
        # Exercise-specific dangers
        if exercise_type == "squat":
            dangers.extend(self.check_squat_dangers(landmarks, angles))
        elif exercise_type == "deadlift":
            dangers.extend(self.check_deadlift_dangers(landmarks, angles))
        elif exercise_type == "bench":
            dangers.extend(self.check_bench_dangers(landmarks, angles))
        
        # Determine overall severity
        severity = self._determine_severity(dangers)
        
        return dangers, severity
    
    def check_knee_valgus(
        self, 
        landmarks: List[Dict],
        angles: Dict[str, float]
    ) -> List[str]:
        """
        Check for knee valgus (knees caving in)
        
        This is a CRITICAL danger pattern
        """
        alerts = []
        
        # Get knee and ankle positions
        left_knee = landmarks[self.landmark_idx.LEFT_KNEE]
        left_ankle = landmarks[self.landmark_idx.LEFT_ANKLE]
        right_knee = landmarks[self.landmark_idx.RIGHT_KNEE]
        right_ankle = landmarks[self.landmark_idx.RIGHT_ANKLE]
        
        # Knee should be roughly over ankle (within threshold)
        left_deviation = abs(left_knee["x"] - left_ankle["x"])
        right_deviation = abs(right_knee["x"] - right_ankle["x"])
        
        # Threshold: if knee is > 30 pixels inward from ankle
        if left_deviation > 30:
            alerts.append("knee_valgus_left")
        
        if right_deviation > 30:
            alerts.append("knee_valgus_right")
        
        return alerts
    
    def check_spine_rounding(
        self,
        landmarks: List[Dict],
        angles: Dict[str, float]
    ) -> List[str]:
        """Check for rounded spine (flexion)"""
        alerts = []
        
        curvature = self._calculate_spine_curvature(landmarks)
        
        # Threshold for dangerous rounding
        if curvature > 50:
            alerts.append("rounded_back")
        
        return alerts
    
    def check_excessive_lean(self, landmarks: List[Dict]) -> List[str]:
        """Check for excessive forward lean"""
        alerts = []
        
        lean = self._calculate_torso_lean(landmarks)
        
        # More than 45° forward lean is risky
        if lean > 45:
            alerts.append("excessive_forward_lean")
        
        return alerts
    
    def check_squat_dangers(
        self,
        landmarks: List[Dict],
        angles: Dict[str, float]
    ) -> List[str]:
        """Squat-specific danger checks"""
        alerts = []
        
        # Check for butt wink (pelvis tuck at bottom)
        # (Requires temporal analysis - skip for now)
        
        # Check heel stability
        left_heel = landmarks[self.landmark_idx.LEFT_HEEL]
        left_ankle = landmarks[self.landmark_idx.LEFT_ANKLE]
        
        # If heel lifts, ankle y-coord gets closer to knee
        # (Simplified check)
        
        return alerts
    
    def check_deadlift_dangers(
        self,
        landmarks: List[Dict],
        angles: Dict[str, float]
    ) -> List[str]:
        """Deadlift-specific danger checks"""
        alerts = []
        
        # Critical: spine rounding in deadlift is very dangerous
        curvature = self._calculate_spine_curvature(landmarks)
        if curvature > 40:  # Stricter than general check
            alerts.append("deadlift_rounded_back_critical")
        
        return alerts
    
    def check_bench_dangers(
        self,
        landmarks: List[Dict],
        angles: Dict[str, float]
    ) -> List[str]:
        """Bench press-specific danger checks"""
        alerts = []
        
        # Check for flared elbows (>90° from body)
        # (Would need shoulder-elbow-torso angle)
        
        # Check for uneven press
        left_elbow = angles.get("left_elbow", 180)
        right_elbow = angles.get("right_elbow", 180)
        
        asymmetry = abs(left_elbow - right_elbow)
        if asymmetry > 20:
            alerts.append("uneven_press")
        
        return alerts
    
    def _determine_severity(self, dangers: List[str]) -> str:
        """Determine overall danger severity from list of alerts"""
        
        if not dangers:
            return "none"
        
        # Critical patterns
        critical = ["deadlift_rounded_back_critical", "spine_fracture_risk"]
        if any(d in dangers for d in critical):
            return "critical"
        
        # High severity
        high = ["rounded_back", "knee_valgus_left", "knee_valgus_right"]
        if any(d in dangers for d in high):
            return "high"
        
        # Medium severity
        medium = ["excessive_forward_lean", "uneven_press"]
        if any(d in dangers for d in medium):
            return "medium"
        
        return "low"
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _midpoint(self, p1: Dict, p2: Dict) -> Tuple[float, float]:
        """Calculate midpoint between two landmarks"""
        return ((p1["x"] + p2["x"]) / 2, (p1["y"] + p2["y"]) / 2)
    
    def _midpoint_tuples(
        self, 
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Calculate midpoint between two tuples"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _distance(self, p1: Dict, p2: Dict) -> float:
        """Euclidean distance between two points"""
        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]
        return sqrt(dx**2 + dy**2)