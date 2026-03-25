"""
Exercise Library - Defines all supported exercises with biomechanics rules

This is the source of truth for:
- Which exercises are supported
- Joint angle definitions per exercise
- Form scoring criteria
- Rep counting logic
- Danger patterns

Add new exercises here as we scale.
"""

from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from enum import Enum


class ExerciseCategory(Enum):
    """Exercise categories for organization"""
    COMPOUND_LOWER = "compound_lower"
    COMPOUND_UPPER = "compound_upper"
    COMPOUND_FULL = "compound_full"
    ISOLATION_LOWER = "isolation_lower"
    ISOLATION_UPPER = "isolation_upper"
    BODYWEIGHT = "bodyweight"
    OLYMPIC = "olympic"


class DifficultyLevel(Enum):
    """AI detection difficulty"""
    EASY = "easy"      # Clear landmarks, simple angles
    MEDIUM = "medium"  # Some occlusion, moderate complexity
    HARD = "hard"      # High occlusion, complex patterns


@dataclass
class JointAngleRange:
    """Expected range of motion for a joint in an exercise"""
    joint_name: str
    min_angle: float  # Minimum angle in ROM
    max_angle: float  # Maximum angle in ROM
    optimal_min: float  # Optimal bottom position
    optimal_max: float  # Optimal top position


@dataclass
class RepPhaseDefinition:
    """Defines how to detect rep phases (up/down/eccentric/concentric)"""
    primary_joint: str  # e.g., "knee", "elbow"
    down_threshold: float  # Angle below this = "down" phase
    up_threshold: float    # Angle above this = "up" phase
    secondary_joint: Optional[str] = None  # For complex movements


@dataclass
class DangerPattern:
    """Defines a dangerous movement pattern"""
    name: str
    description: str
    check_function: str  # Name of function in biomechanics_engine
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class ExerciseDefinition:
    """Complete definition of an exercise"""
    id: str
    name: str
    display_name: str
    category: ExerciseCategory
    difficulty: DifficultyLevel
    
    # Biomechanics
    primary_joints: List[str]  # e.g., ["knee", "hip"]
    secondary_joints: List[str]  # e.g., ["ankle", "spine"]
    angle_ranges: List[JointAngleRange]
    
    # Rep counting
    rep_phase: RepPhaseDefinition
    
    # Form scoring weights
    depth_weight: float = 0.30
    symmetry_weight: float = 0.25
    stability_weight: float = 0.20
    tempo_weight: float = 0.15
    alignment_weight: float = 0.10
    
    # Safety
    danger_patterns: List[DangerPattern] = None
    
    # Metadata
    description: str = ""
    tips: List[str] = None
    common_mistakes: List[str] = None
    
    # Verification settings
    requires_weight_verification: bool = True
    min_weight_kg: float = 20.0  # Minimum weight to be competitive
    max_weight_kg: float = 500.0  # Maximum plausible weight


# ============================================================================
# PHASE 1: THE BIG 3
# ============================================================================

SQUAT = ExerciseDefinition(
    id="squat",
    name="squat",
    display_name="Barbell Back Squat",
    category=ExerciseCategory.COMPOUND_LOWER,
    difficulty=DifficultyLevel.MEDIUM,
    
    primary_joints=["knee", "hip"],
    secondary_joints=["ankle", "spine"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="knee",
            min_angle=50.0,   # Deep squat
            max_angle=180.0,  # Standing
            optimal_min=70.0,  # Parallel or below
            optimal_max=170.0
        ),
        JointAngleRange(
            joint_name="hip",
            min_angle=40.0,
            max_angle=180.0,
            optimal_min=60.0,
            optimal_max=170.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="knee",
        down_threshold=90.0,   # Below parallel
        up_threshold=150.0,    # Nearly locked out
        secondary_joint="hip"
    ),
    
    danger_patterns=[
        DangerPattern(
            name="knee_valgus",
            description="Knees caving inward",
            check_function="check_knee_valgus",
            severity="high"
        ),
        DangerPattern(
            name="excessive_forward_lean",
            description="Torso too far forward",
            check_function="check_forward_lean",
            severity="medium"
        ),
        DangerPattern(
            name="heel_lift",
            description="Heels lifting off ground",
            check_function="check_heel_stability",
            severity="medium"
        ),
        DangerPattern(
            name="butt_wink",
            description="Pelvis tucking under at bottom",
            check_function="check_butt_wink",
            severity="low"
        ),
    ],
    
    depth_weight=0.35,  # Depth is critical for squats
    symmetry_weight=0.25,
    stability_weight=0.20,
    tempo_weight=0.10,
    alignment_weight=0.10,
    
    description="The king of lower body exercises. Builds strength in quads, glutes, and core.",
    tips=[
        "Break at hips and knees simultaneously",
        "Keep chest up throughout movement",
        "Drive through heels on ascent",
        "Maintain neutral spine",
    ],
    common_mistakes=[
        "Not reaching parallel depth",
        "Knees caving inward (valgus)",
        "Rising hips faster than shoulders",
        "Looking down during lift",
    ],
    
    requires_weight_verification=True,
    min_weight_kg=20.0,
    max_weight_kg=500.0
)


BENCH_PRESS = ExerciseDefinition(
    id="bench",
    name="bench",
    display_name="Barbell Bench Press",
    category=ExerciseCategory.COMPOUND_UPPER,
    difficulty=DifficultyLevel.EASY,
    
    primary_joints=["elbow", "shoulder"],
    secondary_joints=["wrist"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="elbow",
            min_angle=40.0,   # Bar touches chest
            max_angle=180.0,  # Lockout
            optimal_min=60.0,
            optimal_max=175.0
        ),
        JointAngleRange(
            joint_name="shoulder",
            min_angle=30.0,
            max_angle=180.0,
            optimal_min=45.0,
            optimal_max=170.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="elbow",
        down_threshold=70.0,   # Bar near chest
        up_threshold=160.0,    # Arms extended
    ),
    
    danger_patterns=[
        DangerPattern(
            name="flared_elbows",
            description="Elbows too far from body",
            check_function="check_elbow_flare",
            severity="medium"
        ),
        DangerPattern(
            name="bouncing_bar",
            description="Bouncing bar off chest",
            check_function="check_bar_bounce",
            severity="high"
        ),
        DangerPattern(
            name="uneven_press",
            description="One arm extending faster",
            check_function="check_press_symmetry",
            severity="medium"
        ),
    ],
    
    depth_weight=0.30,
    symmetry_weight=0.30,  # Symmetry critical for bench
    stability_weight=0.15,
    tempo_weight=0.15,
    alignment_weight=0.10,
    
    description="The ultimate upper body strength builder. Targets chest, shoulders, and triceps.",
    tips=[
        "Retract shoulder blades before unracking",
        "Lower bar to mid-chest",
        "Keep wrists straight",
        "Maintain leg drive throughout",
    ],
    common_mistakes=[
        "Elbows flaring too wide",
        "Bouncing bar off chest",
        "Losing shoulder blade retraction",
        "Lifting butt off bench",
    ],
    
    requires_weight_verification=True,
    min_weight_kg=20.0,
    max_weight_kg=400.0
)


DEADLIFT = ExerciseDefinition(
    id="deadlift",
    name="deadlift",
    display_name="Conventional Deadlift",
    category=ExerciseCategory.COMPOUND_FULL,
    difficulty=DifficultyLevel.HARD,
    
    primary_joints=["hip", "knee"],
    secondary_joints=["spine", "shoulder"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="hip",
            min_angle=40.0,   # Starting position
            max_angle=180.0,  # Lockout
            optimal_min=50.0,
            optimal_max=175.0
        ),
        JointAngleRange(
            joint_name="knee",
            min_angle=90.0,   # Starting position
            max_angle=180.0,  # Lockout
            optimal_min=110.0,
            optimal_max=175.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="hip",
        down_threshold=90.0,   # Starting position
        up_threshold=160.0,    # Full lockout
        secondary_joint="knee"
    ),
    
    danger_patterns=[
        DangerPattern(
            name="rounded_back",
            description="Lumbar spine rounding",
            check_function="check_spine_alignment",
            severity="critical"
        ),
        DangerPattern(
            name="hitching",
            description="Pausing and re-pulling during ascent",
            check_function="check_hitch_pattern",
            severity="medium"
        ),
        DangerPattern(
            name="hips_rising_early",
            description="Hips shooting up before bar moves",
            check_function="check_hip_rise",
            severity="high"
        ),
    ],
    
    depth_weight=0.20,
    symmetry_weight=0.20,
    stability_weight=0.20,
    tempo_weight=0.15,
    alignment_weight=0.25,  # Back alignment is critical
    
    description="The king of all exercises. Total body strength from ground to lockout.",
    tips=[
        "Start with hips above knees, shoulders above bar",
        "Maintain neutral spine throughout",
        "Push through floor with legs first",
        "Lock out hips and knees simultaneously",
    ],
    common_mistakes=[
        "Rounding lower back",
        "Starting with hips too high or too low",
        "Letting bar drift forward",
        "Not achieving full lockout",
    ],
    
    requires_weight_verification=True,
    min_weight_kg=40.0,
    max_weight_kg=600.0
)


# ============================================================================
# PHASE 2: COMPOUND MOVEMENTS
# ============================================================================

OVERHEAD_PRESS = ExerciseDefinition(
    id="ohp",
    name="overhead_press",
    display_name="Overhead Press (OHP)",
    category=ExerciseCategory.COMPOUND_UPPER,
    difficulty=DifficultyLevel.MEDIUM,
    
    primary_joints=["elbow", "shoulder"],
    secondary_joints=["spine"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="elbow",
            min_angle=60.0,
            max_angle=180.0,
            optimal_min=80.0,
            optimal_max=175.0
        ),
        JointAngleRange(
            joint_name="shoulder",
            min_angle=90.0,
            max_angle=180.0,
            optimal_min=100.0,
            optimal_max=175.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="elbow",
        down_threshold=90.0,
        up_threshold=165.0,
    ),
    
    danger_patterns=[
        DangerPattern(
            name="excessive_back_arch",
            description="Hyperextending lower back",
            check_function="check_back_arch",
            severity="high"
        ),
    ],
    
    description="Builds shoulder strength and stability.",
    
    requires_weight_verification=True,
    min_weight_kg=20.0,
    max_weight_kg=200.0
)


PULLUP = ExerciseDefinition(
    id="pullup",
    name="pullup",
    display_name="Pull-up",
    category=ExerciseCategory.BODYWEIGHT,
    difficulty=DifficultyLevel.MEDIUM,
    
    primary_joints=["elbow", "shoulder"],
    secondary_joints=[],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="elbow",
            min_angle=40.0,   # Chin over bar
            max_angle=180.0,  # Dead hang
            optimal_min=50.0,
            optimal_max=175.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="elbow",
        down_threshold=60.0,
        up_threshold=165.0,
    ),
    
    danger_patterns=[
        DangerPattern(
            name="kipping",
            description="Using momentum instead of strength",
            check_function="check_kipping",
            severity="medium"
        ),
    ],
    
    depth_weight=0.35,  # Chin over bar is critical
    symmetry_weight=0.30,
    stability_weight=0.20,
    tempo_weight=0.15,
    
    description="The ultimate back and bicep builder.",
    
    requires_weight_verification=False,  # Bodyweight
    min_weight_kg=0.0,
    max_weight_kg=0.0
)


BARBELL_ROW = ExerciseDefinition(
    id="row",
    name="barbell_row",
    display_name="Barbell Row",
    category=ExerciseCategory.COMPOUND_UPPER,
    difficulty=DifficultyLevel.MEDIUM,
    
    primary_joints=["elbow", "shoulder"],
    secondary_joints=["spine", "hip"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="elbow",
            min_angle=40.0,
            max_angle=180.0,
            optimal_min=50.0,
            optimal_max=170.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="elbow",
        down_threshold=60.0,
        up_threshold=160.0,
    ),
    
    danger_patterns=[],
    
    description="Build a thick back with barbell rows.",
    
    requires_weight_verification=True,
    min_weight_kg=20.0,
    max_weight_kg=300.0
)


# ============================================================================
# PHASE 3: ACCESSORY MOVEMENTS
# ============================================================================

LUNGE = ExerciseDefinition(
    id="lunge",
    name="lunge",
    display_name="Walking Lunge",
    category=ExerciseCategory.ISOLATION_LOWER,
    difficulty=DifficultyLevel.MEDIUM,
    
    primary_joints=["knee", "hip"],
    secondary_joints=["ankle"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="knee",
            min_angle=80.0,
            max_angle=180.0,
            optimal_min=90.0,
            optimal_max=175.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="knee",
        down_threshold=100.0,
        up_threshold=160.0,
    ),
    
    danger_patterns=[],
    
    description="Single-leg strength and stability.",
    
    requires_weight_verification=True,
    min_weight_kg=0.0,
    max_weight_kg=200.0
)


PUSHUP = ExerciseDefinition(
    id="pushup",
    name="pushup",
    display_name="Push-up",
    category=ExerciseCategory.BODYWEIGHT,
    difficulty=DifficultyLevel.EASY,
    
    primary_joints=["elbow"],
    secondary_joints=["shoulder"],
    
    angle_ranges=[
        JointAngleRange(
            joint_name="elbow",
            min_angle=60.0,
            max_angle=180.0,
            optimal_min=70.0,
            optimal_max=175.0
        ),
    ],
    
    rep_phase=RepPhaseDefinition(
        primary_joint="elbow",
        down_threshold=80.0,
        up_threshold=160.0,
    ),
    
    danger_patterns=[
        DangerPattern(
            name="sagging_hips",
            description="Hips dropping below shoulders",
            check_function="check_plank_alignment",
            severity="medium"
        ),
    ],
    
    description="Classic upper body bodyweight exercise.",
    
    requires_weight_verification=False,
    min_weight_kg=0.0,
    max_weight_kg=0.0
)


# ============================================================================
# EXERCISE REGISTRY
# ============================================================================

EXERCISE_REGISTRY: Dict[str, ExerciseDefinition] = {
    # Phase 1: Big 3
    "squat": SQUAT,
    "bench": BENCH_PRESS,
    "deadlift": DEADLIFT,
    
    # Phase 2: Compounds
    "overhead_press": OVERHEAD_PRESS,
    "pullup": PULLUP,
    "barbell_row": BARBELL_ROW,
    
    # Phase 3: Accessories
    "lunge": LUNGE,
    "pushup": PUSHUP,
}


def get_exercise(exercise_id: str) -> Optional[ExerciseDefinition]:
    """Get exercise definition by ID"""
    return EXERCISE_REGISTRY.get(exercise_id)


def get_all_exercises() -> List[ExerciseDefinition]:
    """Get all available exercises"""
    return list(EXERCISE_REGISTRY.values())


def get_exercises_by_category(category: ExerciseCategory) -> List[ExerciseDefinition]:
    """Get exercises filtered by category"""
    return [ex for ex in EXERCISE_REGISTRY.values() if ex.category == category]


def get_phase_1_exercises() -> List[ExerciseDefinition]:
    """Get Phase 1 exercises (The Big 3)"""
    return [SQUAT, BENCH_PRESS, DEADLIFT]


def is_exercise_supported(exercise_id: str) -> bool:
    """Check if exercise is supported"""
    return exercise_id in EXERCISE_REGISTRY


# ============================================================================
# EXERCISE DETECTION (AI)
# ============================================================================

def detect_exercise_from_pose(angles: Dict[str, float], landmarks: List[Dict]) -> str:
    """
    Detect which exercise is being performed based on pose
    
    This is a heuristic-based classifier. In production, consider:
    - Training a small CNN classifier on pose sequences
    - Using temporal patterns (not just single frame)
    - Adding confidence scores
    """
    
    # Extract key angles
    left_knee = angles.get("left_knee", 180)
    right_knee = angles.get("right_knee", 180)
    left_elbow = angles.get("left_elbow", 180)
    right_elbow = angles.get("right_elbow", 180)
    left_hip = angles.get("left_hip", 180)
    right_hip = angles.get("right_hip", 180)
    
    avg_knee = (left_knee + right_knee) / 2
    avg_elbow = (left_elbow + right_elbow) / 2
    avg_hip = (left_hip + right_hip) / 2
    
    # Get landmark positions for context
    try:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip_lm = landmarks[23]
        right_hip_lm = landmarks[24]
        
        # Calculate torso angle (vertical = 0, horizontal = 90)
        shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        hip_y = (left_hip_lm["y"] + right_hip_lm["y"]) / 2
        torso_vertical_dist = abs(shoulder_y - hip_y)
        
    except:
        torso_vertical_dist = 100  # Default
    
    # Detection logic
    
    # SQUAT: Knees bent, torso relatively upright
    if avg_knee < 140 and avg_hip < 140 and torso_vertical_dist > 50:
        return "squat"
    
    # DEADLIFT: Hips lower than shoulders, knees slightly bent
    if avg_hip < 120 and avg_knee > 120 and torso_vertical_dist > 40:
        return "deadlift"
    
    # BENCH PRESS: Horizontal torso, elbows bending
    if avg_elbow < 140 and torso_vertical_dist < 40:
        return "bench"
    
    # OVERHEAD PRESS: Standing, arms overhead
    if avg_elbow < 140 and avg_knee > 160 and torso_vertical_dist > 60:
        shoulder_y_avg = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
        wrist_y_avg = (landmarks[15]["y"] + landmarks[16]["y"]) / 2
        if wrist_y_avg < shoulder_y_avg:  # Wrists above shoulders
            return "overhead_press"
    
    # PULL-UP: Arms overhead, elbows bent
    if avg_elbow < 120 and torso_vertical_dist > 60:
        return "pullup"
    
    # LUNGE: One knee bent much more than other
    knee_asymmetry = abs(left_knee - right_knee)
    if knee_asymmetry > 40:
        return "lunge"
    
    # PUSH-UP: Horizontal, elbows bending
    if avg_elbow < 140 and torso_vertical_dist < 30:
        return "pushup"
    
    return "unknown"
