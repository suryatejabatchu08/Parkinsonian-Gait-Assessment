"""
config.py — Central configuration for the PGSI system.
All tuneable parameters, paths, and constants live here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
SESSIONS_HISTORY_PATH = os.path.join(OUTPUT_DIR, "sessions_history.json")
REPORT_TEMPLATE_DIR = os.path.join(ROOT_DIR, "reporting", "templates")

# ──────────────────────────────────────────────
# Video Preprocessing
# ──────────────────────────────────────────────
TARGET_FPS = 30
TARGET_WIDTH = 640         # Resize width (aspect-ratio preserved)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
BG_SUBTRACTOR_HISTORY = 500
BG_SUBTRACTOR_THRESHOLD = 16

# ──────────────────────────────────────────────
# Pose Estimation
# ──────────────────────────────────────────────
POSE_MODEL = "mediapipe"   # "mediapipe" or "openpose"
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.3
VISIBILITY_THRESHOLD = 0.3  # Min visibility to trust a keypoint

# Path to the MediaPipe PoseLandmarker .task model file
# Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
POSE_MODEL_PATH = os.path.join(ROOT_DIR, "data", "pose_landmarker.task")

# MediaPipe landmark indices used for PGSI features
LANDMARKS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_foot":      31,
    "right_foot":     32,
}

# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────
SMOOTHING_WINDOW = 5          # Savitzky–Golay window for signal smoothing
SMOOTHING_POLY_ORDER = 2
PEAK_MIN_DISTANCE_FRAMES = 10 # Minimum distance between detected heel strikes
PEAK_PROMINENCE = 0.02        # Prominence for scipy.signal.find_peaks (lowered for side-view)

# ──────────────────────────────────────────────
# PGSI Scoring
# ──────────────────────────────────────────────
# Default weights (equal until optimized against UPDRS data)
PGSI_WEIGHTS = {
    "stride":     0.20,
    "posture":    0.20,
    "symmetry":   0.20,
    "variability": 0.20,
    "armswing":   0.20,
}

# Severity thresholds on the 0-100 PGSI scale
SEVERITY_BINS = {
    "Normal":   (0, 25),
    "Mild":     (26, 50),
    "Moderate": (51, 75),
    "Severe":   (76, 100),
}

# ──────────────────────────────────────────────
# Fall Risk
# ──────────────────────────────────────────────
FALL_RISK_POSTURE_THRESHOLD = 70   # Sub-score above this triggers risk
FALL_RISK_VARIABILITY_THRESHOLD = 70

# ──────────────────────────────────────────────
# Supported video formats
# ──────────────────────────────────────────────
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
