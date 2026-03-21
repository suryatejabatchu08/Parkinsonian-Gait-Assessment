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
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.4
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.4
VISIBILITY_THRESHOLD = 0.05  # Very low — accept all detected landmarks for trajectory extraction

# Only require shoulders + hips — far-side limbs always occluded in sagittal view
REQUIRED_LANDMARKS = [11, 12, 23, 24]

# Use mp.solutions.pose — NOT Tasks API (requires no model file download)
# Pin mediapipe==0.10.14 (last version with solutions API)
POSE_MODEL_PATH = ""   # unused — kept for backward compatibility

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
PEAK_MIN_DISTANCE_FRAMES = 5  # Minimum distance between detected heel strikes
PEAK_PROMINENCE = 0.008       # Prominence for scipy.signal.find_peaks

# ──────────────────────────────────────────────
# PGSI Scoring
# ──────────────────────────────────────────────
PGSI_WEIGHTS = {
    "stride":      0.40,   # strongest discriminator
    "posture":     0.25,
    "variability": 0.35,
    # symmetry and armswing removed — unreliable from sagittal monocular video
}

# Severity thresholds on the 0-100 PGSI scale
SEVERITY_BINS = {
    "Normal":   (0, 33),
    "Mild":     (34, 58),
    "Moderate": (59, 78),
    "Severe":   (79, 100),
}

# Reference values for normalization (used by pgsi_scorer.py)
PGSI_HEALTHY_REF = {
    "stride":      0.55,
    "posture":     5.0,
    "variability": 15.0,
}
PGSI_IMPAIRED_REF = {
    "stride":      0.12,
    "posture":     40.0,
    "variability": 85.0,
}
PGSI_HIGHER_IS_WORSE = {
    "stride":      False,
    "posture":     True,
    "variability": True,
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
