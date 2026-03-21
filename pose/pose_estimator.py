"""
pose_estimator.py — Stage 2: Markerless Pose Estimation

Uses mp.solutions.pose (legacy Solutions API) for 33 full-body keypoints per frame.
Requires mediapipe==0.10.14 (last version with solutions API).
No external model file download needed.
"""
import os
import tempfile

# Streamlit Cloud has a read-only venv — mediapipe cannot write model
# files to its package directory. Redirect cache to /tmp before import.
def _configure_mediapipe_tmp():
    tmp_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ.setdefault("MEDIAPIPE_CACHE_DIR", tmp_dir)
    os.environ.setdefault("HOME", tempfile.gettempdir())

_configure_mediapipe_tmp()


import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    VISIBILITY_THRESHOLD,
    LANDMARKS,
    REQUIRED_LANDMARKS,
)


# ─────────────────────────────────────────────
# Skeleton connection pairs for drawing
# ─────────────────────────────────────────────
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
]


@dataclass
class KeypointFrame:
    """Keypoints for a single frame."""
    frame_index: int
    landmarks: Dict[str, Tuple[float, float, float, float]]  # name → (x, y, z, visibility)
    avg_visibility: float = 0.0
    raw_landmarks: Optional[object] = field(default=None, repr=False)

    def get(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Return (x, y, z, visibility) for named landmark, or None."""
        return self.landmarks.get(name)

    def get_xy(self, name: str) -> Optional[Tuple[float, float]]:
        """Return (x, y) for named landmark, or None if not visible."""
        lm = self.landmarks.get(name)
        if lm is None or lm[3] < 0.05:
            return None
        return (lm[0], lm[1])

    def is_visible(self, name: str) -> bool:
        lm = self.landmarks.get(name)
        return lm is not None and lm[3] >= VISIBILITY_THRESHOLD


class PoseEstimator:
    """MediaPipe mp.solutions.pose wrapper for PGSI keypoint extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_path: str = "",
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        num_poses: int = 1,
    ):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=0,          # fastest model
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._frame_timestamp_ms = 0

    # ── core methods ──────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> Optional[KeypointFrame]:
        """Run pose estimation on a single BGR frame.
        Returns KeypointFrame or None if no pose detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark

        # Quality filter: only require shoulders + hips (sagittal view occludes far-side limbs)
        if any(lms[i].visibility < VISIBILITY_THRESHOLD for i in REQUIRED_LANDMARKS):
            return None

        landmarks_dict: Dict[str, Tuple[float, float, float, float]] = {}
        visibility_sum = 0.0

        for name, idx in LANDMARKS.items():
            lm = lms[idx]
            landmarks_dict[name] = (lm.x, lm.y, lm.z, float(lm.visibility))
            visibility_sum += float(lm.visibility)

        avg_vis = visibility_sum / len(LANDMARKS) if LANDMARKS else 0.0

        return KeypointFrame(
            frame_index=frame_index,
            landmarks=landmarks_dict,
            avg_visibility=avg_vis,
            raw_landmarks=lms,
        )

    def process_video_frames(self, frames: List[np.ndarray]) -> List[Optional[KeypointFrame]]:
        """Process a list of frames, returning per-frame keypoint data.
        Missing detections are represented as None."""
        self._frame_timestamp_ms = 0  # Reset timestamp for new video
        keypoints: List[Optional[KeypointFrame]] = []
        for i, frame in enumerate(frames):
            kf = self.process_frame(frame, frame_index=i)
            keypoints.append(kf)
        return keypoints

    def get_valid_keypoints(
        self, keypoint_sequence: List[Optional[KeypointFrame]]
    ) -> List[KeypointFrame]:
        """Filter out frames where pose was not detected."""
        return [kf for kf in keypoint_sequence if kf is not None]

    # ── visualization helpers ─────────────────────────

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoint_frame: KeypointFrame,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """Draw pose landmarks on a BGR frame and return annotated copy."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if keypoint_frame.raw_landmarks is None:
            return annotated

        landmarks = keypoint_frame.landmarks

        # Draw keypoints as green circles
        for name, (x, y, z, vis) in landmarks.items():
            if vis >= 0.05:
                px = int(x * w)
                py = int(y * h)
                cv2.circle(annotated, (px, py), 4, (0, 255, 0), -1)

        # Draw skeleton connections as blue lines
        if draw_connections:
            for name_a, name_b in POSE_CONNECTIONS:
                lm_a = landmarks.get(name_a)
                lm_b = landmarks.get(name_b)
                if (lm_a and lm_b and
                        lm_a[3] >= 0.05 and
                        lm_b[3] >= 0.05):
                    pt_a = (int(lm_a[0] * w), int(lm_a[1] * h))
                    pt_b = (int(lm_b[0] * w), int(lm_b[1] * h))
                    cv2.line(annotated, pt_a, pt_b, (255, 0, 0), 2)

        return annotated

    # ── keypoint trajectory extraction ────────────────

    def extract_trajectory(
        self,
        keypoint_sequence: List[Optional[KeypointFrame]],
        landmark_name: str,
    ) -> np.ndarray:
        """Extract (x, y) trajectory for a named landmark.
        Frames with missing data are interpolated linearly.
        Returns shape (N, 2)."""
        n = len(keypoint_sequence)
        traj = np.full((n, 2), np.nan)

        for i, kf in enumerate(keypoint_sequence):
            if kf is not None:
                xy = kf.get_xy(landmark_name)
                if xy is not None:
                    traj[i] = xy

        # Linear interpolation for missing frames
        for col in range(2):
            valid = ~np.isnan(traj[:, col])
            if valid.sum() >= 2:
                indices = np.arange(n)
                traj[:, col] = np.interp(indices, indices[valid], traj[valid, col])

        return traj

    # ── cleanup ──────────────────────────────────────

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
