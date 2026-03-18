"""
pose_estimator.py — Stage 2: Markerless Pose Estimation

Uses MediaPipe PoseLandmarker (Tasks API) to extract 33 full-body keypoints per frame.
Compatible with mediapipe >= 0.10.9 which removed the legacy mp.solutions API.

Model file required: data/pose_landmarker.task
Download: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    VISIBILITY_THRESHOLD,
    LANDMARKS,
    POSE_MODEL_PATH,
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
        if lm is None or lm[3] < VISIBILITY_THRESHOLD:
            return None
        return (lm[0], lm[1])

    def is_visible(self, name: str) -> bool:
        lm = self.landmarks.get(name)
        return lm is not None and lm[3] >= VISIBILITY_THRESHOLD


def _download_model_if_needed(model_path: str) -> str:
    """Download the PoseLandmarker model if it doesn't exist locally."""
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task"
    )
    print(f"[PGSI] Downloading PoseLandmarker model to {model_path} ...")
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    print("[PGSI] Model downloaded successfully.")
    return model_path


class PoseEstimator:
    """MediaPipe PoseLandmarker (Tasks API) wrapper for PGSI keypoint extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_path: str = POSE_MODEL_PATH,
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        num_poses: int = 1,
    ):
        # Auto-download model if not present
        model_path = _download_model_if_needed(model_path)

        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self._frame_timestamp_ms = 0

    # ── core methods ──────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> Optional[KeypointFrame]:
        """Run pose estimation on a single BGR frame.
        Returns KeypointFrame or None if no pose detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == RunningMode.IMAGE:
            results = self.landmarker.detect(mp_image)
        else:
            # VIDEO mode requires monotonically increasing timestamps
            self._frame_timestamp_ms += 33  # ~30 FPS
            results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        pose_lms = results.pose_landmarks[0]  # First (and usually only) person

        landmarks_dict: Dict[str, Tuple[float, float, float, float]] = {}
        visibility_sum = 0.0

        for name, idx in LANDMARKS.items():
            lm = pose_lms[idx]
            # Tasks API uses .visibility and .presence attributes
            vis = getattr(lm, 'visibility', None)
            if vis is None:
                vis = getattr(lm, 'presence', 0.5)
            landmarks_dict[name] = (lm.x, lm.y, lm.z, float(vis))
            visibility_sum += float(vis)

        avg_vis = visibility_sum / len(LANDMARKS) if LANDMARKS else 0.0

        return KeypointFrame(
            frame_index=frame_index,
            landmarks=landmarks_dict,
            avg_visibility=avg_vis,
            raw_landmarks=pose_lms,
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
        """Draw pose landmarks on a BGR frame and return annotated copy.
        Uses OpenCV drawing (mp.solutions.drawing_utils was removed in newer mediapipe)."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if keypoint_frame.raw_landmarks is None:
            return annotated

        landmarks = keypoint_frame.landmarks

        # Draw keypoints as green circles
        for name, (x, y, z, vis) in landmarks.items():
            if vis >= VISIBILITY_THRESHOLD:
                px = int(x * w)
                py = int(y * h)
                cv2.circle(annotated, (px, py), 4, (0, 255, 0), -1)

        # Draw skeleton connections as blue lines
        if draw_connections:
            for name_a, name_b in POSE_CONNECTIONS:
                lm_a = landmarks.get(name_a)
                lm_b = landmarks.get(name_b)
                if (lm_a and lm_b and
                        lm_a[3] >= VISIBILITY_THRESHOLD and
                        lm_b[3] >= VISIBILITY_THRESHOLD):
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
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
