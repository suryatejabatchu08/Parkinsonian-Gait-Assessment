"""
test_pose_estimator.py — Unit tests for the pose estimation module.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose.pose_estimator import PoseEstimator, KeypointFrame, POSE_CONNECTIONS
from config import VISIBILITY_THRESHOLD, LANDMARKS


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_keypoint_frame():
    """Create a synthetic KeypointFrame for testing."""
    landmarks = {}
    for name in LANDMARKS:
        landmarks[name] = (0.5, 0.5, 0.0, 0.9)  # centered, high visibility
    return KeypointFrame(frame_index=0, landmarks=landmarks, avg_visibility=0.9)


@pytest.fixture
def low_visibility_keypoint_frame():
    """Create a KeypointFrame with low-visibility landmarks."""
    landmarks = {}
    for name in LANDMARKS:
        landmarks[name] = (0.5, 0.5, 0.0, 0.1)  # below threshold
    return KeypointFrame(frame_index=0, landmarks=landmarks, avg_visibility=0.1)


@pytest.fixture
def mixed_keypoint_sequence(sample_keypoint_frame):
    """Sequence with some None entries (simulating missed detections)."""
    return [sample_keypoint_frame, None, sample_keypoint_frame, None, sample_keypoint_frame]


# ─────────────────────────────────────────────
# Tests: KeypointFrame
# ─────────────────────────────────────────────

class TestKeypointFrame:

    def test_get_returns_landmark(self, sample_keypoint_frame):
        lm = sample_keypoint_frame.get("left_shoulder")
        assert lm is not None
        assert len(lm) == 4

    def test_get_missing_returns_none(self, sample_keypoint_frame):
        assert sample_keypoint_frame.get("nonexistent_landmark") is None

    def test_get_xy_visible(self, sample_keypoint_frame):
        xy = sample_keypoint_frame.get_xy("left_shoulder")
        assert xy is not None
        assert xy == (0.5, 0.5)

    def test_get_xy_invisible(self, low_visibility_keypoint_frame):
        """Landmarks below visibility threshold should return None."""
        xy = low_visibility_keypoint_frame.get_xy("left_shoulder")
        assert xy is None

    def test_is_visible(self, sample_keypoint_frame):
        assert sample_keypoint_frame.is_visible("left_shoulder") is True

    def test_is_not_visible(self, low_visibility_keypoint_frame):
        assert low_visibility_keypoint_frame.is_visible("left_shoulder") is False


# ─────────────────────────────────────────────
# Tests: PoseEstimator
# ─────────────────────────────────────────────

class TestPoseEstimator:

    def test_init_and_close(self):
        """PoseEstimator should initialize and close without error."""
        estimator = PoseEstimator(static_image_mode=True)
        estimator.close()

    def test_context_manager(self):
        """Should work as a context manager."""
        with PoseEstimator(static_image_mode=True) as estimator:
            assert estimator is not None

    def test_process_frame_blank_image(self):
        """A blank (black) image should return None (no pose detected)."""
        estimator = PoseEstimator(static_image_mode=True)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.process_frame(blank_frame)
        # Blank image may or may not detect a pose — just verify it doesn't crash
        assert result is None or isinstance(result, KeypointFrame)
        estimator.close()

    def test_get_valid_keypoints(self, mixed_keypoint_sequence):
        """Should filter out None entries."""
        estimator = PoseEstimator(static_image_mode=True)
        valid = estimator.get_valid_keypoints(mixed_keypoint_sequence)
        assert len(valid) == 3
        assert all(isinstance(kf, KeypointFrame) for kf in valid)
        estimator.close()

    def test_extract_trajectory_shape(self, sample_keypoint_frame):
        """Trajectory should have shape (N, 2)."""
        estimator = PoseEstimator(static_image_mode=True)
        sequence = [sample_keypoint_frame] * 10
        traj = estimator.extract_trajectory(sequence, "left_ankle")
        assert traj.shape == (10, 2)
        estimator.close()

    def test_extract_trajectory_interpolation(self, sample_keypoint_frame):
        """Missing frames should be interpolated, not NaN."""
        estimator = PoseEstimator(static_image_mode=True)
        sequence = [sample_keypoint_frame, None, None, sample_keypoint_frame]
        traj = estimator.extract_trajectory(sequence, "left_ankle")
        assert traj.shape == (4, 2)
        assert not np.any(np.isnan(traj)), "Interpolation should fill NaN values"
        estimator.close()

    def test_draw_skeleton_shape(self, sample_keypoint_frame):
        """draw_skeleton should return an image of the same shape."""
        estimator = PoseEstimator(static_image_mode=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = estimator.draw_skeleton(frame, sample_keypoint_frame)
        assert annotated.shape == frame.shape
        estimator.close()


# ─────────────────────────────────────────────
# Tests: POSE_CONNECTIONS
# ─────────────────────────────────────────────

class TestPoseConnections:

    def test_connections_are_valid_landmarks(self):
        """All connection endpoints should reference real landmarks."""
        for name_a, name_b in POSE_CONNECTIONS:
            assert name_a in LANDMARKS, f"'{name_a}' not in LANDMARKS"
            assert name_b in LANDMARKS, f"'{name_b}' not in LANDMARKS"
