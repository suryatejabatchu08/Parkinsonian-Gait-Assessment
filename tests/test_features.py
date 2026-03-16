"""
test_features.py — Unit tests for biomechanical feature extraction.
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.gait_features import GaitFeatureExtractor, GaitCycleInfo
from pose.pose_estimator import PoseEstimator


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _mock_estimator_with_trajectories(trajectory_map):
    """Create a mock PoseEstimator whose extract_trajectory returns pre-defined arrays."""
    estimator = MagicMock(spec=PoseEstimator)

    def mock_extract(keypoints, name):
        return trajectory_map.get(name, np.zeros((100, 2)))

    estimator.extract_trajectory.side_effect = mock_extract
    return estimator


# ─────────────────────────────────────────────
# Tests: Basic utilities
# ─────────────────────────────────────────────

class TestGaitFeatureExtractor:

    def test_init(self):
        extractor = GaitFeatureExtractor(fps=30)
        assert extractor.fps == 30

    def test_smooth(self):
        signal = np.random.randn(50)
        smoothed = GaitFeatureExtractor._smooth(signal)
        assert len(smoothed) == len(signal)

    def test_smooth_short_signal(self):
        """Short signals should pass through without error."""
        signal = np.array([1.0, 2.0])
        smoothed = GaitFeatureExtractor._smooth(signal)
        np.testing.assert_array_equal(smoothed, signal)

    def test_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert GaitFeatureExtractor._euclidean(a, b) == pytest.approx(5.0)


# ─────────────────────────────────────────────
# Tests: Step timing variability
# ─────────────────────────────────────────────

class TestStepTimingVariability:

    def test_step_timing_variability_empty(self):
        extractor = GaitFeatureExtractor()
        gc = GaitCycleInfo(
            left_heel_strikes=np.array([10]),
            right_heel_strikes=np.array([20]),
        )
        cv = extractor.compute_step_timing_variability(gc)
        assert cv == 0.0

    def test_step_timing_variability(self):
        extractor = GaitFeatureExtractor(fps=30)
        gc = GaitCycleInfo(
            left_heel_strikes=np.array([0, 30, 60, 90]),
            right_heel_strikes=np.array([15, 45, 75]),
        )
        cv = extractor.compute_step_timing_variability(gc)
        # All intervals are equal (30 frames = 1s), so CV ≈ 0
        assert cv == pytest.approx(0.0, abs=1.0)

    def test_step_timing_variability_nonzero(self):
        extractor = GaitFeatureExtractor(fps=30)
        gc = GaitCycleInfo(
            left_heel_strikes=np.array([0, 25, 60, 80]),  # uneven intervals
            right_heel_strikes=np.array([10, 50, 75]),
        )
        cv = extractor.compute_step_timing_variability(gc)
        assert cv > 0  # Should have variability


# ─────────────────────────────────────────────
# Tests: Stride Length
# ─────────────────────────────────────────────

class TestStrideLength:

    def test_stride_length_basic(self):
        """Known stride with known hip-ankle distance should produce expected value."""
        extractor = GaitFeatureExtractor(fps=30)
        n_frames = 100

        # Left ankle moves horizontally with stride
        left_ankle = np.zeros((n_frames, 2))
        left_ankle[:, 0] = np.linspace(0.1, 0.5, n_frames)  # x moves
        left_ankle[:, 1] = 0.9  # y stays at bottom

        # Left hip stays above ankle
        left_hip = np.zeros((n_frames, 2))
        left_hip[:, 0] = np.linspace(0.1, 0.5, n_frames)
        left_hip[:, 1] = 0.5  # hip above ankle

        # Right side similar
        right_ankle = np.copy(left_ankle)
        right_hip = np.copy(left_hip)

        traj_map = {
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
            "left_hip": left_hip,
            "right_hip": right_hip,
        }
        estimator = _mock_estimator_with_trajectories(traj_map)

        gc = GaitCycleInfo(
            left_heel_strikes=np.array([0, 50]),
            right_heel_strikes=np.array([25, 75]),
        )
        strides = extractor.compute_stride_lengths(estimator, [], gc)
        assert len(strides) >= 1
        assert all(s > 0 for s in strides)

    def test_stride_length_no_cycles(self):
        """With only one heel strike, should return [0.0]."""
        extractor = GaitFeatureExtractor(fps=30)
        traj_map = {
            "left_ankle": np.ones((10, 2)) * 0.5,
            "right_ankle": np.ones((10, 2)) * 0.5,
            "left_hip": np.ones((10, 2)) * 0.3,
            "right_hip": np.ones((10, 2)) * 0.3,
        }
        estimator = _mock_estimator_with_trajectories(traj_map)
        gc = GaitCycleInfo(
            left_heel_strikes=np.array([5]),
            right_heel_strikes=np.array([7]),
        )
        strides = extractor.compute_stride_lengths(estimator, [], gc)
        np.testing.assert_array_equal(strides, [0.0])


# ─────────────────────────────────────────────
# Tests: Posture Angle
# ─────────────────────────────────────────────

class TestPostureAngle:

    def test_upright_posture_near_zero(self):
        """Perfectly vertical trunk → angle ≈ 0."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 30
        # Shoulders directly above hips in y-axis
        traj_map = {
            "left_shoulder": np.tile([0.4, 0.3], (n, 1)),
            "right_shoulder": np.tile([0.6, 0.3], (n, 1)),
            "left_hip": np.tile([0.4, 0.6], (n, 1)),
            "right_hip": np.tile([0.6, 0.6], (n, 1)),
        }
        estimator = _mock_estimator_with_trajectories(traj_map)
        angles = extractor.compute_posture_angles(estimator, [])
        assert np.mean(angles) == pytest.approx(0.0, abs=5.0)

    def test_forward_lean_positive(self):
        """Shoulders shifted forward (in x) → positive angle."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 30
        # Shoulder midpoint shifted forward in x relative to hip midpoint
        traj_map = {
            "left_shoulder": np.tile([0.5, 0.3], (n, 1)),
            "right_shoulder": np.tile([0.7, 0.3], (n, 1)),
            "left_hip": np.tile([0.4, 0.6], (n, 1)),
            "right_hip": np.tile([0.6, 0.6], (n, 1)),
        }
        estimator = _mock_estimator_with_trajectories(traj_map)
        angles = extractor.compute_posture_angles(estimator, [])
        assert np.mean(angles) > 0


# ─────────────────────────────────────────────
# Tests: Gait Symmetry Index
# ─────────────────────────────────────────────

class TestSymmetryIndex:

    def test_perfect_symmetry_near_zero(self):
        """Identical L/R step lengths and times → SI ≈ 0."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 100
        # Both ankles have identical trajectories
        ankle_traj = np.zeros((n, 2))
        ankle_traj[:, 0] = np.linspace(0.1, 0.5, n)
        ankle_traj[:, 1] = 0.9

        traj_map = {
            "left_ankle": ankle_traj.copy(),
            "right_ankle": ankle_traj.copy(),
        }
        estimator = _mock_estimator_with_trajectories(traj_map)

        # Identical timing
        gc = GaitCycleInfo(
            left_heel_strikes=np.array([0, 30, 60, 90]),
            right_heel_strikes=np.array([0, 30, 60, 90]),
        )
        si_len, si_time = extractor.compute_symmetry_index(estimator, [], gc)
        assert si_len == pytest.approx(0.0, abs=1.0)
        assert si_time == pytest.approx(0.0, abs=1.0)

    def test_asymmetry_detected(self):
        """Different L/R patterns → SI > 0."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 100
        left_ankle = np.zeros((n, 2))
        left_ankle[:, 0] = np.linspace(0.1, 0.5, n)  # large movement
        left_ankle[:, 1] = 0.9

        right_ankle = np.zeros((n, 2))
        right_ankle[:, 0] = np.linspace(0.1, 0.2, n)  # small movement
        right_ankle[:, 1] = 0.9

        traj_map = {"left_ankle": left_ankle, "right_ankle": right_ankle}
        estimator = _mock_estimator_with_trajectories(traj_map)

        gc = GaitCycleInfo(
            left_heel_strikes=np.array([0, 30, 60, 90]),
            right_heel_strikes=np.array([0, 30, 60, 90]),
        )
        si_len, si_time = extractor.compute_symmetry_index(estimator, [], gc)
        assert si_len > 0


# ─────────────────────────────────────────────
# Tests: Arm Swing
# ─────────────────────────────────────────────

class TestArmSwing:

    def test_static_arm_zero_swing(self):
        """A perfectly still arm should have ~0 angular excursion."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 30
        traj_map = {
            "left_shoulder": np.tile([0.3, 0.3], (n, 1)),
            "left_elbow": np.tile([0.3, 0.5], (n, 1)),
            "left_wrist": np.tile([0.3, 0.7], (n, 1)),
            "right_shoulder": np.tile([0.7, 0.3], (n, 1)),
            "right_elbow": np.tile([0.7, 0.5], (n, 1)),
            "right_wrist": np.tile([0.7, 0.7], (n, 1)),
        }
        estimator = _mock_estimator_with_trajectories(traj_map)
        left_swing, right_swing = extractor.compute_arm_swing(estimator, [])
        assert left_swing == pytest.approx(0.0, abs=1.0)
        assert right_swing == pytest.approx(0.0, abs=1.0)

    def test_swinging_arm_positive(self):
        """An arm that moves should have > 0 angular excursion."""
        extractor = GaitFeatureExtractor(fps=30)
        n = 60
        # Left wrist swings back and forth
        left_wrist = np.zeros((n, 2))
        left_wrist[:, 0] = 0.3 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n))
        left_wrist[:, 1] = 0.7

        traj_map = {
            "left_shoulder": np.tile([0.3, 0.3], (n, 1)),
            "left_elbow": np.tile([0.3, 0.5], (n, 1)),
            "left_wrist": left_wrist,
            "right_shoulder": np.tile([0.7, 0.3], (n, 1)),
            "right_elbow": np.tile([0.7, 0.5], (n, 1)),
            "right_wrist": np.tile([0.7, 0.7], (n, 1)),
        }
        estimator = _mock_estimator_with_trajectories(traj_map)
        left_swing, right_swing = extractor.compute_arm_swing(estimator, [])
        assert left_swing > 0

