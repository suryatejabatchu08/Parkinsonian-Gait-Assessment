"""
gait_features.py — Stage 3: Biomechanical Feature Extraction

Computes 5 core Parkinsonian gait features from keypoint sequences:
  1. Stride Length
  2. Posture Angle (forward trunk tilt)
  3. Gait Symmetry Index
  4. Step Timing Variability (CV of inter-heel-strike intervals)
  5. Arm Swing Amplitude
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_FPS,
    SMOOTHING_WINDOW,
    SMOOTHING_POLY_ORDER,
    PEAK_MIN_DISTANCE_FRAMES,
    PEAK_PROMINENCE,
    LANDMARKS,
)

# Re-use the KeypointFrame type
from pose.pose_estimator import KeypointFrame, PoseEstimator


# ─────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────

@dataclass
class GaitCycleInfo:
    """Detected gait cycle boundaries (heel-strike frame indices)."""
    left_heel_strikes: np.ndarray   # frame indices
    right_heel_strikes: np.ndarray  # frame indices


@dataclass
class GaitFeatures:
    """Computed biomechanical features for one video session."""
    stride_lengths: np.ndarray           # per-stride values (normalized)
    mean_stride_length: float
    posture_angles: np.ndarray           # per-frame trunk tilt in degrees
    mean_posture_angle: float
    symmetry_index_step_length: float    # SI for step length
    symmetry_index_step_time: float      # SI for step time
    mean_symmetry_index: float
    step_timing_cv: float                # Coefficient of Variation (%)
    arm_swing_left: float                # Mean angular excursion (degrees)
    arm_swing_right: float
    mean_arm_swing: float
    gait_cycles: GaitCycleInfo


class GaitFeatureExtractor:
    """Extracts the 5 PGSI biomechanical features from pose keypoints."""

    def __init__(self, fps: float = TARGET_FPS):
        self.fps = fps

    # ═══════════════════════════════════════════════
    # 0. Utility helpers
    # ═══════════════════════════════════════════════

    @staticmethod
    def _smooth(signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky–Golay smoothing.
        Handles NaN/Inf values by interpolating them first."""
        if len(signal) < SMOOTHING_WINDOW:
            return signal

        # Replace NaN/Inf with interpolated values before smoothing
        finite_mask = np.isfinite(signal)
        if not np.all(finite_mask):
            if np.sum(finite_mask) < 2:
                # Too few valid points — return zeros
                return np.zeros_like(signal)
            indices = np.arange(len(signal))
            signal = np.interp(indices, indices[finite_mask], signal[finite_mask])

        return savgol_filter(signal, SMOOTHING_WINDOW, SMOOTHING_POLY_ORDER)

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _get_trajectory(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
        name: str,
    ) -> np.ndarray:
        """Shortcut to extract and smooth a landmark trajectory."""
        traj = estimator.extract_trajectory(keypoints, name)

        # Safety: replace any remaining NaN/Inf with 0 before smoothing
        traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

        traj[:, 0] = self._smooth(traj[:, 0])
        traj[:, 1] = self._smooth(traj[:, 1])
        return traj

    # ═══════════════════════════════════════════════
    # 1. Gait Cycle Detection (heel-strike events)
    # ═══════════════════════════════════════════════

    def detect_gait_cycles(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
    ) -> GaitCycleInfo:
        """Detect heel-strike events from ankle vertical (y) trajectories.
        Heel strike ≈ local minima in ankle y position (lowest point = foot on ground
        in normalised MediaPipe coords where y increases downward)."""

        left_ankle_traj = self._get_trajectory(estimator, keypoints, "left_ankle")
        right_ankle_traj = self._get_trajectory(estimator, keypoints, "right_ankle")

        # Heel strike = local maxima in y (foot closest to ground in MediaPipe coords)
        left_y = left_ankle_traj[:, 1]
        right_y = right_ankle_traj[:, 1]

        left_peaks, _ = find_peaks(
            left_y, distance=PEAK_MIN_DISTANCE_FRAMES, prominence=PEAK_PROMINENCE
        )
        right_peaks, _ = find_peaks(
            right_y, distance=PEAK_MIN_DISTANCE_FRAMES, prominence=PEAK_PROMINENCE
        )

        return GaitCycleInfo(
            left_heel_strikes=left_peaks,
            right_heel_strikes=right_peaks,
        )

    # ═══════════════════════════════════════════════
    # 2. Stride Length
    # ═══════════════════════════════════════════════

    def compute_stride_lengths(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
        gait_cycles: GaitCycleInfo,
    ) -> np.ndarray:
        """Compute normalized stride lengths.
        Stride = distance between successive ipsilateral heel strikes,
        normalized by hip-to-ankle distance (height proxy)."""

        left_ankle = self._get_trajectory(estimator, keypoints, "left_ankle")
        right_ankle = self._get_trajectory(estimator, keypoints, "right_ankle")
        left_hip = self._get_trajectory(estimator, keypoints, "left_hip")
        right_hip = self._get_trajectory(estimator, keypoints, "right_hip")

        strides: List[float] = []

        # Left strides
        for i in range(1, len(gait_cycles.left_heel_strikes)):
            idx_prev = gait_cycles.left_heel_strikes[i - 1]
            idx_curr = gait_cycles.left_heel_strikes[i]
            dist = self._euclidean(left_ankle[idx_curr], left_ankle[idx_prev])
            # Height proxy: mean hip-to-ankle distance at these frames
            h1 = self._euclidean(left_hip[idx_prev], left_ankle[idx_prev])
            h2 = self._euclidean(left_hip[idx_curr], left_ankle[idx_curr])
            height_proxy = (h1 + h2) / 2.0
            if height_proxy > 0:
                strides.append(dist / height_proxy)

        # Right strides
        for i in range(1, len(gait_cycles.right_heel_strikes)):
            idx_prev = gait_cycles.right_heel_strikes[i - 1]
            idx_curr = gait_cycles.right_heel_strikes[i]
            dist = self._euclidean(right_ankle[idx_curr], right_ankle[idx_prev])
            h1 = self._euclidean(right_hip[idx_prev], right_ankle[idx_prev])
            h2 = self._euclidean(right_hip[idx_curr], right_ankle[idx_curr])
            height_proxy = (h1 + h2) / 2.0
            if height_proxy > 0:
                strides.append(dist / height_proxy)

        return np.array(strides) if strides else np.array([0.0])

    # ═══════════════════════════════════════════════
    # 3. Posture Angle (Forward Trunk Tilt)
    # ═══════════════════════════════════════════════

    def compute_posture_angles(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
    ) -> np.ndarray:
        """Compute per-frame forward trunk tilt angle (degrees).
        Angle between shoulder-midpoint→hip-midpoint vector and vertical axis."""

        ls = self._get_trajectory(estimator, keypoints, "left_shoulder")
        rs = self._get_trajectory(estimator, keypoints, "right_shoulder")
        lh = self._get_trajectory(estimator, keypoints, "left_hip")
        rh = self._get_trajectory(estimator, keypoints, "right_hip")

        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0

        # Vector from hip midpoint to shoulder midpoint
        trunk_vec = shoulder_mid - hip_mid  # (N, 2): [dx, dy]

        # Vertical axis in image coords (pointing upward) = [0, -1]
        # Angle between trunk vector and vertical
        angles = np.degrees(np.arctan2(trunk_vec[:, 0], -trunk_vec[:, 1]))
        angles = np.abs(angles)  # absolute tilt

        return self._smooth(angles)

    # ═══════════════════════════════════════════════
    # 4. Gait Symmetry Index
    # ═══════════════════════════════════════════════

    def compute_symmetry_index(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
        gait_cycles: GaitCycleInfo,
    ) -> Tuple[float, float]:
        """Compute Symmetry Index for step length and step time.
        SI = |L - R| / (0.5 * (L + R)) * 100
        Returns (SI_step_length, SI_step_time)."""

        left_ankle = self._get_trajectory(estimator, keypoints, "left_ankle")
        right_ankle = self._get_trajectory(estimator, keypoints, "right_ankle")

        # Step lengths
        left_step_lens = []
        for i in range(1, len(gait_cycles.left_heel_strikes)):
            d = self._euclidean(
                left_ankle[gait_cycles.left_heel_strikes[i]],
                left_ankle[gait_cycles.left_heel_strikes[i - 1]],
            )
            left_step_lens.append(d)

        right_step_lens = []
        for i in range(1, len(gait_cycles.right_heel_strikes)):
            d = self._euclidean(
                right_ankle[gait_cycles.right_heel_strikes[i]],
                right_ankle[gait_cycles.right_heel_strikes[i - 1]],
            )
            right_step_lens.append(d)

        L_len = np.mean(left_step_lens) if left_step_lens else 0.0
        R_len = np.mean(right_step_lens) if right_step_lens else 0.0
        denom_len = 0.5 * (L_len + R_len)
        SI_length = (abs(L_len - R_len) / denom_len * 100) if denom_len > 0 else 0.0

        # Step times
        left_step_times = np.diff(gait_cycles.left_heel_strikes) / self.fps
        right_step_times = np.diff(gait_cycles.right_heel_strikes) / self.fps
        L_time = np.mean(left_step_times) if len(left_step_times) > 0 else 0.0
        R_time = np.mean(right_step_times) if len(right_step_times) > 0 else 0.0
        denom_time = 0.5 * (L_time + R_time)
        SI_time = (abs(L_time - R_time) / denom_time * 100) if denom_time > 0 else 0.0

        return float(SI_length), float(SI_time)

    # ═══════════════════════════════════════════════
    # 5. Step Timing Variability (CV)
    # ═══════════════════════════════════════════════

    def compute_step_timing_variability(self, gait_cycles: GaitCycleInfo) -> float:
        """Coefficient of Variation of inter-heel-strike intervals.
        CV = (std / mean) * 100"""
        all_intervals: List[float] = []
        for strikes in [gait_cycles.left_heel_strikes, gait_cycles.right_heel_strikes]:
            if len(strikes) > 1:
                intervals = np.diff(strikes) / self.fps
                all_intervals.extend(intervals.tolist())

        if len(all_intervals) < 2:
            return 0.0

        arr = np.array(all_intervals)
        mean_val = np.mean(arr)
        if mean_val == 0:
            return 0.0
        return float(np.std(arr) / mean_val * 100)

    # ═══════════════════════════════════════════════
    # 6. Arm Swing Amplitude
    # ═══════════════════════════════════════════════

    def compute_arm_swing(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
    ) -> Tuple[float, float]:
        """Compute mean arm swing angular excursion (degrees) for left and right arms.
        Arm swing = angular range of wrist-elbow-shoulder angle across gait cycle."""

        def _arm_angle_series(shoulder_traj, elbow_traj, wrist_traj) -> np.ndarray:
            """Compute per-frame elbow angle (degrees) from 3-joint trajectories."""
            v1 = shoulder_traj - elbow_traj  # elbow→shoulder
            v2 = wrist_traj - elbow_traj     # elbow→wrist
            # Angle at elbow
            cos_angle = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))

        ls = self._get_trajectory(estimator, keypoints, "left_shoulder")
        le = self._get_trajectory(estimator, keypoints, "left_elbow")
        lw = self._get_trajectory(estimator, keypoints, "left_wrist")
        rs = self._get_trajectory(estimator, keypoints, "right_shoulder")
        re = self._get_trajectory(estimator, keypoints, "right_elbow")
        rw = self._get_trajectory(estimator, keypoints, "right_wrist")

        left_angles = _arm_angle_series(ls, le, lw)
        right_angles = _arm_angle_series(rs, re, rw)

        # Angular excursion = max - min over sliding windows
        left_swing = float(np.ptp(left_angles)) if len(left_angles) > 0 else 0.0
        right_swing = float(np.ptp(right_angles)) if len(right_angles) > 0 else 0.0

        return left_swing, right_swing

    # ═══════════════════════════════════════════════
    # Master extraction method
    # ═══════════════════════════════════════════════

    def extract_all(
        self,
        estimator: PoseEstimator,
        keypoints: List[Optional[KeypointFrame]],
    ) -> GaitFeatures:
        """Run full feature extraction pipeline and return GaitFeatures."""

        # Step 1: Gait cycle detection
        gait_cycles = self.detect_gait_cycles(estimator, keypoints)

        # Step 2: Stride length
        strides = self.compute_stride_lengths(estimator, keypoints, gait_cycles)

        # Step 3: Posture angle
        posture = self.compute_posture_angles(estimator, keypoints)

        # Step 4: Symmetry
        si_len, si_time = self.compute_symmetry_index(estimator, keypoints, gait_cycles)

        # Step 5: Step timing variability
        timing_cv = self.compute_step_timing_variability(gait_cycles)

        # Step 6: Arm swing
        arm_l, arm_r = self.compute_arm_swing(estimator, keypoints)

        return GaitFeatures(
            stride_lengths=strides,
            mean_stride_length=float(np.mean(strides)),
            posture_angles=posture,
            mean_posture_angle=float(np.mean(posture)),
            symmetry_index_step_length=si_len,
            symmetry_index_step_time=si_time,
            mean_symmetry_index=(si_len + si_time) / 2.0,
            step_timing_cv=timing_cv,
            arm_swing_left=arm_l,
            arm_swing_right=arm_r,
            mean_arm_swing=(arm_l + arm_r) / 2.0,
            gait_cycles=gait_cycles,
        )
